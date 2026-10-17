r"""Score open-loop rollouts of the global and spatial dynamics through probes.

Both models start from the same held-out frames and roll out the recorded
actions. Probes fit on each representation's encodings of real frames decode
the predicted states into the agent's position and the passable stretches of
the wall, so the global state and the 8x8 grid are scored in the same units.
Errors are also split by whether the agent hit a wall or the border during
the step, since predicting contact is what the door geometry is needed for.

Example::

    python benchmarks/world_models/planning/rollout_eval.py --lewm runs/lewm/model.pt \\
        --spatial runs/spatial --train-data data/train.pt --test-data data/test.pt
"""

import argparse
import json
from pathlib import Path

import torch

from probes import apply_probe, features, fit_probe, frame_targets, passable_bins
from spatial import SpatialDynamics, encode_patches
from two_room import FRAMESKIP
from world_model import LeWM

CONTEXT = 3


def contact(agent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    """Steps ``[N, T]`` where the agent moved more than 4 px less than commanded."""
    speed = 5.0
    commanded = actions.reshape(*actions.shape[:-1], FRAMESKIP, 2).clamp(-1, 1)
    commanded = commanded.sum(-2) * speed
    moved = agent[:, 1:] - agent[:, :-1]
    return (moved - commanded).norm(dim=-1) > 4.0


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lewm", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True, help="Run directory.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    lewm = LeWM().eval()
    lewm.load_state_dict(torch.load(args.lewm))
    spatial_args = json.loads((args.spatial / "train.json").read_text())["args"]
    spatial = SpatialDynamics(history=int(spatial_args["history"])).eval()
    spatial.load_state_dict(torch.load(args.spatial / "model.pt"))

    g = torch.Generator().manual_seed(0)
    train = frame_targets(torch.load(args.train_data), 6, g)
    f_train = features(lewm, train["frames"])
    probes = {
        "global": fit_probe("global", f_train["global"], train, args.epochs),
        "patches": fit_probe("patches", f_train["patches"], train, args.epochs),
    }
    del train, f_train

    test = torch.load(args.test_data)
    k = args.horizon
    clip = slice(0, CONTEXT + k)
    video, agent = test["video"][:, clip], test["agent"][:, clip]
    actions = test["action"][:, : CONTEXT + k - 1]
    future = agent[:, CONTEXT:]
    touched = contact(agent, actions)[:, CONTEXT - 1 :]
    doors = passable_bins(test["door_position"], test["door_size"], test["num_doors"])
    bins = doors.any(1)

    z, _ = lewm.encode(video)
    tokens = encode_patches(lewm, video)
    hs = spatial.history
    predicted = {
        "global": lewm.rollout(
            z[:, :CONTEXT], actions[:, CONTEXT - 1 :], actions[:, : CONTEXT - 1]
        ),
        "patches": spatial.rollout(
            spatial.normalize(tokens[:, CONTEXT - hs : CONTEXT]),
            actions[:, CONTEXT - 1 :],
            actions[:, CONTEXT - hs : CONTEXT - 1],
        )
        * spatial.std
        + spatial.mean,
    }
    sources = {
        "global": {
            "rollout": predicted["global"],
            "copy_last": z[:, CONTEXT - 1 : CONTEXT].expand(-1, k, -1),
            "true_frames": z[:, CONTEXT:],
        },
        "patches": {
            "rollout": predicted["patches"],
            "copy_last": tokens[:, CONTEXT - 1 : CONTEXT].float().expand(-1, k, -1, -1),
            "true_frames": tokens[:, CONTEXT:].float(),
        },
    }
    results = {}
    for level, by_source in sources.items():
        for source, x in by_source.items():
            out = apply_probe(probes[level], x.flatten(0, 1)).reshape(len(x), k, -1)
            err = (out[..., :2] - future).norm(dim=-1)
            pred_bins = out[..., 2:] > 0
            iou = (pred_bins & bins[:, None]).sum(-1) / (pred_bins | bins[:, None]).sum(
                -1
            ).clamp_min(1)
            row = {
                "agent_rmse_px": err.pow(2).mean(0).sqrt().tolist(),
                "agent_rmse_contact": [
                    float(err[:, i][touched[:, i]].pow(2).mean().sqrt())
                    for i in range(k)
                ],
                "agent_rmse_free": [
                    float(err[:, i][~touched[:, i]].pow(2).mean().sqrt())
                    for i in range(k)
                ],
                "passable_iou": iou.mean(0).tolist(),
            }
            results[f"{level} {source}"] = row
            cells = " ".join(f"{v:5.1f}" for v in row["agent_rmse_px"])
            print(
                f"{level:8s} {source:12s} agent RMSE px by step: {cells}  "
                f"contact {row['agent_rmse_contact'][0]:.1f}/{row['agent_rmse_contact'][-1]:.1f}"
                f"  free {row['agent_rmse_free'][0]:.1f}/{row['agent_rmse_free'][-1]:.1f}"
                f"  door IoU {row['passable_iou'][0]:.2f}/{row['passable_iou'][-1]:.2f}",
                flush=True,
            )
    results["contact_fraction"] = touched.float().mean(0).tolist()
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
