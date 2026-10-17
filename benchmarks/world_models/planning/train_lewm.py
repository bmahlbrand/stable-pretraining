"""Train the action-conditioned LeWM on TwoRoom episodes.

Resumes from ``<out>/ckpt.pt`` if it exists, so a run can be restarted with
the same command. Writes ``<out>/model.pt`` and ``<out>/train.json``.

Example::

    python benchmarks/world_models/planning/train_lewm.py --data data/train.pt --out runs/lewm
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from world_model import LeWM


def sample_clips(
    data: dict, n: int, length: int, g: torch.Generator, per_episode: int = 1
) -> tuple:
    """Random ``length``-frame windows and the actions between their frames.

    With ``per_episode > 1`` the ``n`` windows come from ``n / per_episode``
    episodes, ``per_episode`` windows each, so every batch holds several agent
    positions per door layout.
    """
    episodes, frames = data["video"].shape[:2]
    idx = torch.randint(episodes, (n // per_episode,), generator=g)
    idx = idx.repeat_interleave(per_episode)
    start = torch.randint(frames - length + 1, (n,), generator=g)
    t = start[:, None] + torch.arange(length)
    return data["video"][idx[:, None], t], data["action"][idx[:, None], t[:, :-1]]


def train(model, data, args, ckpt_path: Path) -> dict:
    g = torch.Generator().manual_seed(args.seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    warmup = max(1, args.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * s / args.steps))
        ),
    )
    state = {"history": [], "step": 0, "seconds": 0.0}
    if ckpt_path.exists():
        saved = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        g.set_state(saved["gen"])
        torch.set_rng_state(saved["rng"])
        state = saved["state"]
        print(f"resumed at step {state['step']}", flush=True)

    model.train()
    running = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    t0 = time.time() - state["seconds"]
    for step in range(state["step"] + 1, args.steps + 1):
        video, actions = sample_clips(
            data, args.batch, model.history + 1, g, args.clips_per_episode
        )
        out = model(video, actions)
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        for k in running:
            running[k] += float(getattr(out, k).detach())
        if step % args.log_every == 0:
            row = {k: v / args.log_every for k, v in running.items()}
            row.update(step=step, seconds=time.time() - t0)
            state["history"].append(row)
            print(
                f"step {step:6d}  loss {row['loss']:.4f}  pred {row['pred_loss']:.4f}"
                f"  sigreg {row['sigreg_loss']:.3f}  {row['seconds']:.0f}s",
                flush=True,
            )
            running = dict.fromkeys(running, 0.0)
        if step % args.ckpt_every == 0 or step == args.steps:
            state.update(step=step, seconds=time.time() - t0)
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "gen": g.get_state(),
                    "rng": torch.get_rng_state(),
                    "state": state,
                },
                tmp,
            )
            os.replace(tmp, ckpt_path)
    return state


@torch.no_grad()
def validate(model, data, n: int = 1024) -> dict:
    """Teacher-forced prediction loss on held-out clips, and a copy-last baseline."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    video, actions = sample_clips(data, n, model.history + 1, g)
    pred, copy = [], []
    for i in range(0, n, 128):
        z, _ = model.encode(video[i : i + 128])
        p = model.predict(z[:, :-1], actions[i : i + 128])
        pred.append((p - z[:, 1:]).pow(2).mean((0, 2)))
        copy.append((z[:, :-1] - z[:, 1:]).pow(2).mean((0, 2)))
    pred = torch.stack(pred).mean(0)
    copy = torch.stack(copy).mean(0)
    return {"pred_loss": pred.tolist(), "copy_last_loss": copy.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0.09)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument(
        "--clips-per-episode",
        type=int,
        default=1,
        help="Windows per episode in a batch; SIGReg then sees several agent "
        "positions per door layout.",
    )
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=250)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    data = torch.load(args.data)
    torch.manual_seed(args.seed)
    model = LeWM(
        action_dim=data["action"].shape[-1], history=args.history, lamb=args.lamb
    )
    print(
        f"LeWM: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{len(data['video'])} episodes",
        flush=True,
    )
    state = train(model, data, args, args.out / "ckpt.pt")
    torch.save(model.state_dict(), args.out / "model.pt")
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "train": state}
    if args.test_data:
        result["val"] = validate(model, torch.load(args.test_data))
        print("val", result["val"], flush=True)
    (args.out / "train.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
