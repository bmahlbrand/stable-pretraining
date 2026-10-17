r"""Map an 8x8 token grid to the global state, for one planning cost across levels.

Distance between token grids saturates once the agent is a few patches from
the goal, so it gives a planner little to follow from afar, while distance
between global states keeps growing with the distance to the goal. This
readout is trained to predict the frozen encoder's global state from its own
patch tokens (no labels). A spatial rollout's final grid then gets the same
cost as a global rollout: squared distance to the goal's global state.

Example::

    python benchmarks/world_models/planning/readout.py --spatial runs/spatial \
        --global-tokens runs/global_rec --out runs/spatial/readout.pt
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from rollout_eval import load_dynamics


class GridReadout(nn.Module):
    """Normalized token grid ``[B, G*G, D]`` -> global state ``[B, embed_dim]``."""

    def __init__(
        self, tokens: int = 64, dim: int = 128, embed_dim: int = 64, reduce: int = 32
    ):
        super().__init__()
        self.reduce = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, reduce))
        self.net = nn.Sequential(
            nn.Linear(tokens * reduce, 512), nn.GELU(), nn.Linear(512, embed_dim)
        )

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        return self.net(self.reduce(grid).flatten(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--spatial", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--global-tokens",
        type=Path,
        required=True,
        help="Run directory of spatial.py --level global (for its cached states).",
    )
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    spatial = load_dynamics(args.spatial)
    data = {}
    for split in ("train", "test"):
        grids = torch.load(args.spatial / f"{split}_tokens.pt").flatten(0, 1)
        states = torch.load(args.global_tokens / f"{split}_tokens.pt").flatten(0, 1)
        data[split] = (grids, states[:, 0].float())

    torch.manual_seed(0)
    readout = GridReadout()
    opt = torch.optim.AdamW(readout.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=args.steps)
    grids, states = data["train"]
    g = torch.Generator().manual_seed(0)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = torch.randint(len(grids), (args.batch,), generator=g)
        loss = F.mse_loss(readout(spatial.normalize(grids[idx])), states[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0:
            print(
                f"step {step}  mse {float(loss.detach()):.4f}  {time.time() - t0:.0f}s"
            )

    readout.eval()
    grids, states = data["test"]
    with torch.no_grad():
        pred = torch.cat(
            [
                readout(spatial.normalize(grids[i : i + 1024]))
                for i in range(0, len(grids), 1024)
            ]
        )
    mse = F.mse_loss(pred, states)
    print(
        f"held-out mse {float(mse):.4f} (state variance {float(states.var(0).mean()):.3f})"
    )
    torch.save(readout.state_dict(), args.out)


if __name__ == "__main__":
    main()
