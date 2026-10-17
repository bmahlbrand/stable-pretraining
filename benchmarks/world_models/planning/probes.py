r"""What a frozen TwoRoom encoder keeps: agent position and door openings.

Fits small probes on frozen features of training frames and scores them on
held-out episodes, for the global state, the encoder's ``[CLS]`` token, the
8x8 patch tokens, the same features from an untrained encoder, and raw pixels.

Targets:

- Agent position, scored as RMSE in environment pixels (224x224 frame; the
  agent's radius is 7 and the target counts as reached within 16).
- Where the agent's center can cross the wall: the wall's length is split into
  56 bins of 4 px, and a bin is passable if it overlaps a door's span widened
  by the environment's collision margin. Scored as IoU of passable bins, and
  as recall on each door's bins, separately for doors narrower than the
  expert's fit threshold (passable only with precise alignment) and wider ones.

Example::

    python benchmarks/world_models/planning/probes.py --model runs/lewm/model.pt \\
        --train-data data/train.pt --test-data data/test.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from two_room import ENV_SIZE
from world_model import LeWM, to_float

BINS = 56
DOOR_MARGIN = 1.75  # TwoRoomEnv._apply_collisions
FIT = 1.1 * 7.0  # ExpertPolicy door_fit_margin * agent radius


def passable_bins(position: torch.Tensor, size: torch.Tensor, num: torch.Tensor):
    """Per-door passable bins ``[N, MAX_DOOR, BINS]`` (all False for unused slots)."""
    edges = torch.arange(BINS + 1) * (ENV_SIZE / BINS)
    lo = (position - size - DOOR_MARGIN)[..., None]
    hi = (position + size + DOOR_MARGIN)[..., None]
    overlap = (edges[1:] > lo) & (edges[:-1] < hi)
    used = torch.arange(position.shape[1]) < num[:, None]
    return overlap & used[..., None]


def frame_targets(data: dict, frames_per_episode: int, g: torch.Generator) -> dict:
    """Pick frames and their targets: ``frames``, ``agent``, ``bins``, ``door_bins``."""
    n, t = data["video"].shape[:2]
    idx = torch.arange(n).repeat_interleave(frames_per_episode)
    step = torch.randint(t, (len(idx),), generator=g)
    doors = passable_bins(data["door_position"], data["door_size"], data["num_doors"])
    return {
        "frames": data["video"][idx, step],
        "agent": data["agent"][idx, step],
        "bins": doors.any(1)[idx],
        "door_bins": doors[idx],
        "door_size": data["door_size"][idx],
    }


@torch.no_grad()
def features(model: LeWM, frames: torch.Tensor, batch: int = 256) -> dict:
    """Global state, ``[CLS]`` token and patch tokens of uint8 ``frames``."""
    model.eval()
    out = {"global": [], "cls": [], "patches": []}
    for i in range(0, len(frames), batch):
        x = to_float(frames[i : i + batch])
        cls, patches = model.encoder(x)
        out["global"].append(model.projector(cls))
        out["cls"].append(cls)
        out["patches"].append(patches.half())
    return {k: torch.cat(v) for k, v in out.items()}


class VectorProbe(nn.Module):
    """MLP on one feature vector per frame."""

    def __init__(self, dim: int, out: int, hidden: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(x.float())


class PatchProbe(nn.Module):
    """Per-token linear reduction, then an MLP over the flattened grid."""

    def __init__(
        self, tokens: int, dim: int, out: int, hidden: int = 512, reduce: int = 32
    ):
        super().__init__()
        self.reduce = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, reduce))
        self.net = nn.Sequential(
            nn.Linear(tokens * reduce, hidden), nn.GELU(), nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(self.reduce(x.float()).flatten(1))


class PixelProbe(nn.Module):
    """Small CNN on the frame itself: what is recoverable at all."""

    def __init__(self, out: int, hidden: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.Conv2d(64, 32, 1),
        )  # fmt: skip
        self.net = nn.Sequential(
            nn.Linear(32 * 14 * 14, hidden), nn.GELU(), nn.Linear(hidden, out)
        )

    def forward(self, x):
        return self.net(self.conv(to_float(x)).flatten(1))


def make_probe(name: str, x: torch.Tensor, out: int) -> nn.Module:
    if name == "pixels":
        return PixelProbe(out)
    if x.ndim == 3:
        return PatchProbe(x.shape[1], x.shape[2], out)
    return VectorProbe(x.shape[1], out)


@torch.enable_grad()
def fit_probe(name, x_train, train, epochs=30, seed=0) -> nn.Module:
    """Train one probe for agent position and passable bins on frozen features."""
    torch.manual_seed(seed)
    probe = make_probe(name, x_train, 2 + BINS)
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    steps = epochs * (len(x_train) // 256)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
    agent = train["agent"] / ENV_SIZE - 0.5
    bins = train["bins"].float()
    g = torch.Generator().manual_seed(seed)
    probe.train()
    for _ in range(epochs):
        order = torch.randperm(len(x_train), generator=g)
        for i in range(0, len(order) - 255, 256):
            j = order[i : i + 256]
            out = probe(x_train[j])
            loss = F.mse_loss(out[:, :2], agent[j]) * 10
            loss = loss + F.binary_cross_entropy_with_logits(out[:, 2:], bins[j])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return probe.eval()


@torch.no_grad()
def apply_probe(probe: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Probe outputs: agent (x, y) in environment pixels, then passable-bin logits."""
    out = torch.cat([probe(x[i : i + 512]) for i in range(0, len(x), 512)])
    return torch.cat([(out[:, :2] + 0.5) * ENV_SIZE, out[:, 2:]], -1)


def fit_and_score(name, x_train, x_test, train, test, epochs=30, seed=0) -> dict:
    probe = fit_probe(name, x_train, train, epochs, seed)
    return score(apply_probe(probe, x_test), test)


def score(out: torch.Tensor, test: dict) -> dict:
    agent = out[:, :2]
    rmse = (agent - test["agent"]).pow(2).sum(-1).mean().sqrt()
    pred = out[:, 2:] > 0
    truth = test["bins"]
    iou = (pred & truth).sum(-1) / (pred | truth).sum(-1).clamp_min(1)
    door = test["door_bins"]
    hit = (door & pred[:, None]).sum(-1) / door.sum(-1).clamp_min(1)
    used = door.any(-1)
    small = used & (test["door_size"] < FIT)
    wide = used & (test["door_size"] >= FIT)
    return {
        "agent_rmse_px": float(rmse),
        "passable_iou": float(iou.mean()),
        "small_door_recall": float(hit[small].mean()),
        "wide_door_recall": float(hit[wide].mean()),
        "false_open_rate": float((pred & ~truth).sum() / (~truth).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--train-frames", type=int, default=6, help="Per episode.")
    parser.add_argument("--test-frames", type=int, default=4, help="Per episode.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    g = torch.Generator().manual_seed(0)
    train = frame_targets(torch.load(args.train_data), args.train_frames, g)
    test = frame_targets(torch.load(args.test_data), args.test_frames, g)
    trained = LeWM()
    trained.load_state_dict(torch.load(args.model))
    torch.manual_seed(0)
    untrained = LeWM()

    rows = {}
    for tag, model in (("trained", trained), ("untrained", untrained)):
        f_train = features(model, train["frames"])
        f_test = features(model, test["frames"])
        for name in ("global", "cls", "patches"):
            key = f"{tag} {name}"
            rows[key] = fit_and_score(
                name, f_train[name], f_test[name], train, test, args.epochs
            )
            print(key, rows[key], flush=True)
    rows["pixels"] = fit_and_score(
        "pixels", train["frames"], test["frames"], train, test, args.epochs
    )
    print("pixels", rows["pixels"], flush=True)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
