"""Shared evaluation: occupancy probes and probe-decoded latent rollouts.

Any model with ``encode(video) -> [per-level [B, T, tokens, D]]`` and
``rollout(video, num_frames)`` returning per-level embeddings is scored the
same way:

- For every level, a small MLP probe is trained on the encoded frames of
  held-out fit clips to predict the frame's occupancy map: which cells show a
  digit. The same probe architecture is used for every model, so a compact
  embedding is not penalised by a linear read-out. (A frame-level probe can
  locate digits but cannot reliably name them, even from raw pixels, so digit
  class is not scored.)
- The model rolls out from the first ``given`` frames of the test clips and
  the probes decode its predicted embeddings. Every test clip has a digit that
  is hidden behind the occluder at the last given frame and re-emerges during
  the rollout, so finding that digit tests memory.
- References: the probe on the true future frames (read-out ceiling) and
  copying the last given frame.

Scores per horizon, pooled over clips and over cells not under the occluder:
``iou`` of predicted and true occupied cells, ``recall`` of occupied cells,
``false_pos`` on empty cells, and ``reemerged_recall`` on the cells of
re-emerging digits.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from moving_mnist import BACKGROUND, IGNORE


@torch.no_grad()
def encode_clips(model, video: torch.Tensor, batch: int = 32) -> list:
    model.eval()
    parts = [model.encode(video[i : i + batch]) for i in range(0, len(video), batch)]
    return [torch.cat([p[level] for p in parts]) for level in range(len(parts[0]))]


def _flat(z):
    return z.reshape(z.shape[0] * z.shape[1], -1)


class OccupancyProbe:
    """MLP from a frame's embedding to per-cell occupancy logits.

    Args:
        x: Features ``[N, F]``.
        owner: Ownership maps ``[N, cells]`` (``IGNORE`` cells are skipped).
        epochs: Training epochs.
        hidden: Hidden width.
        seed: Seed for initialisation and batching.
    """

    def __init__(self, x, owner, epochs: int = 15, hidden: int = 512, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        torch.manual_seed(seed)
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.net = nn.Sequential(
            nn.Linear(x.shape[1], hidden), nn.GELU(), nn.Linear(hidden, owner.shape[1])
        )
        target = (owner >= 0).float()
        weight = (owner != IGNORE).float()
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)
        xs = (x - self.mu) / self.sd
        for _ in range(epochs):
            for idx in torch.randperm(len(xs), generator=g).split(256):
                loss = F.binary_cross_entropy_with_logits(
                    self.net(xs[idx]), target[idx], weight=weight[idx]
                )
                opt.zero_grad()
                loss.backward()
                opt.step()
        self.net.eval()

    @torch.no_grad()
    def __call__(self, x):
        return self.net((x - self.mu) / self.sd) > 0


def _scores(pred, owner, reemerging):
    """Per-horizon scores; ``pred`` and ``owner`` are ``[clips, frames, cells]``."""
    valid = owner != IGNORE
    occupied = owner >= 0
    empty = owner == BACKGROUND
    back = torch.zeros_like(occupied)
    for digit in range(reemerging.shape[1]):
        back |= (owner == digit) & reemerging[:, digit, None, None]

    def ratio(num, den):
        num, den = num.sum((0, 2)).float(), den.sum((0, 2)).float()
        return torch.where(den > 0, num / den.clamp_min(1), torch.nan).tolist()

    return {
        "iou": ratio(pred & occupied, (pred | occupied) & valid),
        "recall": ratio(pred & occupied, occupied),
        "false_pos": ratio(pred & empty, empty),
        "reemerged_recall": ratio(pred & back, back),
    }


def evaluate(model, data: dict, given: int, horizon: int, num_fit: int) -> dict:
    """Probe and rollout metrics on held-out clips.

    Args:
        model: World model with ``encode`` and ``rollout``.
        data: Fit clips followed by test clips from :class:`MovingMNIST`, with
            ``reemerging`` ``[N, D]`` marking digits that re-emerge in the
            rollout (only read on test clips).
        given: Frames given to the rollout.
        horizon: Frames to roll out and score.
        num_fit: Leading clips used to fit the probes; the rest are test clips.

    Returns:
        Metrics; per-horizon lists are indexed by ``horizon - 1`` and hold
        ``nan`` where a horizon has no cells to score.
    """
    video, owner = data["video"], data["owner"].flatten(2)
    fit, test = slice(0, num_fit), slice(num_fit, None)
    n_test = len(video) - num_fit
    future = owner[test][:, given : given + horizon]
    reemerging = data["reemerging"][test]
    levels = encode_clips(model, video)

    def score(pred):
        return _scores(pred.reshape(n_test, -1, owner.shape[-1]), future, reemerging)

    res = {"level_tokens": [z.shape[2] for z in levels], "true_frames": []}
    probes = []
    for level, z in enumerate(levels):
        probe = OccupancyProbe(
            _flat(z[fit][:, ::2]), _flat(owner[fit][:, ::2]), seed=level
        )
        probes.append(probe)
        true = score(probe(_flat(z[test][:, given : given + horizon])))
        res["true_frames"].append({k: float(np.nanmean(v)) for k, v in true.items()})

    with torch.no_grad():
        rollout = model.rollout(video[test][:, :given], num_frames=given + horizon)
    res["rollout"] = [
        score(probes[level](_flat(z[:, given : given + horizon])))
        for level, z in enumerate(rollout.embeddings)
    ]

    z0 = levels[0][test]
    last = probes[0](_flat(z0[:, given - 1 : given])).reshape(n_test, 1, -1)
    res["reference"] = {
        "probe_on_true_future": score(probes[0](_flat(z0[:, given : given + horizon]))),
        "copy_last_frame": score(last.expand(-1, horizon, -1)),
    }
    return res


def summarize(res: dict, horizons=(1, 2, 4, 8, 16, 32)) -> str:
    n = len(res["reference"]["copy_last_frame"]["iou"])
    hs = [h for h in horizons if h <= n]

    def row(label, values):
        cells = " ".join(f"{values[h - 1]:6.3f}" for h in hs)
        return f"  {label:28s} {cells} | {np.nanmean(values):6.3f}"

    out = [f"tokens per level: {res['level_tokens']}", "probe on true future frames:"]
    for level, scores in enumerate(res["true_frames"]):
        out.append(
            f"  level {level}: " + ", ".join(f"{k} {v:.3f}" for k, v in scores.items())
        )
    for metric in ("iou", "reemerged_recall", "false_pos"):
        out.append(f"{metric} at horizons {hs} | mean")
        for level, scores in enumerate(res["rollout"]):
            out.append(row(f"rollout level {level}", scores[metric]))
        for name, scores in res["reference"].items():
            out.append(row(name, scores[metric]))
    return "\n".join(out)
