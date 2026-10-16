"""Shared evaluation: linear probes and probe-decoded latent rollouts.

Any model with ``encode(video) -> [per-level [B, T, tokens, D]]`` and
``rollout(video, num_frames)`` returning per-level embeddings is scored the
same way:

- Linear probes, fit on encoded frames of held-out clips, read the digit
  position (ridge regression) and class (ridge on one-hot targets) at every
  level.
- The model rolls out from the first ``given`` frames; the same probes decode
  its predicted embeddings, scored per horizon against the true state.
- References: the probe applied to the true future frames (read-out floor),
  copying the last given frame, constant velocity from the true state (no
  walls), and predicting the mean position.
"""

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def encode_clips(model, video: torch.Tensor, batch: int = 64) -> list:
    model.eval()
    parts = [model.encode(video[i : i + batch]) for i in range(0, len(video), batch)]
    return [torch.cat([p[level] for p in parts]) for level in range(len(parts[0]))]


class Ridge:
    """Standardized ridge probe; the penalty is picked on a held-out fifth.

    Args:
        x: Features ``[N, F]``.
        y: Targets ``[N, K]``.
        bounds: Optional ``(low, high)`` to clamp predictions to.
    """

    def __init__(self, x, y, bounds=None, lams=(1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)):
        x, y = x.double(), y.double()
        self.bounds = bounds
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        self.y_mu = y.mean(0)
        xs, yc = (x - self.mu) / self.sd, y - self.y_mu
        cut = int(0.8 * len(x))
        errs = [
            ((xs[cut:] @ self._solve(xs[:cut], yc[:cut], lam) - yc[cut:]) ** 2).mean()
            for lam in lams
        ]
        self.lam = lams[int(torch.stack(errs).argmin())]
        self.w = self._solve(xs, yc, self.lam)

    @staticmethod
    def _solve(x, y, lam):
        a = x.T @ x + lam * len(x) * torch.eye(x.shape[1], dtype=x.dtype)
        return torch.linalg.solve(a, x.T @ y)

    def __call__(self, x):
        out = (((x.double() - self.mu) / self.sd) @ self.w + self.y_mu).float()
        return out if self.bounds is None else out.clamp(*self.bounds)


def _flat(z):
    return z.reshape(z.shape[0] * z.shape[1], -1)


@torch.no_grad()
def evaluate(model, data: dict, given: int, horizon: int, num_fit: int) -> dict:
    """Probe and rollout metrics on held-out clips.

    Args:
        model: World model with ``encode`` and ``rollout``.
        data: Clips from :meth:`MovingMNIST.sample` with at least
            ``given + horizon`` frames.
        given: Frames given to the rollout.
        horizon: Frames to roll out and score.
        num_fit: Clips used to fit the probes; the rest are test clips.

    Returns:
        Metrics; per-horizon lists are indexed by ``horizon - 1``.
    """
    video, pos, label = data["video"], data["position"], data["label"]
    size = video.shape[-1]
    levels = encode_clips(model, video)
    fit, test = slice(0, num_fit), slice(num_fit, None)
    n_test, frames = len(video) - num_fit, video.shape[1]
    onehot = F.one_hot(label, 10).float()[:, None].expand(-1, frames, -1)

    res = {"level_tokens": [z.shape[2] for z in levels]}
    pos_probes, cls_probes = [], []
    for key in ("probe_position_px", "probe_class_acc", "smoothness"):
        res[key] = []
    for z in levels:
        pp = Ridge(_flat(z[fit]), pos[fit].reshape(-1, 2), bounds=(0, size - 1))
        cp = Ridge(_flat(z[fit]), onehot[fit].reshape(-1, 10))
        pos_probes.append(pp)
        cls_probes.append(cp)
        zt = z[test]
        err = (pp(_flat(zt)) - pos[test].reshape(-1, 2)).norm(dim=-1).mean()
        acc = (
            cp(_flat(zt)).argmax(-1) == label[test].repeat_interleave(frames)
        ).float()
        res["probe_position_px"].append(float(err))
        res["probe_class_acc"].append(float(acc.mean()))
        res["smoothness"].append(
            float(((zt[:, 1:] - zt[:, :-1]) ** 2).mean() / zt.var(0).mean())
        )

    future = pos[test][:, given : given + horizon]
    y = label[test][:, None]

    def pos_err(p):
        return (p.reshape(n_test, -1, 2) - future).norm(dim=-1).mean(0).tolist()

    def cls_acc(logits):
        return (logits.reshape(n_test, -1, 10).argmax(-1) == y).float().mean(0).tolist()

    rollout = model.rollout(video[test][:, :given], num_frames=given + horizon)
    res["rollout_position_px"], res["rollout_class_acc"] = [], []
    for level, z in enumerate(rollout.embeddings):
        zf = _flat(z[:, given : given + horizon])
        res["rollout_position_px"].append(pos_err(pos_probes[level](zf)))
        res["rollout_class_acc"].append(cls_acc(cls_probes[level](zf)))

    z0 = levels[0][test]
    true_future = _flat(z0[:, given : given + horizon])
    last = _flat(z0[:, given - 1 : given]).repeat_interleave(horizon, dim=0)
    p = pos[test]
    steps = torch.arange(1, horizon + 1, dtype=torch.float32)[None, :, None]
    velocity = (p[:, given - 1] - p[:, given - 2])[:, None]
    res["reference_position_px"] = {
        "probe_on_true_future": pos_err(pos_probes[0](true_future)),
        "copy_last_frame": pos_err(pos_probes[0](last)),
        "constant_velocity_no_walls": pos_err(
            (p[:, given - 1, None] + steps * velocity).clamp(0, size - 1)
        ),
        "mean_position": pos_err(pos[fit].reshape(-1, 2).mean(0).expand_as(future)),
    }
    res["reference_class_acc"] = {
        "probe_on_true_future": cls_acc(cls_probes[0](true_future)),
        "copy_last_frame": cls_acc(cls_probes[0](last)),
    }
    return res


def summarize(res: dict, horizons=(1, 2, 4, 8, 16, 32)) -> str:
    hs = [
        h for h in horizons if h <= len(res["reference_position_px"]["mean_position"])
    ]

    def row(label, values, fmt):
        cells = " ".join(fmt.format(values[h - 1]) for h in hs)
        return f"  {label:30s} {cells} | {fmt.format(np.mean(values))}"

    out = [
        f"tokens per level: {res['level_tokens']}",
        "probe on true frames, position px: "
        + str([round(v, 2) for v in res["probe_position_px"]]),
        "probe on true frames, class acc:   "
        + str([round(v, 3) for v in res["probe_class_acc"]]),
        "copy-prev MSE / var per level:     "
        + str([round(v, 2) for v in res["smoothness"]]),
        f"position error (px) at horizons {hs} | mean",
    ]
    for level, e in enumerate(res["rollout_position_px"]):
        out.append(row(f"rollout level {level}", e, "{:6.2f}"))
    for k, e in res["reference_position_px"].items():
        out.append(row(k, e, "{:6.2f}"))
    out.append(f"class accuracy at horizons {hs} | mean")
    for level, e in enumerate(res["rollout_class_acc"]):
        out.append(row(f"rollout level {level}", e, "{:6.3f}"))
    for k, e in res["reference_class_acc"].items():
        out.append(row(k, e, "{:6.3f}"))
    return "\n".join(out)
