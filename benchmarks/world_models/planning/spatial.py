r"""Causal action-conditioned dynamics on the frozen encoder's 8x8 patch tokens.

The spatial level is the trained LeWM encoder's patch-token grid, frozen.
A transformer predicts the next grid as a same-grid delta, ``s_{t+1} = s_t +
f(s_{<=t}, a_t)``, with each action applied through AdaLN-zero to every token
of its frame and a block-causal mask over frames. It trains on its own
rollouts: each clip starts from ``history`` encoded grids, rolls out
``rollout`` steps feeding predictions back, and the loss averages the error at
every step, so the model learns to stay on the data manifold over a planning
horizon instead of only one step ahead.

Patch tokens are precomputed once for the whole dataset and standardized per
channel, so the loss weighs channels equally.

Example::

    python benchmarks/world_models/planning/spatial.py --lewm runs/lewm/model.pt \\
        --data data/train.pt --test-data data/test.pt --out runs/spatial
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from stable_pretraining.methods.hlewm import _sincos
from world_model import LeWM, _CausalAdaLNBlock, to_float


@torch.no_grad()
def encode_patches(model: LeWM, video: torch.Tensor, batch: int = 512) -> torch.Tensor:
    """Patch tokens ``[N, T, G*G, D]`` (float16) of uint8 ``video`` ``[N, T, 3, H, W]``."""
    model.eval()
    flat = video.flatten(0, 1)
    out = []
    for i in range(0, len(flat), batch):
        out.append(model.encoder(to_float(flat[i : i + batch]))[1].half())
    out = torch.cat(out)
    return out.reshape(*video.shape[:2], *out.shape[1:])


class SpatialDynamics(nn.Module):
    """Predicts the next patch-token grid from the last ``history`` grids and actions.

    Args:
        token_dim: Patch token size (the encoder width).
        grid: Grid side.
        action_dim: Action size per model step.
        dim: Transformer width.
        depth: Transformer depth.
        heads: Attention heads.
        history: Context grids per prediction.
    """

    def __init__(
        self,
        token_dim: int = 128,
        grid: int = 8,
        action_dim: int = 10,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
        history: int = 1,
    ):
        super().__init__()
        self.grid, self.history = grid, history
        self.tokens = grid * grid
        self.inp = nn.Linear(token_dim, dim)
        rowcol = torch.cartesian_prod(torch.arange(grid), torch.arange(grid))
        self.register_buffer("pos", _sincos(rowcol, dim), persistent=False)
        self.time_embed = nn.Parameter(torch.randn(history, 1, dim) * 0.02)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
        )
        self.blocks = nn.ModuleList(
            [_CausalAdaLNBlock(dim, heads, 4 * dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, token_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        # Standardization of the frozen tokens, set from data by fit_scale().
        self.register_buffer("mean", torch.zeros(token_dim))
        self.register_buffer("std", torch.ones(token_dim))

    def fit_scale(self, tokens: torch.Tensor):
        flat = tokens.reshape(-1, tokens.shape[-1]).float()
        self.mean.copy_(flat.mean(0))
        self.std.copy_(flat.std(0).clamp_min(1e-4))

    def normalize(self, tokens: torch.Tensor) -> torch.Tensor:
        return (tokens.float() - self.mean) / self.std

    def step(self, grids: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Next normalized grid ``[B, G*G, D]``.

        Args:
            grids: Normalized context grids ``[B, L, G*G, D]``, ``L <= history``.
            actions: ``[B, L, action_dim]``, the action taken at each grid.
        """
        b, length, n = grids.shape[:3]
        x = self.inp(grids) + self.pos + self.time_embed[:length]
        c = self.action_encoder(actions)[:, :, None].expand(-1, -1, n, -1)
        frame = torch.arange(length, device=grids.device).repeat_interleave(n)
        mask = frame[:, None] >= frame[None, :]
        x, c = x.flatten(1, 2), c.flatten(1, 2)
        for block in self.blocks:
            x = block(x, c, mask)
        delta = self.out(self.norm(x[:, -n:]))
        return grids[:, -1] + delta

    def rollout(
        self,
        grids: torch.Tensor,
        actions: torch.Tensor,
        past_actions: torch.Tensor = None,
    ) -> torch.Tensor:
        """Predicted normalized grids ``[B, K, G*G, D]`` for ``actions`` ``[B, K, A]``.

        ``grids`` ``[B, h, G*G, D]`` are normalized context grids, most recent
        last; ``past_actions`` ``[B, h - 1, A]`` led between them.
        """
        h = grids.shape[1]
        acts = (
            actions if past_actions is None else torch.cat([past_actions, actions], 1)
        )
        states = list(grids.unbind(1))
        for _ in range(actions.shape[1]):
            lo = max(0, len(states) - self.history)
            states.append(
                self.step(torch.stack(states[lo:], 1), acts[:, lo : len(states)])
            )
        return torch.stack(states[h:], 1)


def sample_clips(tokens, actions, n, length, g):
    episodes, frames = tokens.shape[:2]
    idx = torch.randint(episodes, (n,), generator=g)
    start = torch.randint(frames - length + 1, (n,), generator=g)
    t = start[:, None] + torch.arange(length)
    return tokens[idx[:, None], t], actions[idx[:, None], t[:, :-1]]


def rollout_loss(model, tokens, actions, rollout: int) -> torch.Tensor:
    """Mean squared error per rollout step ``[rollout]`` over normalized tokens."""
    grids = model.normalize(tokens)
    h = model.history
    pred = model.rollout(grids[:, :h], actions[:, h - 1 :], actions[:, : h - 1])
    return (pred - grids[:, h : h + rollout]).pow(2).mean((0, 2, 3))


def train(model, tokens, actions, args, ckpt_path: Path) -> dict:
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
        state = saved["state"]
        print(f"resumed at step {state['step']}", flush=True)
    model.train()
    length = model.history + args.rollout
    running = torch.zeros(args.rollout)
    t0 = time.time() - state["seconds"]
    for step in range(state["step"] + 1, args.steps + 1):
        clip, acts = sample_clips(tokens, actions, args.batch, length, g)
        per_step = rollout_loss(model, clip, acts, args.rollout)
        loss = per_step.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += per_step.detach()
        if step % args.log_every == 0:
            row = {"step": step, "seconds": time.time() - t0}
            row["loss_per_step"] = (running / args.log_every).tolist()
            state["history"].append(row)
            cells = " ".join(f"{v:.3f}" for v in row["loss_per_step"])
            print(
                f"step {step:6d}  loss/step {cells}  {row['seconds']:.0f}s", flush=True
            )
            running.zero_()
        if step % args.ckpt_every == 0 or step == args.steps:
            state.update(step=step, seconds=time.time() - t0)
            tmp = ckpt_path.with_suffix(".tmp")
            torch.save(
                {
                    "model": model.state_dict(),
                    "opt": opt.state_dict(),
                    "sched": sched.state_dict(),
                    "gen": g.get_state(),
                    "state": state,
                },
                tmp,
            )
            os.replace(tmp, ckpt_path)
    return state


@torch.no_grad()
def validate(model, tokens, actions, horizon: int, n: int = 512) -> dict:
    """Rollout error per step against copying the last grid, on held-out clips."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    clip, acts = sample_clips(tokens, actions, n, model.history + horizon, g)
    grids = model.normalize(clip)
    h = model.history
    pred = model.rollout(grids[:, :h], acts[:, h - 1 :], acts[:, : h - 1])
    truth = grids[:, h:]
    return {
        "rollout_mse": (pred - truth).pow(2).mean((0, 2, 3)).tolist(),
        "copy_last_mse": (grids[:, h - 1 : h] - truth).pow(2).mean((0, 2, 3)).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--lewm", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--test-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--history", type=int, default=1)
    parser.add_argument("--rollout", type=int, default=4)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=250)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)

    lewm = LeWM()
    lewm.load_state_dict(torch.load(args.lewm))
    cache = {}
    for split, path in (("train", args.data), ("test", args.test_data)):
        cached = args.out / f"{split}_tokens.pt"
        data = torch.load(path)
        if not cached.exists():
            t0 = time.time()
            torch.save(encode_patches(lewm, data["video"]), cached)
            print(f"encoded {split} in {time.time() - t0:.0f}s", flush=True)
        cache[split] = (torch.load(cached), data["action"])
        del data

    torch.manual_seed(args.seed)
    tokens, actions = cache["train"]
    model = SpatialDynamics(token_dim=tokens.shape[-1], history=args.history)
    model.fit_scale(tokens[:, ::4])
    print(
        f"spatial dynamics: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M "
        f"params, history {args.history}, rollout {args.rollout}",
        flush=True,
    )
    state = train(model, tokens, actions, args, args.out / "ckpt.pt")
    torch.save(model.state_dict(), args.out / "model.pt")
    val = validate(model, *cache["test"], horizon=8)
    print("val", json.dumps(val), flush=True)
    result = {"args": {k: str(v) for k, v in vars(args).items()}, "train": state}
    (args.out / "train.json").write_text(json.dumps({**result, "val": val}))


if __name__ == "__main__":
    main()
