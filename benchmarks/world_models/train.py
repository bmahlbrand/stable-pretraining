"""Train and evaluate a world model on Moving MNIST.

Compares HLeWM (MilliVid schedule), HLeWM's equal-budget full-resolution
schedule, and a LeWorldModel baseline under one configuration: same widths and
depths, batch size, step count, optimizer and evaluation. Training resumes from
``<out>/ckpt.pt`` if it exists, so a run can be restarted with the same
command.

Example::

    python benchmarks/world_models/train.py --model hlewm --seed 0 --out runs/hlewm_s0
    python benchmarks/world_models/train.py --model lewm --seed 0 --out runs/lewm_s0
    python benchmarks/world_models/collect_results.py runs
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

from stable_pretraining.methods import HLeWM
from stable_pretraining.methods.hlewm import HierarchicalSchedule

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from evaluate import evaluate, summarize  # noqa: E402
from lewm import LeWM  # noqa: E402
from moving_mnist import MovingMNIST  # noqa: E402
from utils import get_data_dir  # noqa: E402

MODELS = ("hlewm", "hlewm-fullres", "lewm")
SIZE, PATCH = 32, 8
TOKENS_PER_LEVEL = (16, 4, 1)
NUM_FRAMES_PER_LEVEL = (1, 4, 16)
WIDTHS = {
    "embed_dim": 32,
    "encoder_dim": 64,
    "encoder_depth": 4,
    "encoder_heads": 4,
    "predictor_dim": 64,
    "predictor_depth": 4,
    "predictor_heads": 4,
    "predictor_mlp_dim": 256,
    "projector_hidden_dim": 256,
    "n_slices": 64,
}


def build(args):
    torch.manual_seed(args.seed)
    if args.model == "lewm":
        return LeWM(SIZE, PATCH, 1, history=args.history, lamb=args.lamb, **WIDTHS)
    return HLeWM(
        image_size=SIZE,
        patch_size=PATCH,
        num_levels=3,
        in_channels=1,
        predictor_dim_head=WIDTHS["predictor_dim"] // WIDTHS["predictor_heads"],
        predictor_dropout=0.0,
        schedule="millivid" if args.model == "hlewm" else "full_resolution",
        num_frames_per_level=NUM_FRAMES_PER_LEVEL,
        extra_budget=args.extra_budget,
        lamb=args.lamb,
        sigreg_max_positions=64,
        **WIDTHS,
    )


def given_frames(args) -> int:
    """Frames every model gets before rolling out: what HLeWM's schedule needs."""
    schedule = HierarchicalSchedule.millivid(
        NUM_FRAMES_PER_LEVEL, args.extra_budget, TOKENS_PER_LEVEL
    )
    return schedule.first_target_frame


def clip_frames(model) -> int:
    if isinstance(model, LeWM):
        return model.clip_frames + 8
    return model.schedule.chunk_num_frames + 8


def train(model, data, args, device, ckpt_path):
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
        saved = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["opt"])
        sched.load_state_dict(saved["sched"])
        g.set_state(saved["gen"])
        torch.set_rng_state(saved["rng"])
        state = saved["state"]
        print(f"resumed at step {state['step']}", flush=True)

    model.train()
    frames = clip_frames(model)
    running = {"loss": 0.0, "pred_loss": 0.0, "sigreg_loss": 0.0}
    t0 = time.time() - state["seconds"]
    for step in range(state["step"] + 1, args.steps + 1):
        video = data.sample(args.batch, frames, g)["video"].to(device)
        out = model(video)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--out", required=True, help="Run directory.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lamb", type=float, default=0.09)
    parser.add_argument("--extra-budget", type=int, default=48)
    parser.add_argument("--history", type=int, default=4, help="LeWM context.")
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--eval-clips", type=int, default=768)
    parser.add_argument("--fit-clips", type=int, default=512)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--ckpt-every", type=int, default=250)
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    root = get_data_dir("mnist")
    model = build(args).to(device)
    print(
        f"{args.model} seed {args.seed}: "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
        f"{clip_frames(model)}-frame training clips, device {device}",
        flush=True,
    )
    state = train(model, MovingMNIST(root, train=True), args, device, out / "ckpt.pt")

    given = given_frames(args)
    data = MovingMNIST(root, train=False).sample(
        args.eval_clips, given + args.horizon, torch.Generator().manual_seed(1234)
    )
    model.to("cpu")
    metrics = evaluate(model, data, given, args.horizon, args.fit_clips)
    print(f"\n== {args.model} seed {args.seed}, {given} given frames ==")
    print(summarize(metrics), flush=True)
    with open(out / "results.json", "w") as f:
        json.dump(
            {"args": vars(args), "given": given, "train": state, "eval": metrics}, f
        )
    torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()
