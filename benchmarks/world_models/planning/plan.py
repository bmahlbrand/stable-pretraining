"""Goal-reaching on TwoRoom with random doors: MPC with CEM over a world model.

Each episode samples a door layout with the agent and target in different
rooms; the planner sees the current frame and the goal frame (the agent drawn
at the target). Every replan encodes the current frame, runs CEM over
``horizon`` model steps, executes the first model step (``FRAMESKIP``
environment actions) and warm-starts the next replan from the rest of the
plan. An episode succeeds when the agent gets within 16 px of the target
within ``max_replans`` replans.

Planning runs single-threaded, one episode per process, so latency per replan
is comparable across planners.

Example::

    python benchmarks/world_models/planning/plan.py --model runs/lewm/model.pt --episodes 100
"""

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import torch

from two_room import (
    ACTION_DIM,
    FRAMESKIP,
    SUCCESS_DIST,
    ExpertPolicy,
    TwoRoomEnv,
    goal_image,
    observe,
    reset,
)
from world_model import LeWM

PLAN_SEED = 700_000


def cem(cost_fn, info, horizon, samples, iters, topk, mean=None, generator=None):
    """Cross-entropy method over ``[horizon, ACTION_DIM]`` actions in ``[-1, 1]``.

    Follows stable-worldmodel's ``CEMSolver``: Gaussian samples around the
    mean (the mean itself is always candidate 0), refit to the ``topk``
    cheapest. Candidates are clipped to the action range the model was
    trained on.
    """
    if mean is None:
        mean = torch.zeros(horizon, ACTION_DIM)
    std = torch.ones(horizon, ACTION_DIM)
    for _ in range(iters):
        noise = torch.randn(samples, horizon, ACTION_DIM, generator=generator)
        candidates = (mean + std * noise).clamp(-1, 1)
        candidates[0] = mean
        costs = cost_fn(info, candidates[None])[0]
        elite = candidates[costs.topk(topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
    return mean


class LeWMPlanner:
    """CEM on LeWM's global state; cost is the final state's distance to the goal."""

    def __init__(
        self, model: LeWM, horizon=5, samples=300, iters=30, topk=30, history=1
    ):
        self.model = model.eval()
        self.horizon, self.samples, self.iters, self.topk = (
            horizon,
            samples,
            iters,
            topk,
        )
        self.history = history
        self.generator = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def start(self, frame: torch.Tensor, goal: torch.Tensor):
        self.goal_z = self.model.encode(goal[None])[0]
        self.states = [self.model.encode(frame[None])[0][0]]
        self.actions = []
        self.mean = None

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        h = min(self.history, len(self.states))
        info = {"z": torch.stack(self.states[-h:])[None], "goal_z": self.goal_z}
        if h > 1:
            info["past_actions"] = torch.stack(self.actions[-(h - 1) :])[None]
        plan = cem(
            self.model.get_cost,
            info,
            self.horizon,
            self.samples,
            self.iters,
            self.topk,
            self.mean,
            self.generator,
        )
        self.mean = torch.cat([plan[1:], torch.zeros(1, ACTION_DIM)])
        return plan[0]

    @torch.no_grad()
    def observe(self, action: torch.Tensor, frame: torch.Tensor):
        self.actions.append(action)
        self.states.append(self.model.encode(frame[None])[0][0])


def run_episode(planner, env: TwoRoomEnv, seed: int, max_replans: int) -> dict:
    reset(env, seed)
    start_dist = float((env.agent_position - env.target_position).norm())
    planner.start(observe(env), goal_image(env))
    latency = []
    for replan in range(max_replans):
        t0 = time.perf_counter()
        action = planner.act()
        latency.append(time.perf_counter() - t0)
        done = False
        for k in range(FRAMESKIP):
            _, _, terminated, _, _ = env.step(action[2 * k : 2 * k + 2].numpy())
            if terminated:
                done = True
                break
        if done:
            break
        planner.observe(action, observe(env))
    dist = float((env.agent_position - env.target_position).norm())
    return {
        "seed": seed,
        "success": dist < SUCCESS_DIST,
        "replans": replan + 1,
        "latency": float(np.mean(latency)),
        "start_dist": start_dist,
        "final_dist": dist,
    }


def run_reference(policy: str, env: TwoRoomEnv, seed: int, max_steps: int) -> dict:
    """Expert (no noise) or uniformly random actions with the same step budget."""
    seed = reset(env, seed)
    expert = ExpertPolicy(seed=seed)
    expert.set_env(env)
    rng = np.random.default_rng(seed)
    for _ in range(max_steps):
        info = {
            "state": env.agent_position.numpy(),
            "goal_state": env.target_position.numpy(),
        }
        action = (
            expert.get_action(info) if policy == "expert" else rng.uniform(-1, 1, 2)
        )
        _, _, terminated, _, _ = env.step(action)
        if terminated:
            break
    dist = float((env.agent_position - env.target_position).norm())
    return {"seed": seed, "success": dist < SUCCESS_DIST, "final_dist": dist}


def _worker(job):
    torch.set_num_threads(1)
    kind, seeds, cfg = job
    env = TwoRoomEnv()
    if kind in ("expert", "random"):
        return [
            run_reference(kind, env, s, cfg["max_replans"] * FRAMESKIP) for s in seeds
        ]
    model = LeWM(history=cfg["model_history"])
    model.load_state_dict(torch.load(cfg["model"]))
    planner = LeWMPlanner(
        model, cfg["horizon"], cfg["samples"], cfg["iters"], cfg["topk"], cfg["history"]
    )
    return [run_episode(planner, env, s, cfg["max_replans"]) for s in seeds]


def evaluate(kind: str, cfg: dict, episodes: int, workers: int) -> dict:
    seeds = np.arange(PLAN_SEED, PLAN_SEED + episodes)
    jobs = [(kind, c.tolist(), cfg) for c in np.array_split(seeds, workers) if len(c)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.map(_worker, jobs) for r in part]
    summary = {"success_rate": float(np.mean([r["success"] for r in rows]))}
    if "latency" in rows[0]:
        summary["latency_s"] = float(np.mean([r["latency"] for r in rows]))
        wins = [r["replans"] for r in rows if r["success"]]
        summary["replans_to_success"] = float(np.mean(wins)) if wins else None
    return {"summary": summary, "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--reference", choices=("expert", "random"), default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-replans", type=int, default=40)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--history", type=int, default=1, help="Context frames.")
    parser.add_argument("--model-history", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    kind = args.reference or "lewm"
    t0 = time.time()
    result = evaluate(kind, cfg, args.episodes, args.workers)
    result["config"] = cfg
    print(kind, result["summary"], f"{time.time() - t0:.0f}s", flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
