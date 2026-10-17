"""Goal-reaching on TwoRoom with random doors: MPC with CEM over a world model.

Each episode samples a door layout; the planner sees the current frame and
the goal frame (the agent drawn at the target). Two tasks:

- ``far``: agent and target anywhere in different rooms. Needs routing
  further than a planning horizon, so short-horizon MPC can stall at the wall
  even with perfect dynamics.
- ``near-door``: the start is a few steps before a door on the expert's path
  and the goal ``offset`` steps further along it, so the goal is reachable
  within one horizon but only through the door (the protocol of LeWM's
  evaluation, goals a fixed offset along a dataset trajectory).

Every replan encodes the current frame, runs CEM over
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
    SimDynamics,
    TwoRoomEnv,
    goal_image,
    observe,
    reset,
    reset_near_door,
)
from readout import GridReadout
from rollout_eval import load_dynamics
from world_model import LeWM, to_float

PLAN_SEED = 700_000


def cem(
    cost_fn,
    info,
    horizon,
    samples,
    iters,
    topk,
    mean=None,
    generator=None,
    keep: int = 0,
):
    """Cross-entropy method over ``[horizon, ACTION_DIM]`` actions in ``[-1, 1]``.

    Follows stable-worldmodel's ``CEMSolver``: Gaussian samples around the
    mean (the mean itself is always candidate 0), refit to the ``topk``
    cheapest. Candidates are clipped to the action range the model was
    trained on. With ``keep > 0`` also returns the ``keep`` cheapest
    candidates seen in any iteration, cheapest first, so early, broad
    iterations contribute alternatives the final distribution has abandoned.
    """
    if mean is None:
        mean = torch.zeros(horizon, ACTION_DIM)
    std = torch.ones(horizon, ACTION_DIM)
    seen, seen_costs = [], []
    for _ in range(iters):
        noise = torch.randn(samples, horizon, ACTION_DIM, generator=generator)
        candidates = (mean + std * noise).clamp(-1, 1)
        candidates[0] = mean
        costs = cost_fn(info, candidates[None])[0]
        elite = candidates[costs.topk(topk, largest=False).indices]
        mean, std = elite.mean(0), elite.std(0)
        if keep:
            seen.append(candidates)
            seen_costs.append(costs)
    if not keep:
        return mean
    order = torch.cat(seen_costs).argsort()[:keep]
    return mean, torch.cat(seen)[order]


class GlobalModel:
    """LeWM's own predictor on the global state: the baseline's model and cost."""

    def __init__(self, lewm: LeWM, history: int = 1):
        self.lewm, self.history = lewm.eval(), history

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.lewm.encode(frame[None])[0][0]

    def cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        return self.lewm.get_cost(
            {"z": info["states"], "goal_z": info["goal_z"], **info["past"]},
            candidates,
        )


class DynamicsModel:
    """A ``spatial.py`` dynamics model on frozen tokens of the LeWM encoder.

    Its final predicted state is mapped to the global state (through
    ``readout`` for an 8x8 grid) and scored like LeWM's: squared distance to
    the goal's global state, so every model ranks candidates on one scale.
    """

    def __init__(self, lewm: LeWM, dynamics, readout=None):
        self.lewm, self.dynamics, self.readout = lewm.eval(), dynamics, readout
        self.history = dynamics.history

    def encode(self, frame: torch.Tensor) -> torch.Tensor:
        cls, patches = self.lewm.encoder(to_float(frame[None]))
        tokens = patches if self.readout is not None else self.lewm.projector(cls)[None]
        return self.dynamics.normalize(tokens)[0]

    def cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        n = candidates.shape[1]
        states = info["states"].expand(n, *info["states"].shape[1:])
        past = info["past"].get("past_actions")
        if past is not None:
            past = past.expand(n, -1, -1)
        final = self.dynamics.rollout(states, candidates[0], past)[:, -1]
        if self.readout is not None:
            z = self.readout(final)
        else:
            z = (final * self.dynamics.std + self.dynamics.mean)[:, 0]
        return (z - info["goal_z"]).pow(2).sum(-1)[None]


class LatentPlanner:
    """MPC with CEM on a world model, optionally re-scoring a shortlist.

    Args:
        search: Model whose cost CEM optimizes.
        refine: Optional second model. CEM's final mean plus the
            ``shortlist - 1`` cheapest candidates it saw are re-scored by
            ``refine`` and the cheapest under ``refine`` is executed.
        shortlist: Candidates re-scored by ``refine``.
    """

    def __init__(
        self,
        lewm: LeWM,
        search,
        horizon=5,
        samples=300,
        iters=30,
        topk=30,
        refine=None,
        shortlist=0,
    ):
        self.lewm, self.search, self.refine = lewm, search, refine
        self.horizon, self.samples, self.iters, self.topk = (
            horizon,
            samples,
            iters,
            topk,
        )
        self.shortlist = shortlist
        self.generator = torch.Generator().manual_seed(0)

    @torch.no_grad()
    def start(self, frame: torch.Tensor, goal: torch.Tensor):
        self.goal_z = self.lewm.encode(goal[None])[0]
        self.models = [m for m in (self.search, self.refine) if m is not None]
        self.states = [[m.encode(frame)] for m in self.models]
        self.actions = []
        self.mean = None
        self.decisions = []

    def _info(self, i: int) -> dict:
        model, states = self.models[i], self.states[i]
        h = min(model.history, len(states))
        past = {}
        if h > 1:
            past["past_actions"] = torch.stack(self.actions[-(h - 1) :])[None]
        return {
            "states": torch.stack(states[-h:])[None],
            "goal_z": self.goal_z,
            "past": past,
        }

    @torch.no_grad()
    def act(self) -> torch.Tensor:
        budget = (self.horizon, self.samples, self.iters, self.topk)
        if self.refine is None:
            plan = cem(
                self.search.cost, self._info(0), *budget, self.mean, self.generator
            )
        else:
            mean, best = cem(
                self.search.cost,
                self._info(0),
                *budget,
                self.mean,
                self.generator,
                keep=self.shortlist - 1,
            )
            shortlist = torch.cat([mean[None], best])
            costs = self.refine.cost(self._info(1), shortlist[None])[0]
            pick = int(costs.argmin())
            self.decisions.append(pick != 0)
            plan = shortlist[pick]
        self.mean = torch.cat([plan[1:], torch.zeros(1, ACTION_DIM)])
        return plan[0]

    @torch.no_grad()
    def observe(self, action: torch.Tensor, frame: torch.Tensor):
        self.actions.append(action)
        for model, states in zip(self.models, self.states):
            states.append(model.encode(frame))


class OraclePlanner:
    """CEM with the true dynamics and target: the ceiling for this planner."""

    def __init__(self, env: TwoRoomEnv, horizon=5, samples=300, iters=30, topk=30):
        self.env = env
        self.horizon, self.samples, self.iters, self.topk = (
            horizon,
            samples,
            iters,
            topk,
        )
        self.generator = torch.Generator().manual_seed(0)

    def start(self, frame: torch.Tensor, goal: torch.Tensor):
        self.mean = None

    def cost(self, info: dict, candidates: torch.Tensor) -> torch.Tensor:
        position = info["position"].expand(candidates.shape[1], -1)
        final = info["sim"].rollout(position, candidates[0])[:, -1]
        return (final - info["target"]).pow(2).sum(-1)[None]

    def act(self) -> torch.Tensor:
        info = {
            "sim": SimDynamics(self.env),
            "position": self.env.agent_position[None].clone(),
            "target": self.env.target_position.clone(),
        }
        plan = cem(
            self.cost,
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

    def observe(self, action: torch.Tensor, frame: torch.Tensor):
        pass


def setup(env: TwoRoomEnv, seed: int, cfg: dict) -> int:
    if cfg["task"] == "near-door":
        return reset_near_door(env, seed, cfg["offset"])
    return reset(env, seed)


def run_episode(planner, env: TwoRoomEnv, seed: int, cfg: dict) -> dict:
    max_replans = cfg["max_replans"]
    seed = setup(env, seed, cfg)
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
        "final_agent": env.agent_position.tolist(),
        "target": env.target_position.tolist(),
        "changed_decisions": (
            float(np.mean(planner.decisions))
            if getattr(planner, "decisions", None)
            else None
        ),
    }


def run_reference(policy: str, env: TwoRoomEnv, seed: int, cfg: dict) -> dict:
    """Expert (no noise) or uniformly random actions with the same step budget."""
    max_steps = cfg["max_replans"] * FRAMESKIP
    seed = setup(env, seed, cfg)
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
        return [run_reference(kind, env, s, cfg) for s in seeds]
    budget = cfg["horizon"], cfg["samples"], cfg["iters"], cfg["topk"]
    if kind == "oracle":
        planner = OraclePlanner(env, *budget)
    else:
        planner = build_planner(kind, cfg, budget)
    return [run_episode(planner, env, s, cfg) for s in seeds]


def build_planner(kind: str, cfg: dict, budget: tuple) -> LatentPlanner:
    lewm = LeWM(history=cfg["model_history"])
    lewm.load_state_dict(torch.load(cfg["model"]))
    lewm.eval()

    def model(name):
        if name == "lewm":
            return GlobalModel(lewm, cfg["history"])
        if name == "global-rec":
            return DynamicsModel(lewm, load_dynamics(Path(cfg["global_dynamics"])))
        readout = GridReadout()
        readout.load_state_dict(torch.load(Path(cfg["spatial"]) / "readout.pt"))
        return DynamicsModel(lewm, load_dynamics(Path(cfg["spatial"])), readout.eval())

    if kind == "selective":
        return LatentPlanner(
            lewm,
            model(cfg["search"]),
            *budget,
            refine=model("spatial"),
            shortlist=cfg["shortlist"],
        )
    return LatentPlanner(lewm, model(kind), *budget)


def evaluate(kind: str, cfg: dict, episodes: int, workers: int) -> dict:
    seeds = np.arange(PLAN_SEED, PLAN_SEED + episodes)
    jobs = [(kind, c.tolist(), cfg) for c in np.array_split(seeds, workers) if len(c)]
    with mp.get_context("spawn").Pool(workers) as pool:
        rows = [r for part in pool.map(_worker, jobs) for r in part]
    summary = {"success_rate": float(np.mean([r["success"] for r in rows]))}
    if "latency" in rows[0]:
        summary["latency_s"] = float(np.mean([r["latency"] for r in rows]))
        changed = [
            r["changed_decisions"] for r in rows if r["changed_decisions"] is not None
        ]
        if changed:
            summary["changed_decisions"] = float(np.mean(changed))
        wins = [r["replans"] for r in rows if r["success"]]
        summary["replans_to_success"] = float(np.mean(wins)) if wins else None
    return {"summary": summary, "episodes": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", type=Path, default=None, help="LeWM weights.")
    parser.add_argument(
        "--planner",
        choices=("lewm", "global-rec", "spatial", "selective"),
        default="lewm",
        help="lewm: LeWM's predictor; global-rec / spatial: spatial.py dynamics "
        "on the global state / 8x8 grid; selective: CEM on --search, then the "
        "spatial model re-scores a shortlist.",
    )
    parser.add_argument("--spatial", type=Path, default=None, help="Run directory.")
    parser.add_argument(
        "--global-dynamics", type=Path, default=None, help="Run directory."
    )
    parser.add_argument("--search", choices=("lewm", "global-rec"), default="lewm")
    parser.add_argument("--shortlist", type=int, default=16)
    parser.add_argument(
        "--reference", choices=("expert", "random", "oracle"), default=None
    )
    parser.add_argument("--task", choices=("far", "near-door"), default="near-door")
    parser.add_argument(
        "--offset", type=int, default=25, help="near-door: steps from start to goal."
    )
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument(
        "--max-replans",
        type=int,
        default=None,
        help="Default: 40 for far, 2 * offset / FRAMESKIP for near-door.",
    )
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--history", type=int, default=1, help="Context frames.")
    parser.add_argument("--model-history", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.max_replans is None:
        args.max_replans = 40 if args.task == "far" else 2 * args.offset // FRAMESKIP
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    kind = args.reference or args.planner
    t0 = time.time()
    result = evaluate(kind, cfg, args.episodes, args.workers)
    result["config"] = cfg
    print(kind, result["summary"], f"{time.time() - t0:.0f}s", flush=True)
    if args.out:
        args.out.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
