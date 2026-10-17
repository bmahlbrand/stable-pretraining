"""TwoRoom navigation with randomized doors, from stable-worldmodel.

A red dot moves through two rooms separated by a vertical wall with one to
three doors. Every episode samples the number, size and position of the doors
and puts the agent and the target in different rooms, so reaching the target
means finding a door in the image and steering through it. Doors are passable
at any size, but small ones need the agent's center aligned within about 3 px,
so how precisely a world model represents door geometry matters for planning.

The world model sees frames every ``FRAMESKIP`` environment steps and takes the
``FRAMESKIP`` actions in between as one ``FRAMESKIP * 2``-dimensional action.

Example::

    python benchmarks/world_models/planning/two_room.py --episodes 3000 --seed 0 --out data/train.pt
    python benchmarks/world_models/planning/two_room.py --episodes 500 --seed 100000 --out data/test.pt
"""

import argparse
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="ale-py not found")
from stable_worldmodel.envs.two_room.env import TwoRoomEnv  # noqa: E402
from stable_worldmodel.envs.two_room.expert_policy import ExpertPolicy  # noqa: E402

IMAGE_SIZE = 112
FRAMESKIP = 5
ACTION_DIM = 2 * FRAMESKIP
ENV_SIZE = TwoRoomEnv.IMG_SIZE
WALL = TwoRoomEnv.WALL_CENTER
SUCCESS_DIST = 16.0
# Door centers outside this range can be partly or wholly unreachable, since
# the agent's center is clamped 21 px inside the border.
DOOR_RANGE = (28, 196)
# Agent and target centers stay this far from the wall's center line.
WALL_CLEARANCE = 16
VARIATIONS = [
    "door.number",
    "door.size",
    "door.position",
    "agent.position",
    "target.position",
]


def reset(env: TwoRoomEnv, seed: int) -> int:
    """Reset to a random door layout with agent and target in different rooms.

    Rejection-samples the environment's own variation space, starting from
    ``seed``; returns the seed that was accepted.
    """
    while True:
        env.reset(seed=seed, options={"variation": VARIATIONS})
        agent, target = env.agent_position, env.target_position
        doors = env.door_positions
        if (
            (agent[0] < WALL) != (target[0] < WALL)
            and abs(float(agent[0]) - WALL) > WALL_CLEARANCE
            and abs(float(target[0]) - WALL) > WALL_CLEARANCE
            and bool(((doors >= DOOR_RANGE[0]) & (doors <= DOOR_RANGE[1])).all())
        ):
            return seed
        seed += 1_000_003


def downsample(image: torch.Tensor) -> torch.Tensor:
    """Area-downsample a ``[3, 224, 224]`` uint8 frame to ``IMAGE_SIZE``."""
    x = F.interpolate(image[None].float(), size=IMAGE_SIZE, mode="area")[0]
    return x.round().clamp(0, 255).to(torch.uint8)


def observe(env: TwoRoomEnv) -> torch.Tensor:
    return downsample(env._render_frame(agent_pos=env.agent_position))


def goal_image(env: TwoRoomEnv) -> torch.Tensor:
    return downsample(env._target_img)


def layout(env: TwoRoomEnv) -> dict:
    """Door geometry in environment pixels, padded to ``MAX_DOOR`` slots."""
    k = TwoRoomEnv.MAX_DOOR
    position = torch.zeros(k)
    size = torch.zeros(k)
    position[: env.num_doors] = env.door_positions
    size[: env.num_doors] = env.door_sizes
    return {"num_doors": env.num_doors, "door_position": position, "door_size": size}


class SimDynamics:
    """Batched copy of ``TwoRoomEnv``'s movement and collisions (vertical wall).

    Lets a planner score action candidates with the true dynamics, as an
    upper reference for what a world model's rollouts could achieve.
    """

    def __init__(self, env: TwoRoomEnv):
        assert env.wall_axis == 1, "only the default vertical wall is supported"
        self.speed = float(env.variation_space["agent"]["speed"].value.item())
        radius = float(env.variation_space["agent"]["radius"].value.item())
        half = env.wall_thickness // 2
        self.left = WALL - half - radius
        self.right = WALL + half + radius
        self.lo = env.BORDER_SIZE + radius
        self.hi = ENV_SIZE - env.BORDER_SIZE - radius
        self.doors = [
            (float(c), float(s)) for c, s in zip(env.door_positions, env.door_sizes)
        ]

    def step(self, position: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Next (x, y) ``[N, 2]`` after one environment action ``[N, 2]``."""
        nxt = (position + action.clamp(-1, 1) * self.speed).clamp(self.lo, self.hi)
        x, y = nxt.unbind(-1)
        door = torch.zeros_like(y, dtype=torch.bool)
        for c, s in self.doors:
            door |= (y >= c - s - 1.75) & (y <= c + s + 1.75)
        left = position[:, 0] < WALL
        x = torch.where(left & (x > self.left) & ~door, self.left - 0.5, x)
        x = torch.where(~left & (x < self.right) & ~door, self.right + 0.5, x)
        return torch.stack([x, y], -1)

    def rollout(self, position: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Positions ``[N, K, 2]`` after each model step of ``actions`` ``[N, K, ACTION_DIM]``."""
        out = []
        for k in range(actions.shape[1]):
            for a in actions[:, k].reshape(len(actions), FRAMESKIP, 2).unbind(1):
                position = self.step(position, a)
            out.append(position)
        return torch.stack(out, 1)


def reset_near_door(env: TwoRoomEnv, seed: int, offset: int) -> int:
    """Start ``offset`` environment steps before the goal on a path through a door.

    Runs the noise-free expert from a random layout, then starts the agent
    where the expert was 5 to ``offset - 5`` steps before crossing the wall,
    with the goal where the expert was ``offset`` steps later. The goal is
    reachable within ``offset`` steps but only through the door, which is the
    test from the world model's point of view. Returns the accepted seed.
    """
    while True:
        seed = reset(env, seed)
        expert = ExpertPolicy(seed=seed)
        expert.set_env(env)
        path = [env.agent_position.clone()]
        for _ in range(150):
            info = {
                "state": env.agent_position.numpy(),
                "goal_state": env.target_position.numpy(),
            }
            env.step(expert.get_action(info))
            path.append(env.agent_position.clone())
        path = torch.stack(path)
        side = path[:, 0] < WALL
        crossed = (side != side[0]).nonzero()
        rng = np.random.default_rng(seed)
        if len(crossed):
            start = int(crossed[0]) - int(rng.integers(5, offset - 4))
            if start >= 0 and start + offset < len(path):
                env._set_state(path[start].tolist())
                env._set_goal_state(path[start + offset].tolist())
                return seed
        seed += 1_000_003


class RandomWalk:
    """Persistent random actions: wanders, hits walls, sometimes finds doors."""

    def __init__(self, rng: np.random.Generator, persistence: float = 0.8):
        self.rng = rng
        self.persistence = persistence
        self.action = rng.uniform(-1, 1, 2)

    def get_action(self, info: dict) -> np.ndarray:
        noise = self.rng.normal(0, 0.6, 2)
        self.action = np.clip(self.persistence * self.action + noise, -1, 1)
        return self.action.astype(np.float32)


def run_episode(env: TwoRoomEnv, seed: int, steps: int) -> dict:
    """One episode of ``steps`` environment steps under a random behavior policy.

    Half the episodes follow the stable-worldmodel expert (nearest fitting
    door, then the target) with per-episode Gaussian action noise; the rest
    are persistent random walks. The expert switches to a random walk once it
    reaches the target so the episode keeps moving.
    """
    seed = reset(env, seed)
    rng = np.random.default_rng(seed)
    walk = RandomWalk(rng)
    expert = None
    if rng.random() < 0.5:
        expert = ExpertPolicy(action_noise=float(rng.uniform(0.1, 0.7)), seed=seed)
        expert.set_env(env)
    frames, agent, actions = [observe(env)], [env.agent_position.clone()], []
    reached = False
    for t in range(steps):
        info = {
            "state": env.agent_position.numpy(),
            "goal_state": env.target_position.numpy(),
        }
        policy = walk if expert is None or reached else expert
        action = policy.get_action(info)
        _, _, terminated, _, _ = env.step(action)
        reached |= bool(terminated)
        actions.append(torch.as_tensor(action))
        if (t + 1) % FRAMESKIP == 0:
            frames.append(observe(env))
            agent.append(env.agent_position.clone())
    return {
        "video": torch.stack(frames),
        "agent": torch.stack(agent),
        "action": torch.stack(actions).reshape(-1, ACTION_DIM),
        "target": env.target_position.clone(),
        "goal": goal_image(env),
        "expert": expert is not None,
        **layout(env),
    }


def _worker(args) -> dict:
    seeds, steps = args
    torch.set_num_threads(1)
    env = TwoRoomEnv()
    episodes = [run_episode(env, s, steps) for s in seeds]
    return {
        k: torch.stack([torch.as_tensor(e[k]) for e in episodes]) for k in episodes[0]
    }


def collect(episodes: int, seed: int, steps: int = 100, workers: int = 4) -> dict:
    """Collect ``episodes`` episodes; episode ``i`` starts from seed ``seed + i``.

    Returns:
        Dict with ``video`` ``[N, steps / FRAMESKIP + 1, 3, IMAGE_SIZE,
        IMAGE_SIZE]`` uint8; ``agent`` ``[N, T + 1, 2]`` agent (x, y) in
        environment pixels at each frame; ``action`` ``[N, T, ACTION_DIM]``;
        ``target`` ``[N, 2]``; ``goal`` ``[N, 3, IMAGE_SIZE, IMAGE_SIZE]`` the
        frame with the agent at the target; ``expert`` ``[N]`` behavior
        policy; ``num_doors``, ``door_position`` and ``door_size`` (half
        extent) in environment pixels.
    """
    assert steps % FRAMESKIP == 0
    seeds = np.arange(seed, seed + episodes)
    chunks = [(c.tolist(), steps) for c in np.array_split(seeds, workers * 8) if len(c)]
    with mp.get_context("spawn").Pool(workers) as pool:
        parts = pool.map(_worker, chunks)
    return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    t0 = time.time()
    data = collect(args.episodes, args.seed, args.steps, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, args.out)
    gb = data["video"].numel() / 1e9
    print(f"{args.episodes} episodes, {gb:.2f} GB of frames, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
