"""Single-digit Moving MNIST generated on the fly, with ground-truth state.

A downscaled MNIST digit bounces off the walls of a small frame at constant
speed. Every clip comes with the digit's center position per frame and its
class, so probes can check what an embedding encodes and latent rollouts can
be decoded and scored. Position changes over time (dynamics); the class is
fixed per clip and is only legible at full resolution (detail).
"""

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision


class MovingMNIST:
    """Generator of bouncing-digit clips.

    Args:
        root: Directory for the torchvision MNIST download.
        train: Use MNIST train digits (for training clips) or test digits
            (for held-out evaluation clips).
        size: Frame height and width.
        digit_size: Digit height and width after downscaling from 28 px.
        speed: Range of per-frame speeds in pixels.
    """

    def __init__(
        self,
        root: Path,
        train: bool = True,
        size: int = 32,
        digit_size: int = 14,
        speed: tuple = (0.8, 2.0),
    ):
        mnist = torchvision.datasets.MNIST(root=str(root), train=train, download=True)
        digits = mnist.data.float().div(255).unsqueeze(1)
        self.digits = F.interpolate(digits, size=digit_size, mode="area")
        self.labels = mnist.targets.clone()
        self.size = size
        self.digit_size = digit_size
        self.speed = speed

    def sample(self, n: int, num_frames: int, generator: torch.Generator) -> dict:
        """Sample clips.

        Args:
            n: Number of clips.
            num_frames: Frames per clip.
            generator: Random generator for digits, positions and velocities.

        Returns:
            Dict with ``video`` ``[n, T, 1, size, size]`` in ``[-1, 1]``,
            ``position`` ``[n, T, 2]`` digit centers as (row, col) in pixels,
            and ``label`` ``[n]`` digit classes.
        """
        idx = torch.randint(len(self.digits), (n,), generator=generator)
        hi = float(self.size - self.digit_size)
        corner = hi * torch.rand(n, 2, generator=generator)
        heading = 2 * math.pi * torch.rand(n, generator=generator)
        lo_s, hi_s = self.speed
        speed = lo_s + (hi_s - lo_s) * torch.rand(n, generator=generator)
        vel = torch.stack([heading.sin(), heading.cos()], -1) * speed[:, None]
        corners = []
        for _ in range(num_frames):
            corners.append(corner)
            corner = corner + vel
            over, under = corner > hi, corner < 0
            corner = torch.where(over, 2 * hi - corner, corner)
            corner = torch.where(under, -corner, corner)
            vel = torch.where(over | under, -vel, vel)
        corners = torch.stack(corners, 1)
        return {
            "video": self._render(self.digits[idx], corners) * 2 - 1,
            "position": corners + (self.digit_size - 1) / 2,
            "label": self.labels[idx],
        }

    def _render(self, digits: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        n, t = corners.shape[:2]
        canvas = F.pad(
            digits, (0, self.size - self.digit_size, 0, self.size - self.digit_size)
        )
        canvas = canvas.repeat_interleave(t, dim=0)
        shift = corners.reshape(n * t, 2) * 2 / self.size
        theta = torch.zeros(n * t, 2, 3)
        theta[:, 0, 0] = theta[:, 1, 1] = 1
        theta[:, 0, 2] = -shift[:, 1]
        theta[:, 1, 2] = -shift[:, 0]
        grid = F.affine_grid(theta, list(canvas.shape), align_corners=False)
        frames = F.grid_sample(canvas, grid, align_corners=False)
        return frames.reshape(n, t, 1, self.size, self.size)
