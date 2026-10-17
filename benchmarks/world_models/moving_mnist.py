"""Moving MNIST generated on the fly, with an optional occluder and ground truth.

Downscaled MNIST digits bounce off the frame's walls at constant speed. An
optional vertical band hides whatever passes behind it, so a digit can vanish
for many frames and re-emerge later. With ``textured=True`` frames are RGB:
each clip has its own colored noise texture as background, which the camera
pans across slowly, colored digits, and a textured band, so the scene has real
content besides the digits. Every clip comes with a
per-cell map of which digit (if any) is visible there, which probes use to
check what an embedding encodes and to score latent rollouts, plus each
digit's visible fraction per frame for finding re-emergence events.
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torchvision

BACKGROUND = -1
IGNORE = -2


def fractal_noise(
    n: int, size: int, generator: torch.Generator, octaves: int = 5, base: int = 4
) -> torch.Tensor:
    """Smooth multi-octave noise textures ``[n, 1, size, size]`` in ``[0, 1]``."""
    out = torch.zeros(n, 1, size, size)
    amplitude, total = 1.0, 0.0
    for octave in range(octaves):
        grid = torch.randn(
            n, 1, base * 2**octave, base * 2**octave, generator=generator
        )
        out += amplitude * F.interpolate(grid, size=size, mode="bicubic")
        total += amplitude
        amplitude *= 0.65
    lo = out.amin((-1, -2), keepdim=True)
    hi = out.amax((-1, -2), keepdim=True)
    return (out - lo) / (hi - lo + 1e-6)


class MovingMNIST:
    """Generator of bouncing-digit clips.

    Args:
        root: Directory for the torchvision MNIST download.
        train: Use MNIST train digits (training clips) or test digits
            (held-out evaluation clips).
        size: Frame height and width.
        digit_size: Digit height and width after downscaling from 28 px.
        num_digits: Digits per clip.
        occluder: Optional ``(start, stop)`` columns of an opaque vertical band.
        occluder_value: Band intensity in ``[0, 1]``.
        cell: Label-map cell size in pixels (the finest token's patch size).
        speed: Range of per-frame speeds in pixels.
        textured: RGB frames with textured backgrounds, colored digits and a
            textured band instead of grayscale digits on black.
        pan_speed: Range of per-frame background pan speeds in pixels.
    """

    def __init__(
        self,
        root: Path,
        train: bool = True,
        size: int = 32,
        digit_size: int = 14,
        num_digits: int = 1,
        occluder: Optional[tuple] = None,
        occluder_value: float = 0.5,
        cell: int = 8,
        speed: tuple = (0.8, 2.0),
        textured: bool = False,
        pan_speed: tuple = (0.25, 0.6),
    ):
        mnist = torchvision.datasets.MNIST(root=str(root), train=train, download=True)
        digits = mnist.data.float().div(255).unsqueeze(1)
        self.digits = F.interpolate(digits, size=digit_size, mode="area")
        self.labels = mnist.targets.clone()
        self.size = size
        self.digit_size = digit_size
        self.num_digits = num_digits
        self.cell = cell
        self.speed = speed
        self.occluder_value = occluder_value
        self.band = torch.zeros(size, size, dtype=torch.bool)
        if occluder is not None:
            self.band[:, occluder[0] : occluder[1]] = True
        cells = F.avg_pool2d(self.band[None, None].float(), cell)[0, 0]
        self.ignore_cells = cells > 0.5
        self.textured = textured
        self.pan_speed = pan_speed
        self.channels = 3 if textured else 1

    def sample(
        self, n: int, num_frames: int, generator: torch.Generator, maps: bool = True
    ) -> dict:
        """Sample clips.

        Args:
            n: Number of clips.
            num_frames: Frames per clip.
            generator: Random generator for digits, positions and velocities.
            maps: Also compute ``visible`` and ``owner`` (training needs only
                ``video``).

        Returns:
            Dict with ``video`` ``[n, T, C, size, size]`` in ``[-1, 1]``;
            ``position`` ``[n, T, D, 2]`` digit centers (row, col) in pixels;
            ``label`` ``[n, D]`` digit classes; ``visible`` ``[n, T, D]`` the
            fraction of each digit's ink not hidden by the band; and
            ``owner`` ``[n, T, size / cell, size / cell]``, the index of the
            digit with most visible ink in each cell, ``BACKGROUND`` for empty
            cells and ``IGNORE`` for cells under the band.
        """
        chunk = 64  # rendering memory grows with clips x frames x digits
        if n > chunk:
            parts = [
                self.sample(min(chunk, n - i), num_frames, generator, maps)
                for i in range(0, n, chunk)
            ]
            return {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        d = self.num_digits
        idx = torch.randint(len(self.digits), (n, d), generator=generator)
        hi = float(self.size - self.digit_size)
        corner = hi * torch.rand(n, d, 2, generator=generator)
        heading = 2 * math.pi * torch.rand(n, d, generator=generator)
        lo_s, hi_s = self.speed
        speed = lo_s + (hi_s - lo_s) * torch.rand(n, d, generator=generator)
        vel = torch.stack([heading.sin(), heading.cos()], -1) * speed[..., None]
        corners = []
        for _ in range(num_frames):
            corners.append(corner)
            corner = corner + vel
            over, under = corner > hi, corner < 0
            corner = torch.where(over, 2 * hi - corner, corner)
            corner = torch.where(under, -corner, corner)
            vel = torch.where(over | under, -vel, vel)
        corners = torch.stack(corners, 1)

        per_digit = corners.transpose(1, 2).reshape(n * d, num_frames, 2)
        ink = self._render(self.digits[idx.flatten()], per_digit)
        ink = ink.reshape(n, d, num_frames, self.size, self.size).transpose(1, 2)
        if self.textured:
            frames = self._compose_textured(ink, generator)
        else:
            frames = ink.max(dim=2).values
            frames = torch.where(self.band, self.occluder_value, frames).unsqueeze(2)
        clips = {
            "video": frames * 2 - 1,
            "position": corners + (self.digit_size - 1) / 2,
            "label": self.labels[idx],
        }
        if not maps:
            return clips
        shown = ink * ~self.band
        total = ink.sum((-1, -2)).clamp_min(1e-6)
        visible = shown.sum((-1, -2)) / total

        cell_ink = F.avg_pool2d(shown.flatten(0, 2)[:, None], self.cell)
        cell_ink = cell_ink.reshape(n, num_frames, d, *cell_ink.shape[-2:])
        best, which = cell_ink.max(dim=2)
        owner = torch.where(best > 0.05, which, BACKGROUND)
        owner = torch.where(self.ignore_cells, IGNORE, owner)
        return {**clips, "visible": visible, "owner": owner}

    def sample_reemerging(
        self, n: int, num_frames: int, given: int, generator: torch.Generator
    ) -> dict:
        """Sample clips in which a digit hides at frame ``given - 1`` and returns.

        Each kept clip has a digit that is at least half visible at some frame
        before ``given - 1``, fully hidden at ``given - 1``, and at least half
        visible again before ``num_frames``. ``reemerging`` ``[n, D]`` marks
        those digits.
        """
        kept, count = [], 0
        while count < n:
            clips = self.sample(256, num_frames, generator)
            vis = clips["visible"]
            seen = (vis[:, : given - 1] > 0.5).any(1)
            hidden = vis[:, given - 1] < 0.02
            back = (vis[:, given:] > 0.5).any(1)
            mask = seen & hidden & back
            keep = mask.any(1)
            clips = {k: v[keep] for k, v in clips.items()}
            clips["reemerging"] = mask[keep]
            kept.append(clips)
            count += int(keep.sum())
        return {k: torch.cat([c[k] for c in kept])[:n] for k in kept[0]}

    def _compose_textured(self, ink: torch.Tensor, generator: torch.Generator):
        """RGB frames ``[n, T, 3, H, W]`` in ``[0, 1]`` from digit ink ``[n, T, D, H, W]``."""
        n, t, d, size = *ink.shape[:3], self.size
        span = size + 64
        # Two-tone colored texture with contrast in every channel.
        tone = torch.stack(
            [
                0.3 * torch.rand(n, 3, generator=generator),
                0.3 + 0.3 * torch.rand(n, 3, generator=generator),
            ],
            -1,
        )
        swap = torch.rand(n, 3, 1, generator=generator) < 0.5
        tone = torch.where(swap, tone.flip(-1), tone)
        noise = fractal_noise(n, span, generator)
        tex = tone[..., :1, None] + (tone[..., 1:, None] - tone[..., :1, None]) * noise
        # The camera pans across the texture at constant velocity, staying inside.
        heading = 2 * math.pi * torch.rand(n, generator=generator)
        lo, hi = self.pan_speed
        speed = lo + (hi - lo) * torch.rand(n, generator=generator)
        vel = torch.stack([heading.sin(), heading.cos()], -1) * speed[:, None]
        room = span - size - 1 - vel.abs() * (t - 1)
        start = torch.where(vel < 0, vel.abs() * (t - 1), 0) + room * torch.rand(
            n, 2, generator=generator
        )
        frames = self._crop(
            tex, start[:, None] + vel[:, None] * torch.arange(t)[:, None]
        )
        # Digits are opaque where their ink is; overlapping ink mixes colors.
        colors = 0.65 + 0.35 * torch.rand(n, d, 3, generator=generator)
        weight = ink.sum(2, keepdim=True)
        color = torch.einsum("ntdhw,ndc->ntchw", ink, colors) / weight.clamp_min(1e-6)
        frames = frames + weight.clamp(max=1) * (color - frames)
        wall = 0.35 + 0.3 * fractal_noise(n, size, generator)
        return torch.where(self.band, wall[:, None], frames)

    def _crop(self, tex: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Bilinear frame-sized crops ``[n, T, C, s, s]`` of ``tex`` ``[n, C, S, S]``.

        ``offsets`` ``[n, T, 2]`` are each crop's top-left (row, col) in pixels.
        """
        n, t = offsets.shape[:2]
        s, span = self.size, tex.shape[-1]
        pixels = torch.arange(s, dtype=torch.float32)
        rows = offsets[..., 0, None] + pixels  # [n, T, s]
        cols = offsets[..., 1, None] + pixels
        grid = torch.stack(
            [
                cols[:, :, None, :].expand(-1, -1, s, -1),
                rows[:, :, :, None].expand(-1, -1, -1, s),
            ],
            -1,
        )
        grid = grid * 2 / (span - 1) - 1
        out = F.grid_sample(tex, grid.reshape(n, t * s, s, 2), align_corners=True)
        return out.reshape(n, -1, t, s, s).transpose(1, 2)

    def _render(self, digits: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        """Place ``digits`` ``[N, 1, s, s]`` at ``corners`` ``[N, T, 2]``.

        Bilinear sub-pixel placement: each digit is shifted by the fractional
        part of its corner on a small patch, then pasted at the integer part.
        """
        n, t = corners.shape[:2]
        s = self.digit_size
        corners = corners.reshape(n * t, 2)
        whole = corners.floor()
        fy, fx = (corners - whole).unbind(-1)
        padded = F.pad(digits[:, 0], (1, 1, 1, 1)).repeat_interleave(t, dim=0)
        fy, fx = fy[:, None, None], fx[:, None, None]
        patch = (
            (1 - fy) * (1 - fx) * padded[:, 1:, 1:]
            + (1 - fy) * fx * padded[:, 1:, :-1]
            + fy * (1 - fx) * padded[:, :-1, 1:]
            + fy * fx * padded[:, :-1, :-1]
        )
        canvas = corners.new_zeros(n * t, self.size + 1, self.size + 1)
        offsets = torch.arange(s + 1)
        rows = (whole[:, 0].long()[:, None] + offsets)[:, :, None]
        cols = (whole[:, 1].long()[:, None] + offsets)[:, None, :]
        canvas[torch.arange(n * t)[:, None, None], rows, cols] = patch
        return canvas[:, : self.size, : self.size].reshape(n, t, self.size, self.size)
