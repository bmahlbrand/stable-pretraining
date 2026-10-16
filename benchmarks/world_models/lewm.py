"""LeWorldModel baseline for the world-model benchmark.

Lives in the benchmark rather than the library: it exists to give HLeWM a
faithful flat baseline. It is assembled from the same building blocks as
:mod:`stable_pretraining.methods.hlewm` (transformer block, BatchNorm
projector, Epps-Pulley statistic) so the comparison isolates the design
differences: one embedding per frame and a causal next-frame predictor, versus
HLeWM's token pyramid and MilliVid schedule.

Reference:
    Maes et al. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels." 2026. https://arxiv.org/abs/2603.19312
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.methods.hlewm import (
    _apply_tokenwise,
    _Block,
    _patchify,
    _projector,
    _sincos,
)
from stable_pretraining.methods.lejepa import EppsPulley


@dataclass
class LeWMOutput:
    """Loss terms of one LeWM training call."""

    loss: torch.Tensor
    pred_loss: torch.Tensor
    sigreg_loss: torch.Tensor


@dataclass
class LeWMRollout:
    """Embeddings ``[B, T, 1, D]`` (one level) and which frames were predicted."""

    embeddings: tuple
    predicted: tuple


class LeWM(nn.Module):
    """LeWorldModel without actions.

    A ViT encodes each frame to its ``[CLS]`` token followed by a BatchNorm MLP
    projector. A causal transformer predicts each next frame's embedding from
    the previous ``history`` embeddings. The loss is the next-embedding MSE plus
    ``lamb`` times SIGReg, run per frame position across the batch. Rollouts
    feed predictions back as context.

    Args:
        image_size: Frame height and width.
        patch_size: ViT patch size.
        in_channels: Frame channels.
        embed_dim: Embedding size.
        encoder_dim: Encoder width.
        encoder_depth: Encoder depth.
        encoder_heads: Encoder attention heads.
        predictor_dim: Predictor width.
        predictor_depth: Predictor depth.
        predictor_heads: Predictor attention heads.
        predictor_mlp_dim: Predictor MLP hidden size.
        projector_hidden_dim: Hidden size of both projectors.
        history: Context frames per prediction.
        lamb: SIGReg weight.
        n_slices: SIGReg random projections.
        n_points: SIGReg quadrature points.
    """

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 8,
        in_channels: int = 1,
        embed_dim: int = 32,
        encoder_dim: int = 64,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        predictor_dim: int = 64,
        predictor_depth: int = 4,
        predictor_heads: int = 4,
        predictor_mlp_dim: int = 256,
        projector_hidden_dim: int = 256,
        history: int = 4,
        lamb: float = 0.09,
        n_slices: int = 64,
        n_points: int = 17,
    ):
        super().__init__()
        grid = image_size // patch_size
        self.patch_size = patch_size
        self.patch_embed = nn.Linear(in_channels * patch_size**2, encoder_dim)
        self.cls = nn.Parameter(torch.randn(1, 1, encoder_dim) * 0.02)
        self.register_buffer(
            "rowcol",
            torch.cartesian_prod(torch.arange(grid), torch.arange(grid)),
            persistent=False,
        )
        self.encoder_blocks = nn.ModuleList(
            [
                _Block(encoder_dim, encoder_heads, 4 * encoder_dim)
                for _ in range(encoder_depth)
            ]
        )
        self.encoder_norm = nn.LayerNorm(encoder_dim)
        self.encoder_projector = _projector(
            encoder_dim, projector_hidden_dim, embed_dim
        )

        self.predictor_in = nn.Linear(embed_dim, predictor_dim)
        self.predictor_blocks = nn.ModuleList(
            [
                _Block(predictor_dim, predictor_heads, predictor_mlp_dim)
                for _ in range(predictor_depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_out = nn.Linear(predictor_dim, embed_dim)
        self.predictor_projector = _projector(
            embed_dim, projector_hidden_dim, embed_dim
        )

        self.embed_dim = embed_dim
        self.history = history
        self.lamb = lamb
        self.n_slices = n_slices
        self.epps_pulley = EppsPulley(t_max=3.0, n_points=n_points)
        self.register_buffer("sigreg_step", torch.zeros((), dtype=torch.long))

    @property
    def clip_frames(self) -> int:
        """Frames one training call uses."""
        return self.history + 1

    def embed(self, video: torch.Tensor) -> torch.Tensor:
        """``[B, T, C, H, W]`` -> ``[B, T, embed_dim]``."""
        b, t = video.shape[:2]
        x = self.patch_embed(_patchify(video.flatten(0, 1), self.patch_size))
        x = x + _sincos(self.rowcol, x.shape[-1]).to(x.dtype)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.encoder_blocks:
            x = block(x)
        z = self.encoder_projector(self.encoder_norm(x[:, 0]))
        return z.reshape(b, t, -1)

    def encode(self, video: torch.Tensor) -> list:
        """One level of one token per frame, ``[[B, T, 1, embed_dim]]``.

        Matches the per-level layout of :meth:`HLeWM.encode` so the benchmark
        evaluates both models the same way.
        """
        return [self.embed(video).unsqueeze(2)]

    def predict(self, z: torch.Tensor) -> torch.Tensor:
        """Next-frame embeddings for every position of ``z`` ``[B, L, D]``."""
        length = z.shape[1]
        x = self.predictor_in(z)
        steps = torch.arange(length, device=z.device)[:, None]
        x = x + _sincos(steps, x.shape[-1]).to(x.dtype)
        causal = torch.ones(length, length, dtype=torch.bool, device=z.device).tril()
        for block in self.predictor_blocks:
            x = block(x, causal)
        x = self.predictor_out(self.predictor_norm(x))
        return _apply_tokenwise(self.predictor_projector, x)

    def _sigreg(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            g = torch.Generator(device="cpu").manual_seed(int(self.sigreg_step.item()))
            self.sigreg_step.add_(1)
            directions = torch.randn(self.embed_dim, self.n_slices, generator=g)
            directions = directions / directions.norm(dim=0)
        directions = directions.to(device=z.device, dtype=z.dtype)
        return self.epps_pulley((z @ directions).flatten(1)).mean()

    def forward(self, video: torch.Tensor) -> LeWMOutput:
        """Loss on a random ``history + 1``-frame window of ``[B, T, C, H, W]``."""
        length, t = self.clip_frames, video.shape[1]
        if t < length:
            raise ValueError(f"Clips need at least {length} frames, got {t}.")
        offset = 0
        if self.training and t > length:
            offset = int(torch.randint(0, t - length + 1, (1,)).item())
        z = self.embed(video[:, offset : offset + length])
        pred_loss = F.mse_loss(self.predict(z[:, :-1]), z[:, 1:])
        sigreg_loss = self._sigreg(z)
        return LeWMOutput(pred_loss + self.lamb * sigreg_loss, pred_loss, sigreg_loss)

    @torch.no_grad()
    def rollout(self, video: torch.Tensor, num_frames: int) -> LeWMRollout:
        """Encode the given frames, then predict frames up to ``num_frames``."""
        z = self.embed(video)
        start = z.shape[1]
        if start < self.history:
            raise ValueError(f"Rollout needs at least {self.history} given frames.")
        while z.shape[1] < num_frames:
            nxt = self.predict(z[:, -self.history :])[:, -1:]
            z = torch.cat([z, nxt], dim=1)
        predicted = torch.arange(num_frames, device=video.device) >= start
        return LeWMRollout(embeddings=(z.unsqueeze(2),), predicted=(predicted,))
