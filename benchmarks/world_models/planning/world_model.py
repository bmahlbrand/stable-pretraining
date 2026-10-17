"""Action-conditioned LeWorldModel with a global state and an 8x8 spatial level.

The encoder is a small ViT: its ``[CLS]`` token, after a BatchNorm projector,
is LeWM's global state, and its 8x8 patch tokens are the spatial level that
later milestones build dynamics on. Training is plain LeWM: a causal predictor
with AdaLN-zero action conditioning predicts the next global state from up to
``history`` previous states and actions, with SIGReg on the global states.

Reference:
    Maes et al. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels." 2026. https://arxiv.org/abs/2603.19312
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_pretraining.methods.hlewm import (
    _apply_tokenwise,
    _Attention,
    _Block,
    _patchify,
    _projector,
    _sincos,
)
from stable_pretraining.methods.lejepa import EppsPulley


def to_float(frames: torch.Tensor) -> torch.Tensor:
    """uint8 frames to floats in ``[-1, 1]``."""
    return frames.float() / 127.5 - 1


class ViTEncoder(nn.Module):
    """ViT returning the normalized ``[CLS]`` token and patch tokens."""

    def __init__(
        self,
        image_size: int = 112,
        patch_size: int = 14,
        dim: int = 128,
        depth: int = 4,
        heads: int = 4,
    ):
        super().__init__()
        self.grid = image_size // patch_size
        self.patch_size = patch_size
        self.dim = dim
        self.patch_embed = nn.Linear(3 * patch_size**2, dim)
        self.cls = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        rowcol = torch.cartesian_prod(torch.arange(self.grid), torch.arange(self.grid))
        self.register_buffer("pos", _sincos(rowcol, dim), persistent=False)
        self.blocks = nn.ModuleList([_Block(dim, heads, 4 * dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, frames: torch.Tensor) -> tuple:
        """``[N, 3, H, W]`` in ``[-1, 1]`` -> ``[N, dim]``, ``[N, grid**2, dim]``."""
        x = self.patch_embed(_patchify(frames, self.patch_size)) + self.pos
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]


class _CausalAdaLNBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning and an attention mask."""

    def __init__(self, dim: int, heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(dim, heads, dim // heads)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim), nn.GELU(), nn.Linear(mlp_dim, dim)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor, mask: torch.Tensor):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, -1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1, mask)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


@dataclass
class LeWMLosses:
    """Loss terms of one training call."""

    loss: torch.Tensor
    pred_loss: torch.Tensor
    sigreg_loss: torch.Tensor


class LeWM(nn.Module):
    """Action-conditioned LeWorldModel.

    Args:
        action_dim: Action size per model step.
        image_size: Frame height and width.
        patch_size: ViT patch size; ``image_size / patch_size`` is the spatial
            grid side.
        encoder_dim: Encoder width.
        encoder_depth: Encoder depth.
        encoder_heads: Encoder attention heads.
        embed_dim: Global state size.
        predictor_dim: Predictor width.
        predictor_depth: Predictor depth.
        predictor_heads: Predictor attention heads.
        projector_hidden_dim: Hidden size of both projectors.
        history: Context states per prediction.
        lamb: SIGReg weight.
        n_slices: SIGReg random projections.
    """

    def __init__(
        self,
        action_dim: int = 10,
        image_size: int = 112,
        patch_size: int = 14,
        encoder_dim: int = 128,
        encoder_depth: int = 4,
        encoder_heads: int = 4,
        embed_dim: int = 64,
        predictor_dim: int = 128,
        predictor_depth: int = 4,
        predictor_heads: int = 4,
        projector_hidden_dim: int = 512,
        history: int = 3,
        lamb: float = 0.09,
        n_slices: int = 256,
    ):
        super().__init__()
        self.encoder = ViTEncoder(
            image_size, patch_size, encoder_dim, encoder_depth, encoder_heads
        )
        self.projector = _projector(encoder_dim, projector_hidden_dim, embed_dim)
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, 4 * predictor_dim),
            nn.SiLU(),
            nn.Linear(4 * predictor_dim, predictor_dim),
        )
        self.predictor_in = nn.Linear(embed_dim, predictor_dim)
        self.time_embed = nn.Parameter(torch.randn(1, history, predictor_dim) * 0.02)
        self.predictor_blocks = nn.ModuleList(
            [
                _CausalAdaLNBlock(predictor_dim, predictor_heads, 4 * predictor_dim)
                for _ in range(predictor_depth)
            ]
        )
        self.predictor_norm = nn.LayerNorm(predictor_dim)
        self.predictor_out = nn.Linear(predictor_dim, embed_dim)
        self.pred_proj = _projector(embed_dim, projector_hidden_dim, embed_dim)

        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.history = history
        self.lamb = lamb
        self.n_slices = n_slices
        self.epps_pulley = EppsPulley(t_max=3.0, n_points=17)
        self.register_buffer("sigreg_step", torch.zeros((), dtype=torch.long))

    def encode(self, frames: torch.Tensor) -> tuple:
        """uint8 or float frames ``[..., 3, H, W]`` -> global states and patches.

        Returns:
            ``z`` ``[..., embed_dim]`` and patch tokens ``[..., grid**2,
            encoder_dim]`` (the encoder's last layer, before any projector).
        """
        lead = frames.shape[:-3]
        if frames.dtype == torch.uint8:
            frames = to_float(frames)
        cls, patches = self.encoder(frames.reshape(-1, *frames.shape[-3:]))
        z = self.projector(cls)
        return z.reshape(*lead, -1), patches.reshape(*lead, *patches.shape[1:])

    def predict(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Next global state after each position of ``z`` ``[B, L, D]``.

        ``actions`` ``[B, L, action_dim]``: the action taken at each position.
        ``L`` is at most ``history``.
        """
        length = z.shape[1]
        x = self.predictor_in(z) + self.time_embed[:, :length]
        c = self.action_encoder(actions)
        causal = torch.ones(length, length, dtype=torch.bool, device=z.device).tril()
        for block in self.predictor_blocks:
            x = block(x, c, causal)
        x = self.predictor_out(self.predictor_norm(x))
        return _apply_tokenwise(self.pred_proj, x)

    def _sigreg(self, z: torch.Tensor) -> torch.Tensor:
        """SIGReg on ``[B, T, D]`` across the batch, per time step."""
        with torch.no_grad():
            g = torch.Generator().manual_seed(int(self.sigreg_step.item()))
            self.sigreg_step.add_(1)
            directions = torch.randn(self.embed_dim, self.n_slices, generator=g)
            directions = directions / directions.norm(dim=0)
        directions = directions.to(device=z.device, dtype=z.dtype)
        return self.epps_pulley((z @ directions).flatten(1)).mean()

    def forward(self, video: torch.Tensor, actions: torch.Tensor) -> LeWMLosses:
        """Teacher-forced loss on clips of ``history + 1`` frames.

        Args:
            video: ``[B, history + 1, 3, H, W]``.
            actions: ``[B, history, action_dim]``, action ``t`` leads from
                frame ``t`` to frame ``t + 1``.
        """
        z, _ = self.encode(video)
        pred = self.predict(z[:, :-1], actions)
        pred_loss = F.mse_loss(pred, z[:, 1:])
        sigreg_loss = self._sigreg(z)
        return LeWMLosses(pred_loss + self.lamb * sigreg_loss, pred_loss, sigreg_loss)

    @torch.no_grad()
    def rollout(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        past_actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict global states for an action sequence.

        Args:
            z: Encoded context ``[B, h, D]``, most recent last.
            actions: ``[B, K, action_dim]`` to take from the last context frame.
            past_actions: ``[B, h - 1, action_dim]`` that led between the
                context frames.

        Returns:
            ``[B, K, D]`` predicted states after each action.
        """
        b, h = z.shape[:2]
        acts = (
            actions if past_actions is None else torch.cat([past_actions, actions], 1)
        )
        states = list(z.unbind(1))
        for k in range(actions.shape[1]):
            lo = max(0, len(states) - self.history)
            ctx = torch.stack(states[lo:], 1)
            act = acts[:, lo : len(states)]
            states.append(self.predict(ctx, act)[:, -1])
        return torch.stack(states[h:], 1)

    @torch.no_grad()
    def get_cost(self, info: dict, action_candidates: torch.Tensor) -> torch.Tensor:
        """Squared distance of the final predicted state to the goal state.

        Mirrors stable-worldmodel's ``get_cost(info, action_candidates)``,
        with states encoded once outside the planner.

        Args:
            info: ``z`` ``[B, h, D]`` encoded context, ``goal_z`` ``[B, D]``,
                optional ``past_actions`` ``[B, h - 1, action_dim]``.
            action_candidates: ``[B, S, K, action_dim]``.

        Returns:
            ``[B, S]`` costs.
        """
        b, s = action_candidates.shape[:2]
        z = info["z"].repeat_interleave(s, 0)
        past = info.get("past_actions")
        if past is not None:
            past = past.repeat_interleave(s, 0)
        pred = self.rollout(z, action_candidates.flatten(0, 1), past)
        goal = info["goal_z"].repeat_interleave(s, 0)
        return (pred[:, -1] - goal).pow(2).sum(-1).reshape(b, s)
