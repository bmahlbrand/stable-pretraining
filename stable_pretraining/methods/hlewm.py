"""HLeWM: Hierarchical LeWorldModel.

A JEPA world model whose per-frame embeddings come in a hierarchy of token
budgets, combined with MilliVid's coarse-to-fine rollout schedule.

Each frame is encoded into ``num_levels`` levels of embedding tokens (64, 16
and 4 tokens per frame by default). A predictor fills in the embeddings of
target frames at one level, conditioned on context frames whose levels follow
the schedule: one fine frame, medium frames nearby, many coarse frames further
out. Training follows LeWorldModel: the encoder and predictor are trained end
to end with a next-embedding prediction loss plus SIGReg, with no EMA teacher,
stop-gradient, decoder or pretrained encoder.

Differences from the two source methods:

- From LeWorldModel: frames are embedded as token sets at several levels
  rather than one ``[CLS]`` vector, and the predictor is a masked (non-causal)
  transformer that predicts one schedule step at a time instead of a causal
  next-frame predictor. SIGReg is applied separately to each level.
- From MilliVid: there is no pixel autoencoder and no diffusion. Embeddings
  are learned by prediction alone, so by default each level is encoded only
  from the image at its own resolution (``level_attention="separate"``). This
  prevents a coarse level from absorbing fine detail that would make
  fine-level targets trivially predictable from the same frame's coarse
  context. ``level_attention="joint"`` matches MilliVid's encoder.

References:
    Maes et al. "LeWorldModel: Stable End-to-End Joint-Embedding Predictive
    Architecture from Pixels." 2026. https://arxiv.org/abs/2603.19312

    Chandratreya et al. "MilliVid: Hierarchical Latents for Long-Range
    Consistency in Video Generation." 2026. https://arxiv.org/abs/2606.09056

Example::

    from stable_pretraining.methods import HLeWM

    model = HLeWM(image_size=64, patch_size=8, num_levels=3)
    video = torch.randn(2, model.schedule.chunk_num_frames, 3, 64, 64)

    model.train()
    output = model(video)
    output.loss.backward()

    model.eval()
    with torch.no_grad():
        rollout = model.rollout(video[:, :64], num_frames=112)
    finest = rollout.embeddings[0]  # [B, 112, 64, embed_dim]
"""

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.utils import ModelOutput

from stable_pretraining import Module
from stable_pretraining.methods.lejepa import EppsPulley

UNSEEN = 0
CONTEXT = 1
TARGET = 2


############
# Schedule #
############


class HierarchicalSchedule:
    """Rollout schedule over a chunk of frames and embedding levels.

    A schedule is a sequence of steps. Each step is a ``[num_levels,
    chunk_num_frames]`` array whose entries say whether a (level, frame) pair
    is unseen, context, or a prediction target in that step. Running the steps
    in order over a chunk, then sliding the chunk forward, rolls a video out.

    Use :meth:`millivid` for the coarse-to-fine schedule or
    :meth:`full_resolution` for the finest-level-only baseline.

    Args:
        selections: Step arrays, each of shape ``[num_levels, chunk_num_frames]``.
        groups: One sampling group per step. During training each group is
            sampled with equal probability, then a step uniformly within it.
        tokens_per_level: Tokens per frame at each level, finest first.
    """

    def __init__(
        self,
        selections: Sequence[np.ndarray],
        groups: Sequence[int],
        tokens_per_level: Sequence[int],
    ):
        if len(selections) == 0:
            raise ValueError("A schedule needs at least one step.")
        shape = selections[0].shape
        for selection in selections:
            if selection.shape != shape:
                raise ValueError("All schedule steps must have the same shape.")
            if not np.isin(selection, (UNSEEN, CONTEXT, TARGET)).all():
                raise ValueError("Schedule entries must be UNSEEN, CONTEXT or TARGET.")
            if not (selection == TARGET).any():
                raise ValueError("Every schedule step needs at least one target.")
        if len(tokens_per_level) < shape[0]:
            raise ValueError("tokens_per_level must cover every schedule level.")

        self.selections = tuple(np.asarray(s, dtype=np.int64) for s in selections)
        self.groups = tuple(groups)
        self.num_levels, self.chunk_num_frames = shape
        self.tokens_per_level = tuple(int(n) for n in tokens_per_level[: shape[0]])

        counts: dict[int, int] = {}
        for group in self.groups:
            counts[group] = counts.get(group, 0) + 1
        self.probabilities = np.array(
            [1.0 / (counts[g] * len(counts)) for g in self.groups]
        )

    @classmethod
    def millivid(
        cls,
        num_frames_per_level: Sequence[int],
        extra_budget: int,
        tokens_per_level: Sequence[int],
    ) -> "HierarchicalSchedule":
        """MilliVid's coarse-to-fine schedule.

        Ported from the reference implementation's ``StrategyMilliVid``. The
        coarsest level predicts ``num_frames_per_level[-1]`` frames, then each
        finer level recursively fills in ``num_frames_per_level[l]`` frames,
        down to the finest level. Context at every level is the frames already
        known or predicted at that level, plus ``extra_budget`` tokens' worth of
        extra history at the target level.

        Args:
            num_frames_per_level: Frames predicted per step at each level,
                finest first. Each must be a multiple of the previous one.
            extra_budget: Extra target-level context, in tokens.
            tokens_per_level: Tokens per frame at each level, finest first.

        Returns:
            The schedule.
        """
        windows = [int(w) for w in num_frames_per_level]
        num_levels = len(windows)
        for finer, coarser in zip(windows, windows[1:]):
            if coarser % finer != 0:
                raise ValueError(
                    "Each level's window must be a multiple of the next finer one."
                )
        extra = [
            extra_budget // int(tokens_per_level[lvl]) for lvl in range(num_levels)
        ]
        chunk = sum(windows) + extra[-1]
        starts = [chunk - windows[-1] - windows[lvl] for lvl in range(num_levels)]
        target_level = [-1]

        def overlap(start_a, end_a, start_b, end_b):
            start = start_a if start_b is None else max(start_a, start_b)
            end = end_a if end_b is None else min(end_a, end_b)
            return end - start if start < end else 0

        def traverse(level):
            starts[level] += windows[level]
            target_level[0] = level
            yield
            if level > 0:
                for _ in range(windows[level] // windows[level - 1]):
                    yield from traverse(level - 1)

        def read_step():
            selection = np.zeros((num_levels, chunk), dtype=np.int64)
            left = right = starts[0]
            for level in range(num_levels):
                start = starts[level]
                end = start + windows[level]
                if level == target_level[0]:
                    selection[level, start:end] = TARGET
                    left -= extra[level]
                    selection[level, left : left + extra[level]] = CONTEXT
                    continue
                earlier = overlap(start, end, None, left)
                middle = overlap(start, end, left, right)
                later = overlap(start, end, right, None)
                if earlier + middle + later != windows[level]:
                    raise ValueError("Inconsistent schedule windows.")
                left = left - earlier - middle
                right = right + later
                selection[level, left : left + earlier + middle] = CONTEXT
                selection[level, right - later : right] = CONTEXT
            if left < 0:
                raise ValueError("Schedule context runs past the start of the chunk.")
            return selection, target_level[0]

        steps = [read_step() for _ in traverse(num_levels - 1)]
        return cls([s for s, _ in steps], [g for _, g in steps], tokens_per_level)

    @classmethod
    def full_resolution(
        cls,
        num_frames: int,
        tokens_per_level: Sequence[int],
    ) -> "HierarchicalSchedule":
        """Finest-level-only baseline: first half context, second half targets.

        Matches MilliVid's ``StrategyBaseline``. Pick ``num_frames`` so that
        ``num_frames * tokens_per_level[0]`` equals the hierarchical schedule's
        per-step budget for an equal-compute comparison.

        Args:
            num_frames: Frames per chunk (context plus targets).
            tokens_per_level: Tokens per frame at each level, finest first.

        Returns:
            The schedule.
        """
        if num_frames < 2:
            raise ValueError("num_frames must be at least 2.")
        selection = np.full((1, num_frames), TARGET, dtype=np.int64)
        selection[:, : num_frames // 2] = CONTEXT
        return cls([selection], [0], tokens_per_level)

    @property
    def num_steps(self) -> int:
        return len(self.selections)

    @property
    def tokens_per_step(self) -> tuple[int, ...]:
        """Context plus target tokens in each step."""
        per_level = np.array(self.tokens_per_level)[:, None]
        return tuple(int(((s != UNSEEN) * per_level).sum()) for s in self.selections)

    @property
    def first_target_frame(self) -> int:
        """Earliest frame, within a chunk, that any step predicts."""
        return min(int(np.where(s == TARGET)[1].min()) for s in self.selections)

    @property
    def frames_per_chunk_advance(self) -> int:
        """Frames newly predicted per chunk during rollout."""
        return self.chunk_num_frames - self.first_target_frame

    def sample_step(self, generator: Optional[torch.Generator] = None) -> int:
        """Draw a training step index, weighting sampling groups equally."""
        probs = torch.as_tensor(self.probabilities, dtype=torch.float64)
        return int(torch.multinomial(probs, 1, generator=generator).item())

    def chunk_starts(self, start: int, end: int) -> list[int]:
        """Chunk offsets that roll frames ``start`` to at least ``end`` out.

        Args:
            start: First frame to predict; frames before it are given.
            end: Frame (exclusive) the rollout must reach.

        Returns:
            Absolute frame offsets of each chunk, in rollout order.
        """
        first = start - self.first_target_frame
        if first < 0:
            raise ValueError(
                f"Rollout needs at least {self.first_target_frame} given frames, "
                f"got {start}."
            )
        width = self.frames_per_chunk_advance
        num_chunks = -(-(end - start) // width)
        return [first + i * width for i in range(num_chunks)]


##############
# Components #
##############


def _sincos(positions: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal features for each column of ``positions``, zero-padded to ``dim``."""
    num_axes = positions.shape[-1]
    freqs_per_axis = dim // (2 * num_axes)
    omega = torch.arange(freqs_per_axis, device=positions.device, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(freqs_per_axis - 1, 1)))
    angles = positions.float()[..., None] * omega
    features = torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(-2)
    return F.pad(features, (0, dim - features.shape[-1]))


def _patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    n, c, h, w = images.shape
    p = patch_size
    x = images.reshape(n, c, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5)
    return x.reshape(n, (h // p) * (w // p), c * p * p)


def _projector(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    # BatchNorm rather than LayerNorm: LeWorldModel reports that a final
    # LayerNorm hampers SIGReg.
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


def _apply_tokenwise(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    return module(x.reshape(-1, x.shape[-1])).reshape(*x.shape[:-1], -1)


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout
        self.qkv = nn.Linear(dim, 3 * heads * dim_head, bias=False)
        self.out = nn.Linear(heads * dim_head, dim)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q, k, v = (
            t.reshape(b, s, self.heads, self.dim_head).transpose(1, 2)
            for t in self.qkv(x).chunk(3, dim=-1)
        )
        drop = self.dropout if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=drop)
        return self.out(x.transpose(1, 2).reshape(b, s, -1))


class _Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads, dim // heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        return x + self.mlp(self.norm2(x))


class _AdaLNBlock(nn.Module):
    """Transformer block with per-token AdaLN-zero conditioning, as in LeWorldModel."""

    def __init__(
        self, dim: int, heads: int, dim_head: int, mlp_dim: int, dropout: float
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = _Attention(dim, heads, dim_head, dropout)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, -1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1) + shift1)
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2) + shift2)


class HierarchicalFrameEncoder(nn.Module):
    """ViT that embeds each frame as a pyramid of token levels.

    The frame is downsampled by 2 per level and every level is patchified with
    the same patch size and shared patch weights, so level ``l`` has
    ``(image_size / patch_size / 2**l) ** 2`` tokens. All levels go through one
    transformer, with a learned level embedding and 2D sine-cosine positions.

    Args:
        image_size: Input height and width.
        patch_size: Patch size, used at every level.
        num_levels: Number of levels.
        in_channels: Image channels.
        dim: Transformer width.
        depth: Transformer depth.
        heads: Attention heads.
        embed_dim: Output embedding size per token.
        projector_hidden_dim: Hidden size of the BatchNorm MLP projector.
        level_attention: ``"separate"`` restricts attention to tokens of the
            same level; ``"joint"`` lets all levels attend to each other.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_levels: int,
        in_channels: int = 3,
        dim: int = 192,
        depth: int = 12,
        heads: int = 3,
        embed_dim: int = 192,
        projector_hidden_dim: int = 2048,
        level_attention: Literal["separate", "joint"] = "separate",
    ):
        super().__init__()
        coarsest = patch_size * 2 ** (num_levels - 1)
        if image_size % coarsest != 0:
            raise ValueError(
                f"image_size must be divisible by patch_size * 2**(num_levels - 1) "
                f"= {coarsest}, got {image_size}."
            )
        if level_attention not in ("separate", "joint"):
            raise ValueError("level_attention must be 'separate' or 'joint'.")
        self.patch_size = patch_size
        self.num_levels = num_levels
        self.level_attention = level_attention
        grid = image_size // patch_size
        self.grid_per_level = tuple(grid // 2**lvl for lvl in range(num_levels))
        self.tokens_per_level = tuple(g * g for g in self.grid_per_level)

        self.patch_embed = nn.Linear(in_channels * patch_size**2, dim)
        self.level_embedding = nn.Embedding(num_levels, dim)
        self.blocks = nn.ModuleList([_Block(dim, heads, 4 * dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.projector = _projector(dim, projector_hidden_dim, embed_dim)

        levels, rows, cols = [], [], []
        for level, g in enumerate(self.grid_per_level):
            ij = torch.cartesian_prod(torch.arange(g), torch.arange(g))
            levels.append(torch.full((g * g,), level))
            rows.append(ij[:, 0])
            cols.append(ij[:, 1])
        self.register_buffer("token_level", torch.cat(levels), persistent=False)
        self.register_buffer(
            "token_rowcol",
            torch.stack([torch.cat(rows), torch.cat(cols)], -1),
            persistent=False,
        )

    def forward(self, frames: torch.Tensor) -> list[torch.Tensor]:
        """Embed frames.

        Args:
            frames: Images ``[N, C, H, W]``.

        Returns:
            One tensor per level, finest first, each ``[N, tokens_l, embed_dim]``.
        """
        patches = []
        x = frames
        for level in range(self.num_levels):
            if level > 0:
                x = F.avg_pool2d(x, 2)
            patches.append(_patchify(x, self.patch_size))
        x = self.patch_embed(torch.cat(patches, dim=1))
        x = x + _sincos(self.token_rowcol, x.shape[-1]).to(x.dtype)
        x = x + self.level_embedding(self.token_level)

        mask = None
        if self.level_attention == "separate":
            mask = self.token_level[:, None] == self.token_level[None, :]
        for block in self.blocks:
            x = block(x, mask)
        x = _apply_tokenwise(self.projector, self.norm(x))
        return list(x.split(self.tokens_per_level, dim=1))


class HierarchicalPredictor(nn.Module):
    """Predicts target-level embeddings from scheduled multi-level context.

    Context tokens carry their embeddings; target tokens start from a learned
    query. Every token gets 3D sine-cosine positions over (frame, row, column)
    and per-token AdaLN conditioning on its level, its role (context or
    target) and, optionally, its frame's action.

    Args:
        embed_dim: Embedding size of context and predicted tokens.
        num_levels: Number of levels.
        dim: Transformer width.
        depth: Transformer depth.
        heads: Attention heads.
        dim_head: Size of each attention head.
        mlp_dim: Hidden size of each MLP.
        dropout: Dropout in attention and MLPs.
        projector_hidden_dim: Hidden size of the output BatchNorm MLP.
        action_dim: Per-frame action size, or ``None`` for an unconditional
            model.
    """

    def __init__(
        self,
        embed_dim: int,
        num_levels: int,
        dim: int = 192,
        depth: int = 6,
        heads: int = 16,
        dim_head: int = 64,
        mlp_dim: int = 2048,
        dropout: float = 0.1,
        projector_hidden_dim: int = 2048,
        action_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input = nn.Linear(embed_dim, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.query, std=0.02)
        self.level_embedding = nn.Embedding(num_levels, dim)
        self.role_embedding = nn.Embedding(2, dim)
        self.action_encoder = None
        if action_dim is not None:
            self.action_encoder = nn.Sequential(
                nn.Linear(action_dim, 4 * dim), nn.SiLU(), nn.Linear(4 * dim, dim)
            )
        self.blocks = nn.ModuleList(
            [_AdaLNBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, embed_dim)
        self.projector = _projector(embed_dim, projector_hidden_dim, embed_dim)

    def forward(
        self,
        context: torch.Tensor,
        context_index: torch.Tensor,
        target_index: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict target embeddings.

        Args:
            context: Context embeddings ``[B, S_c, embed_dim]``.
            context_index: ``[S_c, 4]`` (frame, level, row, column) per context
                token, with frames relative to the step's first frame.
            target_index: ``[S_t, 4]`` indices of the target tokens.
            actions: Optional ``[B, F, action_dim]`` actions indexed by the
                same relative frames.

        Returns:
            Predicted target embeddings ``[B, S_t, embed_dim]``.
        """
        b = context.shape[0]
        index = torch.cat([context_index, target_index])
        num_context = context_index.shape[0]
        x = torch.cat(
            [
                self.input(context),
                self.query.expand(b, target_index.shape[0], -1),
            ],
            dim=1,
        )
        x = x + _sincos(index[:, [0, 2, 3]], x.shape[-1]).to(x.dtype)

        role = torch.zeros(index.shape[0], dtype=torch.long, device=index.device)
        role[num_context:] = 1
        c = self.level_embedding(index[:, 1]) + self.role_embedding(role)
        c = c.unsqueeze(0).expand(b, -1, -1)
        if self.action_encoder is not None:
            if actions is None:
                raise ValueError(
                    "This predictor was built with action_dim; pass actions."
                )
            c = c + self.action_encoder(actions)[:, index[:, 0]]

        for block in self.blocks:
            x = block(x, c)
        x = self.output(self.norm(x[:, num_context:]))
        return _apply_tokenwise(self.projector, x)


##########
# Method #
##########


@dataclass
class HLeWMOutput(ModelOutput):
    """Output from the HLeWM forward pass.

    :ivar loss: ``pred_loss + lamb * sigreg_loss``.
    :ivar pred_loss: MSE between predicted and encoded target embeddings.
    :ivar sigreg_loss: SIGReg statistic, averaged over levels.
    :ivar embedding: Detached finest-level embedding, mean-pooled over tokens
        and frames ``[B, embed_dim]``, for probes.
    :ivar level_embeddings: Detached pooled embedding per level, finest first.
    :ivar step_index: Schedule step used for the prediction loss.
    """

    loss: torch.Tensor = None
    pred_loss: torch.Tensor = None
    sigreg_loss: torch.Tensor = None
    embedding: torch.Tensor = None
    level_embeddings: tuple = None
    step_index: int = None


@dataclass
class HLeWMRollout(ModelOutput):
    """Output from :meth:`HLeWM.rollout`.

    :ivar embeddings: Per level, finest first, ``[B, num_frames, tokens_l,
        embed_dim]``. Frames before the rollout start are encoded from the
        given video; later frames are predicted.
    :ivar predicted: Per level, ``[num_frames]`` bool marking predicted frames.
    """

    embeddings: tuple = None
    predicted: tuple = None


class HLeWM(Module):
    """HLeWM: hierarchical LeWorldModel with MilliVid's rollout budgets.

    Architecture:
        - **Encoder**: pyramid ViT emitting ``num_levels`` token levels per frame
        - **Predictor**: AdaLN transformer filling one schedule step's targets
        - **Loss**: ``pred_loss + lamb * mean_l SIGReg(level_l)``

    Each training call samples one schedule step (groups weighted equally, as
    in MilliVid), takes a ``chunk_num_frames`` window of the clip, encodes the
    frames that step uses, and predicts the step's targets. Gradients flow
    through both the context and the target embeddings, as in LeWorldModel.

    SIGReg follows LeWorldModel: for each level and each (frame, token)
    position, the Epps-Pulley test runs across the batch. Up to
    ``sigreg_max_positions`` positions per level are sampled per call to bound
    memory. Because the test's samples are batch elements, the batch size
    matters; LeWorldModel used 128.

    Args:
        image_size: Frame height and width.
        patch_size: Patch size at every level.
        num_levels: Embedding levels; level ``l`` has
            ``(image_size / patch_size / 2**l) ** 2`` tokens.
        in_channels: Frame channels.
        embed_dim: Embedding size per token.
        encoder_dim: Encoder width.
        encoder_depth: Encoder depth.
        encoder_heads: Encoder attention heads.
        level_attention: ``"separate"`` or ``"joint"``; see the module docstring.
        predictor_dim: Predictor width.
        predictor_depth: Predictor depth.
        predictor_heads: Predictor attention heads.
        predictor_dim_head: Predictor head size.
        predictor_mlp_dim: Predictor MLP hidden size.
        predictor_dropout: Predictor dropout.
        projector_hidden_dim: Hidden size of the encoder and predictor projectors.
        action_dim: Per-frame action size, or ``None`` (e.g. for CLEVRER).
        schedule: ``"millivid"`` or ``"full_resolution"``.
        num_frames_per_level: Frames predicted per step at each level for the
            MilliVid schedule, finest first; must have ``num_levels`` entries.
        extra_budget: Extra target-level context tokens for the MilliVid schedule.
        baseline_num_frames: Chunk length for the full-resolution schedule. When
            ``None``, matches the MilliVid schedule's per-step token budget.
        lamb: SIGReg weight.
        n_slices: SIGReg random projections.
        n_points: SIGReg quadrature points.
        sigreg_max_positions: Positions per level sampled for SIGReg.

    Example::

        model = HLeWM(image_size=64, patch_size=8, num_levels=3)
        video = torch.randn(4, 120, 3, 64, 64)

        model.train()
        output = model(video)
        output.loss.backward()

    Example with Manager::

        import types
        import stable_pretraining as spt


        def hlewm_forward(self, batch, stage):
            output = HLeWM.forward(self, batch["video"], batch.get("action"))
            self.log(f"{stage}/loss", output.loss, on_step=True, sync_dist=True)
            self.log(f"{stage}/pred_loss", output.pred_loss, sync_dist=True)
            self.log(f"{stage}/sigreg_loss", output.sigreg_loss, sync_dist=True)
            return {"loss": output.loss, "embedding": output.embedding}


        module = HLeWM()
        module.forward = types.MethodType(hlewm_forward, module)
        module.optim = {
            "optimizer": {"type": "AdamW", "lr": 5e-5, "weight_decay": 1e-3},
            "scheduler": {"type": "LinearWarmupCosineAnnealing"},
        }
        spt.Manager(trainer=trainer, module=module, data=data)()
    """

    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        num_levels: int = 3,
        in_channels: int = 3,
        embed_dim: int = 192,
        encoder_dim: int = 192,
        encoder_depth: int = 12,
        encoder_heads: int = 3,
        level_attention: Literal["separate", "joint"] = "separate",
        predictor_dim: int = 192,
        predictor_depth: int = 6,
        predictor_heads: int = 16,
        predictor_dim_head: int = 64,
        predictor_mlp_dim: int = 2048,
        predictor_dropout: float = 0.1,
        projector_hidden_dim: int = 2048,
        action_dim: Optional[int] = None,
        schedule: Literal["millivid", "full_resolution"] = "millivid",
        num_frames_per_level: Sequence[int] = (3, 12, 48),
        extra_budget: int = 192,
        baseline_num_frames: Optional[int] = None,
        lamb: float = 0.09,
        n_slices: int = 1024,
        n_points: int = 17,
        sigreg_max_positions: int = 256,
    ):
        super().__init__()
        self.encoder = HierarchicalFrameEncoder(
            image_size=image_size,
            patch_size=patch_size,
            num_levels=num_levels,
            in_channels=in_channels,
            dim=encoder_dim,
            depth=encoder_depth,
            heads=encoder_heads,
            embed_dim=embed_dim,
            projector_hidden_dim=projector_hidden_dim,
            level_attention=level_attention,
        )
        self.predictor = HierarchicalPredictor(
            embed_dim=embed_dim,
            num_levels=num_levels,
            dim=predictor_dim,
            depth=predictor_depth,
            heads=predictor_heads,
            dim_head=predictor_dim_head,
            mlp_dim=predictor_mlp_dim,
            dropout=predictor_dropout,
            projector_hidden_dim=projector_hidden_dim,
            action_dim=action_dim,
        )

        tokens = self.encoder.tokens_per_level
        if len(num_frames_per_level) != num_levels:
            raise ValueError("num_frames_per_level needs one entry per level.")
        millivid = HierarchicalSchedule.millivid(
            num_frames_per_level, extra_budget, tokens
        )
        if schedule == "millivid":
            self.schedule = millivid
        elif schedule == "full_resolution":
            if baseline_num_frames is None:
                baseline_num_frames = millivid.tokens_per_step[0] // tokens[0]
            self.schedule = HierarchicalSchedule.full_resolution(
                baseline_num_frames, tokens
            )
        else:
            raise ValueError("schedule must be 'millivid' or 'full_resolution'.")

        self.embed_dim = embed_dim
        self.num_levels = num_levels
        self.action_dim = action_dim
        self.lamb = lamb
        self.n_slices = n_slices
        self.sigreg_max_positions = sigreg_max_positions
        self.epps_pulley = EppsPulley(t_max=3.0, n_points=n_points)
        self.register_buffer("sigreg_step", torch.zeros((), dtype=torch.long))
        self._step_cache: dict = {}

    def _step_layout(self, step_index: int, device: torch.device) -> dict:
        """Frames to encode and token indices for one schedule step, cached."""
        key = (step_index, str(device))
        if key not in self._step_cache:
            selection = self.schedule.selections[step_index]
            visible = np.where((selection != UNSEEN).any(axis=0))[0]
            row_of_frame = {int(f): r for r, f in enumerate(visible)}
            first = int(visible.min())
            grids = self.encoder.grid_per_level
            parts = {CONTEXT: [], TARGET: []}
            for role in (CONTEXT, TARGET):
                for level in range(self.schedule.num_levels):
                    frames = np.where(selection[level] == role)[0]
                    if len(frames) == 0:
                        continue
                    g = grids[level]
                    ij = np.stack(
                        np.meshgrid(np.arange(g), np.arange(g), indexing="ij"), -1
                    )
                    ij = np.tile(ij.reshape(1, -1, 2), (len(frames), 1, 1))
                    f = np.repeat(frames[:, None], g * g, axis=1)[..., None]
                    index = np.concatenate([f - first, np.full_like(f, level), ij], -1)
                    rows = np.array([row_of_frame[int(x)] for x in frames])
                    parts[role].append((level, frames, rows, index.reshape(-1, 4)))
            layout = {
                "visible": torch.as_tensor(visible, device=device),
                "first": first,
            }
            for role, name in ((CONTEXT, "context"), (TARGET, "target")):
                layout[name] = [
                    (
                        level,
                        torch.as_tensor(frames, device=device),
                        torch.as_tensor(rows, device=device),
                    )
                    for level, frames, rows, _ in parts[role]
                ]
                index = [idx for *_, idx in parts[role]]
                layout[f"{name}_index"] = (
                    torch.as_tensor(np.concatenate(index), device=device)
                    if index
                    else torch.zeros((0, 4), dtype=torch.long, device=device)
                )
            self._step_cache[key] = layout
        return self._step_cache[key]

    def encode(self, video: torch.Tensor) -> list[torch.Tensor]:
        """Encode every frame of a clip.

        Args:
            video: ``[B, T, C, H, W]``.

        Returns:
            One tensor per level, finest first, each ``[B, T, tokens_l, embed_dim]``.
        """
        b, t = video.shape[:2]
        levels = self.encoder(video.flatten(0, 1))
        return [z.reshape(b, t, *z.shape[1:]) for z in levels]

    def _gather(self, levels: Sequence[torch.Tensor], parts: list) -> torch.Tensor:
        b = levels[0].shape[0]
        tokens = [
            levels[level][:, rows].reshape(b, -1, self.embed_dim)
            for level, _, rows in parts
        ]
        if not tokens:
            return levels[0].new_zeros(b, 0, self.embed_dim)
        return torch.cat(tokens, dim=1)

    def _sigreg(self, levels: Sequence[torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            seed = int(self.sigreg_step.item())
            self.sigreg_step.add_(1)
            g = torch.Generator(device="cpu").manual_seed(seed)
            directions = torch.randn(self.embed_dim, self.n_slices, generator=g)
            directions = directions / directions.norm(dim=0)
        directions = directions.to(device=levels[0].device, dtype=levels[0].dtype)
        losses = []
        for level, z in enumerate(levels):
            z = z.flatten(1, -2)
            positions = z.shape[1]
            if positions > self.sigreg_max_positions:
                pick = torch.randperm(positions, generator=g)[
                    : self.sigreg_max_positions
                ]
                z = z[:, pick.to(z.device)]
            projected = (z @ directions).flatten(1)
            losses.append(self.epps_pulley(projected).mean())
        return torch.stack(losses).mean()

    def forward(
        self,
        video: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        step_index: Optional[int] = None,
    ) -> HLeWMOutput:
        """Compute the HLeWM loss on one schedule step.

        Args:
            video: Clip ``[B, T, C, H, W]`` with ``T >= schedule.chunk_num_frames``.
                In training a random window of that length is used; otherwise
                the first one.
            actions: Optional ``[B, T, action_dim]`` per-frame actions.
            step_index: Schedule step to use; sampled when ``None``.

        Returns:
            HLeWMOutput with the loss terms and pooled embeddings.
        """
        chunk = self.schedule.chunk_num_frames
        b, t = video.shape[:2]
        if t < chunk:
            raise ValueError(
                f"Clips need at least {chunk} frames for this schedule, got {t}."
            )
        offset = 0
        if self.training and t > chunk:
            offset = int(torch.randint(0, t - chunk + 1, (1,)).item())
        if step_index is None:
            step_index = self.schedule.sample_step()
        layout = self._step_layout(step_index, video.device)

        frames = layout["visible"] + offset
        levels = self.encode(video[:, frames])
        context = self._gather(levels, layout["context"])
        target = self._gather(levels, layout["target"])

        step_actions = None
        if actions is not None:
            start = offset + layout["first"]
            step_actions = actions[:, start : offset + chunk]
        prediction = self.predictor(
            context, layout["context_index"], layout["target_index"], step_actions
        )

        pred_loss = F.mse_loss(prediction, target)
        # Levels the schedule never uses (the full-resolution baseline) would get
        # SIGReg pressure but no prediction pressure, so leave them out.
        sigreg_loss = self._sigreg(levels[: self.schedule.num_levels])
        pooled = tuple(z.detach().mean(dim=(1, 2)) for z in levels)
        return HLeWMOutput(
            loss=pred_loss + self.lamb * sigreg_loss,
            pred_loss=pred_loss,
            sigreg_loss=sigreg_loss,
            embedding=pooled[0],
            level_embeddings=pooled,
            step_index=step_index,
        )

    @torch.no_grad()
    def rollout(
        self,
        video: torch.Tensor,
        num_frames: int,
        actions: Optional[torch.Tensor] = None,
    ) -> HLeWMRollout:
        """Roll the schedule out in embedding space.

        The given frames are encoded at every level. Each schedule step then
        predicts its targets from the embeddings known or predicted so far,
        chunk by chunk, until ``num_frames`` frames are covered at the finest
        level.

        Args:
            video: Given frames ``[B, T0, C, H, W]``. ``T0`` must be at least
                ``schedule.first_target_frame``.
            num_frames: Frames the finest level must reach (exclusive).
            actions: Optional ``[B, >= buffer frames, action_dim]`` actions;
                required when the model was built with ``action_dim``.

        Returns:
            HLeWMRollout with per-level embeddings for frames ``[0, num_frames)``.
        """
        b, start = video.shape[:2]
        chunk_starts = self.schedule.chunk_starts(start, num_frames)
        total = max(num_frames, chunk_starts[-1] + self.schedule.chunk_num_frames)
        given = self.encode(video)
        buffers = []
        predicted = []
        for level, z in enumerate(given[: self.schedule.num_levels]):
            buffer = z.new_zeros(b, total, *z.shape[2:])
            buffer[:, :start] = z
            buffers.append(buffer)
            predicted.append(torch.zeros(total, dtype=torch.bool, device=video.device))

        for chunk_start in chunk_starts:
            for step_index in range(self.schedule.num_steps):
                layout = self._step_layout(step_index, video.device)
                window = [
                    buf[:, chunk_start : chunk_start + self.schedule.chunk_num_frames]
                    for buf in buffers
                ]
                # Context rows index the step's visible frames; map them back
                # to window frames before gathering from the buffers.
                context = torch.cat(
                    [
                        window[level][:, frames].reshape(b, -1, self.embed_dim)
                        for level, frames, _ in layout["context"]
                    ]
                    or [video.new_zeros(b, 0, self.embed_dim)],
                    dim=1,
                )
                step_actions = None
                if actions is not None:
                    first = chunk_start + layout["first"]
                    step_actions = actions[
                        :, first : chunk_start + self.schedule.chunk_num_frames
                    ]
                prediction = self.predictor(
                    context,
                    layout["context_index"],
                    layout["target_index"],
                    step_actions,
                )
                cursor = 0
                for level, frames, _ in layout["target"]:
                    n = len(frames) * self.encoder.tokens_per_level[level]
                    values = prediction[:, cursor : cursor + n].reshape(
                        b,
                        len(frames),
                        self.encoder.tokens_per_level[level],
                        self.embed_dim,
                    )
                    absolute = frames + chunk_start
                    keep = absolute >= start
                    buffers[level][:, absolute[keep]] = values[:, keep]
                    predicted[level][absolute[keep]] = True
                    cursor += n

        return HLeWMRollout(
            embeddings=tuple(buf[:, :num_frames] for buf in buffers),
            predicted=tuple(p[:num_frames] for p in predicted),
        )
