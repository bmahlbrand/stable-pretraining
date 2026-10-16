"""Unit tests for HLeWM: schedule invariants, encoder levels, loss and rollout."""

import numpy as np
import pytest
import torch

from stable_pretraining.methods import HLeWM
from stable_pretraining.methods.hlewm import (
    CONTEXT,
    TARGET,
    UNSEEN,
    HierarchicalFrameEncoder,
    HierarchicalSchedule,
    _token_centers,
)

pytestmark = pytest.mark.unit


def _tiny(**kwargs) -> HLeWM:
    defaults = dict(
        image_size=32,
        patch_size=4,
        num_levels=3,
        embed_dim=16,
        encoder_dim=32,
        encoder_depth=1,
        encoder_heads=2,
        predictor_dim=32,
        predictor_depth=1,
        predictor_heads=2,
        predictor_dim_head=16,
        predictor_mlp_dim=64,
        projector_hidden_dim=32,
        num_frames_per_level=(1, 4, 16),
        extra_budget=64,
        n_slices=16,
        n_points=5,
        sigreg_max_positions=32,
    )
    defaults.update(kwargs)
    torch.manual_seed(0)
    return HLeWM(**defaults)


def _video(model: HLeWM, batch: int = 3, extra_frames: int = 3) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    t = model.schedule.chunk_num_frames + extra_frames
    return torch.randn(batch, t, 3, 32, 32, generator=g)


# --- schedule -------------------------------------------------------------


@pytest.mark.parametrize(
    "windows,extra,base,chunk,steps,tokens",
    [
        ((3, 12, 48, 192), 768, 256, 447, 85, 3840),  # MilliVid paper config
        ((3, 12, 48), 192, 64, 111, 21, 768),  # HLeWM CLEVRER default
        ((1, 4, 16), 64, 64, 37, 21, 256),
    ],
)
def test_millivid_schedule_shape(windows, extra, base, chunk, steps, tokens):
    tpl = [base // 4**level for level in range(len(windows))]
    schedule = HierarchicalSchedule.millivid(windows, extra, tpl)
    assert schedule.chunk_num_frames == chunk
    assert schedule.num_steps == steps
    assert set(schedule.tokens_per_step) == {tokens}
    assert np.isclose(schedule.probabilities.sum(), 1.0)
    # The coarsest level predicts once per chunk; each finer level fills in
    # proportionally more steps.
    groups = np.array(schedule.groups)
    for level in range(len(windows)):
        assert (groups == level).sum() == 4 ** (len(windows) - 1 - level)


@pytest.mark.parametrize(
    "schedule",
    [
        HierarchicalSchedule.millivid((3, 12, 48), 192, [64, 16, 4]),
        HierarchicalSchedule.full_resolution(12, [64, 16, 4]),
    ],
)
def test_rollout_context_is_always_known(schedule):
    """Context only reads frames that were given or predicted by earlier steps."""
    start = schedule.first_target_frame + 2
    end = start + 3 * schedule.frames_per_chunk_advance
    known = set()
    for chunk_start in schedule.chunk_starts(start, end):
        for selection in schedule.selections:
            for level, frame in zip(*np.where(selection == CONTEXT)):
                absolute = chunk_start + frame
                assert absolute < start or (level, absolute) in known
            for level, frame in zip(*np.where(selection == TARGET)):
                known.add((level, chunk_start + frame))
    assert all((0, f) in known for f in range(start, end))


def test_rollout_needs_enough_given_frames():
    schedule = HierarchicalSchedule.millivid((1, 4, 16), 64, [64, 16, 4])
    with pytest.raises(ValueError):
        schedule.chunk_starts(schedule.first_target_frame - 1, 100)


def test_schedule_rejects_bad_windows():
    with pytest.raises(ValueError):
        HierarchicalSchedule.millivid((3, 10), 0, [64, 16])


# --- encoder --------------------------------------------------------------


def test_encoder_level_shapes():
    encoder = HierarchicalFrameEncoder(
        32, 4, 3, dim=32, depth=1, heads=2, embed_dim=16, projector_hidden_dim=32
    )
    levels = encoder(torch.randn(5, 3, 32, 32))
    assert [tuple(z.shape) for z in levels] == [(5, 64, 16), (5, 16, 16), (5, 4, 16)]


def test_separate_levels_ignore_finer_detail():
    """With separate attention, a level only sees the image at its own resolution."""
    torch.manual_seed(0)
    encoder = HierarchicalFrameEncoder(
        32, 4, 3, dim=32, depth=2, heads=2, embed_dim=16, projector_hidden_dim=32
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    # A checkerboard averages to zero over every 2x2 block, so it changes level
    # 0 but leaves the downsampled images of levels 1 and 2 unchanged.
    checker = (torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2 * 2.0 - 1.0
    with torch.no_grad():
        before = encoder(images)
        after = encoder(images + checker)
    assert not torch.allclose(before[0], after[0])
    assert torch.allclose(before[1], after[1], atol=1e-5)
    assert torch.allclose(before[2], after[2], atol=1e-5)

    joint = HierarchicalFrameEncoder(
        32,
        4,
        3,
        dim=32,
        depth=2,
        heads=2,
        embed_dim=16,
        projector_hidden_dim=32,
        level_attention="joint",
    ).eval()
    with torch.no_grad():
        assert not torch.allclose(joint(images)[2], joint(images + checker)[2])


def test_token_centers_line_up_across_levels():
    """A coarser token sits at the center of the finer tokens it covers."""
    centers = _token_centers(torch.tensor([[0, 0], [1, 1], [0, 1]]), torch.arange(3))
    assert centers.tolist() == [[0.0, 0.0], [2.5, 2.5], [1.5, 5.5]]

    encoder = HierarchicalFrameEncoder(
        32, 4, 3, dim=32, depth=1, heads=2, embed_dim=16, projector_hidden_dim=32
    )
    fine = encoder.token_rowcol[encoder.token_level == 0]
    coarse = encoder.token_rowcol[encoder.token_level == 2]
    assert torch.allclose(coarse[0], fine[(fine < 4).all(-1)].mean(0))


def test_encoder_rejects_indivisible_size():
    with pytest.raises(ValueError):
        HierarchicalFrameEncoder(36, 4, 3)


# --- loss -----------------------------------------------------------------


def test_forward_backward_reaches_encoder_and_predictor():
    model = _tiny().train()
    output = model(_video(model))
    assert torch.isfinite(output.loss)
    assert output.embedding.shape == (3, 16)
    assert len(output.level_embeddings) == 3
    output.loss.backward()
    for name in ("encoder.patch_embed.weight", "predictor.query"):
        grad = dict(model.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_every_schedule_step_runs():
    model = _tiny().train()
    video = _video(model, batch=2, extra_frames=0)
    for step in range(model.schedule.num_steps):
        output = model(video, step_index=step)
        assert torch.isfinite(output.loss), step
        assert output.step_index == step


def test_sigreg_sees_only_embeddings_the_step_uses(monkeypatch):
    model = _tiny().train()
    video = _video(model, batch=2, extra_frames=0)
    seen = []

    def record(levels):
        seen.append([z.shape[1] for z in levels])
        return torch.zeros(())

    monkeypatch.setattr(model, "_sigreg", record)
    for step in (0, 2):
        seen.clear()
        model(video, step_index=step)
        used = (model.schedule.selections[step] != UNSEEN).sum(axis=1)
        # A coarse step encodes every chunk frame but uses one fine frame.
        assert seen == [[int(n) for n in used if n > 0]]
        assert used[0] < used.max()


def test_full_resolution_baseline_matches_budget():
    model = _tiny(schedule="full_resolution").train()
    millivid = _tiny()
    assert model.schedule.num_levels == 1
    assert model.schedule.tokens_per_step[0] == millivid.schedule.tokens_per_step[0]
    output = model(_video(model))
    assert torch.isfinite(output.loss)


def test_actions_condition_the_predictor():
    model = _tiny(action_dim=2).train()
    video = _video(model)
    actions = torch.randn(video.shape[0], video.shape[1], 2)
    output = model(video, actions=actions, step_index=0)
    output.loss.backward()
    assert model.predictor.action_encoder[0].weight.grad is not None
    with pytest.raises(ValueError):
        model(video, step_index=0)


def test_short_clip_is_rejected():
    model = _tiny()
    with pytest.raises(ValueError):
        model(torch.randn(2, model.schedule.chunk_num_frames - 1, 3, 32, 32))


# --- rollout --------------------------------------------------------------


def test_rollout_keeps_given_frames_and_fills_the_rest():
    model = _tiny().eval()
    start = model.schedule.first_target_frame + 1
    num_frames = start + 2 * model.schedule.frames_per_chunk_advance
    video = _video(model, batch=2)[:, :start]
    rollout = model.rollout(video, num_frames=num_frames)

    encoded = model.encode(video)
    for level, (z, predicted) in enumerate(zip(rollout.embeddings, rollout.predicted)):
        tokens = model.encoder.tokens_per_level[level]
        assert z.shape == (2, num_frames, tokens, 16)
        assert torch.allclose(z[:, :start], encoded[level])
        assert not predicted[:start].any()
    assert rollout.predicted[0][start:].all()
