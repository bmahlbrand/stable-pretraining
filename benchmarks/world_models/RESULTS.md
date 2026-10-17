# HLeWM evaluation: results (archived, not integrated)

**Verdict.** HLeWM (hierarchical LeWorldModel with MilliVid's coarse-to-fine
schedule) did not beat a flat LeWorldModel baseline on any test we could run,
so it was not merged. The tests were also not able to give the hierarchy a
fair chance: at this scale the JEPA + SIGReg objective that both methods share
discards object information during training (partly on plain backgrounds,
entirely on textured ones). The open question is therefore the objective as
much as the hierarchy.

All runs: CPU only, about 0.55M-parameter models, batch 64, 8,000 steps,
single seed (seed 0), linear or small-MLP probes for read-out. Treat the
numbers as indicative, not definitive.

## What was compared

| Model | Representation per frame | Context for prediction |
|---|---|---|
| HLeWM, MilliVid schedule | 64 / 16 / 4 tokens (fine / medium / coarse) | 3 fine, 12 medium, 48 coarse frames |
| HLeWM, full-resolution schedule | 64 tokens (finest level only) | 3 frames, same per-step token budget |
| LeWM (baseline, `lewm.py`) | one 128-d vector | 16 frames |

Rollouts are decoded with probes trained on the model's own embeddings of real
frames, so every model is scored relative to what its encoder retains.

## Fixes made to HLeWM during evaluation

On the branch, in `stable_pretraining/methods/hlewm.py`:

1. **SIGReg only on used embeddings.** A step encoded every level of every
   frame in its chunk but used a few (level, frame) pairs; the rest got
   SIGReg pressure with no prediction pressure.
2. **Token positions aligned across levels.** Row/column positions were in
   each level's own grid units, so tokens of different levels covering
   different image regions shared positions.
3. **Encode only used (level, frame) pairs** (`forward_levels`): identical
   embeddings in eval mode, 3.8x faster training steps on a 32x32 config and
   about 5x at the default 64/16/4-token layout.

Effect of 1 and 2 on a bouncing-ball toy (position error in px of decoded
rollouts, 32x32 frame, same data, seed and steps):

| Frames ahead | 1 | 4 | 8 | 16 |
|---|---|---|---|---|
| Original code, fine level | 8.6 | 8.2 | 7.9 | 8.6 |
| Original code, coarse level | 5.6 | 5.6 | 5.8 | 6.2 |
| Fixed, fine level | 3.2 | 5.9 | 6.5 | 7.3 |
| Fixed, coarse level | 2.4 | 2.6 | 2.7 | 4.3 |
| Full-resolution schedule | 0.9 | 1.7 | 3.4 | 6.5 |
| Copy last frame | 1.5 | 4.9 | 8.8 | 12.6 |
| Predict mean position | 8.3 | 8.0 | 7.9 | 8.0 |

With the original code the fine level was at chance: its embeddings had become
a hash of position, uncorrelated between consecutive frames. After the fixes
every level carries usable dynamics, but rollouts collapse at the first chunk
boundary (16 frames ahead), because the model only ever trains on encoded
context, never on its own predictions.

## Occluded four-digit Moving MNIST, black background (seed 0)

Four 16-px MNIST digits bounce in a 64x64 frame; an opaque band hides digits
for 13 frames at the median (25 at p90). Each test clip has a digit hidden at
the last given frame that re-emerges during the 32-frame rollout. Scores are
occupancy IoU over an 8x8 cell grid and recall on the re-emerging digits'
cells.

| Higher is better | IoU h=1 | h=4 | h=8 | h=16 | h=32 | Mean IoU | Re-emerging recall |
|---|---|---|---|---|---|---|---|
| LeWM | **0.58** | **0.47** | **0.28** | 0.15 | 0.11 | **0.218** | 0.18 |
| HLeWM, coarse level | 0.32 | 0.30 | 0.27 | **0.19** | **0.15** | 0.207 | **0.33** |
| HLeWM, medium level | 0.32 | 0.30 | 0.24 | 0.16 | 0.14 | 0.189 | 0.23 |
| HLeWM, fine level | 0.29 | 0.22 | 0.18 | 0.15 | 0.14 | 0.163 | 0.22 |
| Full-resolution schedule | 0.32 | 0.30 | 0.21 | 0.07 | 0.06 | 0.132 | 0.06 |
| Copy last frame (HLeWM probe) | 0.51 | 0.33 | 0.19 | 0.12 | 0.08 | 0.163 | 0.14 |

Memory controls (`evaluate.py`; probes fit on 256 clips for this check):

| Re-emerging recall | Own clip | Another clip's truth | Clip-specific |
|---|---|---|---|
| HLeWM, coarse level | 0.28 | 0.23 | +0.05 |
| HLeWM, fine level | 0.27 | 0.27 | 0.00 |
| LeWM | 0.17 | 0.16 | +0.01 |
| Full-resolution schedule | 0.06 | 0.07 | 0.00 |
| Fixed guess: every cell beside the band | 0.50 | - | - |

- On average HLeWM tracks no better than LeWM and is much worse at short
  horizons; it is slightly better 16-32 frames out.
- HLeWM's higher re-emerging recall is mostly a generic guess that digits
  appear next to the band: it barely drops when scored against another clip,
  and the fixed band guess scores higher still. No model clearly remembers a
  specific hidden digit.
- Probes on the true frames read occupancy at 0.54 (HLeWM fine), 0.57 (LeWM),
  0.69 (HLeWM coarse) and 0.34 (full-res) IoU, versus 0.93 from raw pixels:
  training discards much of the spatial information.

## Textured scenes: training discards the digits entirely

Same task with RGB frames: a per-clip two-tone noise texture the camera pans
across, colored digits and a textured band. Probe IoU on the true frames:

| Encoder | Occupancy IoU |
|---|---|
| Raw pixels | 0.78 |
| Untrained full-res encoder | 0.75 |
| Trained full-res encoder (8,000 steps) | 0.002 |
| Trained HLeWM, step 4,750: fine / medium / coarse | 0.009 / 0.005 / 0.001 |

The trained encoders satisfy SIGReg (unit-variance embeddings) by encoding the
background texture and its pan, which is rich and easy to predict, and drop
the small digits. Prediction loss becomes tiny because the digits are no
longer represented. These runs were stopped once this was clear; LeWM was not
run on textured scenes.

## What would be needed to settle the hierarchy question

- An objective that keeps objects in the embedding (e.g. EMA or stop-gradient
  targets, or a reconstruction auxiliary), checked first on LeWM alone.
- Real video at GPU scale (e.g. CLEVRER, HLeWM's intended target), longer
  training and several seeds.
- Rollout training on predicted context (or noise on context) for HLeWM's
  chunk boundaries.

## Reproducing

```bash
# Train and evaluate one model (resumes from <out>/ckpt.pt if present).
python benchmarks/world_models/train.py --model hlewm --seed 0 --steps 8000 --out runs/hlewm_s0
python benchmarks/world_models/train.py --model lewm --seed 0 --steps 8000 --out runs/lewm_s0
python benchmarks/world_models/train.py --model hlewm-fullres --seed 0 --steps 8000 --out runs/fullres_s0

# Re-score with the current evaluation, aggregate, and render rollouts.
python benchmarks/world_models/train.py --model hlewm --seed 0 --eval-only --out runs/hlewm_s0
python benchmarks/world_models/collect_results.py runs --plot runs/results.png
python benchmarks/world_models/render_rollouts.py runs/hlewm_s0 runs/lewm_s0 --out runs/rollouts.mp4
```

Textured scenes are the default (`DATA["textured"]` in `train.py`); the
black-background results above used `textured=False`, the default as of
commit `5228084`. The bouncing-ball toy scripts were exploratory and are not
included.
