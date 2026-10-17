# Hierarchical planning on TwoRoom with random doors: results

**Verdict.** At equal planning time, planning with an 8x8 spatial level did not
beat a global-state model trained the same way. The spatial dynamics predict
more accurately, especially when the agent touches a wall, and re-scoring
LeWM's candidates with them raises LeWM's success from 24% to 43% at the same
time per replan. But training LeWM's global dynamics on their own multistep
rollouts, with no spatial level, reaches 52% at that time. Most of the gain
over LeWM comes from rollout training, not from the spatial level.

Single seed, CPU only, 100 planning episodes per planner (95% intervals about
±10 points); treat differences under ~10 points as noise.

## Setup

- **Environment**: stable-worldmodel's `swm/TwoRoom-v1`, 112x112 frames, one
  model step per 5 environment steps (10-d action). Every episode samples 1-3
  doors (size and position) and puts agent and target in different rooms.
  Data: 3,000 training and 500 test episodes, half noisy expert, half
  persistent random walks (`two_room.py`).
- **Encoder and global state**: action-conditioned LeWM (`world_model.py`),
  ViT with an 8x8 patch grid, `[CLS]` -> BatchNorm projector -> 64-d global
  state, causal AdaLN predictor, SIGReg; 6,000 steps (`train_lewm.py`).
- **Spatial level**: the frozen encoder's 8x8 patch tokens. Dynamics predict
  same-grid deltas with AdaLN action conditioning and train on their own
  4-step rollouts (`spatial.py`). The global ablation is the same model and
  training on the global state as one token (`spatial.py --level global`).
- **Common score**: a readout trained to map the grid to the global state
  (held-out R^2 0.998, `readout.py`). Every planner scores a candidate by the
  squared distance of its final predicted global state to the goal's.
- **Task** (`plan.py --task near-door`): start 5-20 steps before a door on the
  expert's path, goal 25 steps further along it, budget 10 replans (50
  steps). The goal is within one planning horizon (5 model steps) but only
  through the door.

## 1. Plain LeWM training drops the agent when doors vary

With batches of unrelated clips, the encoder satisfied SIGReg with the door
layout (constant within an episode, different across episodes) and dropped
the moving agent: 0.1% of the global state's variance was within episodes,
the agent's position was not decodable (linear RMSE 114 px vs 75 px for
guessing the mean), and the predictor ignored actions. Drawing 8 clips from
each of 8 episodes per batch (`--clips-per-episode 8`) fixed it (50% within
episode variance, 7 px linear RMSE, predictor 16x better than copying the
last state). With that batching the projectors' BatchNorm running statistics
lag the weights, so they are recomputed after training.

## 2. The global state loses the doors; the grid keeps them

MLP probes on frozen features of held-out frames (`probes.py`):

| Features | Agent RMSE (px) | Passable-wall IoU | Small-door recall | Wide-door recall |
|---|---|---|---|---|
| Global state | 4.3 | 0.43 | 0.30 | 0.57 |
| 8x8 patch tokens | 3.0 | 0.88 | 0.51 | 0.97 |
| Raw pixels (CNN) | 4.1 | 0.97 | 0.90 | 0.99 |

Agent RMSE in 224x224 environment pixels (agent radius 7, success within 16).

## 3. Spatial dynamics roll out more accurately, most of all at contact

Probe-decoded agent position error of open-loop rollouts with recorded
actions on held-out episodes (`rollout_eval.py`):

| Agent RMSE (px) | 1 step | 4 | 8 | 8, touching a wall | 8, free |
|---|---|---|---|---|---|
| LeWM predictor (teacher-forced) | 5.7 | 11.7 | 19.6 | 28.1 | 16.1 |
| Global dynamics, rollout-trained | 5.1 | 8.9 | 13.3 | 19.4 | 10.7 |
| 8x8 spatial dynamics, rollout-trained | 4.1 | 6.9 | 9.9 | 13.3 | 8.6 |
| Copy last frame | 19.2 | 57.8 | 81.7 | 74 | 84 |

Rollout training and the spatial level each remove about half of LeWM's gap.
Rollouts keep the door information each representation has (passable IoU
0.88 for the grid, about 0.40 for the global state).

## 4. Planning: success at equal time per replan

CEM (stable-worldmodel's update rule), horizon 5, single-threaded, same 100
episodes for every planner:

| Planner | CEM samples x iters | Time per replan | Success [95% CI] |
|---|---|---|---|
| LeWM predictor | 30 x 5 | 0.13 s | 37% [28, 47] |
| LeWM predictor | 60 x 8 | 0.31 s | 24% [16, 33] |
| LeWM predictor | 100 x 10 | 0.56 s | 29% [20, 38] |
| LeWM predictor | 300 x 30 | 4.8 s | 30% |
| Global dynamics, rollout-trained | 30 x 5 | 0.09 s | 36% [26, 45] |
| **Global dynamics, rollout-trained** | **100 x 10** | **0.30 s** | **52% [42, 62]** |
| Selective: LeWM CEM, spatial re-scores 16 | 30 x 5 | 0.30 s | 43% [33, 53] |
| Selective: global-dynamics CEM, spatial re-scores 16 | 30 x 5 | 0.29 s | 41% [31, 50] |
| Selective: global-dynamics CEM, spatial re-scores 32 | 30 x 5 | 0.48 s | 49% [39, 59] |
| Spatial dynamics for every candidate | 16 x 4 | 0.75 s | 33% [24, 42] |
| Spatial dynamics for every candidate | 30 x 5 | 1.77 s | 47% [37, 57] |
| CEM on the true simulator (planner ceiling) | 100 x 10 | 0.02 s | 77% [69, 85] |
| Expert / random actions | - | - | 100% / 0% |

Paired differences on the same episodes:

- Selective (LeWM search) minus LeWM at the same time: **+19 points**
  [+9, +29].
- Rollout-trained global minus selective (same search model, same time): +11
  [+1, +21].
- Selective with 32 re-scored minus rollout-trained global: -3 [-13, +7]
  at 1.6x the time.
- Spatial for every candidate minus rollout-trained global: -5 [-16, +6] at
  6x the time.

LeWM's success does not grow with planning budget, so its failures are the
model's: 44 of its 71 failures at 100 x 10 end pressed against the wall in
the wrong room. The re-score picks a candidate other than CEM's mean in about
98% of replans; the shortlist is mostly near-duplicates, so this is
re-ranking among similar plans rather than a measure of disagreement.

## What this does and does not show

- The design's premise holds here: a compact global state drops door
  geometry that an 8x8 grid keeps, and grid dynamics predict contact better.
- On this task, at this scale, that accuracy does not buy planning success
  beyond what rollout training of the global model buys. Even true dynamics
  reach only 77% with this CEM and horizon, so the planner is a large part of
  the remaining gap.
- Not tested: tasks where door precision dominates (narrow doors only),
  longer horizons, learned shortlist selection or uncertainty, more seeds,
  GPU-scale encoders.

## Reproducing

```bash
P=benchmarks/world_models/planning
python $P/two_room.py --episodes 3000 --seed 0 --out data/train.pt
python $P/two_room.py --episodes 500 --seed 500000 --out data/test.pt
python $P/train_lewm.py --data data/train.pt --test-data data/test.pt \
    --out runs/lewm --clips-per-episode 8
python $P/probes.py --model runs/lewm/model.pt --train-data data/train.pt --test-data data/test.pt
python $P/spatial.py --lewm runs/lewm/model.pt --data data/train.pt --test-data data/test.pt \
    --out runs/spatial
python $P/spatial.py --level global --lewm runs/lewm/model.pt --data data/train.pt \
    --test-data data/test.pt --out runs/global_rec
python $P/readout.py --spatial runs/spatial --global-tokens runs/global_rec \
    --out runs/spatial/readout.pt
python $P/rollout_eval.py --lewm runs/lewm/model.pt --spatial runs/spatial \
    --global-dynamics runs/global_rec --train-data data/train.pt --test-data data/test.pt
# One planner; see the table for the others (--planner lewm|global-rec|spatial|selective).
python $P/plan.py --planner selective --search lewm --samples 30 --iters 5 --topk 5 \
    --shortlist 16 --model runs/lewm/model.pt --spatial runs/spatial \
    --global-dynamics runs/global_rec
python $P/plan.py --reference oracle --samples 100 --iters 10 --topk 10
```
