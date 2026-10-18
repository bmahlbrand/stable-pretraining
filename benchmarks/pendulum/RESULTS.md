# LeHami on a pixel pendulum

CPU-scale check of LeHami's two headline claims (Yalovetzky, Balestriero &
LeCun, 2026): its leapfrog step preserves latent phase-space volume, and its
rollouts keep track of velocity for longer than an unconstrained predictor.
Script: `lehami_pendulum.py`.

All numbers: one seed, 3,000 training steps at batch 32 on CPU (about 25
minutes per model), 300 training and 60 held-out episodes. Treat differences
under about 0.1 R² as noise.

## Setup

Three models share everything except the predictor: a four-block ConvNet
encoder (the paper's design, narrowed to 16/32/64/128 channels for CPU), a
BatchNorm projector, SIGReg (weight 0.25) and multi-step rollout training on
their own predictions.

| Model | State | Predictor |
|---|---|---|
| `lehami` | 32-d position `q` + 32-d momentum `p` (Savitzky-Golay stencils) | Leapfrog on a separable Hamiltonian `K(p/s) + V(q, u)` |
| `phase-mlp` | same `q`, `p`, same losses | Unconstrained residual MLP `(q, p) + f(q, p, u)` |
| `lewm` | last three 64-d latents | LeWorldModel-style causal transformer |

Each model is rolled out for 60 steps from frame 3 of every held-out episode
without seeing another frame. Probes fit on the model's states of real
training frames read angle and angular velocity off the predicted states. We
use the paper's two probes: distance-weighted 25-nearest-neighbour regression
and ridge regression (alpha 1) on raw features. The latents encode velocity
nonlinearly (on real frames the linear probe reaches R² 0.73, nearest-neighbour
1.00, the same gap the paper reports), so nearest-neighbour is the headline
read-out.

## Conservative pendulum (three rod lengths, 20 Hz)

The rod lengths follow the paper's multi-rod set; the sampling rate is that of
its single-rod set (the paper's multi-rod runs use 10 Hz), so one of our steps
is half a paper step.

Angular-velocity R² of decoded rollouts, nearest-neighbour probe:

| Steps ahead | 1 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| `lehami` | 0.92 | 0.97 | **0.88** | **0.79** | **0.41** |
| `phase-mlp` | 0.93 | 0.95 | 0.76 | -0.19 | -0.10 |
| `lewm` | 0.81 | 0.97 | 0.87 | -0.59 | -0.39 |

Against the paper's LeHami at the same elapsed time (its nearest-neighbour
probe on its own 10 Hz multi-rod runs, three seeds, GPU, 100 epochs):

| Time ahead | 0.5 s | 1 s | 2 s | 3 s | 4 s |
|---|---|---|---|---|---|
| Paper, LeHami | 0.97 | 0.94 | 0.74 | - | 0.09 |
| This run, `lehami` | 0.97 | 0.88 | 0.79 | 0.41 | - |

This CPU run reproduces the paper's numbers; it does not exceed them. The
paper's LeWM baseline is not comparable here: its probe reads a single
frame's latent, which cannot hold velocity, while our `lewm` probe reads its
three-frame window.

Angle R² (sin and cos), nearest-neighbour probe:

| Steps ahead | 1 | 10 | 20 | 40 | 60 |
|---|---|---|---|---|---|
| `lehami` | 1.00 | 0.82 | **0.67** | **0.67** | **0.50** |
| `phase-mlp` | 1.00 | 0.88 | 0.55 | -0.10 | -0.41 |
| `lewm` | 1.00 | 0.90 | 0.65 | -0.15 | -0.52 |

Latent volume, log-determinant per dimension of the k-step rollout's Jacobian
in each model's own state (0 = preserved, negative = contracted):

| Steps | 1 | 10 | 30 | 60 |
|---|---|---|---|---|
| `lehami` | 0.000 | 0.000 | 0.000 | 0.000 |
| `phase-mlp` | -0.03 | -0.36 | -1.12 | -2.32 |
| `lewm` | -0.82 | -7.76 | -22.79 | -45.16 |

- All three track the pendulum about equally well for 10-20 steps (0.5-1 s).
- Beyond that only LeHami keeps reading out velocity and angle; the two
  unconstrained predictors fall to or below the mean-prediction baseline.
- LeHami's volume is exactly preserved by construction; both unconstrained
  predictors contract, the transformer by orders of magnitude.
- With the paper's linear probe LeHami's rollouts read worse (velocity R² 0.19,
  0.28 and -2.4 at 1, 10 and 20 steps): rollouts start from the causal momentum
  estimate, which differs from the centered one the probes are fit on by about
  37%, and a linear read-out is sensitive to that offset while
  nearest-neighbour lookup is not.

## Actuated pendulum

Pending.

## Reproducing

```bash
P=benchmarks/pendulum/lehami_pendulum.py
for m in lehami phase-mlp lewm; do python $P --model $m --out runs/$m; done
python $P --report runs/lehami runs/phase-mlp runs/lewm
# actuated swing-up setting
for m in lehami phase-mlp lewm; do python $P --task actuated --model $m --out runs/act_$m; done
```
