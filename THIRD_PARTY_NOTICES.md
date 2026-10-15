# Third-party components

## Jet

`stable_pretraining/backbone/jet.py` adapts the transformer conditioner and
affine-coupling architecture from [btrude/jet-pytorch](https://github.com/btrude/jet-pytorch),
commit `d71d0dcfdc8b190f9b01795fd188742ce669130d`, distributed under Apache-2.0.
The full license is in `stable_pretraining/backbone/licenses/JET_LICENSE` and is
included in built distributions. The rest of stable-pretraining retains its MIT license.

Changes include BCHW input, indexed permutations, a configurable exponential
scale with a lower floor, FP32-or-higher affine arithmetic and log determinants,
diagnostics, activation checkpointing, and checkpoint configuration validation.
This port uses its own state-dict layout; existing upstream and deepstats
checkpoints require an explicit conversion and cannot be loaded directly.

## MilliVid

The `HierarchicalSchedule.millivid` rollout schedule in
`stable_pretraining/methods/hlewm.py` is ported from `StrategyMilliVid` in
[dcharatan/millivid](https://github.com/dcharatan/millivid), commit
`d68b0de74e5497ff4fd20121f46ed6078ef81eda`, distributed under the MIT license.
The pyramid patchification in `HierarchicalFrameEncoder` follows the same
repository's autoencoder. The license is in
`stable_pretraining/backbone/licenses/MILLIVID_LICENSE`.

## LeWorldModel

The AdaLN-zero predictor block, BatchNorm projectors and SIGReg setup in
`stable_pretraining/methods/hlewm.py` follow
[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm), commit
`8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, distributed under the MIT license.
The license is in `stable_pretraining/backbone/licenses/LEWM_LICENSE`.
