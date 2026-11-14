# Model card: First-Print-Right regression v1

## Intended use

Experimental local prediction of print time for STL orientation comparison under the fixed [calibration profile](../scripts/slicer-profile.ini). The app uses learned **print time only**. Support regression was evaluated but does not pass its deployment gate, so support estimates remain formula-based. This is independent of the optional OpenAI explanation feature.

## Training data and method

- Source: actual PrusaSlicer 2.9.7 G-code estimates from **192 poses of 24 generated calibration solids** (box, L bracket, T profile, tapered cylinder; six sizes per family).
- Six axial and two oblique orientations per solid, including zero bed-contact poses.
- Fixed PLA, 0.2 mm layer, 15% infill, automatic grid support profile; no brim, skirt, raft, or support interface. Full settings and profile hash are recorded in the repository.
- Features: volume, surface area, overhang area, contact area, height, projected support-volume proxy, footprint area. Millimetre units throughout.
- Method: log1p feature/target transforms, train-only standardization, ridge regression, fixed lambda **1**, finite nonnegative predictions. Physically zero support-volume proxy yields zero support prediction.
- Split: **160 training rows / 20 parts**, **32 held-out rows / 4 parts**. No source part appears in both groups. Related size variants remain correlated; this is not a held-out-family benchmark.
- **24/32** held-out rows fall within training feature ranges. Errors below cover all held-out rows, including extrapolation; the browser refuses to extrapolate.
- No hyperparameter search against this holdout was performed. Models are fitted only on the training portion, not refitted on the validation portion after evaluation.

## Held-out results

| Target | Learned MAE | Formula MAE | Mean baseline MAE | Used in app |
| --- | ---: | ---: | ---: | --- |
| Print time | 4.75 min | 15.84 min | 39.39 min | Yes, within range |
| Support material | 0.418 g | 0.388 g | 1.069 g | No; formula retained |

On this small benchmark, time MAE falls by about **70%** relative to the formulas. This is a comparison to slicer estimates under one profile; it does not establish accuracy on physical prints, arbitrary shapes, or other printers. The fitted coefficients, RMSE, per-target gates, split IDs, and training ranges are in [the model artifact](../public/ml/model.json).

## Deployment guards

Each target must match or beat both the heuristic and mean baseline MAE to be enabled. Synthetic data, missing baselines, zero in-range holdout coverage, or failed validation cannot enable a target. Missing/corrupt models, incompatible profiles, invalid features, and any original/feasible candidate outside the feature bounds trigger a whole-search formula fallback. The same enabled target mask is required for every compared orientation.

A model inside feature bounds can still be inaccurate on an unfamiliar shape; bounds are not a confidence interval. Failure risk remains heuristic, and part plastic remains a wall/infill approximation. No print-failure probability is learned. Support highlighting remains geometric.

## Reproduction and expansion

Run `npm run train:ml` and `npm test`. Tests compare the shipped artifact with retraining and verify the slicing-profile hash. See [data collection](data-collection.md) for regenerating actual slicer labels. Gather a larger and more diverse STL corpus, and reserve entirely unseen shape families and physical print outcomes before making broader performance claims.
