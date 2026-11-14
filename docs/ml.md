# ML orientation scoring

The app learns two regression targets from slicer-generated labels: support material in grams and print time in hours. Inference runs locally in the browser, independently of the optional OpenAI explanation service. The STL is not uploaded for prediction.

## Inputs and targets

The shared geometry implementation in `public/analysis.js` measures part volume, surface area, downward overhang area, bed contact area, height, a support-volume proxy, and footprint area for each orientation. These features encode orientation through its geometric effects. Inputs use millimetres, square millimetres, and cubic millimetres. Targets come from actual generated G-code, not the existing handwritten estimates.

The model uses standardized log-transformed features and ridge regression for nonnegative support/time predictions. Its artifact records the feature order, coefficients, feature bounds, training profile, data provenance, model-group split, and validation errors. See `public/ml/model.json` for the actual fitted model and results, and [data collection](data-collection.md) to reproduce the labels.

## Validation and fallback

All orientations of one part stay in the same train or validation group. This prevents orientations of the same mesh leaking across the split. The fixed validation group is reserved for evaluation; hyperparameters are not chosen by searching its results.

Only slicer-trained artifacts that pass the validation gate can be used by the app. Each target is enabled independently only when it beats or matches both the mean predictor and the original formulas on held-out mean absolute error. Targets that fail remain formula-based; the shipped artifact enables only print time. Missing or malformed artifacts, incompatible profiles, failed validation, and any original/feasible candidate outside the feature bounds cause the **entire search** to use geometry formulas. ML scores and formula scores are never mixed. Feature bounds are a basic extrapolation guard, not a guarantee of accuracy for new shapes.

The same estimator supplies the orientation ranking and before/after support and time values. The original material/time weights and bed-contact/stability penalties remain explicit. A fitting orientation always takes priority over a pose outside the build volume. You can select **Geometry formulas** in the review panel to compare the baseline; switching recalculates the orientation and displayed results together.

## What is not learned

- Failure risk remains a geometric heuristic. There is no learned failure probability: that requires recorded successful and failed physical prints, including printer/material/environment data.
- Part plastic uses the existing wall/infill approximation; total plastic combines it with the selected support estimate, and electricity derives from the selected time estimate and the 120 W assumption. Only targets passing their validation gates use ML.
- Slicer labels are simulations under one fixed profile, not measured print outcomes. Generated training shapes and a small validation set are an experimental starting point, not evidence of general performance across arbitrary STL files or printers.
- The original red support visualization is geometric; it is not an ML-generated toolpath.

## Retrain

```sh
npm run train:ml
npm test
```

To train from a different dataset without replacing the deployed artifact:

```sh
node scripts/train-model.mjs path/to/dataset.json path/to/model.json
```

Keep all variants/orientations of the same underlying part under one `modelId`. Use the exact feature extractor and fixed slicing settings for every row. Add genuinely different parts to expand coverage, then regenerate and evaluate the artifact. Changing printer or slicing settings requires a compatible profile and freshly labeled data. Do not relabel synthetic formula outputs as slicer data.
