import { geometryFeatures, heuristicEstimate } from '../analysis.js';

function weightedScore(estimate, evaluation) {
  let score = estimate.supportG * 3 + estimate.hours * 6;
  if (estimate.contactCm2 < 2) score += 3 * (2 - estimate.contactCm2);
  if (evaluation.height / Math.sqrt(Math.max(evaluation.contact, 1)) > 8) score += 4;
  return score;
}

/** Use one estimator for the whole search: never compare ML scores with formula scores. */
export function selectOrientation(prepared, original, candidates, { predictor = null, preferML = true } = {}) {
  const evaluations = [original, ...candidates];
  const feasible = evaluations.filter(e => e.fits);
  const predictions = new Map();
  let mode = 'heuristic';
  let reason = preferML ? 'No validated ML model is available. Using geometry formulas.' : 'Geometry formulas selected.';
  if (preferML && predictor) {
    // Include the original for a meaningful before/after comparison, even when it does not fit.
    const needed = [...new Set([original, ...feasible])];
    let targetMask = null;
    for (const e of needed) {
      const prediction = predictor(geometryFeatures(prepared, e));
      const targets = prediction ? ['supportG', 'hours'].filter(key => Object.hasOwn(prediction, key)) : [];
      if (!targets.length || !targets.every(key => Number.isFinite(prediction[key]) && prediction[key] >= 0)) break;
      if (targetMask !== null && targetMask !== targets.join(',')) break;
      targetMask = targets.join(',');
      predictions.set(e, prediction);
    }
    if (predictions.size === needed.length) {
      mode = 'ml';
      const targets = Object.keys(predictions.get(original));
      reason = `ML predicts ${targets.map(key => key === 'hours' ? 'print time' : 'support material').join(' and ')}; other estimates and failure risk use geometry formulas.`;
    } else {
      predictions.clear();
      reason = 'This part or an orientation is outside the model’s training range. Using geometry formulas for the entire comparison.';
    }
  }
  const estimates = new Map(evaluations.map(e => {
    const base = heuristicEstimate(prepared, e);
    const prediction = mode === 'ml' ? predictions.get(e) : null;
    const supportG = prediction?.supportG ?? base.supportG;
    const hours = prediction?.hours ?? base.hours;
    const result = prediction ? { ...base, supportG, totalG: base.partG + supportG,
      hours, kwh: hours * 0.12 } : base;
    return [e, result];
  }));
  // A non-fitting pose is never eligible when a fitting pose exists.
  const eligible = feasible.length ? feasible : [original];
  let best = eligible.includes(original) ? original : eligible[0];
  let bestScore = weightedScore(estimates.get(best), best);
  for (const e of eligible) {
    const score = weightedScore(estimates.get(e), e);
    if (score < bestScore) { best = e; bestScore = score; }
  }
  if (original.fits && best !== original && bestScore > weightedScore(estimates.get(original), original) * .95) best = original;
  return { best, mode, reason, estimates, targets: mode === 'ml' ? Object.keys(predictions.get(original)) : [] };
}
