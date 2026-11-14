/** Shared training/inference contract. All geometry dimensions are millimetres. */
export const FEATURE_NAMES = Object.freeze([
  'volumeMm3', 'surfaceMm2', 'overhangMm2', 'contactMm2',
  'heightMm', 'supportProxyMm3', 'footprintMm2',
]);
export const TARGET_NAMES = Object.freeze(['supportG', 'hours']);
export const DEFAULT_PROFILE = Object.freeze({
  material: 'PLA', layerMm: 0.2, infill: 0.15,
  supportDensity: 0.15, wallMm: 1.2, printerWatts: 120,
});
const PROFILE_KEYS = Object.keys(DEFAULT_PROFILE);
const nonnegative = value => typeof value === 'number' && Number.isFinite(value) && value >= 0;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const arrayOf = (value, length, predicate) => Array.isArray(value) && value.length === length && value.every(predicate);
const sameNames = (a, b) => arrayOf(a, b.length, (_, i) => a[i] === b[i]);

export function validateProfile(profile) {
  if (!profile || typeof profile.material !== 'string' || !profile.material.trim() ||
      !PROFILE_KEYS.slice(1).every(key => nonnegative(profile[key])) ||
      profile.layerMm <= 0 || profile.wallMm <= 0 || profile.printerWatts <= 0 ||
      profile.infill > 1 || profile.supportDensity > 1 ||
      Object.keys(profile).some(key => !PROFILE_KEYS.includes(key))) {
    throw new Error('Invalid printer/material profile.');
  }
  return profile;
}

export function profilesMatch(a, b) {
  try { validateProfile(a); validateProfile(b); } catch { return false; }
  return PROFILE_KEYS.every(key => a[key] === b[key]);
}

export function validFeatures(features) {
  return Boolean(features && FEATURE_NAMES.every(key => nonnegative(features[key])));
}

function eligibleTargets(model) {
  if (model.provenance.kind !== 'slicer' || model.training.holdoutInRangeRows === 0) return [];
  return TARGET_NAMES.filter(key => nonnegative(model.metrics[key].heuristicMae) &&
    model.metrics[key].mae <= model.metrics[key].heuristicMae &&
    model.metrics[key].mae <= model.metrics[key].meanBaselineMae);
}

export function validateModel(model) {
  if (!model || model.schemaVersion !== 1 || model.type !== 'ridge-log1p' ||
      !sameNames(model.features, FEATURE_NAMES) || !sameNames(model.targets, TARGET_NAMES) ||
      !['slicer', 'synthetic-demo'].includes(model.provenance?.kind) ||
      typeof model.provenance.description !== 'string' || !model.provenance.description.trim() ||
      typeof model.usable !== 'boolean' || typeof model.qualityReason !== 'string' || !Array.isArray(model.enabledTargets)) {
    throw new Error('Unsupported or malformed ML model.');
  }
  validateProfile(model.profile);
  if (!arrayOf(model.scaling?.mean, FEATURE_NAMES.length, finite) ||
      !arrayOf(model.scaling?.scale, FEATURE_NAMES.length, value => finite(value) && value > 0) ||
      !FEATURE_NAMES.every(key => nonnegative(model.ranges?.[key]?.min) &&
        nonnegative(model.ranges[key].max) && model.ranges[key].min <= model.ranges[key].max) ||
      !TARGET_NAMES.every(key => arrayOf(model.coefficients?.[key], FEATURE_NAMES.length + 1, finite))) {
    throw new Error('Invalid ML coefficients, scaling, or feature ranges.');
  }
  if (!Number.isInteger(model.training?.rows) || model.training.rows < 1 ||
      !Number.isInteger(model.training.groups) || model.training.groups < 1 ||
      !Number.isInteger(model.training.holdoutRows) || model.training.holdoutRows < 1 ||
      !Number.isInteger(model.training.holdoutInRangeRows) || model.training.holdoutInRangeRows < 0 || model.training.holdoutInRangeRows > model.training.holdoutRows ||
      !Array.isArray(model.training.trainModelIds) || !Array.isArray(model.training.holdoutModelIds) ||
      model.training.trainModelIds.some(id => model.training.holdoutModelIds.includes(id)) ||
      !TARGET_NAMES.every(key => ['mae', 'rmse', 'meanBaselineMae', 'meanBaselineRmse'].every(metric => nonnegative(model.metrics?.[key]?.[metric])))) {
    throw new Error('Invalid ML evaluation metadata.');
  }
  const eligible = eligibleTargets(model);
  if (!sameNames(model.enabledTargets, eligible) || model.usable !== (eligible.length > 0)) {
    throw new Error('ML model does not meet its declared per-target quality gates.');
  }
  return model;
}

function inRange(model, features) {
  return FEATURE_NAMES.every(key => {
    const { min, max } = model.ranges[key];
    const tolerance = Math.max(1, max) * 1e-8;
    return features[key] >= min - tolerance && features[key] <= max + tolerance;
  });
}

function rawPredict(model, features, targets = TARGET_NAMES) {
  const x = FEATURE_NAMES.map((key, i) => (Math.log1p(features[key]) - model.scaling.mean[i]) / model.scaling.scale[i]);
  const result = {};
  for (const target of targets) {
    const weights = model.coefficients[target];
    const logPrediction = weights[0] + x.reduce((sum, value, i) => sum + value * weights[i + 1], 0);
    const value = Math.max(0, Math.expm1(logPrediction));
    if (!Number.isFinite(value)) return null;
    result[target] = target === 'supportG' && features.supportProxyMm3 <= 1e-9 ? 0 : value;
  }
  return result;
}

/** Validate once at load; each orientation then requires only a small dot product. */
export function createPredictor(model, profile = DEFAULT_PROFILE) {
  validateModel(model);
  const enabled = model.usable && profilesMatch(model.profile, profile);
  return features => enabled && validFeatures(features) && inRange(model, features) ? rawPredict(model, features, model.enabledTargets) : null;
}

export function predict(model, features, profile = DEFAULT_PROFILE) {
  return createPredictor(model, profile)(features);
}

/** Deterministic grouped holdout prevents orientations of a part leaking across splits. */
function splitRows(rows, fraction) {
  const groups = [...new Set(rows.map(row => row.modelId))].sort();
  if (groups.length < 5 || rows.length < 20) throw new Error('Training needs at least 5 distinct models and 20 rows.');
  // FNV-1a order is independent of input order and keeps the split reproducible.
  const hash = value => { let h = 2166136261; for (const c of value) h = Math.imul(h ^ c.charCodeAt(0), 16777619); return h >>> 0; };
  groups.sort((a, b) => hash(a) - hash(b) || a.localeCompare(b));
  const holdoutIds = new Set(groups.slice(0, Math.max(1, Math.floor(groups.length * fraction))));
  return {
    train: rows.filter(row => !holdoutIds.has(row.modelId)),
    holdout: rows.filter(row => holdoutIds.has(row.modelId)),
    trainModelIds: groups.filter(id => !holdoutIds.has(id)),
    holdoutModelIds: [...holdoutIds],
  };
}

/** Partial-pivot Gaussian elimination for the small regularized normal equations. */
function solve(matrix, vector) {
  const n = vector.length;
  const a = matrix.map((row, i) => [...row, vector[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) if (Math.abs(a[row][col]) > Math.abs(a[pivot][col])) pivot = row;
    if (Math.abs(a[pivot][col]) < 1e-12) throw new Error('Training data is numerically singular.');
    [a[col], a[pivot]] = [a[pivot], a[col]];
    const divisor = a[col][col];
    for (let j = col; j <= n; j++) a[col][j] /= divisor;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = a[row][col];
      for (let j = col; j <= n; j++) a[row][j] -= factor * a[col][j];
    }
  }
  return a.map(row => row[n]);
}

function errors(actual, predicted) {
  if (!predicted.every(finite)) throw new Error('Training produced non-finite predictions.');
  const deviations = actual.map((value, i) => value - predicted[i]);
  return {
    mae: deviations.reduce((sum, d) => sum + Math.abs(d), 0) / actual.length,
    rmse: Math.sqrt(deviations.reduce((sum, d) => sum + d * d, 0) / actual.length),
  };
}

export function trainModel(dataset, { lambda = 1, holdoutFraction = 0.2 } = {}) {
  if (!dataset || dataset.schemaVersion !== 1 || !['slicer', 'synthetic-demo'].includes(dataset.provenance?.kind) ||
      typeof dataset.provenance.description !== 'string' || !dataset.provenance.description.trim() || !Array.isArray(dataset.rows)) {
    throw new Error('Invalid training dataset or provenance.');
  }
  validateProfile(dataset.profile);
  if (!finite(lambda) || lambda <= 0 || !finite(holdoutFraction) || holdoutFraction <= 0 || holdoutFraction >= 0.5) throw new Error('Use positive lambda and a holdout fraction between 0 and 0.5.');
  const rows = dataset.rows;
  const hasBaseline = rows.some(row => row?.baseline !== undefined);
  for (const row of rows) {
    if (!row || typeof row.modelId !== 'string' || !row.modelId.trim() || !validFeatures(row.features) ||
        !TARGET_NAMES.every(key => nonnegative(row.targets?.[key])) ||
        (hasBaseline && !TARGET_NAMES.every(key => nonnegative(row.baseline?.[key])))) {
      throw new Error('Invalid training row: modelId, features, targets, and consistent baselines are required.');
    }
  }
  const { train, holdout, trainModelIds, holdoutModelIds } = splitRows(rows, holdoutFraction);
  const logs = train.map(row => FEATURE_NAMES.map(key => Math.log1p(row.features[key])));
  const mean = FEATURE_NAMES.map((_, i) => logs.reduce((sum, row) => sum + row[i], 0) / train.length);
  const scale = mean.map((value, i) => Math.sqrt(logs.reduce((sum, row) => sum + (row[i] - value) ** 2, 0) / train.length) || 1);
  const x = logs.map(row => [1, ...row.map((value, i) => (value - mean[i]) / scale[i])]);
  const n = FEATURE_NAMES.length + 1;
  const gram = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => x.reduce((sum, row) => sum + row[i] * row[j], 0) + (i === j && i > 0 ? lambda : 0)));
  const coefficients = {};
  for (const target of TARGET_NAMES) {
    const rhs = Array.from({ length: n }, (_, i) => x.reduce((sum, row, k) => sum + row[i] * Math.log1p(train[k].targets[target]), 0));
    coefficients[target] = solve(gram, rhs);
  }
  const model = {
    schemaVersion: 1, type: 'ridge-log1p', features: [...FEATURE_NAMES], targets: [...TARGET_NAMES],
    profile: { ...dataset.profile }, provenance: { ...dataset.provenance },
    scaling: { mean, scale }, coefficients,
    ranges: Object.fromEntries(FEATURE_NAMES.map(key => [key, {
      min: train.reduce((value, row) => Math.min(value, row.features[key]), Infinity),
      max: train.reduce((value, row) => Math.max(value, row.features[key]), -Infinity),
    }])),
    training: { lambda, rows: train.length, groups: trainModelIds.length, holdoutRows: holdout.length,
      trainModelIds, holdoutModelIds, split: 'grouped-by-modelId', holdoutInRangeRows: 0 },
    metrics: {}, enabledTargets: [], usable: false, qualityReason: '',
  };
  model.training.holdoutInRangeRows = holdout.filter(row => inRange(model, row.features)).length;
  const predictions = holdout.map(row => rawPredict(model, row.features));
  if (predictions.some(value => value === null)) throw new Error('Training produced non-finite predictions.');
  for (const target of TARGET_NAMES) {
    const actual = holdout.map(row => row.targets[target]);
    const average = train.reduce((sum, row) => sum + row.targets[target], 0) / train.length;
    const meanErrors = errors(actual, holdout.map(() => average));
    model.metrics[target] = { ...errors(actual, predictions.map(row => row[target])), meanBaselineMae: meanErrors.mae, meanBaselineRmse: meanErrors.rmse };
    if (hasBaseline) {
      const baselineErrors = errors(actual, holdout.map(row => row.baseline[target]));
      model.metrics[target].heuristicMae = baselineErrors.mae;
      model.metrics[target].heuristicRmse = baselineErrors.rmse;
    }
  }
  model.enabledTargets = eligibleTargets(model);
  model.usable = model.enabledTargets.length > 0;
  const labels = { supportG: 'support material', hours: 'print time' };
  const disabled = TARGET_NAMES.filter(key => !model.enabledTargets.includes(key));
  model.qualityReason = dataset.provenance.kind !== 'slicer' ? 'Synthetic demonstration data is not eligible for orientation scoring.' :
    !hasBaseline ? 'Heuristic baselines are required to validate an improvement before enabling ML scoring.' :
    model.training.holdoutInRangeRows === 0 ? 'No held-out examples fall within the deployment feature ranges.' :
    !model.usable ? 'Neither target met its held-out MAE gate against the heuristic and mean baselines.' :
    `ML enabled for ${model.enabledTargets.map(key => labels[key]).join(' and ')}: held-out MAE is no worse than the heuristic and mean baselines.` +
      (disabled.length ? ` ${disabled.map(key => labels[key]).join(' and ')} retains the geometry formula because its quality gate failed.` : '');
  return validateModel(model);
}
