import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, writeFile, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { FEATURE_NAMES, DEFAULT_PROFILE, trainModel, validateModel, createPredictor, predict } from '../public/ml/model.js';

// Controlled regression fixture: labels follow an exact log-linear relationship.
// 'slicer' exercises the eligibility gate, not a claim these test labels were sliced.
function fixture() {
  return {
    schemaVersion: 1, provenance: { kind: 'slicer', description: 'Controlled test fixture only; not printer measurements.' },
    profile: { ...DEFAULT_PROFILE },
    rows: Array.from({ length: 60 }, (_, i) => {
      const value = i % 10 + 1;
      return {
        modelId: `part-${Math.floor(i / 10)}`,
        features: Object.fromEntries(FEATURE_NAMES.map((key, j) => [key, value * (j + 1)])),
        targets: { supportG: value, hours: (value + 1) ** 0.5 - 1 },
        baseline: { supportG: value + 5, hours: 8 },
      };
    }),
  };
}

test('ridge learns held-out targets and splits by part, not orientation', () => {
  const dataset = fixture();
  const model = trainModel(dataset);
  assert.equal(model.usable, true);
  assert.deepEqual(model.enabledTargets, ['supportG', 'hours']);
  assert.deepEqual(trainModel(dataset), model, 'deterministic training');
  assert.equal(model.training.rows + model.training.holdoutRows, dataset.rows.length);
  assert.equal(model.training.holdoutInRangeRows, model.training.holdoutRows);
  assert.ok(model.training.trainModelIds.every(id => !model.training.holdoutModelIds.includes(id)));
  for (const target of ['supportG', 'hours']) {
    assert.ok(model.metrics[target].mae < .15);
    assert.ok(model.metrics[target].mae < model.metrics[target].meanBaselineMae);
    assert.ok(model.metrics[target].mae < model.metrics[target].heuristicMae);
  }
  const result = predict(model, dataset.rows[3].features);
  assert.ok(Math.abs(result.supportG - 4) < .2);
});

test('inference falls back on incompatible profiles, unseen ranges and invalid input', () => {
  const dataset = fixture();
  const model = trainModel(dataset);
  const features = dataset.rows[3].features;
  assert.equal(predict(model, features, { ...DEFAULT_PROFILE, infill: .5 }), null);
  assert.equal(predict(model, { ...features, heightMm: 1e6 }), null);
  assert.equal(predict(model, { ...features, heightMm: NaN }), null);
  assert.equal(predict(model, { ...features, heightMm: -1 }), null);
  assert.equal(predict(model, {}), null);
  const trained = createPredictor(model);
  assert.ok(trained(features).hours >= 0);
});

test('demonstration provenance never enables ML scoring', () => {
  const dataset = fixture();
  dataset.provenance.kind = 'synthetic-demo';
  const model = trainModel(dataset);
  assert.equal(model.usable, false);
  assert.equal(predict(model, dataset.rows[0].features), null);
  assert.match(model.qualityReason, /Synthetic/);
});

test('models that fail comparison or lack heuristic labels do not replace estimates', () => {
  const dataset = fixture();
  for (const row of dataset.rows) row.baseline = { ...row.targets };
  assert.equal(trainModel(dataset).usable, false, 'cannot improve on perfect baseline');
  for (const row of dataset.rows) delete row.baseline;
  const model = trainModel(dataset);
  assert.equal(model.usable, false);
  assert.match(model.qualityReason, /baselines are required/);
});

test('per-target gates retain formula support when only print time improves', () => {
  const dataset = fixture();
  for (const row of dataset.rows) row.baseline.supportG = row.targets.supportG;
  const model = trainModel(dataset);
  assert.equal(model.usable, true);
  assert.deepEqual(model.enabledTargets, ['hours']);
  const result = predict(model, dataset.rows[3].features);
  assert.equal(result.supportG, undefined);
  assert.ok(Number.isFinite(result.hours));
  assert.match(model.qualityReason, /support material retains/);
  model.enabledTargets.unshift('supportG');
  assert.throws(() => validateModel(model), /per-target quality/);
});

test('scaling and ranges use training rows only', () => {
  const dataset = fixture();
  const before = trainModel(dataset);
  const heldout = before.training.holdoutModelIds;
  for (const row of dataset.rows) if (heldout.includes(row.modelId)) {
    row.features.volumeMm3 = 1e8;
    row.targets.supportG = 1e8;
  }
  const after = trainModel(dataset);
  assert.deepEqual(after.scaling, before.scaling);
  assert.deepEqual(after.ranges, before.ranges);
  assert.deepEqual(after.coefficients, before.coefficients);
  assert.equal(after.training.holdoutInRangeRows, 0);
});

test('training rejects malformed datasets, mixed baselines and insufficient groups', () => {
  for (const mutate of [
    d => { d.rows[0].features.heightMm = Infinity; },
    d => { d.rows[0].targets.supportG = -1; },
    d => { delete d.rows[0].baseline; },
    d => { d.rows.forEach(row => { row.modelId = 'same-part'; }); },
    d => { d.profile.layerMm = 0; },
    d => { d.profile.infill = 80; },
    d => { d.provenance = {}; },
  ]) {
    const dataset = fixture(); mutate(dataset);
    assert.throws(() => trainModel(dataset));
  }
  assert.throws(() => trainModel(fixture(), { lambda: 0 }));
  assert.throws(() => trainModel(fixture(), { holdoutFraction: .9 }));
});

test('malformed model artifacts and false quality claims are rejected', () => {
  for (const mutate of [
    m => { m.schemaVersion = 99; },
    m => { m.coefficients.supportG[0] = NaN; },
    m => { m.scaling.scale[0] = 0; },
    m => { m.ranges.heightMm.min = 1e10; },
    m => { m.provenance.kind = 'synthetic-demo'; },
    m => { m.metrics.hours.heuristicMae = 0; },
    m => { m.training.holdoutModelIds = m.training.trainModelIds; },
    m => { m.training.holdoutInRangeRows = 0; },
  ]) {
    const model = trainModel(fixture()); mutate(model);
    assert.throws(() => validateModel(model));
  }
});

test('CLI trains an artifact atomically and preserves prior model on failure', async t => {
  const dir = await mkdtemp(join(tmpdir(), 'first-print-ml-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const input = join(dir, 'rows.json'), output = join(dir, 'model.json');
  await writeFile(input, JSON.stringify(fixture()));
  const run = () => spawnSync(process.execPath, ['scripts/train-model.mjs', input, output], { encoding: 'utf8' });
  const success = run();
  assert.equal(success.status, 0, success.stderr);
  const saved = await readFile(output, 'utf8');
  validateModel(JSON.parse(saved));
  await writeFile(input, '{}');
  assert.equal(run().status, 1);
  assert.equal(await readFile(output, 'utf8'), saved);
});
