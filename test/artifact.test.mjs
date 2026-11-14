import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import * as THREE from '../public/vendor/three/build/three.module.js';
import { prepare, evaluate, candidateDirections } from '../public/analysis.js';
import { trainModel, createPredictor } from '../public/ml/model.js';
import { selectOrientation } from '../public/ml/selection.js';

const dataset = JSON.parse(await readFile(new URL('../data/slicer-training.json', import.meta.url)));
const artifact = JSON.parse(await readFile(new URL('../public/ml/model.json', import.meta.url)));

test('shipped model reproduces from real slicer rows and the recorded fixed profile', async () => {
  const profile = await readFile(new URL('../scripts/slicer-profile.ini', import.meta.url));
  assert.equal(createHash('sha256').update(profile).digest('hex'), dataset.provenance.profileSha256);
  assert.equal(dataset.provenance.kind, 'slicer');
  assert.equal(dataset.rows.length, 192);
  assert.equal(new Set(dataset.rows.map(row => row.modelId)).size, 24);
  const retrained = trainModel(dataset);
  // JSON number comparison permits machine roundoff across supported V8/CPU versions.
  const compare = (a, b) => {
    if (typeof a === 'number') return assert.ok(Math.abs(a - b) <= 1e-10 * Math.max(1, Math.abs(a)), `${a} vs ${b}`);
    if (a && typeof a === 'object') {
      assert.deepEqual(Object.keys(a), Object.keys(b));
      for (const key of Object.keys(a)) compare(a[key], b[key]);
    } else assert.equal(a, b);
  };
  compare(artifact, retrained);
  assert.deepEqual(artifact.enabledTargets, ['hours']);
  assert.ok(artifact.metrics.hours.mae < artifact.metrics.hours.heuristicMae);
  assert.ok(artifact.metrics.supportG.mae > artifact.metrics.supportG.heuristicMae);
});

test('real deployed artifact powers a full search and gracefully falls back for unseen scales', () => {
  const run = scale => {
    const prepared = prepare(new THREE.BoxGeometry(26 * scale, 34 * scale, 14 * scale));
    const original = evaluate(prepared, new THREE.Vector3(0, 0, 1));
    return selectOrientation(prepared, original, candidateDirections(prepared).map(u => evaluate(prepared, u)), { predictor: createPredictor(artifact) });
  };
  const result = run(1);
  assert.equal(result.mode, 'ml');
  assert.deepEqual(result.targets, ['hours']);
  for (const values of result.estimates.values()) {
    assert.ok(Number.isFinite(values.hours) && values.hours >= 0);
    assert.ok(values.totalG >= values.supportG);
  }
  assert.equal(run(.01).mode, 'heuristic');
});
