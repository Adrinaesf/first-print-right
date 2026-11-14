import test from 'node:test';
import assert from 'node:assert/strict';
import { selectOrientation } from '../public/ml/selection.js';
import { heuristicEstimate } from '../public/analysis.js';

const prepared = { volume: 100000, surface: 10000 };
const pose = (supportVol, extra = {}) => ({ height: 10, overhang: 100, contact: 400, footprint: [30, 30], supportVol, fits: true, ...extra });

test('learned predictions change ranking and consistently update material, time, and energy', () => {
  const original = pose(20000), alternative = pose(10000);
  assert.equal(selectOrientation(prepared, original, [alternative]).best, alternative);
  const predictor = features => ({ supportG: features.supportProxyMm3 === 20000 ? 1 : 20, hours: 2 });
  const learned = selectOrientation(prepared, original, [alternative], { predictor });
  assert.equal(learned.mode, 'ml');
  assert.equal(learned.best, original);
  const originalEstimate = learned.estimates.get(original);
  assert.equal(originalEstimate.supportG, 1);
  assert.equal(originalEstimate.totalG, heuristicEstimate(prepared, original).partG + 1);
  assert.equal(originalEstimate.hours, 2);
  assert.equal(originalEstimate.kwh, .24);
  assert.equal(originalEstimate.risk, heuristicEstimate(prepared, original).risk);
});

test('one unsupported orientation falls back to formulas for the entire comparison', () => {
  const original = pose(20000), alternative = pose(10000);
  const predictor = features => features.supportProxyMm3 === 20000 ? { supportG: 1, hours: .1 } : null;
  const result = selectOrientation(prepared, original, [alternative], { predictor });
  assert.equal(result.mode, 'heuristic');
  assert.equal(result.best, alternative);
  assert.deepEqual(result.estimates.get(original), heuristicEstimate(prepared, original));
  assert.deepEqual(result.estimates.get(alternative), heuristicEstimate(prepared, alternative));
  const switched = selectOrientation(prepared, original, [alternative], { predictor: () => ({ supportG: 0, hours: 0 }), preferML: false });
  assert.equal(switched.mode, 'heuristic');
  assert.equal(switched.best, result.best);
  assert.deepEqual(switched.estimates, result.estimates);
});

test('printer fit is a hard constraint, independent of predicted cost', () => {
  const original = pose(20000), invalid = pose(1, { fits: false });
  const predictor = features => ({ supportG: features.supportProxyMm3 === 1 ? 0 : 1e9, hours: 1 });
  assert.equal(selectOrientation(prepared, original, [invalid], { predictor }).best, original);
  const fitting = pose(1e10);
  assert.equal(selectOrientation(prepared, invalid, [fitting], { predictor }).best, fitting);
  assert.equal(selectOrientation(prepared, invalid, [], { predictor }).best, invalid);
});

test('meaningful improvement rule preserves original below five percent', () => {
  const original = pose(20000), alternative = pose(10000);
  const result = support => selectOrientation(prepared, original, [alternative], {
    predictor: features => ({ supportG: features.supportProxyMm3 === 20000 ? 10 : support, hours: 1 }),
  });
  assert.equal(result(9.6).best, original);
  assert.equal(result(9).best, alternative);
});

test('non-finite or negative predictions cannot influence orientation scores', () => {
  const original = pose(20000), alternative = pose(10000);
  for (const value of [{ supportG: NaN, hours: 1 }, { supportG: 1, hours: Infinity }, { supportG: -1, hours: 1 }, { supportG: 1, hours: -1 }]) {
    const result = selectOrientation(prepared, original, [alternative], { predictor: () => value });
    assert.equal(result.mode, 'heuristic');
    assert.equal(result.best, alternative);
  }
});

test('time-only ML can change ranking while support and total plastic retain formula estimates', () => {
  const original = pose(20000), alternative = pose(10000);
  assert.equal(selectOrientation(prepared, original, [alternative]).best, alternative);
  const predictor = features => ({ hours: features.supportProxyMm3 === 20000 ? .1 : 10 });
  const result = selectOrientation(prepared, original, [alternative], { predictor });
  assert.equal(result.mode, 'ml');
  assert.equal(result.best, original);
  for (const e of [original, alternative]) {
    const expected = heuristicEstimate(prepared, e), actual = result.estimates.get(e);
    assert.equal(actual.supportG, expected.supportG);
    assert.equal(actual.totalG, expected.totalG);
    assert.equal(actual.partG, expected.partG);
    assert.equal(actual.hours, predictor({ supportProxyMm3: e.supportVol }).hours);
    assert.equal(actual.kwh, actual.hours * .12);
  }
});

test('empty, invalid, or inconsistent partial predictions force one coherent formula fallback', () => {
  const original = pose(20000), alternative = pose(10000);
  for (const predictor of [
    () => ({}),
    () => ({ hours: NaN }),
    () => ({ hours: -1 }),
    () => ({ supportG: Infinity }),
    features => features.supportProxyMm3 === 20000 ? { hours: 1 } : { supportG: 2 },
  ]) {
    const result = selectOrientation(prepared, original, [alternative], { predictor });
    assert.equal(result.mode, 'heuristic');
    assert.equal(result.best, alternative);
    assert.deepEqual(result.estimates.get(original), heuristicEstimate(prepared, original));
    assert.deepEqual(result.estimates.get(alternative), heuristicEstimate(prepared, alternative));
  }
});
