import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import * as THREE from '../public/vendor/three/build/three.module.js';
import { prepare, evaluate, heuristicEstimate, heuristicScore, candidateDirections, geometryFeatures } from '../public/analysis.js';

// The recovered source is retained unchanged: compare extracted analysis against it.
const source = readFileSync(new URL('../original/app.js', import.meta.url), 'utf8');
const settings = source.slice(source.indexOf('const BED ='), source.indexOf('const COLOR_MODEL'));
const functions = source.slice(source.indexOf('function prepare(geometry)'), source.indexOf('/* ================= workflow ================= */'));
const original = vm.runInNewContext(`${settings}\nconst COS_OVER = Math.cos(THREE.MathUtils.degToRad(90 - OVERHANG_DEG));\n${functions}\n({prepare,evaluate,estimate,score,candidateDirections})`, { THREE });
const plain = value => JSON.parse(JSON.stringify(value));

test('geometry extraction preserves original estimates and candidate directions', () => {
  for (const geometry of [new THREE.BoxGeometry(20, 30, 40), new THREE.CylinderGeometry(8, 14, 35, 24), new THREE.TorusGeometry(25, 5, 8, 24)]) {
    const before = original.prepare(geometry.clone());
    const after = prepare(geometry.clone());
    assert.equal(after.volume, before.volume);
    assert.equal(after.surface, before.surface);
    assert.deepEqual([...after.normals], [...before.normals]);
    const directions = candidateDirections(after);
    assert.deepEqual(plain(directions), plain(original.candidateDirections(before)));
    for (const direction of directions) {
      const expected = original.evaluate(before, direction);
      const actual = evaluate(after, direction);
      assert.deepEqual(plain(actual), plain(expected));
      assert.deepEqual(plain(heuristicEstimate(after, actual)), plain(original.estimate(before, expected)));
      assert.equal(heuristicScore(after, actual), original.score(before, expected));
    }
  }
});

test('box measurements use mm units and distinguish bed contact from overhangs', () => {
  const part = prepare(new THREE.BoxGeometry(20, 30, 40));
  const pose = evaluate(part, new THREE.Vector3(0, 0, 1));
  assert.equal(part.volume, 24000);
  assert.equal(part.surface, 5200);
  assert.equal(pose.height, 40);
  assert.equal(pose.contact, 600);
  assert.equal(pose.overhang, 0);
  assert.equal(pose.supportVol, 0);
  assert.deepEqual(geometryFeatures(part, pose), {
    volumeMm3: 24000, surfaceMm2: 5200, overhangMm2: 0,
    contactMm2: 600, heightMm: 40, supportProxyMm3: 0, footprintMm2: 600,
  });
  for (const direction of candidateDirections(part)) {
    assert.ok(Math.abs(direction.length() - 1) < 1e-12);
    assert.ok(Object.values(geometryFeatures(part, evaluate(part, direction))).every(Number.isFinite));
  }
});
