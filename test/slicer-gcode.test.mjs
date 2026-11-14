import test from 'node:test';
import assert from 'node:assert/strict';
import { parseGcode } from '../scripts/slicer-gcode.mjs';

const gramsPerMm = Math.PI * (1.75 / 2) ** 2 * 1.24 / 1000;
const close = (actual, expected) => assert.ok(Math.abs(actual - expected) < 1e-12, `${actual} vs ${expected}`);

test('slicer labels distinguish support/interface extrusion from part plastic and pressure restores', () => {
  const labels = parseGcode(`M83
;TYPE:Perimeter
G1 X10 Y10 E12
;TYPE:Support material
G1 X20 E.5
G1 E-1
G1 E1
G1 X30 Y20 E1.5
;TYPE:Support material interface
G1 Y25 E.25
;TYPE:Internal infill
G1 X10 E20
; estimated printing time (normal mode) = 1h 2m 3s
`);
  close(labels.supportG, 2.25 * gramsPerMm);
  close(labels.hours, 3723 / 3600);
});

test('slicer labels handle absolute extrusion, G92 resets, and switching to relative mode', () => {
  const labels = parseGcode(`M82
G92 E0
;TYPE:Support material
G1 X0 E1
G1 X10 E3
G92 E0
G1 Y10 E.5
G1 E-.5
G1 E.5
M83
G1 X20 E.25
G1 X.25 Y-.5 E.75
; estimated printing time (normal mode) = 1d 2h 3m 4s
`);
  close(labels.supportG, 4.5 * gramsPerMm);
  close(labels.hours, 26 + 3 / 60 + 4 / 3600);
});

test('slicer parsing rejects missing or zero time rather than silently inventing labels', () => {
  assert.throws(() => parseGcode(';TYPE:Support material\nG1 X10 E1'), /Missing/);
  assert.throws(() => parseGcode('; estimated printing time (normal mode) = 0s'), /Invalid/);
  close(parseGcode('; estimated printing time (normal mode) = 1m 30s').supportG, 0);
});
