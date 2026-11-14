/** Generate genuinely slicer-labelled data; never substitutes heuristic targets. */
import * as THREE from '../public/vendor/three/build/three.module.js';
import { prepare, evaluate, geometryFeatures, heuristicEstimate } from '../public/analysis.js';
import { readFile, writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { parseGcode } from './slicer-gcode.mjs';

const root = fileURLToPath(new URL('../', import.meta.url));
const profilePath = join(root, 'scripts/slicer-profile.ini');
const slicer = process.env.PRUSA_SLICER;
if (!slicer) throw new Error('Set PRUSA_SLICER to the PrusaSlicer 2.9.7 CLI executable.');
const output = resolve(process.argv[2] || join(root, 'data/slicer-training.json'));
const work = await mkdtemp(join(tmpdir(), 'first-print-slicing-'));
const keep = process.env.KEEP_SLICER_OUTPUT === '1';

function run(args) {
  const result = spawnSync(slicer, args, { encoding: 'utf8', timeout: 120_000, maxBuffer: 10_000_000 });
  if (result.error || result.status !== 0) throw new Error(`PrusaSlicer failed: ${result.error?.message || result.stderr || result.stdout}`);
  return result.stdout;
}

function shapeGeometry(points, depth) {
  const shape = new THREE.Shape();
  points.forEach(([x, y], i) => i ? shape.lineTo(x, y) : shape.moveTo(x, y));
  shape.closePath();
  return new THREE.ExtrudeGeometry(shape, { depth, bevelEnabled: false, steps: 1 });
}

function models() {
  const models = [];
  for (let i = 0; i < 6; i++) {
    const w = 12 + i * 7, h = 16 + i * 9, d = 6 + i * 4, t = 3 + i;
    models.push({ id: `box-${i}`, geometry: new THREE.BoxGeometry(w, h, d) });
    models.push({ id: `bracket-${i}`, geometry: shapeGeometry([[0, 0], [w, 0], [w, t], [t, t], [t, h], [0, h]], d) });
    models.push({ id: `tee-${i}`, geometry: shapeGeometry([[w*.35, 0], [w*.65, 0], [w*.65, h-t], [w, h-t], [w, h], [0, h], [0, h-t], [w*.35, h-t]], d) });
    const g = new THREE.CylinderGeometry(5 + i*3, 8 + i*4, h, 32);
    g.rotateX(Math.PI/2);
    models.push({ id: `taper-${i}`, geometry: g });
  }
  return models;
}

function stl(geometry) {
  const g = geometry.index ? geometry.toNonIndexed() : geometry;
  const pos = g.attributes.position.array;
  const out = Buffer.alloc(84 + pos.length / 9 * 50);
  out.write('First-Print-Right reproducible calibration geometry');
  out.writeUInt32LE(pos.length / 9, 80);
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3();
  for (let i = 0, o = 84; i < pos.length; i += 9, o += 50) {
    a.fromArray(pos, i); b.fromArray(pos, i+3); c.fromArray(pos, i+6);
    const normal = b.clone().sub(a).cross(c.clone().sub(a)).normalize();
    [normal.x,normal.y,normal.z,...pos.slice(i,i+9)].forEach((value,j) => out.writeFloatLE(value,o+j*4));
  }
  return out;
}


try {
  const version = run(['--help']).split('\n')[0].trim();
  if (!version.includes('2.9.7')) throw new Error(`Expected PrusaSlicer 2.9.7; got ${version}`);
  const profileBytes = await readFile(profilePath);
  const dataset = {
    schemaVersion: 1,
    provenance: {
      kind: 'slicer',
      description: 'PrusaSlicer 2.9.7 estimates from 24 procedural calibration solids in 8 poses each. Labels are slicer estimates, not measured print outcomes; limited demonstration dataset.',
      slicerVersion: version,
      profileFile: 'scripts/slicer-profile.ini',
      profileSha256: createHash('sha256').update(profileBytes).digest('hex'),
      generatedAt: new Date().toISOString(),
      labelMethod: 'Normal-mode estimated duration; positive XY extrusion under Support material/Support material interface TYPE tags, converted with 1.75 mm filament and PLA density 1.24 g/cm3.',
      orientationDirections: [[0,0,1],[0,0,-1],[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[1,1,1],[1,-1,1]],
    },
    profile: { material: 'PLA', layerMm: 0.2, infill: 0.15, supportDensity: 0.15, wallMm: 1.2, printerWatts: 120 },
    rows: [],
  };
  for (const model of models()) {
    const prepared = prepare(model.geometry);
    for (const [pose, vector] of dataset.provenance.orientationDirections.entries()) {
      const up = new THREE.Vector3(...vector).normalize();
      const evaluation = evaluate(prepared, up);
      const g = prepared.geometry.clone().applyQuaternion(new THREE.Quaternion().setFromUnitVectors(up,new THREE.Vector3(0,0,1)));
      g.computeBoundingBox();
      const bounds = g.boundingBox;
      g.translate(-(bounds.min.x+bounds.max.x)/2, -(bounds.min.y+bounds.max.y)/2,-bounds.min.z);
      const file = join(work, `${model.id}-${pose}.stl`), gcode = join(work, `${model.id}-${pose}.gcode`);
      await writeFile(file, stl(g));
      run(['--datadir', join(work,'settings'), '--load', profilePath, '--export-gcode', '--center', '110,110', '--output',gcode,'--threads','2',file]);
      const targets = parseGcode(await readFile(gcode,'utf8'));
      const baseline = heuristicEstimate(prepared,evaluation);
      dataset.rows.push({modelId:model.id, orientation: vector, features:geometryFeatures(prepared,evaluation), targets,baseline:{supportG:baseline.supportG,hours:baseline.hours}});
      console.log(`${dataset.rows.length}/192 ${model.id} pose ${pose}: ${targets.supportG.toFixed(3)} g support, ${(targets.hours*60).toFixed(1)} min`);
    }
  }
  await mkdir(resolve(output,'..'),{recursive:true});
  await writeFile(output,JSON.stringify(dataset,null,2)+'\n');
  console.log(`Saved ${dataset.rows.length} real slicer-labelled rows to ${output}`);
} finally {
  if (keep) console.log(`Slicer outputs retained at ${work}`);
  else await rm(work,{recursive:true,force:true});
}
