import * as THREE from './vendor/three/build/three.module.js';

const BED = { x: 220, y: 220, z: 250 };      // build volume, mm
const OVERHANG_DEG = 45;                       // faces steeper than this need support
const PLA_DENSITY = 1.24;                      // g/cm³
const WALL_MM = 1.2;                           // walls + top/bottom skin, approx
const INFILL = 0.15;
const SUPPORT_DENSITY = 0.15;
const FLOW_MM3_S = 6;                          // average volumetric rate incl. travel
const LAYER_MM = 0.2;
const LAYER_OVERHEAD_S = 1.5;
const PRINTER_WATTS = 120;

const COS_OVER = Math.cos(THREE.MathUtils.degToRad(90 - OVERHANG_DEG));

export function prepare(geometry) {
  let g = geometry.index ? geometry.toNonIndexed() : geometry;
  g.deleteAttribute('normal');
  g.deleteAttribute('uv');
  g.computeBoundingBox();
  // centre the raw model at the origin so rotations are about its middle
  const ctr = g.boundingBox.getCenter(new THREE.Vector3());
  g.translate(-ctr.x, -ctr.y, -ctr.z);

  const pos = g.attributes.position.array;
  const T = pos.length / 9;
  const normals = new Float32Array(T * 3), areas = new Float32Array(T);
  let volume = 0, surface = 0;
  for (let t = 0; t < T; t++) {
    const i = t * 9;
    const ax = pos[i], ay = pos[i + 1], az = pos[i + 2];
    const bx = pos[i + 3], by = pos[i + 4], bz = pos[i + 5];
    const cx = pos[i + 6], cy = pos[i + 7], cz = pos[i + 8];
    const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const len = Math.hypot(nx, ny, nz);
    areas[t] = len / 2;
    surface += len / 2;
    if (len > 0) { nx /= len; ny /= len; nz /= len; }
    normals.set([nx, ny, nz], t * 3);
    volume += (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6;
  }
  g.computeBoundingBox();
  return { geometry: g, pos, normals, areas, T, volume: Math.abs(volume), surface, size: g.boundingBox.getSize(new THREE.Vector3()) };
}

/** Evaluate printing with world-up = direction `u` (unit vector in model space). */
export function evaluate(p, u) {
  const { pos, normals, areas, T } = p;
  const ux = u.x, uy = u.y, uz = u.z;
  // two axes perpendicular to u for the footprint
  const e1 = new THREE.Vector3().crossVectors(u, Math.abs(u.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(1, 0, 0)).normalize();
  const e2 = new THREE.Vector3().crossVectors(u, e1);

  let minZ = Infinity, maxZ = -Infinity, min1 = Infinity, max1 = -Infinity, min2 = Infinity, max2 = -Infinity;
  const n = pos.length / 3;
  for (let v = 0; v < n; v++) {
    const x = pos[v * 3], y = pos[v * 3 + 1], z = pos[v * 3 + 2];
    const h = x * ux + y * uy + z * uz;
    if (h < minZ) minZ = h; if (h > maxZ) maxZ = h;
    const a = x * e1.x + y * e1.y + z * e1.z, b = x * e2.x + y * e2.y + z * e2.z;
    if (a < min1) min1 = a; if (a > max1) max1 = a;
    if (b < min2) min2 = b; if (b > max2) max2 = b;
  }

  let overhang = 0, supportVol = 0, contact = 0;
  for (let t = 0; t < T; t++) {
    const nz = normals[t * 3] * ux + normals[t * 3 + 1] * uy + normals[t * 3 + 2] * uz;
    if (nz >= -COS_OVER) continue;
    const i = t * 9;
    const z0 = pos[i] * ux + pos[i + 1] * uy + pos[i + 2] * uz - minZ;
    const z1 = pos[i + 3] * ux + pos[i + 4] * uy + pos[i + 5] * uz - minZ;
    const z2 = pos[i + 6] * ux + pos[i + 7] * uy + pos[i + 8] * uz - minZ;
    if (Math.max(z0, z1, z2) < 0.3) { if (nz < -0.95) contact += areas[t]; continue; }
    overhang += areas[t];
    supportVol += areas[t] * -nz * ((z0 + z1 + z2) / 3);
  }

  const height = maxZ - minZ, fx = max1 - min1, fy = max2 - min2;
  const fits = height <= BED.z && ((fx <= BED.x && fy <= BED.y) || (fx <= BED.y && fy <= BED.x));
  return { u: u.clone(), height, footprint: [fx, fy], overhang, supportVol, contact, fits };
}

export function heuristicEstimate(p, e) {
  const shell = Math.min(p.volume, p.surface * WALL_MM);
  const partMm3 = shell + INFILL * (p.volume - shell);
  const supportMm3 = e.supportVol * SUPPORT_DENSITY;
  const grams = (mm3) => (mm3 / 1000) * PLA_DENSITY;
  const seconds = (partMm3 + supportMm3) / FLOW_MM3_S + (e.height / LAYER_MM) * LAYER_OVERHEAD_S;
  const hours = seconds / 3600;

  let risk = 0;
  const overRatio = e.overhang / Math.max(p.surface, 1);
  if (overRatio > 0.15) risk += 2; else if (overRatio > 0.04) risk += 1;
  if (e.overhang > 5000) risk += 1;                 // > 50 cm² printed over support
  if (e.contact < 20) risk += 1;
  if (e.contact < 20 && e.height > 15) risk += 1;
  if (e.height / Math.sqrt(Math.max(e.contact, 1)) > 8) risk += 1;
  if (!e.fits) risk += 3;
  const riskLabel = risk >= 3 ? 'High' : risk >= 2 ? 'Medium' : 'Low';

  return {
    supportG: grams(supportMm3), partG: grams(partMm3), totalG: grams(partMm3 + supportMm3),
    hours, kwh: hours * PRINTER_WATTS / 1000,
    overhangCm2: e.overhang / 100, contactCm2: e.contact / 100, heightMm: e.height,
    risk: riskLabel, fits: e.fits,
  };
}

export function heuristicScore(p, e) {
  const est = heuristicEstimate(p, e);
  let s = est.supportG * 3 + est.hours * 6;          // waste weighted above time
  if (est.contactCm2 < 2) s += 3 * (2 - est.contactCm2); // little to hold it down
  if (e.height / Math.sqrt(Math.max(e.contact, 1)) > 8) s += 4; // tall and narrow
  if (!e.fits) s += 1e6;
  return s;
}

export function candidateDirections(p) {
  const dirs = [];
  const add = (x, y, z) => { const v = new THREE.Vector3(x, y, z); if (v.lengthSq() > 0) dirs.push(v.normalize()); };
  for (const x of [-1, 0, 1]) for (const y of [-1, 0, 1]) for (const z of [-1, 0, 1]) add(x, y, z);
  // even spread over the sphere
  const N = 240, golden = Math.PI * (3 - Math.sqrt(5));
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 2, r = Math.sqrt(1 - y * y), th = golden * i;
    add(Math.cos(th) * r, y, Math.sin(th) * r);
  }
  // largest flat faces placed face-down on the plate
  const groups = new Map();
  for (let t = 0; t < p.T; t++) {
    const key = [0, 1, 2].map(k => Math.round(p.normals[t * 3 + k] * 50)).join(',');
    const gr = groups.get(key) || { a: 0, n: new THREE.Vector3() };
    gr.a += p.areas[t];
    gr.n.x += p.normals[t * 3] * p.areas[t]; gr.n.y += p.normals[t * 3 + 1] * p.areas[t]; gr.n.z += p.normals[t * 3 + 2] * p.areas[t];
    groups.set(key, gr);
  }
  [...groups.values()].sort((a, b) => b.a - a.a).slice(0, 24).forEach(gr => add(-gr.n.x, -gr.n.y, -gr.n.z));
  return dirs;
}


/** Same measured features for slicer training and browser inference (mm units). */
export function geometryFeatures(p, e) {
  return { volumeMm3: p.volume, surfaceMm2: p.surface, overhangMm2: e.overhang,
    contactMm2: e.contact, heightMm: e.height, supportProxyMm3: e.supportVol,
    footprintMm2: e.footprint[0] * e.footprint[1] };
}
