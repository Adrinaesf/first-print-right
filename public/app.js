import * as THREE from 'three';
import { prepare, evaluate, heuristicEstimate, candidateDirections } from './analysis.js';
import { createPredictor, validateModel, profilesMatch, DEFAULT_PROFILE } from './ml/model.js';
import { selectOrientation } from './ml/selection.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

/* ================= settings (assumptions shown to the user) ================= */
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

const COLOR_MODEL = new THREE.Color('#4fbf7a');
const COLOR_OVER = new THREE.Color('#f04f4f');
const COS_OVER = Math.cos(THREE.MathUtils.degToRad(90 - OVERHANG_DEG)); // 0.707

/* ================= state ================= */
const state = {
  name: '', prep: null,
  original: null, best: null,        // evaluations
  qBest: new THREE.Quaternion(),
  applyRotation: true,
  view: 'optimized',
  showOverhang: true,
  report: null,
  analysis: null,
  candidates: [],
};

const $ = (id) => document.getElementById(id);

/* ================= AI status ================= */
let aiAvailable = false;
fetch('/api/status').then(r => r.json()).then(s => {
  aiAvailable = s.ai;
  const pill = $('aiPill');
  pill.classList.add(s.ai ? 'on' : 'off');
  pill.querySelector('span').textContent = s.ai ? `AI · ${s.model} · ${s.effort}` : 'AI off · add API key';
}).catch(() => { $('aiPill').classList.add('off'); $('aiPill').querySelector('span').textContent = 'AI offline'; });

/* ================= local regression model ================= */
let mlModel = null, mlPredictor = null;
const modelReady = fetch('/ml/model.json', { signal: AbortSignal.timeout(5000) })
  .then(async response => {
    if (!response.ok) throw new Error('Model unavailable');
    const model = await response.json();
    validateModel(model);
    mlModel = model;
    if (!profilesMatch(model.profile, DEFAULT_PROFILE)) throw new Error('Incompatible ML profile');
    if (model.usable) mlPredictor = createPredictor(model, DEFAULT_PROFILE);
    $('mlStatus').textContent = model.usable
      ? `ML model ready · ${model.enabledTargets.map(key => key === 'hours' ? 'print time' : 'support material').join(' and ')} predictions run in your browser.`
      : 'ML model did not pass validation. Geometry formulas remain active.';
  }).catch(() => { $('mlStatus').textContent = 'ML model unavailable. Geometry formulas remain active.'; });

function applyEstimator() {
  state.analysis = selectOrientation(state.prep, state.original, state.candidates, {
    predictor: mlPredictor, preferML: $('estimatorSelect').value === 'auto',
  });
  state.best = state.analysis.best;
  state.qBest.setFromUnitVectors(state.best.u, new THREE.Vector3(0, 0, 1));
}

function estimate(p, evaluation) {
  return state.analysis?.estimates.get(evaluation) ?? heuristicEstimate(p, evaluation);
}

/* ================= viewer ================= */
let renderer, scene, camera, controls, meshOriginal, meshOptimized;

function initViewer() {
  if (renderer) return;
  const host = $('viewport');
  renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.prepend(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color('#2b2c31');
  scene.fog = new THREE.Fog('#2b2c31', 900, 1800);

  camera = new THREE.PerspectiveCamera(35, 1, 1, 5000);
  camera.up.set(0, 0, 1);
  camera.position.set(260, -380, 300);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.set(0, 0, 30);
  controls.maxDistance = 1500;

  scene.add(new THREE.HemisphereLight(0xffffff, 0x3a3b40, 1.1));
  const key = new THREE.DirectionalLight(0xffffff, 1.7);
  key.position.set(180, -240, 420);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -180, right: 180, top: 180, bottom: -180, near: 10, far: 1200 });
  key.shadow.bias = -0.0005;
  key.shadow.normalBias = 0.8;   // removes speckles on large flat faces
  scene.add(key);
  const fill = new THREE.DirectionalLight(0xbcd2ff, 0.5);
  fill.position.set(-300, 200, 150);
  scene.add(fill);

  buildPrinter();

  new ResizeObserver(resize).observe(host);
  resize();
  (function loop() {
    requestAnimationFrame(loop);
    controls.update();
    renderer.render(scene, camera);
  })();
}

function resize() {
  const host = $('viewport');
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

/** Build plate with grid texture, printable volume outline and axes. */
function buildPrinter() {
  const S = 1024;
  const c = document.createElement('canvas');
  c.width = c.height = S;
  const g = c.getContext('2d');
  g.fillStyle = '#3b3d43';
  g.fillRect(0, 0, S, S);
  const px = S / BED.x;
  for (let mm = 0; mm <= BED.x; mm += 10) {
    const major = mm % 50 === 0;
    g.strokeStyle = major ? 'rgba(255,255,255,.22)' : 'rgba(255,255,255,.08)';
    g.lineWidth = major ? 2 : 1;
    const p = Math.round(mm * px) + .5;
    g.beginPath(); g.moveTo(p, 0); g.lineTo(p, S); g.stroke();
    g.beginPath(); g.moveTo(0, p); g.lineTo(S, p); g.stroke();
  }
  g.strokeStyle = 'rgba(23,201,100,.55)';
  g.lineWidth = 6;
  g.strokeRect(3, 3, S - 6, S - 6);
  g.fillStyle = 'rgba(255,255,255,.18)';
  g.font = '600 30px Inter, sans-serif';
  g.fillText('FIRST-PRINT-RIGHT', 28, S - 30);
  g.font = '500 22px Inter, sans-serif';
  g.fillText(`${BED.x} × ${BED.y} mm`, S - 190, S - 30);

  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const plate = new THREE.Mesh(
    new THREE.BoxGeometry(BED.x, BED.y, 4),
    [
      new THREE.MeshStandardMaterial({ color: '#2a2b30' }), new THREE.MeshStandardMaterial({ color: '#2a2b30' }),
      new THREE.MeshStandardMaterial({ color: '#2a2b30' }), new THREE.MeshStandardMaterial({ color: '#2a2b30' }),
      new THREE.MeshStandardMaterial({ map: tex, roughness: .85, metalness: .05 }),
      new THREE.MeshStandardMaterial({ color: '#1e1f23' }),
    ]
  );
  plate.position.z = -2;
  plate.receiveShadow = true;
  scene.add(plate);

  // printable volume
  const box = new THREE.LineSegments(
    new THREE.EdgesGeometry(new THREE.BoxGeometry(BED.x, BED.y, BED.z)),
    new THREE.LineBasicMaterial({ color: 0x8a8d96, transparent: true, opacity: .22 })
  );
  box.position.z = BED.z / 2;
  scene.add(box);

  // axes at the front-left corner
  const o = new THREE.Vector3(-BED.x / 2, -BED.y / 2, 0.3);
  [[1, 0, 0, 0xe5484d], [0, 1, 0, 0x30a46c], [0, 0, 1, 0x3e8ef7]].forEach(([x, y, z, col]) => {
    const geo = new THREE.BufferGeometry().setFromPoints([o, o.clone().add(new THREE.Vector3(x, y, z).multiplyScalar(30))]);
    scene.add(new THREE.Line(geo, new THREE.LineBasicMaterial({ color: col })));
  });

  // soft floor under the printer
  const floor = new THREE.Mesh(new THREE.CircleGeometry(1400, 64), new THREE.MeshBasicMaterial({ color: '#26272b' }));
  floor.position.z = -4.2;
  scene.add(floor);
}

function makeMesh(geometry) {
  const mat = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: .55, metalness: .05, flatShading: false });
  const mesh = new THREE.Mesh(geometry, mat);
  mesh.castShadow = true;
  mesh.receiveShadow = true;
  return mesh;
}

/** Colour each triangle by whether it needs support in its current pose. */
function paintOverhangs(geometry, minZ) {
  const pos = geometry.attributes.position.array;
  const tris = pos.length / 9;
  let colors = geometry.attributes.color?.array;
  if (!colors) {
    colors = new Float32Array(pos.length);
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  }
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), n = new THREE.Vector3();
  for (let t = 0; t < tris; t++) {
    const i = t * 9;
    a.fromArray(pos, i); b.fromArray(pos, i + 3); c.fromArray(pos, i + 6);
    n.subVectors(c, b).cross(a.clone().sub(b)).normalize();
    const lowest = Math.min(a.z, b.z, c.z);
    const over = state.showOverhang && n.z < -COS_OVER && lowest - minZ > 0.3;
    const col = over ? COLOR_OVER : COLOR_MODEL;
    for (let k = 0; k < 3; k++) col.toArray(colors, i + k * 3);
  }
  geometry.attributes.color.needsUpdate = true;
}

/** Translucent columns under every overhang: a preview of the support the slicer would add. */
function buildSupports(geometry) {
  const pos = geometry.attributes.position.array;
  const tris = pos.length / 9;
  const A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3(), n = new THREE.Vector3();
  const vkey = (i) => `${Math.round(pos[i] * 100)},${Math.round(pos[i + 1] * 100)},${Math.round(pos[i + 2] * 100)}`;
  const edges = new Map();   // only the outline of each overhang region gets a wall
  let count = 0;
  for (let t = 0; t < tris && count < 200000; t++) {
    const i = t * 9;
    A.fromArray(pos, i); B.fromArray(pos, i + 3); C.fromArray(pos, i + 6);
    n.subVectors(C, B).cross(A.clone().sub(B)).normalize();
    if (!(n.z < -COS_OVER) || Math.min(A.z, B.z, C.z) <= 0.3) continue;
    count++;
    const idx = [i, i + 3, i + 6], keys = idx.map(vkey);
    for (let k = 0; k < 3; k++) {
      const k0 = keys[k], k1 = keys[(k + 1) % 3];
      const key = k0 < k1 ? k0 + '|' + k1 : k1 + '|' + k0;
      const e = edges.get(key);
      if (e) e.n++; else edges.set(key, { n: 1, a: idx[k], b: idx[(k + 1) % 3] });
    }
  }
  const out = [];
  for (const e of edges.values()) {
    if (e.n !== 1) continue;
    const ax = pos[e.a], ay = pos[e.a + 1], az = pos[e.a + 2], bx = pos[e.b], by = pos[e.b + 1], bz = pos[e.b + 2];
    out.push(ax, ay, az, ax, ay, 0, bx, by, bz, bx, by, bz, ax, ay, 0, bx, by, 0);
  }
  if (!out.length) return null;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(out, 3));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({ color: 0xf04f4f, transparent: true, opacity: 0.2, depthWrite: false, side: THREE.DoubleSide }));
  m.name = 'supports';
  m.renderOrder = 2;
  return m;
}

/** Put a geometry on the plate: rotate, drop to z=0, centre in x/y. */
function placeOnBed(source, quaternion) {
  const g = source.clone();
  g.applyQuaternion(quaternion);
  g.computeBoundingBox();
  const bb = g.boundingBox;
  g.translate(-(bb.min.x + bb.max.x) / 2, -(bb.min.y + bb.max.y) / 2, -bb.min.z);
  g.computeVertexNormals();
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

function showMeshes() {
  [meshOriginal, meshOptimized].forEach(m => {
    if (!m) return;
    scene.remove(m);
    m.traverse(o => { o.geometry?.dispose(); o.material?.dispose(); });
  });
  const base = state.prep.geometry;
  const gOrig = placeOnBed(base, new THREE.Quaternion());
  const gOpt = placeOnBed(base, state.applyRotation ? state.qBest : new THREE.Quaternion());
  paintOverhangs(gOrig, 0);
  paintOverhangs(gOpt, 0);
  meshOriginal = makeMesh(gOrig);
  meshOptimized = makeMesh(gOpt);
  [meshOriginal, meshOptimized].forEach(m => {
    const sup = buildSupports(m.geometry);
    if (sup) { sup.visible = state.showOverhang; m.add(sup); }
  });
  scene.add(meshOriginal, meshOptimized);
  setView(state.view, false);
}

function setView(which, refit = true) {
  state.view = which;
  meshOriginal.visible = which === 'original';
  meshOptimized.visible = which === 'optimized';
  $('tabOriginal').classList.toggle('active', which === 'original');
  $('tabOptimized').classList.toggle('active', which === 'optimized');
  if (refit) fitView();
}

function currentMesh() { return state.view === 'original' ? meshOriginal : meshOptimized; }

function fitView(dir = null) {
  const m = currentMesh();
  if (!m) return;
  const bb = m.geometry.boundingBox;
  const center = bb.getCenter(new THREE.Vector3());
  const r = Math.max(bb.getSize(new THREE.Vector3()).length() / 2, 25);
  const d = (r / Math.sin(THREE.MathUtils.degToRad(camera.fov / 2))) * 1.15;
  const v = (dir || camera.position.clone().sub(controls.target)).clone().normalize();
  animateCamera(center.clone().add(v.multiplyScalar(d)), center);
}

function presetView(name) {
  const dirs = {
    iso: new THREE.Vector3(0.55, -0.8, 0.6),
    front: new THREE.Vector3(0, -1, 0.0001),
    top: new THREE.Vector3(0.0001, -0.0001, 1),
    side: new THREE.Vector3(1, 0, 0.0001),
  };
  fitView(dirs[name]);
}

let camAnim = null;
function animateCamera(toPos, toTarget) {
  const fromPos = camera.position.clone(), fromT = controls.target.clone();
  const t0 = performance.now(), dur = 450;
  cancelAnimationFrame(camAnim);
  const step = (now) => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    camera.position.lerpVectors(fromPos, toPos, e);
    controls.target.lerpVectors(fromT, toTarget, e);
    if (k < 1) camAnim = requestAnimationFrame(step);
  };
  camAnim = requestAnimationFrame(step);
}

/* ================= workflow ================= */
const tick = () => new Promise(r => setTimeout(r, 260));

async function runAnalysis(geometry, name) {
  await modelReady;
  state.analysis = null;
  showWorkspace();
  state.name = name;
  $('pAnalysing').hidden = false; $('pReview').hidden = true; $('pDone').hidden = true; $('panelFoot').hidden = true;
  $('vpBusy').hidden = false;
  const items = [...$('checklist').children];
  items.forEach(li => li.className = '');
  const mark = (i) => { items.forEach((li, k) => li.className = k < i ? 'ok' : k === i ? 'run' : ''); };

  mark(0); await tick();
  const p = prepare(geometry);
  state.prep = p;
  const s = p.size;
  $('vpFile').innerHTML = `<b>${escapeHtml(name)}</b>${s.x.toFixed(1)} × ${s.y.toFixed(1)} × ${s.z.toFixed(1)} mm · ${p.T.toLocaleString()} triangles`;

  mark(1); await tick();
  const up = new THREE.Vector3(0, 0, 1);
  state.original = evaluate(p, up);
  state.view = 'original';
  state.qBest.identity();
  state.applyRotation = true;
  showMeshes();
  fitView(new THREE.Vector3(0.55, -0.8, 0.6));

  mark(2); await tick();
  const dirs = candidateDirections(p);
  state.candidates = [];
  for (let i = 0; i < dirs.length; i++) {
    state.candidates.push(evaluate(p, dirs[i]));
    if (i % 40 === 39) await new Promise(r => setTimeout(r));
  }
  applyEstimator();
  state.tested = dirs.length;

  mark(3); await tick();
  mark(4);
  $('vpBusy').hidden = true;
  state.view = 'optimized';
  showMeshes();
  presetView('iso');
  renderReview();
  setStep(2);
}

function changed() { return state.best !== state.original; }

function chosenEval() { return state.applyRotation && changed() ? state.best : state.original; }

function renderReview() {
  const p = state.prep;
  const before = estimate(p, state.original);
  const after = estimate(p, chosenEval());
  const isChange = changed() && state.applyRotation;
  const usingML = state.analysis.mode === 'ml';
  const mlSupport = usingML && state.analysis.targets.includes('supportG');
  const mlTime = usingML && state.analysis.targets.includes('hours');
  $('estimatorLabel').textContent = usingML ? 'ML + formulas' : 'formula estimates';
  $('estimatorNote').textContent = state.analysis.reason + (usingML ? ' Experimental: trained on generated parts sliced with a fixed PLA profile, not real print outcomes.' : ' Failure risk is a heuristic, not a measured probability.');

  // verdict
  const savedG = before.totalG - after.totalG;
  const v = $('verdict');
  if (isChange && savedG > 0.05) {
    const pct = Math.round((before.supportG - after.supportG) / Math.max(before.supportG, 0.01) * 100);
    v.className = 'verdict';
    v.innerHTML = `<div class="big">−${fmtG(savedG)}</div><div><strong>Less plastic, ${before.risk === after.risk ? 'same risk' : (before.risk === 'Low' || (before.risk === 'Medium' && after.risk === 'High') ? 'higher failure risk' : 'lower failure risk')}</strong><p>Support material down about ${pct}%. Estimated print time ${fmtTime(before.hours)} → ${fmtTime(after.hours)}.</p></div>`;
  } else if (isChange) {
    v.className = 'verdict';
    v.innerHTML = `<div class="big">✓</div><div><strong>Orientation updated</strong><p>Selected using support, time, and stability scores. Review the estimates below.</p></div>`;
  } else {
    v.className = 'verdict neutral';
    v.innerHTML = `<div class="big">✓</div><div><strong>${changed() ? 'Original orientation kept' : 'Already well oriented'}</strong><p>${changed() ? 'You turned off the rotation.' : `We tested ${state.tested} orientations and none was meaningfully better.`}</p></div>`;
  }

  // metrics
  const row = (label, b, a, fmt, lowerIsBetter = true) => {
    const d = a - b, good = lowerIsBetter ? d < 0 : d > 0;
    const show = Math.abs(d) > 1e-3 && fmt(Math.abs(d)) !== fmt(0);
    return `<tr><td>${label}</td><td>${fmt(b)}</td><td class="after">${fmt(a)}</td><td class="delta ${show ? (good ? 'good' : 'bad') : ''}">${show ? (d < 0 ? '−' : '+') + fmt(Math.abs(d)) : '—'}</td></tr>`;
  };
  $('metrics').innerHTML =
    `<tr><th>&nbsp;</th><th>Before</th><th>After</th><th>Change</th></tr>` +
    `<tr><td>Failure risk · heuristic</td><td><span class="risk ${before.risk}">${before.risk}</span></td><td class="after"><span class="risk ${after.risk}">${after.risk}</span></td><td class="delta">—</td></tr>` +
    row(mlSupport ? 'Support · ML' : 'Support material', before.supportG, after.supportG, fmtG) +
    row('Total plastic', before.totalG, after.totalG, fmtG) +
    row(mlTime ? 'Print time · ML' : 'Print time', before.hours, after.hours, fmtTime) +
    row('Electricity', before.kwh, after.kwh, (x) => x.toFixed(2) + ' kWh') +
    row('Overhang area', before.overhangCm2, after.overhangCm2, (x) => x.toFixed(1) + ' cm²') +
    row('Bed contact', before.contactCm2, after.contactCm2, (x) => x.toFixed(1) + ' cm²', false);

  // changes
  const ch = [];
  if (changed()) {
    ch.push(`<label class="change"><input type="checkbox" id="chkRotate" ${state.applyRotation ? 'checked' : ''}><div><strong>Rotate to the best orientation</strong><span>${describeRotation()}</span></div></label>`);
  }
  ch.push(`<label class="change"><input type="checkbox" checked disabled><div><strong>Place flat in the centre of the plate</strong><span>Lowest point at 0 mm so the first layer sticks.</span></div></label>`);
  $('changes').innerHTML = ch.join('');
  $('chkRotate')?.addEventListener('change', (e) => {
    state.applyRotation = e.target.checked;
    showMeshes(); renderReview();
  });

  // settings
  const supportsOn = after.overhangCm2 > 0.5;
  const brim = !usingML && (after.contactCm2 < 3 || after.heightMm / Math.sqrt(Math.max(after.contactCm2 * 100, 1)) > 5);
  $('settings').innerHTML = [
    `Supports <b>${usingML ? 'Auto · grid' : supportsOn ? 'On · tree' : 'Off'}</b>`,
    `Brim <b>${brim ? 'On · 5 mm' : 'Off'}</b>`,
    `Layer <b>${LAYER_MM} mm</b>`,
    `Infill <b>${Math.round(INFILL * 100)}%</b>`,
    `Material <b>PLA</b>`,
    ...(usingML ? ['Profile <b>fixed calibration</b>'] : []),
  ].map(t => `<span class="chip">${t}</span>`).join('');

  state.report = {
    file: state.name,
    estimator: state.analysis.mode,
    model_version: usingML ? mlModel.schemaVersion : null,
    ml_targets: state.analysis.targets,
    size_mm: [p.size.x, p.size.y, p.size.z].map(r1),
    orientations_tested: state.tested,
    rotation_applied: isChange,
    rotation: describeRotation(),
    before: round(before), after: round(after),
    suggested_settings: { supports: usingML ? 'auto-grid' : supportsOn ? 'tree' : 'off', brim, layer_mm: LAYER_MM, infill: INFILL, material: 'PLA' },
    assumptions: `PLA ${PLA_DENSITY} g/cm³, ${WALL_MM} mm walls, ${INFILL * 100}% infill, ${SUPPORT_DENSITY * 100}% support density, ${FLOW_MM3_S} mm³/s average, ${PRINTER_WATTS} W printer, overhang limit ${OVERHANG_DEG}°`,
  };
  requestExplanation();

  $('pAnalysing').hidden = true; $('pReview').hidden = false; $('pDone').hidden = true; $('panelFoot').hidden = false;
}

function describeRotation() {
  const old = state.original, nw = state.best;
  if (nw === old) return 'No change';
  const tilt = Math.round(THREE.MathUtils.radToDeg(nw.u.angleTo(new THREE.Vector3(0, 0, 1))));
  const what = tilt >= 175 ? 'Flip it upside down' : Math.abs(tilt - 90) <= 5 ? 'Lay it on its side' : `Tilt it ${tilt}°`;
  return `${what} · height ${old.height.toFixed(0)} → ${nw.height.toFixed(0)} mm`;
}

let explainSeq = 0;
async function requestExplanation() {
  const seq = ++explainSeq;
  const box = $('ai');
  $('aiTag').textContent = aiAvailable ? 'AI' : 'summary';
  box.innerHTML = '<div class="skel" style="width:92%"></div><div class="skel" style="width:80%"></div><div class="skel" style="width:86%"></div><div class="skel" style="width:60%"></div>';
  if (!aiAvailable) return renderAI(localExplanation(), 'AI is off. Add an API key in .env for a written explanation.');
  try {
    const res = await fetch('/api/explain', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(state.report) });
    const data = await res.json();
    if (seq !== explainSeq) return;
    if (!res.ok) throw new Error(data.error || res.statusText);
    renderAI(data);
  } catch (err) {
    if (seq !== explainSeq) return;
    renderAI(localExplanation(), `AI unavailable: ${err.message}`);
  }
}

function renderAI(d, note = '') {
  $('ai').innerHTML =
    `<p class="headline">${escapeHtml(d.headline)}</p>` +
    `<ul>${d.reasons.map(r => `<li>${escapeHtml(r)}</li>`).join('')}</ul>` +
    (d.tips?.length ? `<p class="tips"><b>Tips:</b> ${d.tips.map(escapeHtml).join(' ')}</p>` : '') +
    (note ? `<p class="note">${escapeHtml(note)}</p>` : '');
}

function localExplanation() {
  const r = state.report;
  if (!r.rotation_applied) return {
    headline: 'No rotation is applied.',
    reasons: [`Overhang area is about ${r.after.overhangCm2.toFixed(1)} cm².`, `Bed contact is about ${r.after.contactCm2.toFixed(1)} cm².`],
    tips: [],
  };
  return {
    headline: 'Review the support, time, and stability trade-offs below.',
    reasons: [
      `Estimated support changes from ${fmtG(r.before.supportG)} to ${fmtG(r.after.supportG)}.`,
      `Bed contact changes from ${r.before.contactCm2.toFixed(1)} to ${r.after.contactCm2.toFixed(1)} cm².`,
      `Estimated print time changes from ${fmtTime(r.before.hours)} to ${fmtTime(r.after.hours)}.`,
    ],
    tips: [],
  };
}

function download() {
  const g = meshOptimized.geometry.clone();
  g.deleteAttribute('color');
  const mesh = new THREE.Mesh(g);
  const data = new STLExporter().parse(mesh, { binary: true });
  const blob = new Blob([data], { type: 'model/stl' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = state.name.replace(/\.stl$/i, '') + '_optimized.stl';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  return a.download;
}

function confirmAndDownload() {
  setView('optimized');
  const file = download();
  const before = estimate(state.prep, state.original), after = estimate(state.prep, chosenEval());
  const saved = before.totalG - after.totalG;
  $('doneText').innerHTML = `<b>${escapeHtml(file)}</b> is downloading.` +
    (saved > 0.05 ? `<br>This print should use about <b>${fmtG(saved)}</b> less plastic.<br>Estimated print time: <b>${fmtTime(before.hours)} → ${fmtTime(after.hours)}</b>.` : '<br>Open it in your slicer and print.');
  $('pReview').hidden = true; $('pDone').hidden = false;
  $('confirmBtn').hidden = true;
  $('restartBtn').textContent = 'Check another part';
  setStep(3);
}

/* ================= UI plumbing ================= */
function setStep(n) {
  [...$('stepper').children].forEach(li => {
    const k = +li.dataset.step;
    li.className = k === n ? 'active' : k < n ? 'done' : '';
  });
}

function showWorkspace() {
  $('uploadView').hidden = true;
  $('workspace').hidden = false;
  initViewer();
  resize();
}

function restart() {
  $('workspace').hidden = true;
  $('uploadView').hidden = false;
  $('confirmBtn').hidden = false;
  $('restartBtn').textContent = 'Start over';
  $('fileInput').value = '';
  explainSeq++;
  setStep(1);
}

function uploadError(msg) { const e = $('uploadError'); e.textContent = msg; e.hidden = !msg; }

async function handleFile(file) {
  uploadError('');
  if (!file) return;
  if (!/\.stl$/i.test(file.name)) return uploadError('Please choose an .stl file.');
  if (file.size > 100 * 1024 * 1024) return uploadError('That file is over 100 MB.');
  try {
    const buf = await file.arrayBuffer();
    const geometry = new STLLoader().parse(buf);
    if (!geometry.attributes.position || geometry.attributes.position.count < 3) throw new Error('empty');
    await runAnalysis(geometry, file.name);
  } catch (err) {
    console.error(err);
    restart();
    uploadError('We could not read that STL. Try exporting it again from your CAD tool.');
  }
}

/** Sample: a stove knob saved upside down (thin shaft on the plate, wide grip on top). */
function sampleGeometry() {
  const parts = [];
  const shaft = new THREE.CylinderGeometry(6, 6, 20, 48); shaft.rotateX(Math.PI / 2); shaft.translate(0, 0, 10); parts.push(shaft);
  const grip = new THREE.CylinderGeometry(22, 22, 14, 96); grip.rotateX(Math.PI / 2); grip.translate(0, 0, 27); parts.push(grip);
  const clean = parts.map(g => { const n = g.toNonIndexed(); n.deleteAttribute('normal'); n.deleteAttribute('uv'); return n; });
  return mergeGeometries(clean);
}

$('fileInput').addEventListener('change', (e) => handleFile(e.target.files[0]));
const dz = $('dropzone');
['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.add('drag'); }));
['dragleave', 'drop'].forEach(t => dz.addEventListener(t, (e) => { e.preventDefault(); dz.classList.remove('drag'); }));
dz.addEventListener('drop', (e) => handleFile(e.dataTransfer.files[0]));
window.addEventListener('dragover', (e) => e.preventDefault());
window.addEventListener('drop', (e) => e.preventDefault());

$('estimatorSelect').addEventListener('change', () => {
  if (!state.analysis) return;
  state.applyRotation = true;
  applyEstimator();
  showMeshes();
  presetView('iso');
  renderReview();
});
$('sampleBtn').addEventListener('click', () => runAnalysis(sampleGeometry(), 'stove-knob-sample.stl'));
$('tabOriginal').addEventListener('click', () => setView('original'));
$('tabOptimized').addEventListener('click', () => setView('optimized'));
$('toolFit').addEventListener('click', () => fitView());
$('toolOpen').addEventListener('click', restart);
document.querySelectorAll('[data-view]').forEach(b => b.addEventListener('click', () => presetView(b.dataset.view)));
$('toolOverhang').addEventListener('click', (e) => {
  state.showOverhang = !state.showOverhang;
  e.currentTarget.classList.toggle('on', state.showOverhang);
  [meshOriginal, meshOptimized].forEach(m => {
    paintOverhangs(m.geometry, 0);
    const sup = m.getObjectByName('supports');
    if (sup) sup.visible = state.showOverhang;
  });
});
$('confirmBtn').addEventListener('click', confirmAndDownload);
$('againDownload').addEventListener('click', download);
$('restartBtn').addEventListener('click', restart);

/* ================= helpers ================= */
function fmtG(g) { return g >= 100 ? `${Math.round(g)} g` : `${g.toFixed(1)} g`; }
function fmtTime(h) { const m = Math.max(0, Math.round(h * 60)); return m >= 60 ? `${Math.floor(m / 60)} h ${m % 60} m` : `${m} min`; }
function r1(x) { return Math.round(x * 10) / 10; }
function round(o) { const out = {}; for (const [k, v] of Object.entries(o)) out[k] = typeof v === 'number' ? Math.round(v * 1e6) / 1e6 : v; return out; }
function escapeHtml(s) { return String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c])); }
