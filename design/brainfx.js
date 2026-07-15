// NeuroScan — decorative anatomical BRAIN (vanilla Three.js via the page import
// map). The dominant object is a real sculpted anatomical brain GLB (visible gyri
// & sulci) rendered as a semi-transparent slate shell; neural fibres + travelling
// pulses live INSIDE it and only enhance the anatomy. Faint EEG traces scroll
// right→left behind it and calm toward the cursor. Sits half-off the RIGHT edge,
// behind all content, radial-faded. Palette only:
//   #273751 navy · #627890 slate · #D6B896 sand · #642226 accent · black · white.
// Brain model: Science Museum Group, CC BY-SA 4.0 (self-hosted /brain.glb).
// Decorative + aria-hidden; hides itself on reduced-motion / no-WebGL / error.

import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { ImprovedNoise } from 'three/addons/math/ImprovedNoise.js';

const NAVY = 0x273751, SLATE = 0x627890, SAND = 0xd6b896, WARN = 0x642226, GREY = 0x9aa0a8, GOLD = 0xe8a320;

const canvas = document.getElementById('brainfx');
const reduce = window.matchMedia('(prefers-reduced-motion:reduce)').matches;
const finePointer = window.matchMedia('(pointer:fine)').matches;
const hide = () => { if (canvas) canvas.style.display = 'none'; };

if (!canvas || reduce) hide();
else { try { run(); } catch (e) { console.warn('[brainfx] disabled:', e); hide(); } }

function run() {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.1;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(38, 1, 0.1, 100);
  camera.position.set(0, 0, 8);
  const brain = new THREE.Group();
  scene.add(brain);

  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.05).texture;
  const key = new THREE.DirectionalLight(0xffffff, 2.4); key.position.set(4, 6, 6); scene.add(key);
  const fill = new THREE.DirectionalLight(SLATE, 1.1); fill.position.set(-5, -1, 2); scene.add(fill);
  const back = new THREE.DirectionalLight(SAND, 0.6); back.position.set(-2, 3, -5); scene.add(back);
  scene.add(new THREE.AmbientLight(0x51617a, 0.55));

  // ---------- the anatomical brain model (dominant) ----------
  // holographic brain: translucent additive cyan + fresnel edge glow + contour lines
  const HOLO = new THREE.Color(0x6fe0ff);
  const brainMat = new THREE.MeshBasicMaterial({ color: HOLO, transparent: true, opacity: 0.12, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
  const wireMat = new THREE.LineBasicMaterial({ color: HOLO, transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending, depthWrite: false });
  const rimMat = new THREE.ShaderMaterial({
    uniforms: { uColor: { value: HOLO } },
    vertexShader: `varying vec3 vN; varying vec3 vV; void main(){ vec4 wp=modelMatrix*vec4(position,1.0); vN=normalize(mat3(modelMatrix)*normal); vV=normalize(cameraPosition-wp.xyz); gl_Position=projectionMatrix*viewMatrix*wp; }`,
    fragmentShader: `varying vec3 vN; varying vec3 vV; uniform vec3 uColor; void main(){ float f=pow(1.0-abs(dot(vN,vV)),2.2); gl_FragColor=vec4(uColor, f*0.8); }`,
    transparent: true, blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthWrite: false,
  });
  let radii = new THREE.Vector3(1.5, 1.2, 1.9);              // filled in once the model is measured
  let maxHoriz = 2.0;                                        // brain's largest horizontal half-extent (any spin angle)
  let modelReady = false;

  new GLTFLoader().load('/brain3.glb', (gltf) => {
    const model = gltf.scene;
    // keep the model's own materials, just make them translucent
    model.traverse((o) => { if (o.isMesh) { o.castShadow = o.receiveShadow = false; const mats = Array.isArray(o.material) ? o.material : [o.material]; mats.forEach((m) => { m.transparent = true; m.opacity = 0.55; m.depthWrite = false; }); } });
    const box = new THREE.Box3().setFromObject(model);
    const size = box.getSize(new THREE.Vector3()), center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);                              // centre it at the group origin
    const pivot = new THREE.Group(); pivot.add(model);
    const s = 4.2 / Math.max(size.x, size.y, size.z); pivot.scale.setScalar(s);
    brain.add(pivot);
    radii = size.clone().multiplyScalar(s * 0.42);   // fill more of the brain volume
    maxHoriz = Math.max(size.x, size.z) * s * 0.5;   // half-width as it rotates, for edge clamping
    resize();                                          // re-clamp position now that we know the real size
    buildInternals();    // interior neural fibres + travelling electrical-activity signals
    modelReady = true;
  }, undefined, (err) => { console.warn('[brainfx] model load failed:', err); hide(); });

  // ---------- interior neural fibres + pulses (subordinate) ----------
  const noise = new ImprovedNoise();
  const cSlate = new THREE.Color(SLATE), cSand = new THREE.Color(SAND), cWhite = new THREE.Color(0xffffff), cWarn = new THREE.Color(WARN), cGold = new THREE.Color(GOLD), cTmp = new THREE.Color();
  const fiberPaths = [];
  let fiberMat = null, partMat = null, partGeo = null, pPos = null, parts = [];
  const PN = 100;

  function buildInternals() {
    const inside = (p) => (p.x * p.x) / (radii.x * radii.x) + (p.y * p.y) / (radii.y * radii.y) + (p.z * p.z) / (radii.z * radii.z) <= 1;
    const randInside = () => { let p; for (let g = 0; g < 40; g++) { p = new THREE.Vector3((Math.random() * 2 - 1) * radii.x, (Math.random() * 2 - 1) * radii.y, (Math.random() * 2 - 1) * radii.z); if (inside(p)) return p; } return p; };
    const S = 0.7, dir = new THREE.Vector3(), nd = new THREE.Vector3();
    const flow = (p, out) => { const a = noise.noise(p.x * S, p.y * S, p.z * S); const b = noise.noise(p.y * S + 31, p.z * S - 5, p.x * S + 3); const c = noise.noise(p.z * S + 8, p.x * S + 19, p.y * S - 12); out.set(a, b * 0.7, c); if (out.lengthSq() < 1e-6) out.set(0, 0, 1); return out.normalize(); };
    const segPos = [], segCol = [], stepLen = radii.length() * 0.045;
    for (let f = 0; f < 240; f++) {
      const pts = []; let p = randInside(); flow(p, dir); const steps = 26 + (Math.random() * 26 | 0);
      for (let i = 0; i < steps; i++) { pts.push(p.clone()); flow(p, nd); dir.lerp(nd, 0.45).normalize(); const np = p.clone().addScaledVector(dir, stepLen); if (!inside(np)) { dir.multiplyScalar(-1); const q = p.clone().addScaledVector(dir, stepLen); if (!inside(q)) break; p = q; } else p = np; }
      if (pts.length < 4) continue;
      const warm = Math.random() * 0.45;                    // catchy gold, brightness varied (low blue → never washes to white)
      for (let i = 0; i < pts.length - 1; i++) { const a = pts[i], b = pts[i + 1]; segPos.push(a.x, a.y, a.z, b.x, b.y, b.z); cTmp.copy(cGold).multiplyScalar(0.72 + warm * 0.85); segCol.push(cTmp.r, cTmp.g, cTmp.b, cTmp.r, cTmp.g, cTmp.b); }
      fiberPaths.push(pts);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(segPos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(segCol, 3));
    fiberMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0, blending: THREE.AdditiveBlending, depthWrite: false });
    // no fibre lines drawn — only the glowing sparkle signals below (paths stay, just used to move the sparkles)

    const tex = (() => { const c = document.createElement('canvas'); c.width = c.height = 64; const gg = c.getContext('2d'); const gr = gg.createRadialGradient(32, 32, 0, 32, 32, 32); gr.addColorStop(0, 'rgba(255,212,110,1)'); gr.addColorStop(0.35, 'rgba(240,168,32,0.55)'); gr.addColorStop(1, 'rgba(240,168,32,0)'); gg.fillStyle = gr; gg.fillRect(0, 0, 64, 64); const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace; return t; })();
    pPos = new Float32Array(PN * 3); const pCol = new Float32Array(PN * 3); parts = [];
    for (let i = 0; i < PN; i++) { const path = fiberPaths[(Math.random() * fiberPaths.length) | 0] || fiberPaths[0]; const col = Math.random() < 0.08 ? cWarn.clone() : cGold.clone().multiplyScalar(0.85 + Math.random() * 0.4); pCol[i * 3] = col.r; pCol[i * 3 + 1] = col.g; pCol[i * 3 + 2] = col.b; parts.push({ path, u: Math.random(), speed: 0.05 + Math.random() * 0.12 }); }
    partGeo = new THREE.BufferGeometry(); partGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3)); partGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
    partMat = new THREE.PointsMaterial({ size: 0.16, map: tex, vertexColors: true, transparent: true, opacity: 0.34, blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, sizeAttenuation: true });
    const act = new THREE.Group(); act.position.y = radii.y * 0.32; brain.add(act);   // lift the sparkle group up, into the cerebrum
    const pts = new THREE.Points(partGeo, partMat); pts.renderOrder = 3; act.add(pts);   // draw over the shell so far-side sparkles show too
  }

  // ---------- EEG traces behind the brain ----------
  const M = 6, N = 150, EEGZ = -3.4, VIRT_W = 1500, Rw = 0.55;
  const traces = [];
  for (let i = 0; i < M; i++) { const pos = new THREE.BufferAttribute(new Float32Array(N * 3), 3); const g = new THREE.BufferGeometry(); g.setAttribute('position', pos); scene.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x9fb6cf, transparent: true, opacity: 0.5 + Math.random() * 0.15 }))); traces.push({ pos, y0: 0, lane: 1, speed: 30 + Math.random() * 40, pw: 6.2832 / (360 + Math.random() * 240), ph: Math.random() * 6.283, pph: Math.random() * 6.283 }); }
  let halfW = 6, halfH = 3.6; const xs = new Float32Array(N);
  function layoutEeg() { halfH = Math.tan((camera.fov * Math.PI / 180) / 2) * (camera.position.z - EEGZ); halfW = halfH * camera.aspect; const usableH = halfH * 1.8, step = usableH / (M + 1); for (let i = 0; i < N; i++) xs[i] = -halfW * 1.1 + (i / (N - 1)) * halfW * 2.2; for (let ti = 0; ti < M; ti++) { const tr = traces[ti]; tr.lane = step * 0.4; tr.y0 = usableH / 2 - (ti + 1) * step; const a = tr.pos.array; for (let i = 0; i < N; i++) { a[i * 3] = xs[i]; a[i * 3 + 1] = tr.y0; a[i * 3 + 2] = EEGZ; } tr.pos.needsUpdate = true; } }
  function updateEeg(t) { for (let ti = 0; ti < M; ti++) { const tr = traces[ti], a = tr.pos.array, u0 = t * tr.speed; for (let i = 0; i < N; i++) { const u = (xs[i] + halfW * 1.1) / (halfW * 2.2) * VIRT_W + u0; const healthy = Math.sin(u * 0.05 + tr.ph) * tr.lane * 0.28 + Math.sin(u * 0.11) * tr.lane * 0.1; const pk = Math.max(0, Math.sin(u * tr.pw + tr.pph)); let infl = 0; if (mouseActive) { const d = xs[i] - mouseWX; infl = Math.exp(-(d * d) / (2 * Rw * Rw)); } let vv = healthy + Math.pow(pk, 14) * tr.lane * 0.7 + infl * tr.lane * 0.95; if (vv > tr.lane) vv = tr.lane; else if (vv < -tr.lane) vv = -tr.lane; a[i * 3 + 1] = tr.y0 + vv; } tr.pos.needsUpdate = true; } }

  let mouseWX = 9999, mouseActive = false; const ndc = new THREE.Vector3(), rd = new THREE.Vector3();
  if (finePointer) { window.addEventListener('pointermove', (e) => { const r = canvas.getBoundingClientRect(); ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -(((e.clientY - r.top) / r.height) * 2 - 1), 0.5).unproject(camera); rd.copy(ndc).sub(camera.position).normalize(); mouseWX = camera.position.x + rd.x * ((EEGZ - camera.position.z) / rd.z); mouseActive = true; }, { passive: true }); document.addEventListener('mouseleave', () => { mouseActive = false; }); }

  // ---------- grab-to-rotate: drag empty background to turn the brain by hand ----------
  // Offsets add on top of the auto-spin, so releasing resumes rotation from the new angle.
  let userYaw = 0, userPitch = 0, dragging = false, lastX = 0, lastY = 0;
  const BLOCK = 'input,button,a,label,select,textarea,.sidebar,.empty-card,.panel,.hero,.report-body,.topbar .pill';
  if (finePointer) {
    window.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || (e.target.closest && e.target.closest(BLOCK))) return;
      dragging = true; lastX = e.clientX; lastY = e.clientY; document.body.style.cursor = 'grabbing';
    });
    window.addEventListener('pointermove', (e) => {
      if (!dragging) return;
      userYaw += (e.clientX - lastX) * 0.008;
      userPitch = Math.max(-0.6, Math.min(0.6, userPitch + (e.clientY - lastY) * 0.006));
      lastX = e.clientX; lastY = e.clientY;
    }, { passive: true });
    const endDrag = () => { if (dragging) { dragging = false; document.body.style.cursor = ''; } };
    window.addEventListener('pointerup', endDrag);
    window.addEventListener('pointercancel', endDrag);
  }

  let halfW0v = 6, halfH0v = 3.6, maxShiftV = 4, posInit = false, wantScale = 0.95; const tgt = new THREE.Vector3();
  const pxToWorldX = (px) => Math.min(((px / innerWidth) * 2 - 1) * halfW0v, maxShiftV);
  const pyToWorldY = (py) => (1 - (py / innerHeight) * 2) * halfH0v;

  // Where the brain should sit, depending on app state:
  //  · empty state  → centred in the gap beside the upload box (vertically ~middle)
  //  · results shown → up beside the risk score, in the right margin next to the content
  function computeTarget(out) {
    const narrow = innerWidth < 820;
    if (narrow) { wantScale = 0.72; out.set(0, -0.35, 0); return; }
    const results = document.getElementById('results');
    const showingResults = results && !results.classList.contains('hidden');
    wantScale = showingResults ? 0.36 : 0.88;                // slightly smaller in empty state so it never clips the edge
    maxShiftV = halfW0v - maxHoriz * wantScale - 0.15;       // let the smaller brain sit further right
    let px, wy = -0.35;
    if (showingResults) {
      const main = document.querySelector('.main');
      let rightPx = innerWidth * 0.55;
      if (main) { const r = main.getBoundingClientRect(); if (r.right > 0) rightPx = r.right; }
      px = rightPx + (innerWidth - rightPx) * 0.42 - 150;    // 0.42 of the gap, then a fixed 150px left
      const hero = document.getElementById('hero');
      if (hero) { const r = hero.getBoundingClientRect(); wy = pyToWorldY(r.top + r.height / 2); }
    } else {
      const panel = document.getElementById('uploadPanel');
      let rightPx = innerWidth * 0.62;
      if (panel) { const r = panel.getBoundingClientRect(); if (r.right > 0) rightPx = r.right; }
      px = (rightPx + innerWidth) / 2 - 20;
    }
    out.set(Math.max(0, pxToWorldX(px)), wy, 0);
  }

  function resize() {
    const w = Math.max(1, canvas.clientWidth), h = Math.max(1, canvas.clientHeight);
    renderer.setSize(w, h, false); camera.aspect = w / h; camera.updateProjectionMatrix();
    const narrow = innerWidth < 820;
    halfW0v = Math.tan((camera.fov * Math.PI / 180) / 2) * camera.position.z * camera.aspect;
    halfH0v = halfW0v / camera.aspect;
    const baseScale = narrow ? 0.72 : 0.95;
    maxShiftV = halfW0v - maxHoriz * baseScale - 0.15;
    computeTarget(tgt);
    if (!posInit) { brain.position.copy(tgt); brain.scale.setScalar(wantScale); posInit = true; }
    layoutEeg();
  }
  resize();
  window.addEventListener('resize', resize, { passive: true });

  const clock = new THREE.Clock(); let nextSync = 7, syncAt = -10, raf = 0; const tmp = new THREE.Vector3();
  function frame() {
    raf = requestAnimationFrame(frame);
    const t = clock.getElapsedTime();
    computeTarget(tgt); brain.position.lerp(tgt, 0.06);       // glide to the state's anchor (empty box / risk score)
    brain.scale.setScalar(brain.scale.x + (wantScale - brain.scale.x) * 0.06);  // shrink/grow to fit that space
    brain.rotation.y = t * (Math.PI * 2 / 100) + userYaw;    // slower, one turn ~100s + hand-turn
    brain.rotation.x = -0.05 + Math.sin(t * 0.07) * 0.03 + userPitch;
    if (t > nextSync) { syncAt = t; nextSync = t + 9 + Math.random() * 5; }
    const sync = Math.max(0, 1 - (t - syncAt) / 0.9);
    if (modelReady && fiberMat) {
      fiberMat.opacity = 0.18 + 0.14 * sync;
      partMat.size = 0.16 + 0.1 * sync; partMat.opacity = 0.3 + 0.14 * sync;
      for (let i = 0; i < PN; i++) { const pr = parts[i]; pr.u += pr.speed * 0.016; if (pr.u >= 1) pr.u -= 1; const path = pr.path, fi = pr.u * (path.length - 1), i0 = fi | 0, a = path[i0], b = path[Math.min(i0 + 1, path.length - 1)]; tmp.copy(a).lerp(b, fi - i0); pPos[i * 3] = tmp.x; pPos[i * 3 + 1] = tmp.y; pPos[i * 3 + 2] = tmp.z; }
      partGeo.attributes.position.needsUpdate = true;
    }
    updateEeg(t);
    renderer.render(scene, camera);
  }
  frame();
  document.addEventListener('visibilitychange', () => { if (document.hidden) { cancelAnimationFrame(raf); raf = 0; } else if (!raf) { clock.getDelta(); frame(); } });
}
