// NeuroScan frontend — talks to the Python backend (serve_ui.py) which reuses src/.
// Single clinical tool for a neurologist: upload an EDF, analyze, read the report.
// Infrastructure choices are fixed to safe defaults (general cross-subject model,
// local offline LLM, RAG on) — the clinician never picks a model or a provider.
const $ = (id) => document.getElementById(id);
const state = { lastAnalysis: null, uploadName: null };

// Fixed backend defaults (hidden from the clinician).
const MODEL = 'cross';            // general cross-subject model
const PROVIDER = 'foundry_local'; // local, offline LLM
const LOCAL_MODEL = 'phi-3.5-mini';
const USE_RAG = true;             // ground the report in the local knowledge base

function toast(msg) {
  const t = document.createElement('div');
  t.className = 'toast'; t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3800);
}

// file-picker icons: an upload arrow by default, a "generation" sparkle once added
const ICON_UPLOAD = '<svg viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>';
const ICON_SPARKLE = '<svg viewBox="0 0 24 24"><path d="m12 3-1.9 5.8a2 2 0 0 1-1.3 1.3L3 12l5.8 1.9a2 2 0 0 1 1.3 1.3L12 21l1.9-5.8a2 2 0 0 1 1.3-1.3L21 12l-5.8-1.9a2 2 0 0 1-1.3-1.3z"/><path d="M5 3v4"/><path d="M19 17v4"/><path d="M3 5h4"/><path d="M17 19h4"/></svg>';

// ---------- init ----------
function init() {
  $('uploadInput').addEventListener('change', e => {
    const f = e.target.files[0];
    state.uploadName = f?.name || null;
    $('fileText').textContent = f ? f.name : 'Choose an EDF file…';
    $('fileIcon').innerHTML = f ? ICON_SPARKLE : ICON_UPLOAD;
    $('fileLabel').classList.toggle('has-file', !!f);
  });

  // sensitivity (threshold)
  $('threshold').addEventListener('input', e => {
    const v = parseFloat(e.target.value).toFixed(2);
    $('thVal').textContent = v; $('thLegend').textContent = v;
    if (state.lastAnalysis) renderChart(state.lastAnalysis, parseFloat(v)); // live re-threshold visual
  });

  $('analyzeBtn').addEventListener('click', analyze);
  $('reportBtn').addEventListener('click', generateReport);
  $('printBtn').addEventListener('click', () => window.print());  // Save as PDF
  setupCursorGlow();
}

// A restrained ambient glow that trails the cursor. Off on touch / reduced-motion.
function setupCursorGlow() {
  if (!window.matchMedia('(pointer:fine)').matches) return;
  if (window.matchMedia('(prefers-reduced-motion:reduce)').matches) return;
  const g = document.createElement('div');
  g.className = 'cursor-glow';
  g.setAttribute('aria-hidden', 'true');
  document.body.appendChild(g);
  let tx = innerWidth / 2, ty = innerHeight / 2, x = tx, y = ty, shown = false;
  addEventListener('pointermove', e => {
    tx = e.clientX; ty = e.clientY;
    if (!shown) { shown = true; g.style.opacity = '1'; }
  }, { passive: true });
  (function loop() {
    x += (tx - x) * 0.09; y += (ty - y) * 0.09;   // eased trailing
    g.style.setProperty('--mx', x + 'px');
    g.style.setProperty('--my', y + 'px');
    requestAnimationFrame(loop);
  })();
}

// ---------- analyze ----------
async function analyze() {
  const threshold = parseFloat($('threshold').value);
  const f = $('uploadInput').files[0];
  if (!f) { toast('Choose an EDF file to upload.'); return; }

  const payload = { model: MODEL, threshold, upload: await fileToBase64(f), filename: f.name };

  showState('loading');
  $('analyzeBtn').disabled = true;
  try {
    const res = await fetch('/api/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    state.lastAnalysis = data;
    renderResults(data, threshold);
    showState('results');
  } catch (e) {
    showState('empty');
    toast('Analysis failed: ' + e.message);
  } finally {
    $('analyzeBtn').disabled = false;
  }
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result.split(',')[1]);
    r.onerror = reject;
    r.readAsDataURL(file);
  });
}

function showState(s) {
  $('emptyState').classList.toggle('hidden', s !== 'empty');
  $('loadingState').classList.toggle('hidden', s !== 'loading');
  $('results').classList.toggle('hidden', s !== 'results');
}

// ---------- render results ----------
const RISK_META = {
  High:     { title: 'Seizure-like activity detected', badge: 'High epilepsy risk',     color: 'var(--high)' },
  Moderate: { title: 'Some abnormal activity',          badge: 'Moderate epilepsy risk', color: 'var(--moderate)' },
  Low:      { title: 'No seizure-like activity',        badge: 'Low epilepsy risk',      color: 'var(--low)' },
};

function renderResults(d, threshold) {
  const a = d.assessment;
  const meta = RISK_META[a.risk_level] || RISK_META.Low;
  const hero = $('hero');
  hero.className = 'hero ' + a.risk_level.toLowerCase();
  $('heroBadge').querySelector('.dot').style.background = meta.color;
  $('heroBadgeText').textContent = meta.badge;
  $('heroTitle').textContent = meta.title;
  $('heroSub').textContent = `${a.n_episodes} episode${a.n_episodes === 1 ? '' : 's'} · ${d.filename}`;
  $('heroScore').textContent = a.risk_score;

  const ICONS = {
    score: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    windows: '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6v6H9z"/>',
    episodes: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    peak: '<path d="M12 2 15 8l6 .5-4.5 4 1.5 6L12 15l-6 3.5 1.5-6L3 8.5 9 8z"/>',
  };
  const metric = (label, value, small, foot, icon) => `
    <div class="card metric">
      <div class="top"><div class="label">${label}</div>
        <div class="icon"><svg viewBox="0 0 24 24" fill="none" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${icon}</svg></div></div>
      <div class="value">${value}${small ? `<small>${small}</small>` : ''}</div>
      <div class="foot">${foot}</div>
    </div>`;
  const pc = Number(a.peak_confidence).toFixed(2);
  $('metrics').innerHTML =
    metric('RISK SCORE', a.risk_score, '/100', `Peak confidence ${pc}`, ICONS.score) +
    metric('ABNORMAL WINDOWS', a.n_abnormal_windows, '/' + a.n_windows, `${a.pct_abnormal}% of recording`, ICONS.windows) +
    metric('EPISODES', a.n_episodes, '', `${a.abnormal_seconds}s total abnormal`, ICONS.episodes) +
    metric('PEAK CONFIDENCE', pc, '', 'Max P(seizure) in recording', ICONS.peak);

  // episodes
  const eps = a.episodes || [];
  $('episodes').innerHTML = eps.length ? eps.map(e =>
    `<div class="ep"><div><div class="rng mono">${e.start_sec}s – ${e.end_sec}s</div><div class="len">duration ${e.end_sec - e.start_sec}s</div></div><div class="tag red">Model</div></div>`
  ).join('') : '<div class="muted-empty">No seizure-like activity detected.</div>';

  // report caption
  $('reportCaption').innerHTML = `Generated locally on this machine · grounded with clinical references. Draft for the neurologist to review, edit, and sign off.`;
  $('reportBody').classList.add('hidden');
  $('reportBtn').disabled = false;

  renderChart(d, threshold);
}

// ---------- SVG chart ----------
function renderChart(d, threshold) {
  const probs = d.probs || [];
  const W = 1000, H = 240;
  const n = probs.length;
  const x = i => n <= 1 ? 0 : (i / (n - 1)) * W;
  const y = p => H - p * H;
  const win = d.window_sec || 5;

  // recompute detected spans at current threshold (visual live update)
  const spans = [];
  let start = null;
  probs.forEach((p, i) => {
    if (p > threshold && start === null) start = i;
    else if (p <= threshold && start !== null) { spans.push([start, i]); start = null; }
  });
  if (start !== null) spans.push([start, n]);

  let line = '', area = '';
  probs.forEach((p, i) => { const cmd = i === 0 ? 'M' : 'L'; line += `${cmd}${x(i).toFixed(1)},${y(p).toFixed(1)} `; });
  area = `M0,${H} ` + probs.map((p, i) => `L${x(i).toFixed(1)},${y(p).toFixed(1)}`).join(' ') + ` L${W},${H} Z`;

  const det = spans.map(([a, b]) => {
    const x1 = x(a), x2 = x(Math.min(b, n - 1));
    return `<rect x="${x1.toFixed(1)}" y="0" width="${Math.max(2, x2 - x1).toFixed(1)}" height="${H}" fill="#f87171" fill-opacity=".14"/>`;
  }).join('');
  const thY = y(threshold);

  $('chart').innerHTML = `
    <svg viewBox="0 0 ${W} ${H}" width="100%" preserveAspectRatio="none" style="display:block">
      <defs>
        <linearGradient id="area" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#3b82f6" stop-opacity=".28"/><stop offset="1" stop-color="#3b82f6" stop-opacity="0"/></linearGradient>
      </defs>
      <g stroke="#1e293b" stroke-width="1"><line x1="0" y1="60" x2="${W}" y2="60"/><line x1="0" y1="120" x2="${W}" y2="120"/><line x1="0" y1="180" x2="${W}" y2="180"/></g>
      ${det}
      <line x1="0" y1="${thY.toFixed(1)}" x2="${W}" y2="${thY.toFixed(1)}" stroke="#64748b" stroke-width="1.5" stroke-dasharray="6 5"/>
      <path d="${area}" fill="url(#area)"/>
      <path d="${line}" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>
    </svg>`;

  // axis ticks
  const total = n * win;
  const ticks = 5;
  let axis = '';
  for (let i = 0; i < ticks; i++) axis += `<span>${Math.round((i / (ticks - 1)) * total)}s</span>`;
  $('axis').innerHTML = axis;

  // hover tooltip — show the time + probability under the cursor
  const host = $('chart');                       // its innerHTML was just replaced, so re-add the tip
  const tip = document.createElement('div');
  tip.className = 'chart-tip';
  host.appendChild(tip);
  host.onmousemove = e => {
    const r = host.getBoundingClientRect();
    const frac = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
    const i = Math.round(frac * (n - 1));
    tip.textContent = `${Math.round(i * win)}s · P=${Number(probs[i] ?? 0).toFixed(2)}`;
    tip.style.left = (e.clientX - r.left) + 'px';
    tip.style.opacity = '1';
  };
  host.onmouseleave = () => { tip.style.opacity = '0'; };
}

// ---------- report ----------
async function generateReport() {
  if (!state.lastAnalysis) return;
  const btn = $('reportBtn');
  const original = btn.innerHTML;
  btn.disabled = true;
  btn.innerHTML = '<span class="spin"></span> Generating…';
  try {
    const body = {
      file: state.lastAnalysis.file_key,
      provider: PROVIDER,
      use_rag: USE_RAG,
      local_model: LOCAL_MODEL,
      api_key: null,
    };
    const res = await fetch('/api/report', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
    if (!res.ok) throw new Error((await res.json()).error || res.statusText);
    const data = await res.json();
    $('reportBody').innerHTML = `<div class="src">Source: <code>${data.source}</code></div>` + formatReport(data.text);
    $('reportBody').classList.remove('hidden');
  } catch (e) {
    toast('Report failed: ' + e.message);
  } finally {
    btn.disabled = false; btn.innerHTML = original;
  }
}

// Structure the clinical report into labelled sections. The report labels its parts
// ("Findings", "Risk Assessment", "Recommendation", …) as plain text — with a colon
// (template) or an em-dash (LLM), and sometimes with a markdown header. This detects
// those labels regardless of style and turns each into a section heading, so the
// report reads as distinct blocks instead of one dense wall of text.
const REPORT_SECTIONS =
  /^(EEG Analysis Report|Clinical Impression|Impression|Findings|Risk Assessment|Recommendation|Summary)\s*[:—–-]*\s*(.*)$/i;

function formatReport(md) {
  const esc = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  // wrap every number (and the risk level) so it sparkles on hover — no colour, no box
  const highlight = t => t
    .replace(/\b(High|Moderate|Low)\b(?=\s*\(?\s*(?:score|\d))/g, '<span class="hl">$1</span>')
    .replace(/\d+(?:\.\d+)?[%s]?/g, m => `<span class="hl">${m}</span>`);
  const inline = s => highlight(esc(s)).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/`(.+?)`/g, '<code>$1</code>');
  let html = '', inList = false, openSec = false;
  const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };
  for (let raw of md.split('\n')) {
    let l = raw.trim();
    if (!l) continue;
    l = l.replace(/^#{1,6}\s*/, '');                 // tolerate markdown headers too
    const m = l.match(REPORT_SECTIONS);
    if (m) {
      closeList();
      if (openSec) html += '</div>';
      html += `<h4>${inline(m[1])}</h4><div class="rp-sec">`;
      openSec = true;
      if (m[2].trim()) html += `<p>${inline(m[2].trim())}</p>`;
    } else if (/^[-*]\s/.test(l)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += `<li>${inline(l.replace(/^[-*]\s/, ''))}</li>`;
    } else {
      closeList();
      html += `<p>${inline(l)}</p>`;
    }
  }
  closeList();
  if (openSec) html += '</div>';
  return html;
}

init();