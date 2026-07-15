// Drag & drop for the glass upload card. Assigns a dropped file to the hidden
// file input and fires its 'change' event, so the existing app.js flow (filename
// display, state) runs unchanged. Also toggles the .dragover glass state.
(function () {
  const label = document.getElementById('fileLabel');
  const input = document.getElementById('uploadInput');
  if (!label || !input) return;

  const stop = (e) => { e.preventDefault(); e.stopPropagation(); };
  ['dragenter', 'dragover'].forEach((ev) => label.addEventListener(ev, (e) => { stop(e); label.classList.add('dragover'); }));
  ['dragleave', 'dragend'].forEach((ev) => label.addEventListener(ev, (e) => { stop(e); label.classList.remove('dragover'); }));

  const assign = (e, el) => {
    stop(e);
    el.classList.remove('dragover');
    const f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (!f) return;
    try { const dt = new DataTransfer(); dt.items.add(f); input.files = dt.files; } catch (_) { /* older browsers */ }
    input.dispatchEvent(new Event('change', { bubbles: true }));
  };
  label.addEventListener('drop', (e) => assign(e, label));

  // the big empty-state panel doubles as an upload target (click or drop)
  const panel = document.getElementById('uploadPanel');
  if (panel) {
    panel.addEventListener('click', () => input.click());
    panel.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
    ['dragenter', 'dragover'].forEach((ev) => panel.addEventListener(ev, (e) => { stop(e); panel.classList.add('dragover'); }));
    ['dragleave', 'dragend'].forEach((ev) => panel.addEventListener(ev, (e) => { stop(e); panel.classList.remove('dragover'); }));
    panel.addEventListener('drop', (e) => assign(e, panel));
  }
})();
