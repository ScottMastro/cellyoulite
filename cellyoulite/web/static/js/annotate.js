const wellSel = document.getElementById('well-sel');
const tpSel = document.getElementById('tp-sel');
const alignedBox = document.getElementById('aligned');
const clearBtn = document.getElementById('clear');
const copyFromSel = document.getElementById('copy-from');
const statusEl = document.getElementById('status');
const img = document.getElementById('img');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let wells = [];
let timepoints = [];
let currentKey = null;     // image key for backend
let imgW = 0, imgH = 0;    // natural image pixel size
let circles = [];          // {cx,cy,r,star} in image coords
let pendingCentre = null;  // first click while placing a new circle
let mouse = null;
let dirty = false;
let selectedIdx = -1;      // index of currently-selected circle, or -1

function setStatus(text, isDirty=false) {
  statusEl.textContent = text;
  statusEl.classList.toggle('dirty', isDirty);
  dirty = isDirty;
}

async function loadGrid() {
  const r = await fetch('/api/grid');
  const data = await r.json();
  wells = data.wells || [];
  wellSel.innerHTML = '';
  for (const w of wells) {
    const opt = document.createElement('option');
    opt.value = [w.mount_id, w.batch, w.folder_name].join('|');
    // Well names repeat across batches, so name the batch in the list.
    opt.textContent = `${w.batch} · ${w.folder_name}`;
    opt.dataset.batch = w.batch;
    opt.dataset.folder = w.folder_name;
    wellSel.appendChild(opt);
  }
  // Fire-and-forget alignment warm-up; wells whose cache is already valid
  // are nearly instant. Status bar reports when it finishes.
  warmAlignments();
  if (wells.length) loadWell();
}

async function warmAlignments() {
  setStatus('warming alignments…', false);
  try {
    const r = await fetch('/api/warm-alignments', {method: 'POST'});
    const data = await r.json();
    if (data.n_recomputed > 0) {
      setStatus(`aligned: ${data.n_recomputed} recomputed, ${data.n_wells - data.n_recomputed} cached`, false);
    } else {
      setStatus(`aligned: all ${data.n_wells} wells cached`, false);
    }
  } catch (e) {
    setStatus('warm-up failed', false);
  }
}

async function loadWell() {
  if (!wellSel.value) return;
  const [mid, batch, folder] = wellSel.value.split('|');
  const r = await fetch(`/api/well?mount_id=${encodeURIComponent(mid)}`
    + `&batch=${encodeURIComponent(batch)}&folder_name=${encodeURIComponent(folder)}`);
  const data = await r.json();
  timepoints = data.timepoints || [];
  tpSel.innerHTML = '';
  for (const tp of timepoints) {
    const opt = document.createElement('option');
    opt.value = tp.key;
    opt.dataset.label = tp.label;
    opt.dataset.tIdx = tp.t_idx;
    opt.textContent = `t${String(tp.t_idx).padStart(2,'0')} · ${tp.label}`;
    tpSel.appendChild(opt);
  }
  populateCopyFrom();
  if (timepoints.length) loadFrame();
}

function populateCopyFrom() {
  // Mirror the timepoint list (excluding the current one when loadFrame runs).
  copyFromSel.innerHTML = '<option value="">copy from…</option>';
  for (const tp of timepoints) {
    const opt = document.createElement('option');
    opt.value = tp.key;
    opt.dataset.label = tp.label;
    opt.textContent = `t${String(tp.t_idx).padStart(2,'0')} · ${tp.label}`;
    copyFromSel.appendChild(opt);
  }
}

async function loadFrame() {
  if (!tpSel.value) return;
  // Flush any pending edits before switching frames so we never lose work.
  if (dirty) {
    try { await saveAnnotations(); } catch (e) { /* surfaced in status */ }
  }
  currentKey = tpSel.value;
  const aligned = alignedBox.checked ? 1 : 0;
  img.src = `/api/image?key=${encodeURIComponent(currentKey)}&aligned=${aligned}`;
  img.onload = async () => {
    imgW = img.naturalWidth; imgH = img.naturalHeight;
    canvas.width = imgW; canvas.height = imgH;
    canvas.style.width = img.clientWidth + 'px';
    canvas.style.height = img.clientHeight + 'px';
    await loadAnnotations();
    pendingCentre = null;
    draw();
  };
}

// Annotations are stored per batch + well + frame.
function annUrl(k) {
  return `/api/annotations?batch=${encodeURIComponent(k.batch)}`
       + `&well=${encodeURIComponent(k.well)}&label=${encodeURIComponent(k.label)}`;
}

function annotationKey() {
  const wopt = wellSel.options[wellSel.selectedIndex];
  const topt = tpSel.options[tpSel.selectedIndex];
  return { batch: wopt.dataset.batch, well: wopt.dataset.folder,
           label: topt.dataset.label };
}

async function loadAnnotations() {
  const k = annotationKey();
  const r = await fetch(annUrl(k));
  const data = await r.json();
  circles = data.circles || [];
  setStatus(`${circles.length} loaded`, false);
}

async function saveAnnotations() {
  const k = annotationKey();
  const aligned = alignedBox.checked ? 1 : 0;
  const r = await fetch(`${annUrl(k)}&aligned=${aligned}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({circles, key: currentKey, image_w: imgW, image_h: imgH}),
  });
  if (!r.ok) { setStatus('save FAILED', true); return; }
  setStatus(`saved ${circles.length}`, false);
}

function eventToImageCoords(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.clientX - rect.left) * (imgW / rect.width);
  const y = (e.clientY - rect.top)  * (imgH / rect.height);
  return {x, y};
}

function hitCircle(x, y) {
  // Hit-test in band: |distance - r| < r * 0.25, or inside if very small.
  for (let i = circles.length - 1; i >= 0; --i) {
    const c = circles[i];
    const d = Math.hypot(x - c.cx, y - c.cy);
    if (Math.abs(d - c.r) <= Math.max(6, c.r * 0.25)) return i;
  }
  return -1;
}

canvas.addEventListener('mousemove', (e) => {
  mouse = eventToImageCoords(e);
  if (pendingCentre) draw();
});
canvas.addEventListener('mouseleave', () => { mouse = null; if (pendingCentre) draw(); });

canvas.addEventListener('contextmenu', (e) => {
  e.preventDefault();
  const {x, y} = eventToImageCoords(e);
  const hit = hitCircle(x, y);
  if (hit >= 0) {
    circles.splice(hit, 1);
    setStatus(`${circles.length} (deleted one)`, true);
    draw();
  }
});

canvas.addEventListener('click', (e) => {
  const {x, y} = eventToImageCoords(e);
  // If clicking on an existing circle and we're not mid-creation, select it.
  if (!pendingCentre) {
    const hit = hitCircle(x, y);
    if (hit >= 0) {
      selectedIdx = hit;
      draw();
      return;
    }
    // empty space click while something is selected → deselect
    if (selectedIdx >= 0) {
      selectedIdx = -1;
      draw();
      return;
    }
    pendingCentre = {x, y};
  } else {
    const r = Math.hypot(x - pendingCentre.x, y - pendingCentre.y);
    if (r >= 5) {
      circles.push({cx: pendingCentre.x, cy: pendingCentre.y, r, star: false});
      selectedIdx = circles.length - 1;
      setStatus(`${circles.length} (★ ${circles.filter(c=>c.star).length})`, true);
    }
    pendingCentre = null;
  }
  draw();
});

function _activeIsInput() {
  const el = document.activeElement;
  if (!el) return false;
  const t = el.tagName;
  return t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA';
}

document.addEventListener('keydown', (e) => {
  if (_activeIsInput()) return;
  if (e.key === 'Escape') {
    pendingCentre = null;
    selectedIdx = -1;
    draw();
    return;
  }
  if (selectedIdx < 0) return;
  const c = circles[selectedIdx];
  const step = e.shiftKey ? 10 : 1;
  let handled = true;
  if (e.key === 'ArrowLeft') {
    if (e.shiftKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {} // unreachable
    c.cx -= step;
  } else if (e.key === 'ArrowRight') {
    c.cx += step;
  } else if (e.key === 'ArrowUp') {
    if (e.shiftKey) { c.r = Math.max(2, c.r + 1); }
    else { c.cy -= step; }
  } else if (e.key === 'ArrowDown') {
    if (e.shiftKey) { c.r = Math.max(2, c.r - 1); }
    else { c.cy += step; }
  } else if (e.key === 's' || e.key === 'S') {
    c.star = !c.star;
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    circles.splice(selectedIdx, 1);
    selectedIdx = -1;
  } else {
    handled = false;
  }
  if (handled) {
    e.preventDefault();
    setStatus(`${circles.length} (★ ${circles.filter(c=>c.star).length})`, true);
    draw();
  }
});

function draw() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  for (let i = 0; i < circles.length; ++i) {
    const c = circles[i];
    const sel = i === selectedIdx;
    ctx.lineWidth = sel ? 5 : 3;
    ctx.strokeStyle = c.star ? '#ffd24a' : '#5cf';
    // semitransparent fill so it's clearly an annotated region without
    // hiding the underlying image
    ctx.fillStyle = c.star ? 'rgba(255, 210, 74, 0.18)'
                            : 'rgba(85, 204, 255, 0.15)';
    ctx.beginPath(); ctx.arc(c.cx, c.cy, c.r, 0, 2*Math.PI);
    ctx.fill();
    ctx.stroke();
    // centre cross
    ctx.beginPath();
    ctx.moveTo(c.cx-8, c.cy); ctx.lineTo(c.cx+8, c.cy);
    ctx.moveTo(c.cx, c.cy-8); ctx.lineTo(c.cx, c.cy+8);
    ctx.stroke();
    if (sel) {
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 1;
      ctx.setLineDash([4,4]);
      ctx.beginPath(); ctx.arc(c.cx, c.cy, c.r + 4, 0, 2*Math.PI); ctx.stroke();
      ctx.setLineDash([]);
    }
    if (c.star) {
      ctx.fillStyle = '#ffd24a';
      ctx.font = '24px sans-serif';
      ctx.fillText('★', c.cx + c.r + 4, c.cy - c.r + 4);
    }
  }
  if (pendingCentre && mouse) {
    const r = Math.hypot(mouse.x - pendingCentre.x, mouse.y - pendingCentre.y);
    ctx.strokeStyle = '#fa6'; ctx.lineWidth = 2;
    ctx.setLineDash([6,4]);
    ctx.beginPath(); ctx.arc(pendingCentre.x, pendingCentre.y, r, 0, 2*Math.PI); ctx.stroke();
    ctx.setLineDash([]);
  }
}

wellSel.addEventListener('change', loadWell);
tpSel.addEventListener('change', loadFrame);
alignedBox.addEventListener('change', loadFrame);
copyFromSel.addEventListener('change', async () => {
  const opt = copyFromSel.options[copyFromSel.selectedIndex];
  const srcKey = opt.value;
  if (!srcKey) return;
  if (srcKey === currentKey) {
    copyFromSel.value = '';
    return;
  }
  if (circles.length && !confirm(`Replace current ${circles.length} circle(s) with those from t${opt.dataset.label}?`)) {
    copyFromSel.value = '';
    return;
  }
  const wopt = wellSel.options[wellSel.selectedIndex];
  const r = await fetch(annUrl({batch: wopt.dataset.batch, well: wopt.dataset.folder,
                                label: opt.dataset.label}));
  const data = await r.json();
  const incoming = data.circles || [];
  // Deep copy so editing the new ones doesn't write through to anything.
  circles = incoming.map(c => ({cx: c.cx, cy: c.cy, r: c.r, star: !!c.star}));
  selectedIdx = -1;
  setStatus(`copied ${circles.length} from t${opt.dataset.label}`, circles.length > 0);
  copyFromSel.value = '';
  draw();
});

clearBtn.addEventListener('click', () => {
  if (!circles.length) return;
  if (!confirm('clear all circles on this frame?')) return;
  circles = [];
  setStatus('cleared', true);
  draw();
});

window.addEventListener('resize', () => {
  if (!imgW) return;
  canvas.style.width = img.clientWidth + 'px';
  canvas.style.height = img.clientHeight + 'px';
});

// Autosave loop: every 2s, if dirty and no in-progress placement, flush to disk.
setInterval(() => {
  if (dirty && !pendingCentre && currentKey) {
    saveAnnotations();
  }
}, 2000);

// Best-effort flush on page unload.
window.addEventListener('beforeunload', (e) => {
  if (!dirty) return;
  // Use sendBeacon for fire-and-forget delivery.
  const k = annotationKey();
  const payload = JSON.stringify({circles, key: currentKey, image_w: imgW, image_h: imgH});
  navigator.sendBeacon(
    annUrl(k),
    new Blob([payload], {type: 'application/json'}),
  );
});

loadGrid();
