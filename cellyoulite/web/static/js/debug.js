const PARAMS = [
  // [key, label, min, max, step, default]
  ['r_min',          'r min',          5,   80,  1,    30],
  ['r_max',          'r max',          40,  300, 5,    140],
  ['r_step',         'r step',         2,   30,  1,    10],
  ['illum_sigma',    'illum σ',        20,  500, 10,   250],
  ['nms_factor',     'NMS factor',     0.3, 2.0, 0.05, 0.9],
  ['score_frac',     'score frac',     0.02,1.0, 0.01, 0.20],
  ['ring_band',      'ring band',      0.05,0.5, 0.01, 0.18],
  ['halo_factor',    'halo factor',    1.0, 2.0, 0.01, 1.35],
  ['contrast_floor', 'contrast floor', 0,   30,  0.5,  6.0],
];

const paramsEl = document.getElementById('params');
const wellSel = document.getElementById('well-sel');
const tpSel = document.getElementById('tp-sel');
const alignedBox = document.getElementById('aligned');
const resetBtn = document.getElementById('reset');
const statusEl = document.getElementById('status');
const img = document.getElementById('img');
const imgBefore = document.getElementById('img-before');
const compareBox = document.getElementById('compare');
const panesEl = document.getElementById('panes');
const afterCaption = document.getElementById('after-caption');
let stage = 'flat';
let timepoints = [];
let pending = null;
let reqSeq = 0;            // monotonically increasing; latest wins
const imgWrap = document.getElementById('img-wrap');

// Snapshot history: every successful refresh appends one entry. Arrow keys
// flip through them so the user can compare "before vs after" for a single
// parameter change without re-issuing a request.
const HISTORY_MAX = 25;
const history = [];
let historyIdx = -1;       // -1 = no history; otherwise points at the entry currently displayed

const values = {};
for (const [k, label, mn, mx, step, def] of PARAMS) {
  values[k] = def;
  const row = document.createElement('div');
  row.className = 'row';
  row.innerHTML = `
    <label for="p_${k}">${label}</label>
    <input type="range" id="p_${k}" min="${mn}" max="${mx}" step="${step}" value="${def}" />
    <span class="val" id="v_${k}">${def}</span>
  `;
  paramsEl.appendChild(row);
  const inp = document.getElementById('p_'+k);
  const val = document.getElementById('v_'+k);
  inp.addEventListener('input', () => {
    values[k] = parseFloat(inp.value);
    val.textContent = inp.value;
    schedule();
  });
}

for (const btn of document.querySelectorAll('.stage-toggle button')) {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.stage-toggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    stage = btn.dataset.stage;
    schedule();
  });
}

resetBtn.addEventListener('click', () => {
  for (const [k, label, mn, mx, step, def] of PARAMS) {
    values[k] = def;
    document.getElementById('p_'+k).value = def;
    document.getElementById('v_'+k).textContent = def;
  }
  schedule();
});

async function loadGrid() {
  const r = await fetch('/api/grid');
  const data = await r.json();
  const wells = data.wells || [];
  wellSel.innerHTML = '';
  for (const w of wells) {
    const opt = document.createElement('option');
    opt.value = w.mount_id + '|' + w.folder_name;
    opt.textContent = w.folder_name;
    wellSel.appendChild(opt);
  }
  if (wells.length) loadWell();
}

async function loadWell() {
  if (!wellSel.value) return;
  const [mid, folder] = wellSel.value.split('|');
  const r = await fetch(`/api/well?mount_id=${encodeURIComponent(mid)}&folder_name=${encodeURIComponent(folder)}`);
  const data = await r.json();
  timepoints = data.timepoints || [];
  tpSel.innerHTML = '';
  for (const tp of timepoints) {
    const opt = document.createElement('option');
    opt.value = tp.key;
    opt.textContent = `t${String(tp.t_idx).padStart(2,'0')} · ${tp.label}`;
    tpSel.appendChild(opt);
  }
  if (timepoints.length) schedule();
}

function schedule() {
  if (pending) clearTimeout(pending);
  pending = setTimeout(refresh, 80);
}

async function refresh() {
  if (!tpSel.value) return;
  const myReq = ++reqSeq;
  const q = new URLSearchParams({key: tpSel.value, stage,
    aligned: alignedBox.checked ? '1' : '0'});
  for (const [k] of PARAMS) q.set(k, values[k]);
  imgWrap.classList.add('loading');
  statusEl.classList.add('busy');
  statusEl.textContent = `computing ${stage}…`;
  const t0 = performance.now();
  const url = '/api/detect-debug?' + q.toString();
  let r;
  try {
    r = await fetch(url);
  } catch (e) {
    if (myReq !== reqSeq) return;
    statusEl.textContent = 'fetch error';
    statusEl.classList.remove('busy');
    imgWrap.classList.remove('loading');
    return;
  }
  if (myReq !== reqSeq) return;  // a newer request superseded us — drop
  if (!r.ok) {
    statusEl.textContent = 'error';
    statusEl.classList.remove('busy');
    imgWrap.classList.remove('loading');
    return;
  }
  const acc = r.headers.get('X-Accepted');
  const rej = r.headers.get('X-Rejected');
  const pre = r.headers.get('X-PreContrast');
  const blob = await r.blob();
  if (myReq !== reqSeq) return;
  // Don't revoke the previous blob URL here — it's referenced by the
  // history entry we just left. pushHistory() handles revocation when
  // entries fall off the front of the ring or get truncated.
  img.src = URL.createObjectURL(blob);
  img.onload = () => {
    if (myReq !== reqSeq) return;
    imgWrap.classList.remove('loading');
    statusEl.classList.remove('busy');
    const ms = (performance.now() - t0).toFixed(0);
    pushHistory({
      blobUrl: img.src,
      stage,
      params: {...values},
      key: tpSel.value,
      counts: {acc, rej, pre},
      ms,
    });
    renderHistoryStatus();
  };
  afterCaption.textContent = `after · ${stage}`;
}

function pushHistory(snap) {
  // If we're behind the tip, drop the forward history before appending.
  if (historyIdx >= 0 && historyIdx < history.length - 1) {
    for (let i = historyIdx + 1; i < history.length; ++i) {
      if (history[i].blobUrl && history[i].blobUrl.startsWith('blob:')) {
        URL.revokeObjectURL(history[i].blobUrl);
      }
    }
    history.length = historyIdx + 1;
  }
  history.push(snap);
  while (history.length > HISTORY_MAX) {
    const dropped = history.shift();
    if (dropped.blobUrl && dropped.blobUrl.startsWith('blob:')) {
      URL.revokeObjectURL(dropped.blobUrl);
    }
  }
  historyIdx = history.length - 1;
}

function applySnapshot(snap) {
  // Reflect snapshot params on the sliders so the user can see what produced this image.
  for (const [k] of PARAMS) {
    if (snap.params[k] === undefined) continue;
    values[k] = snap.params[k];
    const inp = document.getElementById('p_'+k);
    const val = document.getElementById('v_'+k);
    inp.value = snap.params[k];
    val.textContent = snap.params[k];
  }
  // Show the snapshot's image WITHOUT refetching.
  stage = snap.stage;
  document.querySelectorAll('.stage-toggle button').forEach(b => {
    b.classList.toggle('active', b.dataset.stage === stage);
  });
  img.src = snap.blobUrl;
  afterCaption.textContent = `after · ${snap.stage}`;
}

function renderHistoryStatus() {
  if (historyIdx < 0 || !history.length) return;
  const snap = history[historyIdx];
  const tag = historyIdx === history.length - 1 ? '' : ' · FROZEN';
  statusEl.textContent =
    `${snap.stage} · ${snap.ms}ms · accepted ${snap.counts.acc} · ` +
    `rejected ${snap.counts.rej} · pre ${snap.counts.pre}` +
    ` · history ${historyIdx + 1}/${history.length}${tag}`;
}

document.addEventListener('keydown', (e) => {
  const t = document.activeElement?.tagName;
  if (t === 'INPUT' || t === 'SELECT' || t === 'TEXTAREA') return;
  if (e.key === 'ArrowLeft' && historyIdx > 0) {
    historyIdx -= 1; applySnapshot(history[historyIdx]); renderHistoryStatus(); e.preventDefault();
  } else if (e.key === 'ArrowRight' && historyIdx < history.length - 1) {
    historyIdx += 1; applySnapshot(history[historyIdx]); renderHistoryStatus(); e.preventDefault();
  }
});

function updateBefore() {
  if (!tpSel.value) return;
  const aligned = alignedBox.checked ? 1 : 0;
  imgBefore.src = `/api/image?key=${encodeURIComponent(tpSel.value)}&aligned=${aligned}`;
}

function updateCompareMode() {
  panesEl.classList.toggle('compare-on', compareBox.checked);
  panesEl.classList.toggle('compare-off', !compareBox.checked);
  if (compareBox.checked) updateBefore();
}

wellSel.addEventListener('change', loadWell);
tpSel.addEventListener('change', () => { updateBefore(); schedule(); });
alignedBox.addEventListener('change', () => { updateBefore(); schedule(); });
compareBox.addEventListener('change', updateCompareMode);

loadGrid();
