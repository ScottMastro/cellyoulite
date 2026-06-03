// Per-well growth plot: log2(area / area@t0) over time, one curve per organoid.
import { $ } from "../core/dom.js";

let _tracks = [];   // last fetched organoids (all, accepted + rejected)
const opts = { autoScale: false, starsOnly: false, showRejected: false };

export function initGrowth() {
  const wire = (id, key) => {
    const el = $(id);
    if (el) el.onchange = () => { opts[key] = el.checked; render(); };
  };
  wire("growth-autoscale", "autoScale");
  wire("growth-stars-only", "starsOnly");
  wire("growth-show-rejected", "showRejected");
}

export async function fetchGrowthPlot(qs) {
  try {
    // Fetch every organoid (accepted + rejected); the toggles below decide
    // what to draw, so changing a toggle re-renders without re-fetching.
    const r = await fetch(`/api/well-growth?${qs}&valid_only=0`);
    if (!r.ok) { _tracks = []; render(); return; }
    const data = await r.json();
    _tracks = (data.available && data.tracks) ? data.tracks : [];
  } catch (e) {
    _tracks = [];
  }
  render();
}

function render() {
  const ctrls = $("growth-controls");
  if (!_tracks.length) {
    $("growth-plot").innerHTML = "";
    if (ctrls) ctrls.hidden = true;
    return;
  }
  if (ctrls) ctrls.hidden = false;
  // Accepted always shown; rejected only when asked (drawn dotted).
  let vis = _tracks.filter(t => t.accepted || (opts.showRejected && !t.accepted));
  if (opts.starsOnly) vis = vis.filter(t => t.starred);
  renderGrowthPlot(vis);
}

export function renderGrowthPlot(tracks) {
  const W = 880, H = 320, PAD_L = 50, PAD_R = 16, PAD_T = 14, PAD_B = 30;
  // log2(area/area@t0): 0 = baseline, +1 = doubled, -1 = halved.
  let maxMin = 0, dataMin = Infinity, dataMax = -Infinity;
  for (const t of tracks) for (const p of t.points) {
    if (p.minutes > maxMin) maxMin = p.minutes;
    const lf = Math.log2(Math.max(1e-6, p.norm));
    if (lf < dataMin) dataMin = lf;
    if (lf > dataMax) dataMax = lf;
  }
  if (!isFinite(dataMin)) { dataMin = -1.5; dataMax = 1.5; }
  if (maxMin === 0) maxMin = 1;
  // Default: a fixed log2 ±1.5 window (extended if data exceeds it). Auto-scale:
  // fit tightly to the data with a little padding.
  let yMin, yMax;
  if (opts.autoScale) {
    const pad = Math.max(0.1, (dataMax - dataMin) * 0.08);
    yMin = dataMin - pad; yMax = dataMax + pad;
  } else {
    yMin = Math.min(-1.5, dataMin); yMax = Math.max(1.5, dataMax);
  }
  const yRange = (yMax - yMin) || 1;
  const x = m => PAD_L + (W - PAD_L - PAD_R) * (m / maxMin);
  const y = n => H - PAD_B - (H - PAD_T - PAD_B) * ((Math.log2(Math.max(1e-6, n)) - yMin) / yRange);
  const lines = tracks.map(t => {
    const hue = (t.id * 137.508) % 360;
    const pts = t.points.map(p => `${x(p.minutes).toFixed(1)},${y(p.norm).toFixed(1)}`).join(" ");
    // Rejected organoids (only present when "show rejected" is on) are dotted.
    const dash = t.accepted ? "" : ` stroke-dasharray="3 3"`;
    return `<polyline points="${pts}" stroke="hsl(${hue.toFixed(0)},85%,60%)" `
         + `fill="none" stroke-width="1.4" stroke-opacity="0.85"${dash} `
         + `data-track-id="${t.id}">`
         + `</polyline>`;
  }).join("");
  const xTicks = [0, 0.25, 0.5, 0.75, 1].map(f => {
    const m = Math.round(maxMin * f);
    const hours = (m / 60).toFixed(m >= 60 ? 1 : 2);
    const xp = x(m).toFixed(1);
    return `<g><line x1="${xp}" y1="${H - PAD_B}" x2="${xp}" y2="${H - PAD_B + 4}" stroke="#888"/>`
         + `<text x="${xp}" y="${H - PAD_B + 18}" text-anchor="middle" fill="#aaa" font-size="10">${hours}h</text></g>`;
  }).join("");
  // log2 ticks at every 0.5, spanning the actual range; baseline (0) bold.
  const yTickVals = [];
  const lo = Math.floor(yMin * 2) / 2, hi = Math.ceil(yMax * 2) / 2;
  for (let v = lo; v <= hi + 1e-9; v += 0.5) yTickVals.push(parseFloat(v.toFixed(2)));
  const yTicks = yTickVals.map(v => {
    // Pass v as ratio: 2^v.
    const yp = y(Math.pow(2, v)).toFixed(1);
    const isBase = Math.abs(v) < 1e-6;
    const stroke = isBase ? "#666" : "#222";
    const dash   = isBase ? ""     : 'stroke-dasharray="2 4"';
    return `<g><line x1="${PAD_L - 4}" y1="${yp}" x2="${PAD_L}" y2="${yp}" stroke="#888"/>`
         + `<text x="${PAD_L - 6}" y="${(parseFloat(yp) + 3).toFixed(1)}" text-anchor="end" `
         + `fill="${isBase ? '#cdf' : '#aaa'}" font-size="10">${v.toFixed(1)}</text>`
         + `<line x1="${PAD_L}" y1="${yp}" x2="${W - PAD_R}" y2="${yp}" stroke="${stroke}" ${dash}/></g>`;
  }).join("");
  const scaleLabel = opts.autoScale ? "auto-scaled" : "log₂ ±1.5";
  $("growth-plot").innerHTML = `
    <div class="growth-plot-head">log<sub>2</sub>(area / area@t0) · <strong>${tracks.length}</strong> organoids · ${scaleLabel} · click a curve to inspect</div>
    <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" class="growth-svg">
      <rect x="${PAD_L}" y="${PAD_T}" width="${W - PAD_L - PAD_R}" height="${H - PAD_T - PAD_B}" fill="#0d1117" stroke="#2a2f38"/>
      ${yTicks}
      ${xTicks}
      ${lines}
      <text x="${PAD_L - 38}" y="${PAD_T + 10}" fill="#aaa" font-size="10">log₂</text>
      <text x="${W - PAD_R}" y="${H - PAD_B + 18}" text-anchor="end" fill="#aaa" font-size="10">time</text>
    </svg>
  `;
  // Click a curve → scroll the organoid list to that organoid and flash it.
  $("growth-plot").querySelectorAll("polyline[data-track-id]").forEach(el => {
    el.style.cursor = "pointer";
    el.onclick = () => jumpToTrack(parseInt(el.getAttribute("data-track-id")));
  });
}

export function jumpToTrack(trackId) {
  const row = document.querySelector(`#all-tracks-list .atr-row[data-track-id="${trackId}"]`);
  if (!row) return;
  row.scrollIntoView({ behavior: "smooth", block: "center" });
  row.classList.add("atr-flash");
  setTimeout(() => row.classList.remove("atr-flash"), 1200);
}
