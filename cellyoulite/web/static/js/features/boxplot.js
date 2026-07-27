// Per-treatment growth boxplots (one well's treatment, or all treatments).
import { $ } from "../core/dom.js";
import { state } from "../core/state.js";

function setDrill(treatment) {
  // null = the "all conditions" overview; a treatment name = zoomed in.
  state.boxDrill = treatment;
  refreshBoxplot();
}

// The Growth distributions are independent of the playback selection: they
// always pool every well, showing all conditions by default and a single
// condition when the user clicks one to zoom in.
export async function refreshBoxplot() {
  const wrap = $("boxplot-card");
  const drill = state.boxDrill;
  const byRep = $("box-by-rep").checked ? 1 : 0;
  const starsOnly = $("box-stars-only").checked ? 1 : 0;
  if ($("box-back")) $("box-back").hidden = !drill;
  const treatmentQs = drill ? `&treatment=${encodeURIComponent(drill)}` : "";
  // Treatment names repeat across batches, so pooling them would mix two
  // experiments into one distribution. Follow the grid's batch selection.
  const batchQs = state.batch ? `&batch=${encodeURIComponent(state.batch)}` : "";
  const url = `/api/boxplot-data?by_replicate=${byRep}&stars_only=${starsOnly}${treatmentQs}${batchQs}`;
  // Keep the current plot in place while the (cached, fast) fetch runs — don't
  // collapse it to a placeholder, which shrinks the page and makes the scroll
  // jump up.
  try {
    const r = await fetch(url);
    wrap.hidden = false;
    if (!r.ok) { $("boxplot").innerHTML = `<div class="hint">error: ${r.statusText}</div>`; return; }
    renderBoxplot(await r.json());
  } catch (e) { $("boxplot").innerHTML = `<div class="hint">error: ${e}</div>`; }
}

// Scale: server always returns log2(area/area@t0); the client transforms
// for display so swapping scales doesn't refetch.
//   fold   → 2^x         (1 = baseline, 2 = doubled, 0.5 = halved)
//   pct    → (2^x - 1)·100  (% change; 0 = baseline, +100 = doubled)
//   log2   → x          (raw, symmetric around 0)
function scaleTransform(scale) {
  if (scale === "fold") {
    return { f: v => Math.pow(2, v),
             unit: "×",     baseline: 1, defaultLo: 0.5, defaultHi: 2.5,
             yLabel: "fold", tickFmt: v => v.toFixed(v < 10 ? 2 : 1) + "×",
             slopeFmt: rate => (rate >= 0 ? "+" : "") + (Math.pow(2, rate) - 1).toFixed(3) + "×/h" };
  }
  if (scale === "pct") {
    return { f: v => (Math.pow(2, v) - 1) * 100,
             unit: "%",     baseline: 0, defaultLo: -50, defaultHi: 150,
             yLabel: "% Δ", tickFmt: v => (v >= 0 ? "+" : "") + v.toFixed(0) + "%",
             slopeFmt: rate => (rate >= 0 ? "+" : "") + ((Math.pow(2, rate) - 1) * 100).toFixed(1) + "%/h" };
  }
  return { f: v => v,
           unit: "log₂", baseline: 0, defaultLo: -1.5, defaultHi: 1.5,
           yLabel: "log₂", tickFmt: v => v.toFixed(1),
           slopeFmt: rate => (rate >= 0 ? "+" : "") + rate.toFixed(3) + " log₂/h" };
}

// Treatment colour — golden-angle hue, stable per name.
function hueForTreatment(name) {
  let h = 0;
  for (let i = 0; i < name.length; ++i) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return (h * 137.508) % 360;
}

function renderBoxplot(data) {
  const treatments = data.treatments || [];
  const minutes = data.minutes || [];
  const boxes = data.boxes || [];
  if (!boxes.length) {
    const msg = $("box-stars-only").checked ? "no starred organoids" : "no accepted organoids yet";
    $("boxplot").innerHTML = `<div class="hint" style="padding:8px">${msg}</div>`;
    $("box-summary").textContent = "";
    return;
  }

  const scale = scaleTransform($("box-scale").value);
  // Transform every box's stats once, so renderers don't need to know
  // about the scale; they just plot what's given to them.
  const tBoxes = boxes.map(b => ({
    treatment: b.treatment,
    minutes: b.minutes,
    replicate: b.replicate,
    n: b.n,
    median: scale.f(b.median),
    q1: scale.f(b.q1),
    q3: scale.f(b.q3),
    lo: scale.f(b.lo),
    hi: scale.f(b.hi),
    outliers: (b.outliers || []).map(scale.f),
    _log2_median: b.median,   // kept so the slope fit stays exact
    _log2_minutes: b.minutes,
  }));
  let yMin = scale.defaultLo, yMax = scale.defaultHi;
  for (const b of tBoxes) {
    yMin = Math.min(yMin, b.lo);
    yMax = Math.max(yMax, b.hi);
    for (const o of b.outliers) { yMin = Math.min(yMin, o); yMax = Math.max(yMax, o); }
  }

  // Summary
  const totalN = boxes.reduce((a, b) => a + (b.n || 0), 0);
  $("box-summary").innerHTML =
    `<strong>${treatments.length}</strong> treatment(s) · `
    + `<strong>${minutes.length}</strong> timepoints · `
    + `<strong>${totalN}</strong> organoid-frame observations`;

  const replicates = data.replicates || [];
  const byRep = !!data.by_replicate;
  const series = byRep && replicates.length ? replicates : [null];

  if (state.boxDrill || treatments.length === 1) {
    const svg = renderBoxSubplot({
      title: treatments[0] || "",
      boxes: tBoxes, minutes, treatments, series, byRep, scale,
      yMin, yMax,
      width: 880, height: 360, large: true,
    });
    $("boxplot").innerHTML = svg;
    return;
  }

  const byTreat = new Map(treatments.map(t => [t, []]));
  for (const b of tBoxes) {
    if (byTreat.has(b.treatment)) byTreat.get(b.treatment).push(b);
  }
  const subWidth = 420, subHeight = 230;
  const tiles = treatments.map(t => {
    const sub = renderBoxSubplot({
      title: t,
      boxes: byTreat.get(t),
      minutes, treatments: [t], series, byRep, scale,
      yMin, yMax,
      width: subWidth, height: subHeight, large: false,
    });
    const safe = t.replace(/"/g, "&quot;");
    return `<div class="boxplot-facet" data-treatment="${safe}" title="click to zoom">${sub}</div>`;
  }).join("");
  $("boxplot").innerHTML = `<div class="boxplot-grid">${tiles}</div>`;
  // Click a facet → zoom into that condition.
  $("boxplot").querySelectorAll(".boxplot-facet[data-treatment]").forEach(el => {
    el.onclick = () => setDrill(el.dataset.treatment);
  });
}

function renderBoxSubplot({ title, boxes, minutes, treatments, series, byRep, scale,
                            yMin, yMax, width, height, large }) {
  const W = width, H = height;
  const PAD_L = large ? 50 : 38, PAD_R = 12, PAD_T = large ? 18 : 22, PAD_B = large ? 50 : 36;
  const plotW = W - PAD_L - PAD_R, plotH = H - PAD_T - PAD_B;
  const yRange = yMax - yMin;
  const y = v => H - PAD_B - plotH * ((v - yMin) / yRange);
  const slotW = plotW / Math.max(1, minutes.length);
  const minutesIdx = new Map(minutes.map((m, i) => [m, i]));
  series = series && series.length ? series : [null];
  // Per-slot grouping: (treatment × series). In single-treatment subplots
  // that's just `series.length` boxes per slot; in the multi-treatment view
  // (no facets) it's treatments × series.
  const groupCount = (treatments.length || 1) * series.length;
  const boxW = Math.max(3, Math.min(large ? 30 : 18, (slotW - 4) / Math.max(1, groupCount)));

  // X ticks — fewer in small subplots.
  const xStride = large ? 1 : Math.max(1, Math.ceil(minutes.length / 5));
  const xTicks = minutes.map((m, i) => {
    if (i % xStride !== 0 && i !== minutes.length - 1) return "";
    const cx = PAD_L + slotW * i + slotW / 2;
    const hours = (m / 60).toFixed(m >= 60 ? 1 : 2);
    return `<g><line x1="${cx}" y1="${H - PAD_B}" x2="${cx}" y2="${H - PAD_B + 3}" stroke="#888"/>`
         + `<text x="${cx}" y="${H - PAD_B + 14}" text-anchor="middle" fill="#aaa" font-size="9">${hours}h</text></g>`;
  }).join("");

  // Y ticks — adaptive step per scale so we don't end up with 30 ticks
  // on a percent axis. Aim for ~6-8 ticks total.
  const span = yMax - yMin;
  const targetTicks = large ? 8 : 5;
  const rawStep = span / targetTicks;
  // Snap to a 1/2/5 × 10^k.
  const exp10 = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const mant = rawStep / exp10;
  const niceMant = mant < 1.5 ? 1 : mant < 3 ? 2 : mant < 7 ? 5 : 10;
  const tickStep = niceMant * exp10;
  const baseline = scale.baseline;
  const yTicks = [];
  // Walk outward from the baseline in both directions so it's always
  // exactly on a tick (and visually emphasised).
  for (let v = baseline; v <= yMax + 1e-9; v += tickStep) yTicks.push(v);
  for (let v = baseline - tickStep; v >= yMin - 1e-9; v -= tickStep) yTicks.push(v);
  yTicks.sort((a, b) => a - b);
  const yTickHtml = yTicks.map(v => {
    const yp = y(v).toFixed(1);
    const isBase = Math.abs(v - baseline) < tickStep * 0.01;
    return `<g><line x1="${PAD_L - 3}" y1="${yp}" x2="${PAD_L}" y2="${yp}" stroke="#888"/>`
         + `<text x="${PAD_L - 5}" y="${(parseFloat(yp)+3).toFixed(1)}" text-anchor="end" `
         + `fill="${isBase ? '#cdf' : '#aaa'}" font-size="9">${scale.tickFmt(v)}</text>`
         + `<line x1="${PAD_L}" y1="${yp}" x2="${W - PAD_R}" y2="${yp}" stroke="${isBase ? '#666' : '#222'}" `
         + `${isBase ? "" : 'stroke-dasharray="2 4"'} /></g>`;
  }).join("");

  const boxesHtml = (boxes || []).map(b => {
    const ti = minutesIdx.get(b.minutes);
    if (ti == null) return "";
    const tIdxInGroup = Math.max(0, treatments.indexOf(b.treatment));
    const sIdx = byRep && b.replicate != null
      ? Math.max(0, series.indexOf(b.replicate)) : 0;
    const slotIdx = tIdxInGroup * series.length + sIdx;
    const slotCx = PAD_L + slotW * ti + slotW / 2;
    const groupOffset = (slotIdx - (groupCount - 1) / 2) * boxW;
    const cx = slotCx + groupOffset;
    const x0 = cx - boxW / 2;
    const hue = hueForTreatment(b.treatment);
    // Replicates: shift lightness around the treatment's hue so they read
    // as a family (same color, different brightness).
    const lightness = byRep && b.replicate != null
      ? 40 + (sIdx * 18) : 55;
    const color = `hsl(${hue.toFixed(0)},75%,${lightness}%)`;
    const yMed = y(b.median).toFixed(1);
    const yQ1 = y(b.q1).toFixed(1);
    const yQ3 = y(b.q3).toFixed(1);
    const yLo = y(b.lo).toFixed(1);
    const yHi = y(b.hi).toFixed(1);
    const outlierDots = (b.outliers || []).map(o =>
      `<circle cx="${cx}" cy="${y(o).toFixed(1)}" r="1.4" fill="${color}" fill-opacity="0.7"/>`
    ).join("");
    return `<g>
      <line x1="${cx}" x2="${cx}" y1="${yHi}" y2="${yQ3}" stroke="${color}" stroke-width="1"/>
      <line x1="${cx}" x2="${cx}" y1="${yQ1}" y2="${yLo}" stroke="${color}" stroke-width="1"/>
      <line x1="${cx - boxW/3}" x2="${cx + boxW/3}" y1="${yHi}" y2="${yHi}" stroke="${color}" stroke-width="1"/>
      <line x1="${cx - boxW/3}" x2="${cx + boxW/3}" y1="${yLo}" y2="${yLo}" stroke="${color}" stroke-width="1"/>
      <rect x="${x0.toFixed(1)}" y="${yQ3}" width="${boxW.toFixed(1)}" height="${(parseFloat(yQ1) - parseFloat(yQ3)).toFixed(1)}"
            fill="${color}" fill-opacity="0.22" stroke="${color}" stroke-width="1.2"/>
      <line x1="${x0.toFixed(1)}" x2="${(x0+boxW).toFixed(1)}" y1="${yMed}" y2="${yMed}" stroke="${color}" stroke-width="2"/>
      ${outlierDots}
      <title>${b.treatment}${b.replicate != null ? ` · r${b.replicate}` : ""} · ${(b.minutes/60).toFixed(2)}h\nn=${b.n} · med=${b.median.toFixed(2)} · q1=${b.q1.toFixed(2)} · q3=${b.q3.toFixed(2)}</title>
    </g>`;
  }).join("");

  // Linear fit of medians for each (treatment, replicate) series — slope
  // in log2-fold per HOUR (minutes → hours via *60).
  function _linFit(xs, ys) {
    const n = xs.length;
    if (n < 2) return null;
    const mx = xs.reduce((a, b) => a + b, 0) / n;
    const my = ys.reduce((a, b) => a + b, 0) / n;
    let num = 0, den = 0;
    for (let i = 0; i < n; ++i) {
      num += (xs[i] - mx) * (ys[i] - my);
      den += (xs[i] - mx) ** 2;
    }
    if (den < 1e-9) return null;
    const m = num / den;
    return { slope: m, intercept: my - m * mx };
  }

  const seriesBoxes = new Map();
  for (const b of (boxes || [])) {
    const repKey = byRep && b.replicate != null ? b.replicate : -1;
    const k = `${b.treatment}|${repKey}`;
    if (!seriesBoxes.has(k)) seriesBoxes.set(k, { treatment: b.treatment, replicate: repKey, xs: [], ys: [] });
    const ent = seriesBoxes.get(k);
    // Slope: always fit in log2 space (exact log2_median preserved on
    // each transformed box), then format per the chosen scale.
    ent.xs.push(b._log2_minutes != null ? b._log2_minutes : b.minutes);
    ent.ys.push(b._log2_median != null ? b._log2_median : b.median);
  }
  const slopeChips = [];
  let chipRowY = PAD_T + 12;
  for (const ent of seriesBoxes.values()) {
    const fit = _linFit(ent.xs, ent.ys);
    if (!fit) continue;
    const hue = hueForTreatment(ent.treatment);
    const sIdx = ent.replicate >= 0 ? Math.max(0, series.indexOf(ent.replicate)) : 0;
    const lightness = ent.replicate >= 0 ? 40 + sIdx * 18 : 55;
    const color = `hsl(${hue.toFixed(0)},75%,${lightness}%)`;
    const slopePerHour = fit.slope * 60;
    const label = (ent.replicate >= 0 ? `r${ent.replicate} ` : "") + scale.slopeFmt(slopePerHour);
    slopeChips.push(`<text x="${W - PAD_R - 6}" y="${chipRowY}" text-anchor="end" `
                  + `fill="${color}" font-size="${large ? 11 : 10}" font-weight="700">${label}</text>`);
    chipRowY += (large ? 13 : 12);
  }

  const titleHue = hueForTreatment(title);
  const titleHtml = title ? `<text x="${W/2}" y="13" text-anchor="middle" `
                          + `fill="hsl(${titleHue.toFixed(0)},75%,55%)" font-size="${large ? 13 : 11}" `
                          + `font-weight="700">${title}</text>` : "";

  return `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet" class="boxplot-svg">
    <rect x="${PAD_L}" y="${PAD_T}" width="${plotW}" height="${plotH}" fill="#0d1117" stroke="#2a2f38"/>
    ${titleHtml}
    ${yTickHtml}
    ${xTicks}
    ${boxesHtml}
    ${slopeChips.join("")}
    <text x="${PAD_L - 28}" y="${PAD_T + 9}" fill="#aaa" font-size="9">${scale.yLabel}</text>
  </svg>`;
}

export function initBoxplot() {
  $("box-back").onclick = () => setDrill(null);
  $("box-stars-only").onchange = () => refreshBoxplot();
  $("box-by-rep").onchange = () => refreshBoxplot();
  $("box-scale").onchange = () => refreshBoxplot();
  $("box-csv").onclick = async () => {
    const btn = $("box-csv");
    const orig = btn.textContent;
    btn.disabled = true; btn.textContent = "…";
    try {
      const params = new URLSearchParams();
      if (state.boxDrill) params.set("treatment", state.boxDrill);
      if (state.batch) params.set("batch", state.batch);
      const q = params.toString();
      const url = q ? `/api/growth-csv?${q}` : `/api/growth-csv`;
      const r = await fetch(url);
      if (!r.ok) throw new Error(r.statusText);
      const blob = await r.blob();
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = objectUrl;
      const fname = state.boxDrill
        ? `growth_${state.boxDrill.replace(/\s+/g, "_").replace(/\+/g, "p")}.csv`
        : "growth_all.csv";
      a.download = fname;
      a.click();
      setTimeout(() => URL.revokeObjectURL(objectUrl), 60000);
      btn.textContent = "✓";
      setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 800);
    } catch (e) {
      btn.textContent = "err";
      setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1500);
    }
  };
}
