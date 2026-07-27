// Experiment chooser: one row per treatment, each holding its replicate
// thumbnails with segmentation / tracking / validation badges. Clicking a
// thumbnail opens that well in the viewer.
import { $, escapeHtml, setStatus, renderPills } from "../core/dom.js";
import { imgUrl } from "../core/api.js";
import { state, wellKey } from "../core/state.js";
import * as boxplot from "./boxplot.js";
import * as well from "./well.js";

export async function refreshGrid() {
  setStatus("live", "scanning…");
  try {
    const qs = state.batch ? `?batch=${encodeURIComponent(state.batch)}` : "";
    const r = await fetch(`/api/grid${qs}`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    state.grid = data;
    state.batches = data.batches || [];
    state.batchCounts = data.batch_counts || {};
    // A batch that has gone away (renamed on disk) falls back to "all".
    if (state.batch && !state.batches.includes(state.batch)) state.batch = null;
    renderBatchTabs();
    renderPills("grid-pills", [
      ["treatments", data.treatments.length],
      ["replicates", data.replicates.length],
      ["wells", data.n_wells],
      ["images", data.n_images],
    ]);
    renderThumbGrid(data);
    $("grid-card").hidden = data.n_wells === 0;
    setStatus("ok", data.n_wells ? "ready" : "no wells");
  } catch (e) {
    setStatus("err", String(e.message || e));
  }
}

// Curate and Results each get a row, both driving the one state.batch, so the
// two views can never disagree about which batch is on screen.
const _BATCH_TAB_HOSTS = ["batch-tabs", "batch-tabs-results"];

export function renderBatchTabs() {
  // Nothing to filter between until a second batch exists.
  const hide = state.batches.length < 2;
  const total = Object.values(state.batchCounts).reduce((a, b) => a + b, 0);
  const tab = (value, label, n) =>
    `<button class="batch-tab${value === state.batch ? " active" : ""}"`
    + ` data-batch="${value === null ? "" : escapeHtml(value)}"`
    + ` title="${escapeHtml(label)} — ${n} well${n === 1 ? "" : "s"}">`
    + `${escapeHtml(label)}<span class="batch-tab-n">${n}</span></button>`;
  const html = [tab(null, "All", total)].concat(
    state.batches.map(b => tab(b, b, state.batchCounts[b] ?? 0))).join("");

  for (const id of _BATCH_TAB_HOSTS) {
    const el = $(id);
    if (!el) continue;
    el.hidden = hide;
    if (hide) continue;
    el.innerHTML = html;
    el.querySelectorAll(".batch-tab").forEach(b => {
      b.onclick = () => selectBatch(b.dataset.batch || null);
    });
  }
}

export function selectBatch(name) {
  if (state.batch === name) return;
  state.batch = name;
  // Refresh both views: the well list, and the distributions, which pool
  // organoids per treatment and must not mix batches.
  refreshGrid();
  boxplot.refreshBoxplot();
}

function thumbHtml(w, treatment) {
  const k = wellKey(w.batch, w.folder_name);
  const st = state.cellposeStatus.get(k);
  const tracked = state.trackDoneByWell.get(k) === true;
  const humanValidated = state.humanValidatedByWell.get(k) === true;
  let badge = "";
  if (st) {
    let cls;
    if (st.n_done === 0) cls = "todo";
    else if (st.n_done < st.n_total) cls = "partial";
    else if (!tracked) cls = "partial";
    else cls = "done";
    const title = `${st.n_done}/${st.n_total} segmented`
      + (tracked ? " · tracked" : " · tracking pending");
    badge = `<span class="cp-badge ${cls}" title="${title}">${st.n_done}/${st.n_total}</span>`;
  }
  const valCls = humanValidated ? "validated" : "unvalidated";
  const valGlyph = humanValidated ? "✓" : "?";
  const valTitle = humanValidated ? "validated by human" : "not yet validated";
  const valChip = `<span class="val-chip ${valCls}" title="${valTitle}">${valGlyph}</span>`;
  return `<div class="thumb"
               data-mount="${w.mount_id}"
               data-batch="${escapeHtml(w.batch)}"
               data-folder-name="${escapeHtml(w.folder_name)}"
               data-key="${w.thumb_key}">
            <img loading="lazy" src="${imgUrl(w.thumb_key, 128)}" alt="${escapeHtml(treatment)} r${w.replicate}">
            <span class="thumb-tag">r${w.replicate}</span>
            ${badge}
            ${valChip}
          </div>`;
}

export function renderThumbGrid(data) {
  const { replicates, wells } = data;
  // Group wells into rows keyed by (batch, treatment) — the same treatment
  // name in two batches is two different experiments and gets its own row.
  const rowKeys = [];
  const rowMeta = new Map();
  for (const w of wells) {
    const k = `${w.mount_id}|${w.batch}|${w.treatment}`;
    if (!rowMeta.has(k)) {
      rowKeys.push(k);
      rowMeta.set(k, { mount_id: w.mount_id, batch: w.batch, treatment: w.treatment });
    }
  }
  rowKeys.sort((a, b) => {
    const A = rowMeta.get(a), B = rowMeta.get(b);
    return A.batch.localeCompare(B.batch) || A.treatment.localeCompare(B.treatment);
  });
  const byCell = new Map(
    wells.map(w => [`${w.mount_id}|${w.batch}|${w.treatment}|${w.replicate}`, w]));
  // Only worth naming the batch on each row when more than one is on screen.
  const showBatch = new Set(wells.map(w => w.batch)).size > 1;

  $("thumb-grid").innerHTML = rowKeys.map(rk => {
    const { mount_id, batch, treatment } = rowMeta.get(rk);
    const cells = replicates.map(r => {
      const w = byCell.get(`${mount_id}|${batch}|${treatment}|${r}`);
      return (w && w.thumb_key) ? thumbHtml(w, treatment) : `<div class="thumb empty"></div>`;
    }).join("");
    const label = showBatch
      ? `${escapeHtml(treatment)}<span class="exp-row-batch">${escapeHtml(batch)}</span>`
      : escapeHtml(treatment);
    return `<div class="exp-row">
              <div class="exp-row-label" title="${escapeHtml(batch + " · " + treatment)}">${label}</div>
              <div class="exp-row-thumbs">${cells}</div>
            </div>`;
  }).join("");

  $("thumb-grid").querySelectorAll(".thumb[data-key]").forEach(el => {
    el.onclick = () => well.openWell(el.dataset.mount, el.dataset.batch, el.dataset.folderName);
  });
}
