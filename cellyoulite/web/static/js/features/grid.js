// Experiment chooser: one row per treatment, each holding its replicate
// thumbnails with segmentation / tracking / validation badges. Clicking a
// thumbnail opens that well in the viewer.
import { $, escapeHtml, setStatus, renderPills } from "../core/dom.js";
import { imgUrl } from "../core/api.js";
import { state } from "../core/state.js";
import * as well from "./well.js";

export async function refreshGrid() {
  setStatus("live", "scanning…");
  try {
    const r = await fetch(`/api/grid`);
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    state.grid = data;
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

function thumbHtml(w, treatment) {
  const st = state.cellposeStatus.get(w.folder_name);
  const tracked = state.trackDoneByWell.get(w.folder_name) === true;
  const humanValidated = state.humanValidatedByWell.get(w.folder_name) === true;
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
               data-folder-name="${w.folder_name}"
               data-key="${w.thumb_key}">
            <img loading="lazy" src="${imgUrl(w.thumb_key, 128)}" alt="${escapeHtml(treatment)} r${w.replicate}">
            <span class="thumb-tag">r${w.replicate}</span>
            ${badge}
            ${valChip}
          </div>`;
}

export function renderThumbGrid(data) {
  const { replicates, wells } = data;
  // Group wells into rows keyed by (mount, treatment).
  const rowKeys = [];
  const rowMeta = new Map();
  for (const w of wells) {
    const k = `${w.mount_id}|${w.treatment}`;
    if (!rowMeta.has(k)) {
      rowKeys.push(k);
      rowMeta.set(k, { mount_id: w.mount_id, mount_alias: w.mount_alias, treatment: w.treatment });
    }
  }
  rowKeys.sort((a, b) => {
    const A = rowMeta.get(a), B = rowMeta.get(b);
    return A.mount_alias.localeCompare(B.mount_alias) || A.treatment.localeCompare(B.treatment);
  });
  const byCell = new Map(wells.map(w => [`${w.mount_id}|${w.treatment}|${w.replicate}`, w]));

  $("thumb-grid").innerHTML = rowKeys.map(rk => {
    const { mount_id, treatment } = rowMeta.get(rk);
    const cells = replicates.map(r => {
      const w = byCell.get(`${mount_id}|${treatment}|${r}`);
      return (w && w.thumb_key) ? thumbHtml(w, treatment) : `<div class="thumb empty"></div>`;
    }).join("");
    return `<div class="exp-row">
              <div class="exp-row-label" title="${escapeHtml(treatment)}">${escapeHtml(treatment)}</div>
              <div class="exp-row-thumbs">${cells}</div>
            </div>`;
  }).join("");

  $("thumb-grid").querySelectorAll(".thumb[data-key]").forEach(el => {
    el.onclick = () => well.openWell(el.dataset.mount, el.dataset.folderName);
  });
}
