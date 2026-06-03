// Experiment grid: thumbnails per (treatment × replicate) with segmentation
// / tracking / validation badges, plus a large hover preview.
import { $, setStatus, renderPills } from "../core/dom.js";
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
    // Counts are shown in the data card pills, so keep the header minimal.
    setStatus("ok", data.n_wells ? "ready" : "no wells");
  } catch (e) {
    setStatus("err", String(e.message || e));
  }
}

export function renderThumbGrid(data) {
  const { treatments, replicates, wells } = data;
  // Row key = (mount_id, treatment). Each well belongs to exactly one row.
  const rowKeys = [];
  const rowMeta = new Map();   // rowKey -> { mount_id, mount_alias, treatment }
  for (const w of wells) {
    const k = `${w.mount_id}|${w.treatment}`;
    if (!rowMeta.has(k)) {
      rowKeys.push(k);
      rowMeta.set(k, { mount_id: w.mount_id, mount_alias: w.mount_alias, treatment: w.treatment });
    }
  }
  rowKeys.sort((a, b) => {
    const A = rowMeta.get(a), B = rowMeta.get(b);
    return (A.mount_alias.localeCompare(B.mount_alias)) ||
           (A.treatment.localeCompare(B.treatment));
  });
  const byCell = new Map(wells.map(w => [`${w.mount_id}|${w.treatment}|${w.replicate}`, w]));

  const grid = $("thumb-grid");
  grid.style.gridTemplateColumns = `auto repeat(${replicates.length}, 88px)`;

  let html = `<div class="th corner"></div>`;
  for (const r of replicates) html += `<div class="th col">r${r}</div>`;
  for (const rk of rowKeys) {
    const { mount_id, mount_alias, treatment } = rowMeta.get(rk);
    const label = treatment;
    html += `<div class="th row">${label}</div>`;
    for (const r of replicates) {
      const w = byCell.get(`${mount_id}|${treatment}|${r}`);
      if (!w || !w.thumb_key) {
        html += `<div class="thumb empty"></div>`;
      } else {
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
        // Human-validated check-circle in the top-left of the thumb.
        const valCls = humanValidated ? "validated" : "unvalidated";
        const valGlyph = humanValidated ? "✓" : "?";
        const valTitle = humanValidated ? "validated by human" : "not yet validated";
        const valChip = `<span class="val-chip ${valCls}" title="${valTitle}">${valGlyph}</span>`;
        html += `<div class="thumb"
                      data-mount="${w.mount_id}"
                      data-folder-name="${w.folder_name}"
                      data-tx="${w.treatment}"
                      data-rep="${w.replicate}"
                      data-key="${w.thumb_key}">
                   <img loading="lazy" src="${imgUrl(w.thumb_key, 128)}" alt="${treatment} r${r}">
                   <span class="thumb-tag">r${r}</span>
                   ${badge}
                   ${valChip}
                 </div>`;
      }
    }
  }
  grid.innerHTML = html;

  grid.querySelectorAll(".thumb[data-key]").forEach(el => {
    el.onclick = () => well.openWell(el.dataset.mount, el.dataset.folderName);
    el.addEventListener("mouseenter", onThumbEnter);
    el.addEventListener("mouseleave", hidePreview);
    el.addEventListener("mousemove", positionPreview);
  });
}

// ---------------- hover preview ----------------

const preview = $("thumb-preview");
const previewImg = $("thumb-preview-img");
const previewCap = $("thumb-preview-cap");

function onThumbEnter(e) {
  const el = e.currentTarget;
  previewImg.src = imgUrl(el.dataset.key, 512);
  previewCap.textContent = `${el.dataset.tx} · r${el.dataset.rep}`;
  preview.classList.add("show");
  positionPreview(e);
}
function positionPreview(e) {
  const PAD = 16;
  const W = 332, H = 360;
  let x = e.clientX + PAD;
  let y = e.clientY;
  if (x + W > window.innerWidth) x = e.clientX - W - PAD;
  if (y - H / 2 < 8) y = H / 2 + 8;
  if (y + H / 2 > window.innerHeight - 8) y = window.innerHeight - H / 2 - 8;
  preview.style.left = `${x}px`;
  preview.style.top  = `${y}px`;
}
function hidePreview() { preview.classList.remove("show"); }
