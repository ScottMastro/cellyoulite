// "Data" card: add raw images (folder upload), add segmentation results
// (bundle import), and download a subset of experiments' images.
import { $, escapeHtml } from "../core/dom.js";
import { state } from "../core/state.js";
import { refreshAll } from "./polling.js";

function updateDlCount() {
  const n = $("dl-list").querySelectorAll(".dl-item input:checked").length;
  $("dl-count").textContent = `${n} selected`;
}

// subset image-download picker (experiments grouped by treatment)
function openDownloadModal() {
  const wells = (state.grid && state.grid.wells) || [];
  if (!wells.length) { $("data-status").textContent = "no experiments to download yet"; return; }
  const byTreat = {};
  for (const w of wells) (byTreat[w.treatment] ||= []).push(w.folder_name);
  $("dl-list").innerHTML = Object.keys(byTreat).sort().map(tr => {
    const items = byTreat[tr].sort().map(fn =>
      `<label class="dl-item"><input type="checkbox" value="${escapeHtml(fn)}"> ${escapeHtml(fn)}</label>`
    ).join("");
    return `<div class="dl-group">`
         + `<label class="dl-group-head"><input type="checkbox" class="dl-group-all"> <strong>${escapeHtml(tr)}</strong></label>`
         + `${items}</div>`;
  }).join("");
  $("dl-select-all").checked = false;
  $("dl-list").querySelectorAll(".dl-item input").forEach(c =>
    c.addEventListener("change", updateDlCount));
  $("dl-list").querySelectorAll(".dl-group-all").forEach(g =>
    g.addEventListener("change", (e) => {
      e.target.closest(".dl-group").querySelectorAll(".dl-item input")
        .forEach(c => c.checked = e.target.checked);
      updateDlCount();
    }));
  updateDlCount();
  $("dl-modal").classList.remove("hidden");
}

export function initData() {
  $("add-images").onchange = async (e) => {
    const files = [...e.target.files];
    e.target.value = "";
    if (!files.length) return;
    const st = $("data-status");
    const imgs = files.filter(f => /\.(tif|tiff|png|jpe?g|bmp)$/i.test(f.name));
    if (!imgs.length) { st.textContent = "no image files in that folder"; return; }
    st.textContent = `uploading ${imgs.length} image(s)…`;
    const fd = new FormData();
    // 3rd arg keeps the relative path so the server can route by experiment name.
    for (const f of imgs) fd.append("files", f, f.webkitRelativePath || f.name);
    try {
      const r = await fetch("/api/upload-images", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) { st.textContent = `error: ${d.detail || r.statusText}`; return; }
      st.textContent = `added ${d.n_files} image(s) to ${d.n_experiments} experiment(s)`
        + (d.skipped ? ` · skipped ${d.skipped}` : "");
      await refreshAll();
    } catch (err) { st.textContent = `error: ${err}`; }
  };

  $("add-seg").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const st = $("data-status");
    st.textContent = `importing ${file.name}…`;
    try {
      const r = await fetch("/api/bundle-import", {
        method: "POST",
        headers: { "Content-Type": "application/gzip" },
        body: file,
      });
      const d = await r.json();
      if (!r.ok) { st.textContent = `error: ${d.detail || r.statusText}`; return; }
      const ex = d.updated_experiments || [];
      st.textContent = ex.length
        ? `merged results for ${ex.length} experiment(s): ${ex.join(", ")}`
        : "imported (no experiments detected in bundle)";
      await refreshAll();
    } catch (err) { st.textContent = `error: ${err}`; }
  };

  $("download-images").onclick = openDownloadModal;
  $("dl-close").onclick = () => $("dl-modal").classList.add("hidden");
  $("dl-select-all").onchange = (e) => {
    $("dl-list").querySelectorAll("input[type=checkbox]").forEach(c => c.checked = e.target.checked);
    updateDlCount();
  };
  $("dl-go").onclick = () => {
    const chosen = [...$("dl-list").querySelectorAll(".dl-item input:checked")].map(c => c.value);
    if (!chosen.length) { $("dl-count").textContent = "select at least one"; return; }
    const params = chosen.map(x => `exp=${encodeURIComponent(x)}`).join("&");
    window.location.href = `/api/download-images?${params}`;
    $("dl-modal").classList.add("hidden");
  };
}
