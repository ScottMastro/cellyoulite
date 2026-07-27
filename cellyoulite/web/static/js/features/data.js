// "Data" card: add raw images (folder upload), add segmentation results
// (bundle import), and download a subset of experiments' images.
import { $, escapeHtml } from "../core/dom.js";
import { state } from "../core/state.js";
import { refreshAll } from "./polling.js";

function updateDlCount() {
  const n = $("dl-list").querySelectorAll(".dl-item input:checked").length;
  $("dl-count").textContent = `${n} selected`;
}

// Uploads and imports may need a batch that does not exist yet, so they ask
// for a name. Downloads never do — they always target an existing batch, and
// get a picker inside the dialog instead (see openDownloadModal).
function askBatch(prompt) {
  if (state.batch) return state.batch;
  if (state.batches.length === 1) return state.batches[0];
  const answer = window.prompt(prompt, "");
  const name = (answer || "").trim();
  if (name && (name.includes("/") || name.startsWith("."))) {
    $("data-status").textContent = "batch name can't contain / or start with .";
    return null;
  }
  return name || null;
}

// subset image-download picker (experiments grouped by treatment)
function openDownloadModal() {
  const batches = state.batches || [];
  if (!batches.length) { $("data-status").textContent = "no experiments to download yet"; return; }
  // Default to whatever the list is filtered to; on "All", the first batch.
  const batch = (state.batch && batches.includes(state.batch)) ? state.batch : batches[0];

  const sel = $("dl-batch");
  const multi = batches.length > 1;
  sel.hidden = !multi;
  $("dl-batch-label").hidden = !multi;
  if (multi) {
    sel.innerHTML = batches.map(b =>
      `<option value="${escapeHtml(b)}"${b === batch ? " selected" : ""}>${escapeHtml(b)}</option>`
    ).join("");
    // Switching batch re-lists that batch's experiments in place.
    sel.onchange = () => renderDownloadList(sel.value);
  }
  renderDownloadList(batch);
  $("dl-modal").classList.remove("hidden");
}

async function renderDownloadList(batch) {
  $("dl-modal").dataset.batch = batch;
  let wells = ((state.grid && state.grid.wells) || []).filter(w => w.batch === batch);
  if (!wells.length) {
    // The list is filtered to a different batch, so this one's wells aren't
    // loaded. Fetch just that batch rather than making the user go and switch.
    $("dl-list").innerHTML = `<div class="hint" style="padding:8px">loading…</div>`;
    try {
      const r = await fetch(`/api/grid?batch=${encodeURIComponent(batch)}`);
      if (r.ok) wells = (await r.json()).wells || [];
    } catch (e) { /* fall through to the empty message below */ }
  }
  if (!wells.length) {
    $("dl-list").innerHTML =
      `<div class="hint" style="padding:8px">no experiments in `
      + `${escapeHtml(batch)}</div>`;
    updateDlCount();
    return;
  }
  const byTreat = {};
  // Download works on source directories, and several wells can share one.
  for (const w of wells) {
    const dir = w.source_dir || w.folder_name;
    if (!(byTreat[w.treatment] ||= []).includes(dir)) byTreat[w.treatment].push(dir);
  }
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
}

export function initData() {
  $("add-images").onchange = async (e) => {
    const files = [...e.target.files];
    e.target.value = "";
    if (!files.length) return;
    const st = $("data-status");
    const imgs = files.filter(f => /\.(tif|tiff|png|jpe?g|bmp)$/i.test(f.name));
    if (!imgs.length) { st.textContent = "no image files in that folder"; return; }
    const batch = askBatch("Add these images to which batch? (name a new one to create it)");
    if (!batch) { st.textContent = "upload cancelled — no batch chosen"; return; }
    st.textContent = `uploading ${imgs.length} image(s) to ${batch}…`;
    const fd = new FormData();
    // 3rd arg keeps the relative path so the server can route by experiment name.
    for (const f of imgs) fd.append("files", f, f.webkitRelativePath || f.name);
    try {
      const r = await fetch(`/api/upload-images?batch=${encodeURIComponent(batch)}`,
                            { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) { st.textContent = `error: ${d.detail || r.statusText}`; return; }
      st.textContent = `added ${d.n_files} image(s) to ${d.n_experiments} experiment(s)`
        + ` in ${d.batch}` + (d.skipped ? ` · skipped ${d.skipped}` : "");
      await refreshAll();
    } catch (err) { st.textContent = `error: ${err}`; }
  };

  $("add-seg").onchange = async (e) => {
    const file = e.target.files[0];
    e.target.value = "";
    if (!file) return;
    const st = $("data-status");
    const batch = askBatch("Import these results into which batch? (name a new one to create it)");
    if (!batch) { st.textContent = "import cancelled — no batch chosen"; return; }
    st.textContent = `importing ${file.name} into ${batch}…`;
    try {
      const r = await fetch(`/api/bundle-import?batch=${encodeURIComponent(batch)}`, {
        method: "POST",
        headers: { "Content-Type": "application/gzip" },
        body: file,
      });
      const d = await r.json();
      if (!r.ok) { st.textContent = `error: ${d.detail || r.statusText}`; return; }
      st.textContent = d.db_tracks
        ? `imported ${d.n_files} file(s) into ${d.batch}`
          + ` · ${d.db_tracks} organoids, ${d.db_alignments} alignments`
        : `imported ${d.n_files} file(s) into ${d.batch} (no organoids found)`;
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
    const batch = $("dl-modal").dataset.batch || "";
    const params = [`batch=${encodeURIComponent(batch)}`]
      .concat(chosen.map(x => `exp=${encodeURIComponent(x)}`)).join("&");
    window.location.href = `/api/download-images?${params}`;
    $("dl-modal").classList.add("hidden");
  };
}
