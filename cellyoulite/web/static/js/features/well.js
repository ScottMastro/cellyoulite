// Well detail: image viewer, filmstrip, frame scrubbing/playback, cellpose
// overlay + track colouring, per-track validation, and GIF export.
import { $, setStatus } from "../core/dom.js";
import { alignFlag, imgUrl, edgesUrl } from "../core/api.js";
import { state } from "../core/state.js";
import * as grid from "./grid.js";
import * as growth from "./growth.js";
import * as boxplot from "./boxplot.js";

// Longest-side cap for the playback viewer. The base image is displayed at
// the canvas height (CSS), so a shrunk JPEG looks identical but loads far
// faster; vector overlays stay crisp regardless.
const VIEWER_PX = 1100;

// Kick off background loads of every frame at the viewer size so the first
// playback is smooth (warms both the browser cache and the server disk cache).
function prewarmFrames(well, aligned) {
  // Prime the browser cache with both the base JPEG and the edges overlay PNG
  // for every frame, so playback swaps from cache instead of waiting on a
  // per-frame server render (the overlay lag). The server disk-caches the
  // edges PNG too, so this is cheap after the first pass.
  const hideInvalid = $("toggle-deactivated").checked ? 0 : 1;
  for (const tp of well.timepoints) {
    new Image().src = imgUrl(tp.key, VIEWER_PX, aligned);
    new Image().src = edgesUrl(tp.key, aligned, hideInvalid);
  }
}

export async function openWell(mountId, folderName) {
  stopPlay();
  document.querySelectorAll(".thumb.selected").forEach(e => e.classList.remove("selected"));
  document.querySelector(`.thumb[data-mount="${CSS.escape(mountId)}"][data-folder-name="${CSS.escape(folderName)}"]`)?.classList.add("selected");

  $("well-card").hidden = false;
  $("tracks-card").hidden = false;

  const qs = `mount_id=${encodeURIComponent(mountId)}&folder_name=${encodeURIComponent(folderName)}`;
  const r = await fetch(`/api/well?${qs}`);
  const data = await r.json();
  if (!r.ok) { setStatus("err", data.detail || r.statusText); return; }
  state.well = data;

  const scrub = $("scrub");
  scrub.min = 0; scrub.max = data.timepoints.length - 1; scrub.value = 0;
  scrub.oninput = () => showFrame(parseInt(scrub.value));

  try {
    const ar = await fetch(`/api/well-align?${qs}`);
    if (ar.ok) state.wellAlignCache = await ar.json();
  } catch (e) {}
  // Try to pull the tracks file for this well; absent if tracking hasn't
  // been run yet. Used to colour circles by track ID.
  state.tracks = null;
  state.validation = {};   // {track_id: bool} overrides
  $("growth-plot").innerHTML = "";
  $("track-stitch").hidden = true;
  $("all-tracks").hidden = true;
  try {
    const tr = await fetch(`/api/tracks?${qs}`);
    if (tr.ok) {
      const td = await tr.json();
      if (td.available) state.tracks = td;
    }
  } catch (e) {}
  try {
    const vr = await fetch(`/api/track-validation?${qs}`);
    if (vr.ok) {
      const vd = await vr.json();
      for (const [k, v] of Object.entries(vd.overrides || {})) {
        state.validation[parseInt(k)] = !!v;
      }
      state.humanValidatedByWell.set(state.well.folder_name, !!vd.human_validated);
    }
  } catch (e) {}
  renderAllTracks();
  refreshValidateButton();
  growth.fetchGrowthPlot(qs);
  boxplot.refreshBoxplot();
  renderFilmstrip();
  showFrame(0);
  prewarmFrames(data, alignFlag());
  syncChooserHeight();
  $("well-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

export function renderFilmstrip() {
  if (!state.well) return;
  const a = alignFlag();
  const fs = $("filmstrip");
  const st = state.cellposeStatus.get(state.well.folder_name);
  const labelsDone = st ? st.labels_done : new Set();
  fs.innerHTML = state.well.timepoints.map((tp, i) => {
    const done = labelsDone.has(tp.label);
    return `<div class="frame ${done ? 'cp-done' : 'cp-todo'}" data-i="${i}" title="${tp.label}${done ? ' · cellpose ✓' : ' · cellpose pending'}">
       <img loading="lazy" src="${imgUrl(tp.key, 192, a)}" alt="t${i}">
       <span class="frame-tag">${i}</span>
       <span class="frame-cp-dot"></span>
     </div>`;
  }).join("");
  fs.querySelectorAll(".frame").forEach(el => {
    el.onclick = () => showFrame(parseInt(el.dataset.i));
  });
}

async function showFrame(i) {
  if (!state.well) return;
  const tps = state.well.timepoints;
  if (i < 0 || i >= tps.length) return;
  state.tIdx = i;
  const tp = tps[i];
  $("scrub").value = i;
  $("scrub-label").textContent = `${i + 1} / ${tps.length}`;
  $("well-frame-label").textContent =
    `${state.well.treatment} · r${state.well.replicate} · t${i} · ${tp.label}`;

  document.querySelectorAll("#filmstrip .frame.active").forEach(e => e.classList.remove("active"));
  const frameEl = document.querySelector(`#filmstrip .frame[data-i="${i}"]`);
  if (frameEl) {
    frameEl.classList.add("active");
    const fs = $("filmstrip");
    // Center the active frame in the reel. Use rect math (not offsetLeft, whose
    // offsetParent isn't the scroll container, so the target came out far too
    // large and the reel pinned right — leaving the highlight off the left).
    // Recenter instantly while playing; smooth can't keep up with the cadence.
    const target = fs.scrollLeft
      + (frameEl.getBoundingClientRect().left - fs.getBoundingClientRect().left)
      - (fs.clientWidth - frameEl.clientWidth) / 2;
    fs.scrollTo({ left: target, behavior: state.playTimer ? "auto" : "smooth" });
  }

  const a = alignFlag();
  const nextSrc = imgUrl(tp.key, VIEWER_PX, a);
  // When deactivated organoids are hidden, also drop their segmentation from
  // the server-rendered edges PNG (not just the SVG circle/label).
  const hideInvalid = $("toggle-deactivated").checked ? 0 : 1;
  const nextEdges = edgesUrl(tp.key, a, hideInvalid);
  // Preload the base JPEG and the edges overlay together and swap only once
  // BOTH are ready, so the overlay never lands a frame behind the image. Both
  // are browser- and server-disk-cached (prewarmed on open), so during playback
  // this is near-instant.
  const pre = new Image();
  const preE = new Image();
  let ready = 0;
  const swap = () => {
    if (++ready < 2) return;
    $("viewer-img").src = nextSrc;
    $("layer-edges-img").src = nextEdges;
  };
  pre.onload = pre.onerror = swap;
  preE.onload = preE.onerror = swap;
  pre.src = nextSrc;
  preE.src = nextEdges;

  // Cellpose circles for this frame (single source of truth for the
  // overlay). Empty if the frame isn't segmented yet.
  try {
    const cpKey = `key=${encodeURIComponent(tp.key)}&aligned=${a}`;
    const r2 = await fetch(`/api/cellpose-circles?${cpKey}`);
    const d2 = await r2.json();
    if (r2.ok && d2.cached) {
      $("viewer-svg").setAttribute("viewBox", `0 0 ${d2.width} ${d2.height}`);
      $("layer-cellpose").innerHTML = renderTrackedCircles(d2.circles, i, a);
    } else {
      $("layer-cellpose").innerHTML = "";
    }
    applyToggles();
  } catch (e) {
    $("layer-cellpose").innerHTML = "";
    setStatus("err", String(e.message || e));
  }
}

// ---------- track coloring ----------
// Build a per-frame map from (cx, cy, r) → {track_id, valid} by indexing
// the tracks JSON. We match by rounded center distance < 3px, which is
// tight enough to be reliable and tolerant of float rounding.
function detectionsForFrame(t_idx, alignedFlag) {
  if (!state.tracks) return [];
  const out = [];
  for (const tr of state.tracks.tracks) {
    for (const d of tr.detections) {
      if (d.t_idx !== t_idx) continue;
      out.push({ cx: d.cx, cy: d.cy, r: d.r,
                 track_id: tr.id, valid: tr.valid, starred: tr.starred,
                 anchor_cx: tr.anchor_cx, anchor_cy: tr.anchor_cy });
      break;
    }
  }
  return out;
}

function renderTrackedCircles(circles, t_idx, alignedFlag) {
  if (!state.tracks) {
    // Tracking hasn't run — render solid yellow.
    return circles.map(c =>
      `<circle cx="${c.cx}" cy="${c.cy}" r="${c.r}" class="circ-untracked" />`
    ).join("");
  }
  const trackDets = detectionsForFrame(t_idx, alignedFlag);
  // When the viewer is unaligned, /api/cellpose-circles already returns
  // shifted coords; but state.tracks lives in aligned coords. To match we
  // add the placement offset back when alignedFlag === 0.
  const placement = alignedFlag ? null : (state.wellAlignCache && state.wellAlignCache.placements
      ? state.wellAlignCache.placements[t_idx] : null);
  // Text label size scales with the circle so it stays readable on tiny
  // organoids without overpowering big ones. Capped at a sensible max.
  // A starred organoid gets a gold ★ prefix on its id.
  const labelFor = (cx, cy, r, txt, color, extra = "", starred = false) => {
    const fs = Math.min(40, Math.max(14, r * 0.8));
    const star = starred ? `<tspan fill="#ffd24a">★</tspan>` : "";
    return `<text x="${cx}" y="${cy}" text-anchor="middle" `
         + `dominant-baseline="central" `
         + `font-size="${fs.toFixed(1)}" `
         + `class="circ-label ${extra}" fill="${color}">${star}${txt}</text>`;
  };
  // The id label is pinned to the organoid's fixed anchor (its first-frame
  // centroid, from the DB) so it stays put instead of jumping with the moving
  // organoid. Anchors are in aligned coords; in the unaligned view we shift them
  // by this frame's placement to match the displayed (raw) image.
  const anchorPos = (td, fx, fy) => {
    if (td.anchor_cx == null || td.anchor_cy == null) return [fx, fy];
    return placement
      ? [td.anchor_cx - placement[1], td.anchor_cy - placement[0]]
      : [td.anchor_cx, td.anchor_cy];
  };
  const circHtml = [];
  const labelHtml = [];
  for (const c of circles) {
    const cxA = placement ? c.cx + placement[1] : c.cx;
    const cyA = placement ? c.cy + placement[0] : c.cy;
    let best = null;
    for (const td of trackDets) {
      const d = Math.hypot(td.cx - cxA, td.cy - cyA);
      if (d < 3 && (best == null || d < best[1])) best = [td, d];
    }
    if (!best) {
      circHtml.push(`<circle cx="${c.cx}" cy="${c.cy}" r="${c.r}" class="circ-untracked" />`);
      labelHtml.push(labelFor(c.cx, c.cy, c.r, "?", "#ffd24a"));
      continue;
    }
    const td = best[0];
    const [lx, ly] = anchorPos(td, c.cx, c.cy);
    // Acceptance follows the user's manual overrides too, so toggling a
    // validation switch flips the colour live without a refresh.
    const accepted = state.validation.hasOwnProperty(td.track_id)
      ? state.validation[td.track_id] : td.valid;
    if (!accepted) {
      circHtml.push(`<circle cx="${c.cx}" cy="${c.cy}" r="${c.r}" `
                  + `data-track-id="${td.track_id}" class="circ-invalid" />`);
      labelHtml.push(labelFor(lx, ly, c.r, td.track_id, "#bbbbbb", "circ-label-invalid", td.starred));
      continue;
    }
    const hue = (td.track_id * 137.508) % 360;
    const color = `hsl(${hue.toFixed(0)}, 85%, 60%)`;
    circHtml.push(`<circle cx="${c.cx}" cy="${c.cy}" r="${c.r}" `
                + `data-track-id="${td.track_id}" `
                + `stroke="${color}" fill="${color}" />`);
    labelHtml.push(labelFor(lx, ly, c.r, td.track_id, color, "", td.starred));
  }
  // Side-effect: also pop the labels into their dedicated layer so they
  // remain visible when the "circles" toggle is off.
  $("layer-cellpose-labels").innerHTML = labelHtml.join("");
  return circHtml.join("");
}

function isTrackAccepted(t) {
  // Manual override wins; otherwise fall back to the script's auto-validity.
  return state.validation.hasOwnProperty(t.id)
    ? state.validation[t.id] : !!t.valid;
}

function renderAllTracks() {
  const wrap = $("all-tracks");
  const listEl = $("all-tracks-list");
  if (!state.tracks || !state.tracks.tracks.length) {
    wrap.hidden = true; listEl.innerHTML = ""; return;
  }
  wrap.hidden = false;
  const annotated = $("atr-annotated").checked;
  const variant = annotated ? "seg" : "raw";
  // Sort: accepted first, then by length descending.
  const sorted = [...state.tracks.tracks].sort((a, b) => {
    const ax = isTrackAccepted(a) ? 0 : 1;
    const bx = isTrackAccepted(b) ? 0 : 1;
    if (ax !== bx) return ax - bx;
    return b.n_detections - a.n_detections;
  });
  const nAcc = sorted.filter(isTrackAccepted).length;
  $("all-tracks-count").textContent =
    `${nAcc} accepted / ${sorted.length - nAcc} rejected (of ${sorted.length})`;
  const qs = `mount_id=${encodeURIComponent(state.well.mount_id)}`
           + `&folder_name=${encodeURIComponent(state.well.folder_name)}`;
  listEl.innerHTML = sorted.map(t => {
    const hue = (t.id * 137.508) % 360;
    const accepted = isTrackAccepted(t);
    const stitchUrl = `/api/track-stitch?${qs}&track_id=${t.id}&variant=${variant}`;
    return `<div class="atr-row ${accepted ? "atr-good" : "atr-bad"}" data-track-id="${t.id}">
      <label class="atr-switch" title="${accepted ? "accepted" : "rejected"}">
        <input type="checkbox" ${accepted ? "checked" : ""}>
        <span class="atr-switch-knob" style="--hue: ${hue.toFixed(0)}"></span>
      </label>
      <div class="atr-info">
        <div class="atr-id-row">
          <span class="atr-id" style="background: hsl(${hue.toFixed(0)},85%,60%)">#${t.id}</span>
          <button class="atr-star${t.starred ? " on" : ""}" data-track-id="${t.id}" title="mark as exemplar">★</button>
        </div>
        <span class="atr-meta">n=${t.n_detections}/${state.tracks.n_frames} · t${t.first_t}–t${t.last_t}</span>
        <button class="atr-gif" data-track-id="${t.id}" title="download GIF">GIF ↓</button>
      </div>
      <img class="atr-img" src="${stitchUrl}" loading="lazy"
           alt="organoid ${t.id}" />
    </div>`;
  }).join("");

  // Wire row interactions.
  listEl.querySelectorAll(".atr-row").forEach(row => {
    const id = parseInt(row.dataset.trackId);
    const cb = row.querySelector("input[type=checkbox]");
    cb.onchange = async () => {
      state.validation[id] = cb.checked;
      row.classList.toggle("atr-good", cb.checked);
      row.classList.toggle("atr-bad", !cb.checked);
      // Force any server-rendered overlay PNGs to re-fetch with the new
      // validation state by bumping the URL version param.
      state.overlayVersion += 1;
      showFrame(state.tIdx);
      await saveValidation();
      growth.fetchGrowthPlot(qs);
      boxplot.refreshBoxplot();
    };
    row.querySelector(".atr-star").onclick = async (e) => {
      e.stopPropagation();
      const star = e.currentTarget;
      const next = !star.classList.contains("on");
      star.classList.toggle("on", next);
      const tk = state.tracks && state.tracks.tracks.find(x => x.id === id);
      if (tk) tk.starred = next;
      // Reflect the ★ in the playback overlay immediately (not just on the next
      // frame change).
      if (state.well) showFrame(state.tIdx);
      try {
        await fetch(`/api/track-star?${qs}&track_id=${id}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ starred: next, user: state.user }),
        });
      } catch (err) {
        star.classList.toggle("on", !next);   // revert on failure
        if (tk) tk.starred = !next;
        if (state.well) showFrame(state.tIdx);
      }
    };
    row.querySelector(".atr-img").onclick = () => showTrackStitch(id);
    row.querySelector(".atr-gif").onclick = async (e) => {
      e.stopPropagation();
      const btn = e.currentTarget;
      const orig = btn.textContent;
      btn.disabled = true; btn.textContent = "…";
      try {
        const variant = $("atr-annotated").checked ? "seg" : "raw";
        const r = await fetch(`/api/track-gif?${qs}&track_id=${id}&variant=${variant}`);
        if (!r.ok) throw new Error(r.statusText);
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${state.well.folder_name.replace(/\s+/g, "_")}_organoid_${id}_${variant}.gif`;
        a.click();
        setTimeout(() => URL.revokeObjectURL(url), 60000);
        btn.textContent = "✓";
        setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 800);
      } catch (err) {
        btn.textContent = "err";
        setTimeout(() => { btn.textContent = orig; btn.disabled = false; }, 1500);
      }
    };
  });
}

async function saveValidation() {
  if (!state.well) return;
  const qs = `mount_id=${encodeURIComponent(state.well.mount_id)}`
           + `&folder_name=${encodeURIComponent(state.well.folder_name)}`;
  try {
    await fetch(`/api/track-validation?${qs}`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({overrides: state.validation, user: state.user}),
    });
  } catch (e) {}
}

function refreshValidateButton() {
  if (!state.well) return;
  const validated = state.humanValidatedByWell.get(state.well.folder_name) === true;
  const btn = $("validate-well"), statusEl = $("validate-status");
  if (validated) {
    btn.textContent = "Un-validate";
    btn.classList.remove("primary"); btn.classList.add("ghost");
    statusEl.textContent = "✓ signed off";
  } else {
    btn.textContent = "Validate well";
    btn.classList.add("primary"); btn.classList.remove("ghost");
    statusEl.textContent = "review the tracks above, then validate";
  }
}

async function showTrackStitch(trackId) {
  if (!state.well) return;
  const qs = `mount_id=${encodeURIComponent(state.well.mount_id)}`
           + `&folder_name=${encodeURIComponent(state.well.folder_name)}`
           + `&track_id=${trackId}`;
  const wrap = $("track-stitch");
  wrap.hidden = false;
  $("track-stitch-title").textContent = `organoid ${trackId} · loading…`;
  $("track-stitch-img").removeAttribute("src");
  try {
    const r = await fetch(`/api/track-stitch?${qs}`);
    if (!r.ok) { $("track-stitch-title").textContent = `organoid ${trackId} · error`; return; }
    const blob = await r.blob();
    const url = URL.createObjectURL(blob);
    $("track-stitch-img").src = url;
    $("track-stitch-title").textContent = `organoid ${trackId}`;
  } catch (e) {
    $("track-stitch-title").textContent = `organoid ${trackId} · ${e.message || e}`;
  }
}

// Self-rescheduling play loop. Each step awaits showFrame so overlays
// (cellpose circles + edges + track colours) have time to land before the
// next frame, and the wait between frames is `speed_ms - render_ms` so a
// slow renderer just plays at its real speed instead of piling up.
async function _playStep() {
  if (!state.playTimer || !state.well) return;
  const next = (state.tIdx + 1) % state.well.timepoints.length;
  const t0 = performance.now();
  try { await showFrame(next); } catch (e) {}
  if (!state.playTimer) return;
  const wait = Math.max(0, parseInt($("speed").value) - (performance.now() - t0));
  state.playTimer = setTimeout(_playStep, wait);
}
function startPlay() {
  if (!state.well) return;
  $("play-btn").textContent = "❚❚";
  state.playTimer = setTimeout(_playStep, 0);
}
function stopPlay() {
  if (state.playTimer) { clearTimeout(state.playTimer); state.playTimer = null; }
  $("play-btn").textContent = "▶";
}

const applyToggles = () => {
  $("layer-edges-img").style.display = $("toggle-edges").checked ? "" : "none";
  $("layer-cellpose").style.display = $("toggle-cellpose").checked ? "" : "none";
  $("layer-cellpose-labels").style.display = $("toggle-ids").checked ? "" : "none";
  // Hide rejected/deactivated organoids (circles + labels) unless asked for.
  const stack = document.querySelector(".image-stack");
  if (stack) stack.classList.toggle("hide-deact", !$("toggle-deactivated").checked);
};

// Match the experiments list height to the playback viewer so the two columns
// line up. Cleared (back to the CSS fallback) when no well is open or while
// fullscreen, where the chooser isn't visible.
function syncChooserHeight() {
  const grid = $("thumb-grid");
  const viewer = document.querySelector(".viewer");
  const card = $("well-card");
  if (state.well && viewer && !card.hidden && !card.classList.contains("fs")) {
    const h = viewer.offsetHeight;
    grid.style.height = h + "px";
    grid.style.maxHeight = h + "px";
  } else {
    grid.style.height = "";
    grid.style.maxHeight = "";
  }
}

export function initWell() {
  // Click on a circle in the SVG → fetch a wide stitch PNG for that track.
  $("viewer-svg").addEventListener("click", async (e) => {
    const c = e.target.closest("circle[data-track-id]");
    if (!c || !state.well) return;
    const trackId = parseInt(c.getAttribute("data-track-id"));
    await showTrackStitch(trackId);
  });
  // Pause playback while the cursor is over the viewer so the user can
  // actually aim at a circle. Resume on leave if they were playing.
  $("viewer-svg").addEventListener("mouseenter", () => {
    if (state.playTimer) { state.pauseOnHoverWasPlaying = true; stopPlay(); }
  });
  $("viewer-svg").addEventListener("mouseleave", () => {
    if (state.pauseOnHoverWasPlaying) { state.pauseOnHoverWasPlaying = false; startPlay(); }
  });
  $("track-stitch-close").onclick = () => { $("track-stitch").hidden = true; };

  $("atr-annotated").onchange = () => renderAllTracks();

  $("validate-well").onclick = async () => {
    if (!state.well) return;
    const validated = state.humanValidatedByWell.get(state.well.folder_name) === true;
    const next = !validated;
    const qs = `mount_id=${encodeURIComponent(state.well.mount_id)}`
             + `&folder_name=${encodeURIComponent(state.well.folder_name)}`;
    try {
      const r = await fetch(`/api/track-validation?${qs}`, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({human_validated: next, user: state.user}),
      });
      if (!r.ok) { setStatus("err", `validate: ${r.statusText}`); return; }
      state.humanValidatedByWell.set(state.well.folder_name, next);
      refreshValidateButton();
      if (state.grid) grid.renderThumbGrid(state.grid);
    } catch (e) { setStatus("err", String(e)); }
  };

  $("export-gif").onclick = async () => {
    if (!state.well) return;
    const btn = $("export-gif"), statusEl = $("export-gif-status");
    btn.disabled = true;
    btn.textContent = "Rendering…";
    statusEl.textContent = "this may take 10–30s on large wells";
    const qs = `mount_id=${encodeURIComponent(state.well.mount_id)}`
             + `&folder_name=${encodeURIComponent(state.well.folder_name)}`;
    try {
      const r = await fetch(`/api/well-gif?${qs}`);
      if (!r.ok) {
        statusEl.textContent = `error: ${r.statusText}`;
        return;
      }
      const blob = await r.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${state.well.folder_name.replace(/\s+/g, "_")}_tracked.gif`;
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      statusEl.textContent = `downloaded ${Math.round(blob.size / 1024)} KB`;
    } catch (e) {
      statusEl.textContent = `error: ${e}`;
    } finally {
      btn.disabled = false;
      btn.textContent = "Export GIF (accepted organoids)";
    }
  };

  $("play-btn").onclick = () => { state.playTimer ? stopPlay() : startPlay(); };
  $("speed").onchange = () => { /* picked up on the next _playStep tick */ };

  // Fullscreen: blow the viewer pane up to fill the window. The image/overlays
  // scale via CSS (height:100% + the SVG viewBox), so no reload is needed.
  const setFullscreen = (on) => {
    const card = $("well-card");
    card.classList.toggle("fs", on);
    const btn = $("viewer-fs");
    btn.textContent = on ? "✕" : "⛶";
    btn.title = on ? "exit fullscreen (Esc)" : "fullscreen";
    syncChooserHeight();
  };
  $("viewer-fs").onclick = () => setFullscreen(!$("well-card").classList.contains("fs"));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && $("well-card").classList.contains("fs")) setFullscreen(false);
  });

  $("well-close").onclick = () => {
    stopPlay();
    setFullscreen(false);
    $("well-card").hidden = true;
    $("tracks-card").hidden = true;
    syncChooserHeight();
  };

  window.addEventListener("resize", syncChooserHeight);

  $("toggle-edges").onchange = applyToggles;
  $("toggle-cellpose").onchange = applyToggles;
  $("toggle-ids").onchange = applyToggles;
  // Re-render the frame so the edges PNG refetches with/without the deactivated
  // organoids' segmentation; applyToggles (inside showFrame) handles the circles.
  $("toggle-deactivated").onchange = () => { if (state.well) showFrame(state.tIdx); };
  $("toggle-align").onchange = async () => {
    if (!state.well) return;
    if (alignFlag()) {
      const qs = `mount_id=${encodeURIComponent(state.well.mount_id)}&folder_name=${encodeURIComponent(state.well.folder_name)}`;
      try { await fetch(`/api/well-align?${qs}`); } catch (e) {}
    }
    renderFilmstrip();
    showFrame(state.tIdx);
  };
}
