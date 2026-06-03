// Status polling: keep the grid/viewer in sync with results on disk.
// Analysis is offshored, so there are no running jobs to poll — uploads and
// validation are the only things that change state.
import { state } from "../core/state.js";
import * as grid from "./grid.js";
import * as well from "./well.js";

export async function refreshCellposeStatus() {
  try {
    const r = await fetch("/api/cellpose-status");
    if (!r.ok) return false;
    const data = await r.json();
    // Detect whether anything changed before re-rendering.
    let changed = false;
    const next = new Map();
    for (const w of (data.wells || [])) {
      const prev = state.cellposeStatus.get(w.folder_name);
      if (!prev || prev.n_done !== w.n_done || prev.n_total !== w.n_total) {
        changed = true;
      }
      next.set(w.folder_name, {
        n_total: w.n_total,
        n_done: w.n_done,
        labels_done: new Set(w.labels_done || []),
      });
    }
    if (state.cellposeStatus.size !== next.size) changed = true;
    state.cellposeStatus = next;
    return changed;
  } catch (e) { return false; }
}

export async function refreshTrackStatus() {
  try {
    const r = await fetch("/api/track-status");
    if (!r.ok) return false;
    const data = await r.json();
    state.trackStatus = { n_done: data.n_done || 0, n_total: data.n_total || 0 };
    const next = new Map();
    const nextVal = new Map();
    let changed = false;
    for (const w of (data.wells || [])) {
      const prev = state.trackDoneByWell.get(w.folder_name);
      const prevHv = state.humanValidatedByWell.get(w.folder_name);
      if (prev !== !!w.done) changed = true;
      if (prevHv !== !!w.human_validated) changed = true;
      next.set(w.folder_name, !!w.done);
      nextVal.set(w.folder_name, !!w.human_validated);
    }
    if (state.trackDoneByWell.size !== next.size) changed = true;
    state.trackDoneByWell = next;
    state.humanValidatedByWell = nextVal;
    return changed;
  } catch (e) { return false; }
}

export async function refreshAll() {
  // Pull cellpose AND track status BEFORE rendering the grid so the first
  // paint of the well thumbnails already knows which wells are tracked.
  await Promise.all([refreshCellposeStatus(), refreshTrackStatus()]);
  await grid.refreshGrid();
}

async function combinedPoll() {
  const trackChanged = await refreshTrackStatus();
  const cpChanged = await refreshCellposeStatus();
  if (cpChanged || trackChanged) {
    if (state.grid) grid.renderThumbGrid(state.grid);
    if (state.well) { well.renderFilmstrip(); well.updateWellCpStatus(); }
  }
  setTimeout(combinedPoll, 5000);
}

export function startPolling() {
  setTimeout(combinedPoll, 1500);
}
