// Shared mutable app state. ES modules each have their own scope, so the
// former cross-function globals live here as one mutable object that every
// feature module imports and reads/writes (same pattern as before, no window.*).
export const state = {
  grid: null,                 // last /api/grid response
  well: null,                 // open well {mount_id, treatment, replicate, folder_name, timepoints, …}
  tIdx: 0,                    // current frame index in the open well
  tracks: null,               // tracks JSON for the open well, or null
  validation: {},             // track_id -> manual override bool
  cellposeStatus: new Map(),  // folder_name -> {n_total, n_done, labels_done:Set}
  playTimer: null,            // setTimeout handle for the play loop
  overlayVersion: 0,          // bumped to bust server-rendered overlay caches
  trackStatus: { n_done: 0, n_total: 0 },
  trackDoneByWell: new Map(), // folder_name -> bool
  humanValidatedByWell: new Map(), // folder_name -> bool
  wellAlignCache: null,       // last /api/well-align response for the open well
  pauseOnHoverWasPlaying: false,
  boxDrill: null,             // Growth distributions: null = all conditions; else a treatment name (zoomed)
  user: null,                 // selected profile name
};
