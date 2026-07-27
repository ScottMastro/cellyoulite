-- CellxYou Lite — schema v1. Applied when PRAGMA user_version < 1.
-- This is the v1 snapshot only; later columns and tables (batch scoping,
-- annotations) are added by the steps in migrate.py, which is where the
-- current shape actually lives. IF NOT EXISTS keeps it safe to (re)apply
-- on a partial DB.

-- Wells are keyed by folder_name today (mount_id is ignored in current paths).
CREATE TABLE IF NOT EXISTS well (
  id          INTEGER PRIMARY KEY,
  folder_name TEXT NOT NULL UNIQUE,   -- e.g. "DMSO r1"
  treatment   TEXT,
  replicate   INTEGER,
  first_seen  TEXT NOT NULL           -- ISO timestamp
);

-- Reproducible cache. Keyed by fingerprint so a stale row triggers recompute.
CREATE TABLE IF NOT EXISTS alignment (
  well_id       INTEGER PRIMARY KEY REFERENCES well(id) ON DELETE CASCADE,
  fingerprint   TEXT NOT NULL,        -- SHA1 of (rel-path,mtime,size)+version
  cache_version INTEGER NOT NULL,     -- mirrors _CACHE_VERSION in align.py
  canvas_h      INTEGER NOT NULL,
  canvas_w      INTEGER NOT NULL,
  offsets       TEXT NOT NULL,        -- JSON [[dy,dx],...]
  placements    TEXT NOT NULL,        -- JSON [[y0,x0],...]
  created_at    TEXT NOT NULL
);

-- Reproducible cache. One row per linked track within a well.
CREATE TABLE IF NOT EXISTS track (
  id            INTEGER PRIMARY KEY,
  well_id       INTEGER NOT NULL REFERENCES well(id) ON DELETE CASCADE,
  track_num     INTEGER NOT NULL,     -- the per-well integer "id" from tracks JSON
  n_detections  INTEGER NOT NULL,
  first_t       INTEGER NOT NULL,
  last_t        INTEGER NOT NULL,
  auto_valid    INTEGER NOT NULL,     -- pipeline's `valid` flag (bool)
  edge_clipped  INTEGER NOT NULL,     -- pipeline's `edge_clipped` flag (bool)
  source_fingerprint TEXT,            -- ties tracks to the segmentation/align inputs
  starred       INTEGER NOT NULL DEFAULT 0,   -- track-level star
  starred_by    TEXT,
  starred_at    TEXT,
  UNIQUE (well_id, track_num)
);

-- Per-frame detections. Rows (not a blob) so growth curves can be queried.
CREATE TABLE IF NOT EXISTS detection (
  track_id  INTEGER NOT NULL REFERENCES track(id) ON DELETE CASCADE,
  t_idx     INTEGER NOT NULL,
  label     TEXT NOT NULL,            -- "00d04h15m"
  cx        REAL NOT NULL,
  cy        REAL NOT NULL,
  r         REAL NOT NULL,
  area_px   INTEGER,
  PRIMARY KEY (track_id, t_idx)
);

-- Current filter/validation decision per track (authored — never auto-dropped).
CREATE TABLE IF NOT EXISTS track_filter (
  track_id    INTEGER PRIMARY KEY REFERENCES track(id) ON DELETE CASCADE,
  accepted    INTEGER NOT NULL,       -- override bool (true=accepted)
  decided_by  TEXT NOT NULL,          -- username, or "auto"
  decided_at  TEXT NOT NULL,
  reason      TEXT                     -- "manual", "min_frac", "legacy-import", ...
);

-- Well-level human sign-off (replaces the human_validated JSON flag).
CREATE TABLE IF NOT EXISTS well_validation (
  well_id      INTEGER PRIMARY KEY REFERENCES well(id) ON DELETE CASCADE,
  validated    INTEGER NOT NULL,
  validated_by TEXT,
  validated_at TEXT
);

CREATE TABLE IF NOT EXISTS app_user (
  username   TEXT PRIMARY KEY,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_track_well ON track(well_id);
CREATE INDEX IF NOT EXISTS idx_detection_track ON detection(track_id);
