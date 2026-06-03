# CellxYou Lite — conventions for contributors (and Claude)

## Terminology: "track" → "organoid" (in-progress rename)

We are migrating the user-facing and code vocabulary from **track** to
**organoid**. A "track" is internally the time-linked trajectory of a single
organoid across frames — to users it is simply that organoid over time, so we
call it an **organoid** everywhere.

**Trajectory (do this as you touch code):**
- New code, UI text, comments, and identifiers should use **organoid**, not track.
- When you edit a file/function that still says "track", rename it toward
  "organoid" as part of that change (boy-scout rule) — provided it stays a safe,
  self-contained change.

**Done (low-hanging fruit):** user-facing text in the templates / JS strings.

**Not yet migrated (change opportunistically, each needs care):**
- The `/api/track*` HTTP routes (rename frontend + backend together; safe since
  the API is internal to the web UI, but it is churn — do it when touching them).
- The SQLite tables/columns `track`, `detection`, `track_filter`, `track_num`,
  `track_id` (needs a `user_version` migration in `db/migrate.py`).
- The analysis `tracks/*.json` format and the offline tools that read/write it
  (`scripts/cellpose_track.py`, `pipeline/track_stitch.py`, `analyze_cli.py`,
  the bundle/backfill paths). Changing the JSON keys breaks existing bundles and
  needs a re-backfill — coordinate before doing it.

When you migrate one of these, delete its bullet here.

## Other standing rules
- All code changes go through git (commit → push → pull); never edit code on the
  droplet. Deploy = pull + `systemctl restart cellyoulite`. See `DEPLOY.md`.
- Keep code clean and modular: no dead code; DB access stays in `db/repo.py`
  (no SQL in route handlers); split concerns into modules.
- SQLite schema changes go through `db/migrate.py` (`user_version`-gated steps);
  `db/schema.sql` is the v1 snapshot — add later columns via a migration step,
  don't rewrite the snapshot.
