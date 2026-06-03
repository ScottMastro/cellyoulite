# Deploying updates

All code changes go through git — never edit code on the droplet. Deploy is a
pull + restart. Data (the `data/` images and the SQLite DB) lives only on the
droplet and is never committed.

## Routine deploy (code only)

```bash
ssh root@142.93.158.91
cd /opt/cellyoulite
sudo -u cellyoulite git pull
sudo systemctl restart cellyoulite
```

On startup the app auto-runs SQLite migrations (`PRAGMA user_version`), so a
schema bump needs no manual step — just the restart.

## First-time / one-shot database backfill

The DB (`cellyoulite.db`) is the source of truth for alignment, tracks,
detections, validation/filters (with provenance), track stars, and users.
The legacy JSON files (`.align_cache/`, `tracks/*.json`,
`tracks/*__validation.json`, `users.json`) are **import sources only** — the
server no longer reads or writes them at request time.

Run the backfill **once** on a host that already has the legacy JSON on disk,
to populate the DB from it. It is idempotent (re-runnable): reproducible caches
are upserted; authored curation (filters, sign-off, users) is inserted only if
absent, so it never clobbers later decisions.

```bash
cd /opt/cellyoulite
sudo -u cellyoulite env CELLYOULITE_DATA=/opt/cellyoulite/data \
  python -m scripts.backfill_db
# prints e.g. backfill complete: {'well': 21, 'alignment': 21, 'track': ...}
```

After the initial backfill, no run-once step is needed for subsequent deploys.

## How new analysis results reach the DB

A user uploads raw images → another user downloads them, runs
`cellyoulite-analyze` offline, then uploads the result bundle via **Add
segmentation data**. `POST /api/bundle-import` extracts the bundle and ingests
its alignment + tracks/detections straight into the DB (cellpose mask PNGs stay
on disk as binary). No backfill re-run is required for uploads.
