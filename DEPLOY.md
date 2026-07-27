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

## One-off: converting to the batched layout (v3/v4)

Needed **once**, when moving from the pre-batch release. Wells are now scoped
to a batch, because two datasets can use the same folder name ("DMSO r1"
exists in both the May and July plates). The directories have to move to match,
or one batch's results will overwrite another's.

Do this with the service stopped, and take a database snapshot first — v3
rebuilds the `well` table, and every alignment, organoid and filter cascades
off `well.id`:

```bash
ssh root@142.93.158.91
cd /opt/cellyoulite
sudo systemctl stop cellyoulite
cp cellyoulite.db "cellyoulite.db.pre-v3-$(date +%Y%m%d-%H%M%S)"

sudo -u cellyoulite git pull
# Dry run first — prints what would move, changes nothing.
sudo -u cellyoulite python -m scripts.reorganize_batches --batch "CA1 T1"
sudo -u cellyoulite python -m scripts.reorganize_batches --batch "CA1 T1" --apply

sudo systemctl start cellyoulite          # migrations v3 + v4 run here
sudo -u cellyoulite python -m scripts.import_annotations
```

`reorganize_batches` moves `data/`, `tracks/`, `.cellpose_cache/` and
`annotations/` under a batch directory. It is a rename within the filesystem,
so the images are never copied, and alignment fingerprints hash only the
immediate parent directory name — the cached alignments stay valid.
`.align_cache/` is deliberately left flat: its entries are already keyed by
fingerprint, which differs between batches.

`import_annotations` loads the hand-drawn ground truth into the database
(migration v4). It skips frames already present, so it is safe to re-run.

Both steps are idempotent and report "nothing to move" / "skipped" on a repeat.

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
  python -m scripts.backfill_db --batch "CA1 T1"
# prints e.g. backfill complete: {'well': 21, 'alignment': 21, 'track': ...}
```

It reads the flat pre-batch layout (`tracks/<well>.json`, `data/<well>/`), so
`--batch` says which batch those wells belong to. Run it *before*
`reorganize_batches`, not after.

After the initial backfill, no run-once step is needed for subsequent deploys.

## How new analysis results reach the DB

A user uploads raw images → another user downloads them, runs
`cellyoulite-analyze` offline, then uploads the result bundle via **Add
segmentation data**. `POST /api/bundle-import` extracts the bundle and ingests
its alignment + tracks/detections straight into the DB (cellpose mask PNGs stay
on disk as binary). No backfill re-run is required for uploads.
