"""Load an already-extracted results bundle into the database. No web upload.

"Add segmentation data" in the browser holds the whole bundle in memory, which
is fine for a modest upload and not fine for hundreds of megabytes on a small
host. On the droplet, copy the bundle over and stream it in with tar instead:

    scp results.tar.gz root@host:/tmp/
    cd /opt/cellyoulite
    sudo -u cellyoulite tar -xzf /tmp/results.tar.gz \\
        --transform 's|^\\(\\.align_cache\\)/|\\1/|' -C .    # see below
    sudo -u cellyoulite python -m scripts.ingest_results --batch "CA1 T2"

Bundles are flat — tracks/<well>.json — because the offline tool knows nothing
about batches. Extract them under the batch directory so one batch's results
can never land on another's:

    tar -xzf results.tar.gz -C . --one-top-level=.unpack
    mv .unpack/tracks          "tracks/CA1 T2"
    mv .unpack/.cellpose_cache ".cellpose_cache/CA1 T2"
    cp -r .unpack/.align_cache/. .align_cache/     # flat, keyed by fingerprint

Idempotent: re-running upserts the reproducible caches and leaves authored
curation alone.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from cellyoulite.db.migrate import migrate
from cellyoulite.io.ingest import ingest_batch


def _data_dir() -> Path:
    env = os.environ.get("CELLYOULITE_DATA")
    if env and Path(env).is_dir():
        return Path(env)
    return Path("data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", required=True,
                    help='batch whose extracted results to load, e.g. "CA1 T2"')
    ap.add_argument("--tracks-root", default="tracks",
                    help="where tracks/<batch>/ lives (default %(default)s)")
    args = ap.parse_args()

    migrate()
    data_dir = _data_dir()
    bdir = data_dir / args.batch.replace("/", "_")
    if not bdir.is_dir():
        raise SystemExit(f"no images for that batch at {bdir} — the raw data "
                         "has to be in place before the results can be ingested")

    out = ingest_batch(args.batch, data_dir, Path(args.tracks_root))
    print(f"ingested {args.batch!r}: {out['wells']} wells, "
          f"{out['db_alignments']} alignments, {out['db_tracks']} organoids")


if __name__ == "__main__":
    main()
