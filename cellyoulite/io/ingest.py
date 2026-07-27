"""Translate an extracted results bundle into database rows.

The bundle carries reproducible caches — alignment, organoids, detections —
as files. This walks them into the DB, which is the source of truth at request
time. Cellpose mask PNGs stay on disk; only their circle metadata is in tracks.

Lives here rather than in the server so it can run from a command line. The
web import route holds the whole upload in memory, which is fine for a browser
upload and not fine for a multi-hundred-megabyte bundle on a small host —
there, copy the file over, extract it with tar, and call this.
"""
from __future__ import annotations

import json
from pathlib import Path

from cellyoulite.db import repo
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import (
    _CACHE_VERSION,
    _fingerprint,
    compute_alignment_cached,
    is_alignment_cached,
)


def safe_name(name: str) -> str:
    """A path segment that cannot escape its parent."""
    return name.replace("/", "_")


def tracks_json(tracks_root: Path, batch: str, well_folder: str) -> Path:
    return tracks_root / safe_name(batch) / f"{safe_name(well_folder)}.json"


def ingest_batch(batch: str, data_dir: Path, tracks_root: Path) -> dict:
    """Load one batch's extracted results into the DB. Idempotent.

    Wells are re-discovered from data/<batch> rather than taken from the
    bundle's file names: a plate folder holds several positions, so its
    directory name is not a well name.
    """
    bdir = Path(data_dir) / safe_name(batch)
    if not bdir.is_dir():
        return {"db_alignments": 0, "db_tracks": 0, "wells": 0}
    n_align = n_tracks = n_wells = 0
    for w in discover_grid(bdir).wells:
        n_wells += 1
        paths = [tp.path for tp in w.timepoints]
        if paths and is_alignment_cached(paths):
            al = compute_alignment_cached(paths)
            repo.set_alignment(
                batch, w.folder_name, _fingerprint(paths), _CACHE_VERSION,
                al.canvas_shape[0], al.canvas_shape[1],
                [list(o) for o in al.offsets], [list(p) for p in al.placements],
                treatment=w.treatment, replicate=w.replicate)
            n_align += 1
        tpath = tracks_json(Path(tracks_root), batch, w.folder_name)
        if tpath.is_file():
            try:
                tdata = json.loads(tpath.read_text())
            except (OSError, ValueError):
                tdata = None
            if tdata:
                src = _fingerprint(paths) if paths else None
                n_tracks += repo.set_tracks(batch, w.folder_name, tdata, src,
                                            treatment=w.treatment,
                                            replicate=w.replicate)
    return {"db_alignments": n_align, "db_tracks": n_tracks, "wells": n_wells}
