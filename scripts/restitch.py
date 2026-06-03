"""Regenerate the cached per-organoid stitch PNGs (raw/seg/diff/shape/both)
for every well, from the EXISTING tracks (DB) + cached cellpose masks.

This does NOT re-run segmentation or tracking, so organoid ids — and the
curation keyed to them (validation, stars) — are preserved. Use it to warm /
refresh the stitch cache after the rendering code changes (e.g. the 4-row
layout), so the click-to-inspect view is served from disk instead of rendered
on demand.

Run from the project root (where data/, tracks/, .align_cache/,
.cellpose_cache/ live):

    python -m scripts.restitch
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
from skimage.io import imread

from cellyoulite.db import repo
from cellyoulite.db.migrate import migrate
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import compute_alignment_cached, paste_onto_canvas
from cellyoulite.pipeline.track_stitch import render_track_strips

_VARIANTS = ("raw", "seg", "diff", "shape", "both")


def _data_dir() -> Path:
    env = os.environ.get("CELLYOULITE_DATA")
    if env and Path(env).is_dir():
        return Path(env)
    return Path("data")


def main() -> None:
    migrate()
    ddir = _data_dir()
    if not ddir.is_dir():
        print(f"no data dir: {ddir}")
        return
    spec = discover_grid(ddir)
    tracks_root = Path("tracks")
    cp_root = Path(".cellpose_cache")
    n_wells = n_tracks = 0
    for w in spec.wells:
        data = repo.get_tracks(w.folder_name)
        if not data or not data.get("tracks"):
            print(f"  {w.folder_name:24s} no tracks — skip")
            continue
        safe = w.folder_name.replace("/", "_")
        paths = [tp.path for tp in w.timepoints]
        align = compute_alignment_cached(paths)
        idx_by_label = {tp.label: i for i, tp in enumerate(w.timepoints)}
        # Cache aligned frames + masks per well so each is read once, not once
        # per track that references it.
        frame_cache: dict[str, object] = {}
        mask_cache: dict[str, object] = {}

        def load_frame(label, _w=w, _a=align, _idx=idx_by_label, _c=frame_cache):
            if label not in _c:
                i = _idx.get(label)
                _c[label] = (paste_onto_canvas(imread(_w.timepoints[i].path),
                                               _a.placements[i], _a.canvas_shape)
                             if i is not None and _a.placements else None)
            return _c[label]

        def load_mask(label, _safe=safe, _c=mask_cache):
            if label not in _c:
                p = cp_root / _safe / f"{label}.mask.png"
                _c[label] = cv2.imread(str(p), cv2.IMREAD_UNCHANGED) if p.is_file() else None
            return _c[label]

        outdir = tracks_root / safe
        outdir.mkdir(parents=True, exist_ok=True)
        n = 0
        for t in data["tracks"]:
            strips = render_track_strips(
                track_id=t["id"], detections=t.get("detections", []),
                load_frame=load_frame, load_mask=load_mask)
            if not strips:
                continue
            for v in _VARIANTS:
                if v in strips:
                    (outdir / f"track_{t['id']}_{v}.png").write_bytes(strips[v])
            n += 1
        n_wells += 1
        n_tracks += n
        print(f"  {w.folder_name:24s} {n} organoids restitched")
    print(f"done: {n_tracks} organoids across {n_wells} wells")


if __name__ == "__main__":
    main()
