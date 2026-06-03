"""Pre-warm the per-well alignment cache across the whole dataset.

For each well, computes (if needed) and persists the LK+RANSAC alignment
to .align_cache/<well>__<fingerprint>.json. Already-cached wells are
near-instant (just a file-exists check via fingerprint).

Prints per-well progress so a UI subscriber can render a progress bar.

Usage:
    python scripts/align_batch.py                  # every well
    python scripts/align_batch.py --wells "DMSO r1"
    python scripts/align_batch.py --force          # ignore cache
"""
from __future__ import annotations

import argparse
import time

from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import (
    compute_alignment, compute_alignment_cached, is_alignment_cached, _save_cached,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data")
    ap.add_argument("--wells", nargs="*", default=None,
                    help="folder names to process; default = all")
    ap.add_argument("--force", action="store_true",
                    help="recompute even when fingerprint cache exists")
    args = ap.parse_args()

    spec = discover_grid(args.folder)
    wells = spec.wells
    if args.wells:
        wanted = set(args.wells)
        wells = [w for w in wells if w.folder_name in wanted]
        missing = wanted - {w.folder_name for w in wells}
        if missing:
            print(f"WARN: not found: {sorted(missing)}")

    print(f"aligning {len(wells)} well(s)")
    t_start = time.perf_counter()
    n_recomputed = 0
    for i, w in enumerate(wells, 1):
        paths = [tp.path for tp in w.timepoints]
        cached = is_alignment_cached(paths) and not args.force
        t0 = time.perf_counter()
        if args.force:
            align = compute_alignment(paths)
            _save_cached(paths, align)
        else:
            compute_alignment_cached(paths)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if not cached:
            n_recomputed += 1
        tag = "cache" if cached else f"{elapsed_ms}ms"
        print(f"[{i:3d}/{len(wells):3d}] {w.folder_name:24s}  {tag}", flush=True)

    elapsed = time.perf_counter() - t_start
    print(f"\ndone in {elapsed:.1f}s — recomputed {n_recomputed}/{len(wells)}")


if __name__ == "__main__":
    main()
