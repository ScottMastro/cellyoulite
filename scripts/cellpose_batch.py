"""Run Cellpose cyto3 across many wells/timepoints in parallel.

Loads Cellpose once per worker process, then iterates over its slice of
(well, t_idx) tasks. Detections are written to a per-frame JSON cache so
re-running is cheap.

Usage:
    python scripts/cellpose_batch.py                    # all wells, 8 workers
    python scripts/cellpose_batch.py --wells "DMSO r1" "012 r2"
    python scripts/cellpose_batch.py --workers 4
    python scripts/cellpose_batch.py --gpu              # single GPU worker
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from skimage.io import imread

from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import compute_alignment_cached, paste_onto_canvas


_CACHE_ROOT = Path.cwd() / ".cellpose_cache"


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


def _masks_to_circles(masks: np.ndarray) -> list[dict]:
    out = []
    for label in np.unique(masks):
        if label == 0:
            continue
        ys, xs = np.where(masks == label)
        if ys.size < 50:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        r = float(np.sqrt(ys.size / np.pi))
        if r < 18 or r > 200:
            continue
        out.append({"cx": cx, "cy": cy, "r": r, "area_px": int(ys.size)})
    return out


def _aligned_frame(well, t_idx, align):
    raw = imread(well.timepoints[t_idx].path)
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=-1)
    if raw.dtype != np.uint8:
        raw = np.clip(raw[..., :3], 0, 255).astype(np.uint8)
    return paste_onto_canvas(raw[..., :3], align.placements[t_idx],
                              align.canvas_shape, fill=0)


# Module-level state per worker — model is heavy, load it once and reuse.
_WORKER_MODEL = None


def _ensure_model(gpu: bool):
    """Load Cellpose-SAM (cpsam) in the worker."""
    global _WORKER_MODEL
    if _WORKER_MODEL is not None:
        return _WORKER_MODEL
    from cellpose import models
    _WORKER_MODEL = models.CellposeModel(gpu=gpu, pretrained_model="cpsam")
    return _WORKER_MODEL


def _cache_path(well_folder: str, label: str) -> Path:
    safe_well = well_folder.replace("/", "_")
    p = _CACHE_ROOT / safe_well
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{label}.json"


def _process_one(args: tuple) -> dict:
    """Worker: do one (well, t_idx). Returns a small result dict."""
    well_folder, t_idx, label, paths, placement, canvas_shape, gpu, force = args
    out_path = _cache_path(well_folder, label)
    if out_path.is_file() and not force:
        try:
            data = json.loads(out_path.read_text())
            return {"well": well_folder, "t_idx": t_idx, "label": label,
                    "n_circles": len(data.get("circles", [])),
                    "cached": True, "ms": 0}
        except (OSError, ValueError):
            pass  # corrupt cache — re-run

    model = _ensure_model(gpu)
    raw = imread(paths[t_idx])
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=-1)
    if raw.dtype != np.uint8:
        raw = np.clip(raw[..., :3], 0, 255).astype(np.uint8)
    frame = paste_onto_canvas(raw[..., :3], placement, canvas_shape, fill=0)
    img = _to_gray(frame).astype(np.float32)

    t0 = time.perf_counter()
    masks, _, _ = model.eval(img, diameter=None)
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    circles = _masks_to_circles(masks)
    out_path.write_text(json.dumps({
        "well": well_folder, "label": label, "t_idx": t_idx,
        "model": "cpsam", "circles": circles,
    }, indent=2))
    return {"well": well_folder, "t_idx": t_idx, "label": label,
            "n_circles": len(circles), "cached": False, "ms": elapsed_ms}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data")
    ap.add_argument("--wells", nargs="*", default=None,
                    help="folder names to process; default = all")
    ap.add_argument("--workers", type=int, default=4,
                    help="parallel worker processes; each holds ~1.4 GB Cellpose-SAM "
                         "(cpsam, ViT backbone) + working memory. Default 4. "
                         "Use 1 on tight RAM.")
    ap.add_argument("--max-tasks-per-child", type=int, default=20,
                    help="recycle each worker after this many frames to bound "
                         "memory growth from accumulated allocations.")
    ap.add_argument("--gpu", action="store_true",
                    help="use GPU; forces workers=1 (single process)")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if cache file exists")
    args = ap.parse_args()

    spec = discover_grid(args.folder)
    wells = spec.wells
    if args.wells:
        wanted = set(args.wells)
        wells = [w for w in wells if w.folder_name in wanted]
        missing = wanted - {w.folder_name for w in wells}
        if missing:
            print(f"WARN: not found: {sorted(missing)}")

    # Pre-warm the disk alignment cache serially so workers don't all stomp on it.
    print(f"pre-warming alignment for {len(wells)} well(s)...")
    from cellyoulite.pipeline.align import is_alignment_cached
    aligns: dict[str, object] = {}
    for i, w in enumerate(wells, 1):
        paths = [tp.path for tp in w.timepoints]
        was_cached = is_alignment_cached(paths)
        t0 = time.perf_counter()
        aligns[w.folder_name] = compute_alignment_cached(paths)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        tag = "cache" if was_cached else f"{elapsed_ms}ms"
        print(f"  align [{i:2d}/{len(wells):2d}] {w.folder_name:24s}  {tag}")

    # Build the task list.
    tasks = []
    for w in wells:
        align = aligns[w.folder_name]
        paths = [tp.path for tp in w.timepoints]
        for i, tp in enumerate(w.timepoints):
            tasks.append((w.folder_name, i, tp.label, paths,
                          align.placements[i], align.canvas_shape,
                          args.gpu, args.force))
    print(f"queued {len(tasks)} frame(s) across {len(wells)} well(s) — "
          f"workers={1 if args.gpu else args.workers}")

    workers = 1 if args.gpu else max(1, args.workers)
    est_gb = workers * 1.6  # rough: ~1.4 GB cpsam ViT + ~200 MB working set
    print(f"estimated peak RAM: ~{est_gb:.1f} GB ({workers} workers × ~0.9 GB)")
    t_start = time.perf_counter()
    done = 0
    if workers == 1:
        for task in tasks:
            r = _process_one(task)
            done += 1
            tag = "cache" if r["cached"] else f"{r['ms']}ms"
            print(f"[{done:3d}/{len(tasks):3d}] {r['well']:24s} "
                  f"{r['label']:10s}  n={r['n_circles']:3d}  {tag}")
    else:
        # `max_tasks_per_child` recycles workers periodically to release any
        # accumulated allocator fragmentation / leaked tensors. Requires Py3.11+.
        kw = {"max_workers": workers}
        try:
            with ProcessPoolExecutor(max_tasks_per_child=args.max_tasks_per_child, **kw) as pool:
                for r in pool.map(_process_one, tasks, chunksize=1):
                    done += 1
                    tag = "cache" if r["cached"] else f"{r['ms']}ms"
                    print(f"[{done:3d}/{len(tasks):3d}] {r['well']:24s} "
                          f"{r['label']:10s}  n={r['n_circles']:3d}  {tag}")
        except TypeError:
            # older Python: max_tasks_per_child not supported
            with ProcessPoolExecutor(**kw) as pool:
                for r in pool.map(_process_one, tasks, chunksize=1):
                    done += 1
                    tag = "cache" if r["cached"] else f"{r['ms']}ms"
                    print(f"[{done:3d}/{len(tasks):3d}] {r['well']:24s} "
                          f"{r['label']:10s}  n={r['n_circles']:3d}  {tag}")

    elapsed = time.perf_counter() - t_start
    print(f"\ndone in {elapsed:.1f}s — cache at {_CACHE_ROOT.resolve()}/")


if __name__ == "__main__":
    main()
