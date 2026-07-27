"""Re-derive circle detections from cached Cellpose masks. No GPU, no model.

Segmentation writes two things per frame: the per-instance label mask
(`<label>.mask.png`) and the circles JSON derived from it. The mask is stored
*unfiltered* — every instance Cellpose found — and the size gates are applied
afterwards. So changing those gates only means re-reading the masks:

    segment (GPU, ~70 min)  ->  mask.png        <- unchanged
                                    |
                                    v
                            masks_to_circles     <- this script, seconds
                                    |
                                    v
                            <label>.json  ->  track (~20 s)

Sweep the floor to see what each threshold yields, changing nothing:

    python -m scripts.recircle --sweep 18 14 12 10 8

Then commit to one:

    python -m scripts.recircle --min-radius 12 --apply
    python -m scripts.cellpose_track            # re-link, ~20 s

Existing JSON is only rewritten with --apply, and the mask sidecars are never
touched, so this is always reversible by re-running at the old threshold.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

from cellyoulite.pipeline.segment import (
    MAX_RADIUS_PX,
    MIN_AREA_PX,
    MIN_RADIUS_PX,
    masks_to_circles,
)

_CACHE_ROOT = Path(".cellpose_cache")


def _mask_frames(root: Path, batch: str | None) -> list[Path]:
    """Every cached mask PNG, optionally within one batch directory."""
    base = root / batch if batch else root
    if not base.is_dir():
        raise SystemExit(f"no cache at {base}")
    return sorted(base.rglob("*.mask.png"))


def _radii(mask_path: Path) -> np.ndarray:
    m = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if m is None:
        return np.empty(0)
    lab, cnt = np.unique(m, return_counts=True)
    return np.sqrt(cnt[lab != 0] / np.pi)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", default=None,
                    help="only this batch directory under .cellpose_cache/")
    ap.add_argument("--min-radius", type=float, default=MIN_RADIUS_PX)
    ap.add_argument("--max-radius", type=float, default=MAX_RADIUS_PX)
    ap.add_argument("--min-area", type=int, default=MIN_AREA_PX)
    ap.add_argument("--sweep", type=float, nargs="*", default=None,
                    help="report the yield at each of these radii and exit")
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the circles JSON (default: report only)")
    ap.add_argument("--um-per-px", type=float, default=2.827,
                    help="only used to print radii in microns (default %(default)s)")
    args = ap.parse_args()

    frames = _mask_frames(_CACHE_ROOT, args.batch)
    if not frames:
        raise SystemExit("no cached masks found — run segmentation first")
    print(f"{len(frames)} cached frame(s)"
          + (f" in batch {args.batch!r}" if args.batch else ""))

    if args.sweep:
        radii = [_radii(f) for f in frames]
        total = sum(r.size for r in radii)
        print(f"\n{total} mask instances\n")
        print(f"{'min radius':>12} {'diameter':>12} {'kept':>10} {'share':>8} {'per frame':>10}")
        for thr in args.sweep:
            n = sum(int(((r >= thr) & (r <= args.max_radius)).sum()) for r in radii)
            print(f"{thr:>9.0f} px {2*thr*args.um_per_px:>9.0f} um "
                  f"{n:>10} {100*n/total:>7.1f}% {n/len(frames):>10.1f}")
        print("\nreport only — pick one and re-run with --min-radius N --apply")
        return

    n_before = n_after = 0
    changed = 0
    per_frame = []
    for f in frames:
        m = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if m is None:
            print(f"  unreadable, skipping: {f}")
            continue
        circles = masks_to_circles(m, min_radius_px=args.min_radius,
                                   max_radius_px=args.max_radius,
                                   min_area_px=args.min_area)
        jf = f.with_name(f.name.replace(".mask.png", ".json"))
        try:
            data = json.loads(jf.read_text()) if jf.is_file() else {}
        except (OSError, ValueError):
            data = {}
        n_before += len(data.get("circles", []))
        n_after += len(circles)
        per_frame.append(len(circles))
        if len(circles) != len(data.get("circles", [])):
            changed += 1
        if args.apply:
            data.update({"circles": circles, "has_mask": True,
                         "min_radius_px": args.min_radius,
                         "max_radius_px": args.max_radius,
                         "min_area_px": args.min_area})
            data.setdefault("model", "cpsam")
            jf.write_text(json.dumps(data, indent=2))

    verb = "rewrote" if args.apply else "would rewrite"
    print(f"\nmin_radius={args.min_radius} px "
          f"({2*args.min_radius*args.um_per_px:.0f} um diameter)")
    print(f"  detections: {n_before} -> {n_after}"
          f"  ({n_after/max(1, len(per_frame)):.1f}/frame, "
          f"median {statistics.median(per_frame) if per_frame else 0:.0f})")
    print(f"  {verb} {changed} frame(s)")
    if args.apply:
        print("\nnext: python -m scripts.cellpose_track   # re-link, ~20 s")
    else:
        print("\nreport only — pass --apply to write")


if __name__ == "__main__":
    main()
