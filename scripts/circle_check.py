"""Per-frame stills of the circle detector output.

Writes one annotated PNG per aligned timepoint at full resolution.

Layout:
    circle_check/<well>/t00.png  t01.png  ...

Usage:
    python scripts/circle_check.py                       # default well
    python scripts/circle_check.py --well "DMSO r1"
    python scripts/circle_check.py --all                 # every well
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from skimage.io import imread

from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import compute_alignment_cached as compute_alignment, paste_onto_canvas
from cellyoulite.pipeline.circle_methods import detect_circles


_COLOR = (255, 90, 90)


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _annotate(frame: np.ndarray, circles, label: str) -> np.ndarray:
    out = frame.copy()
    for cx, cy, r in circles:
        cv2.circle(out, (int(round(cx)), int(round(cy))), int(round(r)),
                   _COLOR, thickness=2, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (int(round(cx)), int(round(cy))), _COLOR,
                       cv2.MARKER_CROSS, markerSize=14, thickness=2)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), thickness=-1)
    cv2.putText(out, f"{label}  n={len(circles)}",
                (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, _COLOR, 2, cv2.LINE_AA)
    return out


def make_well_stills(well, out_dir: Path) -> None:
    paths = [tp.path for tp in well.timepoints]
    align = compute_alignment(paths)
    raw = [_to_rgb_u8(imread(p)) for p in paths]
    aligned = [paste_onto_canvas(r, align.placements[i], align.canvas_shape, fill=0)
               for i, r in enumerate(raw)]

    out_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(aligned):
        circles = detect_circles(frame)
        stamp = _annotate(frame, circles, f"t{i:02d}")
        imageio.imwrite(out_dir / f"t{i:02d}.png", stamp)
    print(f"  wrote {len(aligned)} stills to {out_dir.name}/")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data")
    ap.add_argument("--out", default="circle_check")
    ap.add_argument("--well", default="DMSO r1")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    spec = discover_grid(args.folder)
    wells = spec.wells if args.all else [w for w in spec.wells if w.folder_name == args.well]
    if not wells:
        raise SystemExit(f"no well {args.well!r}; available: "
                         + ", ".join(w.folder_name for w in spec.wells))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"writing stills for {len(wells)} well(s) to {out.resolve()}/")
    for w in wells:
        safe = w.folder_name.replace("/", "_").replace(" ", "_").replace("+", "p")
        make_well_stills(w, out / safe)


if __name__ == "__main__":
    main()
