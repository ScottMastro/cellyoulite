"""Visual sanity-check for the per-well alignment.

For each well in the data folder, write a side-by-side GIF: raw timeseries on
the left, registered timeseries on the right. The right pane should look
"locked" — only the burned-in timer/scale-bar should move, the organoids should
stay put. The left pane shows the raw drift for comparison.

Usage:
    python scripts/align_check.py                    # all wells, default folder
    python scripts/align_check.py --folder /path     # custom folder
    python scripts/align_check.py --well "DMSO r1"   # one well only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from skimage.io import imread
from skimage.transform import resize

from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import compute_alignment, paste_onto_canvas


def _to_rgb_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _resize_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img
    new_h = int(round(h * width / w))
    out = resize(img, (new_h, width), preserve_range=True, anti_aliasing=True)
    return out.astype(np.uint8)


def _label(frame: np.ndarray, text: str) -> np.ndarray:
    """Burn a tiny text caption into the top-left corner (no font dep: pixel block)."""
    # Cheap visual marker: just a dark strip + nothing else. Real text would need a
    # font; the filename in the GIF name already identifies the well.
    f = frame.copy()
    f[:18, :max(40, len(text) * 6), :] = 0
    return f


def _pad_to_same_height(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = max(a.shape[0], b.shape[0])
    def pad(x):
        if x.shape[0] == h:
            return x
        out = np.zeros((h, x.shape[1], 3), dtype=np.uint8)
        out[:x.shape[0]] = x
        return out
    return pad(a), pad(b)


def make_well_gif(well, out_path: Path, *, width: int = 380, fps: int = 6) -> None:
    paths = [tp.path for tp in well.timepoints]
    align = compute_alignment(paths)

    raw_frames = [_to_rgb_u8(imread(p)) for p in paths]
    aligned_frames = [
        paste_onto_canvas(rf, align.placements[i], align.canvas_shape, fill=0)
        for i, rf in enumerate(raw_frames)
    ]

    raw_small = [_resize_to_width(f, width) for f in raw_frames]
    al_small = [_resize_to_width(f, width) for f in aligned_frames]

    gap = np.zeros((1, 6, 3), dtype=np.uint8)  # placeholder column
    composed: list[np.ndarray] = []
    for i, (r, a) in enumerate(zip(raw_small, al_small)):
        r, a = _pad_to_same_height(r, a)
        sep = np.full((r.shape[0], 6, 3), 64, dtype=np.uint8)
        frame = np.concatenate([_label(r, "raw"), sep, _label(a, "aligned")], axis=1)
        composed.append(frame)

    imageio.mimsave(out_path, composed, format="GIF", duration=1.0 / fps, loop=0)
    # also report numerical drift
    dys = [o[0] for o in align.offsets]
    dxs = [o[1] for o in align.offsets]
    print(f"  {well.folder_name:30s}  dy ∈ [{min(dys):+d}, {max(dys):+d}]  "
          f"dx ∈ [{min(dxs):+d}, {max(dxs):+d}]  canvas {align.canvas_shape}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data", help="root images folder")
    ap.add_argument("--out", default="align_check", help="output folder for GIFs")
    ap.add_argument("--well", default=None, help='only one well, e.g. "DMSO r1"')
    ap.add_argument("--width", type=int, default=380)
    ap.add_argument("--fps", type=int, default=6)
    args = ap.parse_args()

    spec = discover_grid(args.folder)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    wells = spec.wells
    if args.well:
        wells = [w for w in wells if w.folder_name == args.well]
        if not wells:
            raise SystemExit(f"no well named {args.well!r} — try one of: "
                             + ", ".join(w.folder_name for w in spec.wells))

    print(f"writing {len(wells)} GIFs to {out.resolve()}/")
    for w in wells:
        safe = w.folder_name.replace("/", "_").replace(" ", "_").replace("+", "p")
        make_well_gif(w, out / f"{safe}.gif", width=args.width, fps=args.fps)
    print("done.")


if __name__ == "__main__":
    main()
