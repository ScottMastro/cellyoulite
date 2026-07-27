"""Render per-organoid stitched strips (one panel per frame).

Variants per organoid:
  - "raw"  — cropped raw frames with the timepoint label on top.
  - "seg"  — same crop with the organoid's mask instance tinted in its hue.
  - "diff"  — growth/loss of each frame vs the first-frame baseline mask
              (centroid-aligned), tinted ON the raw organoid: growth blue,
              loss orange (so changes can be checked against the real image).
  - "shape" — the same growth/loss as a silhouette on black (retained grey),
              isolating the shape change.
  - "both"  — raw + seg + diff + shape stacked (the click-to-inspect view).

Used by both the tracking script (pre-generates and caches per-organoid PNGs
when a well is processed) and the FastAPI server (falls back to on-demand
rendering for any organoid that's not in the cache).
"""
from __future__ import annotations

import re
from typing import Callable

import cv2
import numpy as np
from skimage.segmentation import find_boundaries


_DAY_PREFIX = re.compile(r"^(\d+)d")


def _make_label_fmt(labels: list[str]):
    """Returns a function that strips the leading 'NNd' day prefix from each
    label when every label shares the same day (so '00d04h45m' becomes
    '04h45m', but '01d00h00m' / '02d00h00m' stay full)."""
    days = set()
    for lbl in labels:
        m = _DAY_PREFIX.match(lbl)
        if m:
            days.add(m.group(1))
        else:
            days.add(None)
    if len(days) == 1:
        return lambda l: _DAY_PREFIX.sub("", l)
    return lambda l: l


SEP_W = 2
LABEL_STRIP_H = 18
HEADER_H = 22
ROW_SEP = 3
FILL_ALPHA = 0.32   # interior tint strength on the segmented panel


def hue_for_track(track_id: int) -> tuple[int, int, int]:
    """Golden-angle HSL → RGB (note: returns rgb, not bgr)."""
    h = (track_id * 137.508) % 360
    s, l = 0.85, 0.60
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = l - c / 2
    if   h < 60:  r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return int((r + m) * 255), int((g + m) * 255), int((b + m) * 255)


def _crop_to(frame: np.ndarray, cx: int, cy: int, side: int) -> np.ndarray:
    H, W = frame.shape[:2]
    x0 = max(0, cx - side // 2); x1 = min(W, x0 + side)
    y0 = max(0, cy - side // 2); y1 = min(H, y0 + side)
    crop = frame[y0:y1, x0:x1]
    if crop.shape[0] != side or crop.shape[1] != side:
        if crop.ndim == 3:
            full = np.zeros((side, side, crop.shape[2]), dtype=crop.dtype)
        else:
            full = np.zeros((side, side), dtype=crop.dtype)
        full[:crop.shape[0], :crop.shape[1]] = crop
        crop = full
    return crop


def _text_metrics(side: int) -> tuple[float, int, int, int]:
    """Font scale + thickness + label/header strip heights, all relative to the
    crop size so the text reads the same at any organoid scale."""
    fs = max(0.32, min(0.85, side / 240.0))
    th = 1 if fs < 0.75 else 2
    strip_h = max(13, int(round(side * 0.18)))
    header_h = max(16, int(round(side * 0.20)))
    return fs, th, strip_h, header_h


def _label_strip(width: int, text: str, strip_h: int,
                 font_scale: float, thickness: int) -> np.ndarray:
    strip = np.zeros((strip_h, width, 3), dtype=np.uint8)
    y = strip_h - max(3, int(strip_h * 0.25))
    cv2.putText(strip, text, (4, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return strip


def make_label_fmt(labels: list[str]):
    """Public form of the day-prefix stripper (also used by the server)."""
    return _make_label_fmt(labels)


def _hcat(panels: list[np.ndarray]) -> np.ndarray:
    sep = np.full((panels[0].shape[0], SEP_W, 3), 36, dtype=np.uint8)
    pieces: list[np.ndarray] = []
    for i, p in enumerate(panels):
        if i > 0:
            pieces.append(sep)
        pieces.append(p)
    return np.concatenate(pieces, axis=1)


def _header(width: int, text: str, header_h: int,
            font_scale: float, thickness: int) -> np.ndarray:
    hdr = np.zeros((header_h, width, 3), dtype=np.uint8)
    y = header_h - max(4, int(header_h * 0.28))
    cv2.putText(hdr, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                (220, 220, 220), thickness, cv2.LINE_AA)
    return hdr


_GROW = (60, 130, 255)   # blue
_LOSS = (255, 140, 30)   # orange
_KEEP = (150, 150, 150)  # retained (silhouette panel only)


def _diff_masks(base_bool, cur_bool):
    """(growth, loss) boolean masks vs the baseline; either may be None."""
    if base_bool is not None and cur_bool is not None:
        return cur_bool & ~base_bool, base_bool & ~cur_bool
    if cur_bool is not None:
        return cur_bool, None
    if base_bool is not None:
        return None, base_bool
    return None, None


def _diff_overlay(raw_crop: np.ndarray, base_bool, cur_bool) -> np.ndarray:
    """Per-frame growth/loss vs the first-frame baseline mask (centroid-aligned,
    so it's a pure size/shape change) tinted ON the raw organoid crop — so the
    change can be checked against the actual image, not just the segmentation.
    New growth is tinted blue, loss orange; retained organoid left as raw."""
    out = raw_crop.astype(np.float32)
    growth, loss = _diff_masks(base_bool, cur_bool)
    alpha = 0.5
    if growth is not None and growth.any():
        out[growth] = out[growth] * (1 - alpha) + np.array(_GROW, np.float32) * alpha
    if loss is not None and loss.any():
        out[loss] = out[loss] * (1 - alpha) + np.array(_LOSS, np.float32) * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def _shape_panel(base_bool, cur_bool, side: int) -> np.ndarray:
    """The growth/loss silhouette on black: retained area grey, new growth blue,
    loss orange. Complements the on-image overlay by isolating the shape change."""
    panel = np.zeros((side, side, 3), dtype=np.uint8)
    growth, loss = _diff_masks(base_bool, cur_bool)
    if base_bool is not None and cur_bool is not None:
        panel[base_bool & cur_bool] = _KEEP
    if growth is not None:
        panel[growth] = _GROW
    if loss is not None:
        panel[loss] = _LOSS
    return panel


def _seg_overlay(raw_crop: np.ndarray, mask_crop: np.ndarray,
                  label_here: int, tr_rgb: np.ndarray) -> np.ndarray:
    out = raw_crop.astype(np.float32)
    inst = mask_crop == label_here
    if inst.any():
        out[inst] = out[inst] * (1 - FILL_ALPHA) + tr_rgb * FILL_ALPHA
        bnd = find_boundaries(inst, mode="thick")
        out[bnd] = tr_rgb
    return np.clip(out, 0, 255).astype(np.uint8)


def _png_bytes(rgb: np.ndarray) -> bytes:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def render_track_gif(
    *,
    track_id: int,
    detections: list[dict],
    load_frame: Callable[[str], np.ndarray | None],
    load_mask: Callable[[str], np.ndarray | None],
    well_name: str | None = None,
    variant: str = "seg",
    pad: float = 1.4,
    fps: int = 4,
) -> bytes:
    """Animated GIF of one track: each detection becomes a single frame
    cropped + (optionally) tinted to the track's hue.
    variant='raw' → bare crop. 'seg' → with mask overlay."""
    import io
    import imageio.v2 as imageio
    if not detections:
        return b""
    largest = max(detections, key=lambda d: d.get("r", 0))
    side = max(32, int(round(2 * largest["r"] * pad)))
    tr_rgb = np.array(hue_for_track(track_id), dtype=np.float32)

    fmt_label = _make_label_fmt([d["label"] for d in detections])
    frames: list[np.ndarray] = []
    for d in detections:
        frame = load_frame(d["label"])
        if frame is None:
            continue
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        cx, cy = int(round(d["cx"])), int(round(d["cy"]))
        crop = _crop_to(frame[..., :3], cx, cy, side)
        if variant == "seg":
            mask = load_mask(d["label"])
            if mask is not None and mask.shape[0] == frame.shape[0] and mask.shape[1] == frame.shape[1]:
                label_here = int(mask[max(0, min(mask.shape[0] - 1, int(round(d["cy"])))),
                                        max(0, min(mask.shape[1] - 1, int(round(d["cx"]))))])
                if label_here != 0:
                    mask_crop = _crop_to(mask, cx, cy, side)
                    crop = _seg_overlay(crop, mask_crop, label_here, tr_rgb)
        # Two-line label strip ABOVE the image. Line 1: well name (treatment
        # + replicate). Line 2: timepoint. Track id is internal; not shown.
        strip_h = LABEL_STRIP_H * 2 if well_name else LABEL_STRIP_H
        strip = np.zeros((strip_h, crop.shape[1], 3), dtype=np.uint8)
        tr_col = (int(tr_rgb[0]), int(tr_rgb[1]), int(tr_rgb[2]))
        disp = fmt_label(d["label"])
        if well_name:
            cv2.putText(strip, well_name, (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, tr_col, 1, cv2.LINE_AA)
            cv2.putText(strip, disp, (4, 13 + LABEL_STRIP_H),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (200, 200, 200), 1, cv2.LINE_AA)
        else:
            cv2.putText(strip, disp, (4, 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, tr_col, 1, cv2.LINE_AA)
        frames.append(np.vstack([strip, crop]))

    if not frames:
        return b""
    buf = io.BytesIO()
    imageio.mimsave(buf, frames, format="GIF",
                     duration=1.0 / max(1, fps), loop=0)
    return buf.getvalue()


def render_track_strips(
    *,
    track_id: int,
    detections: list[dict],
    load_frame: Callable[[str], np.ndarray | None],
    load_mask: Callable[[str], np.ndarray | None],
    pad: float = 1.4,
) -> dict[str, bytes]:
    """Render 'raw', 'seg', and 'diff' strips for a single organoid, plus a
    stacked 'both' (raw+seg+diff) for the click-to-inspect view.

    Each detection needs: label, cx, cy, r (and area_px when available). The
    header text scales with the crop size and reports the first→last size
    change. The 'diff' strip compares each frame's mask to the first-frame
    baseline (centroid-aligned), tinting growth blue / loss orange ON the raw
    organoid crop; 'shape' is the same growth/loss as a silhouette on black.
    Returns {'raw','seg','diff','shape','both': png_bytes, 'side','n': int}."""
    if not detections:
        return {}
    dets = sorted(detections, key=lambda d: d.get("t_idx", 0))
    largest = max(dets, key=lambda d: d.get("r", 0))
    side = max(32, int(round(2 * largest["r"] * pad)))
    tr_rgb = np.array(hue_for_track(track_id), dtype=np.float32)
    font_scale, thickness, strip_h, header_h = _text_metrics(side)

    fmt_label = _make_label_fmt([d["label"] for d in dets])
    raw_panels: list[np.ndarray] = []
    seg_panels: list[np.ndarray] = []
    diff_panels: list[np.ndarray] = []
    shape_panels: list[np.ndarray] = []
    base_bool = None   # first-frame mask instance (baseline for the diff)
    for d in dets:
        frame = load_frame(d["label"])
        if frame is None:
            continue
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        cx, cy = int(round(d["cx"])), int(round(d["cy"]))
        raw_crop = _crop_to(frame[..., :3], cx, cy, side)
        # Seg overlay + boolean instance: pick the mask label at the centroid.
        seg_crop = raw_crop.copy()
        inst_bool = None
        mask = load_mask(d["label"])
        if mask is not None and mask.shape[0] == frame.shape[0] and mask.shape[1] == frame.shape[1]:
            label_here = int(mask[max(0, min(mask.shape[0] - 1, int(round(d["cy"])))),
                                    max(0, min(mask.shape[1] - 1, int(round(d["cx"]))))])
            if label_here != 0:
                mask_crop = _crop_to(mask, cx, cy, side)
                inst_bool = mask_crop == label_here
                seg_crop = _seg_overlay(raw_crop, mask_crop, label_here, tr_rgb)
        if base_bool is None and inst_bool is not None:
            base_bool = inst_bool
        diff_crop = _diff_overlay(raw_crop, base_bool, inst_bool)
        shape_crop = _shape_panel(base_bool, inst_bool, side)
        strip = _label_strip(side, fmt_label(d["label"]), strip_h, font_scale, thickness)
        raw_panels.append(np.vstack([strip, raw_crop]))
        seg_panels.append(np.vstack([strip, seg_crop]))
        diff_panels.append(np.vstack([strip, diff_crop]))
        shape_panels.append(np.vstack([strip, shape_crop]))

    if not raw_panels:
        return {}

    raw_row = _hcat(raw_panels)
    seg_row = _hcat(seg_panels)
    diff_row = _hcat(diff_panels)
    shape_row = _hcat(shape_panels)

    # First → last size change (mask area, falling back to the circle's area).
    def _area(d):
        a = d.get("area_px")
        if a:
            return float(a)
        r = float(d.get("r", 0) or 0)
        return 3.14159 * r * r
    a0, a1 = _area(dets[0]), _area(dets[-1])
    pct = ((a1 - a0) / a0 * 100.0) if a0 else 0.0
    # ASCII only — cv2's Hershey font renders "·"/"Δ"/"±" as "?".
    size_txt = f"{'+' if pct >= 0 else '-'}{abs(pct):.0f}%"
    hdr_text = f"organoid {track_id} | n={len(raw_panels)} | size {size_txt}"
    hdr = _header(raw_row.shape[1], hdr_text, header_h, font_scale, thickness)

    def _sep(w):
        return np.full((ROW_SEP, w, 3), 36, dtype=np.uint8)
    s = raw_row.shape[1]
    both = np.vstack([hdr, raw_row, _sep(s), seg_row, _sep(s),
                       diff_row, _sep(s), shape_row])
    return {"raw": _png_bytes(np.vstack([hdr, raw_row])),
            "seg": _png_bytes(np.vstack([hdr, seg_row])),
            "diff": _png_bytes(np.vstack([hdr, diff_row])),
            "shape": _png_bytes(np.vstack([hdr, shape_row])),
            "both": _png_bytes(both),
            "side": side, "n": len(raw_panels)}


# ---------------------- list thumbnails ----------------------
# The organoid list shows one row per organoid at ~88 px tall. Serving the
# full-resolution strip there is ~0.5 MB apiece, so rows get a small JPEG
# cached on disk. Shared with scripts/restitch.py, which pre-generates them so
# the first real request is already warm.

THUMB_PX = 128


def thumb_tag(batch: str, well: str, track_id: int, variant: str,
              thumb_px: int, mtime_ns: int) -> str:
    """Cache key for one thumbnail. Includes the source strip's mtime, so a
    re-rendered strip invalidates its thumbnail automatically."""
    import hashlib
    return hashlib.sha1(
        f"{batch}|{well}|{track_id}|{variant}|{thumb_px}|{mtime_ns}".encode()
    ).hexdigest()


def write_thumb(strip_path, out_path, thumb_px: int = THUMB_PX) -> bool:
    """Downscale a stitch strip to `thumb_px` tall as JPEG. Returns False if
    the strip can't be read; the caller then falls back to the full strip.
    Written to a temp file and renamed, so a reader never sees a partial JPEG."""
    arr = cv2.imread(str(strip_path), cv2.IMREAD_COLOR)
    if arr is None:
        return False
    h, w = arr.shape[:2]
    if h > thumb_px:
        arr = cv2.resize(arr, (max(1, round(w * thumb_px / h)), thumb_px),
                         interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".jpg.tmp")
    tmp.write_bytes(buf.tobytes())
    tmp.replace(out_path)
    return True
