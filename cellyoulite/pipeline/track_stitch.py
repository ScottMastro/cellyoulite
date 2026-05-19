"""Render per-track stitched strips (one panel per frame).

Two variants per track:
  - "raw"  — cropped raw frames with the timepoint label on top.
  - "seg"  — same crop with the track's mask instance tinted in its hue.

Used by both the tracking script (pre-generates and caches per-track PNGs
when a well is processed) and the FastAPI server (falls back to on-demand
rendering for any track that's not in the cache).
"""
from __future__ import annotations

import re
from pathlib import Path
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


def _label_strip(width: int, text: str) -> np.ndarray:
    strip = np.zeros((LABEL_STRIP_H, width, 3), dtype=np.uint8)
    cv2.putText(strip, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1, cv2.LINE_AA)
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


def _header(width: int, text: str) -> np.ndarray:
    hdr = np.zeros((HEADER_H, width, 3), dtype=np.uint8)
    cv2.putText(hdr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (220, 220, 220), 1, cv2.LINE_AA)
    return hdr


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
    """Render both 'raw' and 'seg' strips for a single track.
    Each detection in `detections` must have: label, cx, cy, r.
    Returns {'raw': png_bytes, 'seg': png_bytes, 'side': int, 'n': int}."""
    if not detections:
        return {}
    largest = max(detections, key=lambda d: d.get("r", 0))
    side = max(32, int(round(2 * largest["r"] * pad)))
    tr_rgb_tuple = hue_for_track(track_id)
    tr_rgb = np.array(tr_rgb_tuple, dtype=np.float32)

    fmt_label = _make_label_fmt([d["label"] for d in detections])
    raw_panels: list[np.ndarray] = []
    seg_panels: list[np.ndarray] = []
    for d in detections:
        frame = load_frame(d["label"])
        if frame is None:
            continue
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        cx, cy = int(round(d["cx"])), int(round(d["cy"]))
        raw_crop = _crop_to(frame[..., :3], cx, cy, side)
        # Seg overlay: pick the mask label at this detection's centroid.
        seg_crop = raw_crop.copy()
        mask = load_mask(d["label"])
        if mask is not None and mask.shape[0] == frame.shape[0] and mask.shape[1] == frame.shape[1]:
            label_here = int(mask[max(0, min(mask.shape[0] - 1, int(round(d["cy"])))),
                                    max(0, min(mask.shape[1] - 1, int(round(d["cx"]))))])
            if label_here != 0:
                mask_crop = _crop_to(mask, cx, cy, side)
                seg_crop = _seg_overlay(raw_crop, mask_crop, label_here, tr_rgb)
        strip = _label_strip(side, fmt_label(d["label"]))
        raw_panels.append(np.vstack([strip, raw_crop]))
        seg_panels.append(np.vstack([strip, seg_crop]))

    if not raw_panels:
        return {}

    raw_row = _hcat(raw_panels)
    seg_row = _hcat(seg_panels)
    hdr_text = (f"track {track_id} · n={len(raw_panels)} · "
                f"crop={side}px (largest r={largest['r']:.0f})")
    hdr = _header(raw_row.shape[1], hdr_text)
    raw_full = np.vstack([hdr, raw_row])
    seg_full = np.vstack([hdr, seg_row])
    # Both-row variant used by the click-to-inspect endpoint.
    both = np.vstack([hdr,
                       raw_row,
                       np.full((ROW_SEP, raw_row.shape[1], 3), 36, dtype=np.uint8),
                       seg_row])
    return {"raw": _png_bytes(raw_full),
            "seg": _png_bytes(seg_full),
            "both": _png_bytes(both),
            "side": side, "n": len(raw_panels)}
