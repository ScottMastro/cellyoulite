"""Phase-correlation alignment for a single well's timeseries.

Strategy: register each frame against the *previous* one and accumulate the
offsets, rather than registering every frame against frame 0. Pairwise
registration is much more robust here because the organoids are growing —
direct-to-reference correlation gets pulled around by the expanding edges,
whereas frame-to-frame the organoid barely changes shape, so the correlation
peak is driven by the actual translation.

Before correlating we high-pass each frame (subtract a heavy Gaussian blur)
to flatten the slow illumination gradient and suppress the burned-in timer /
scale-bar in the corners, then take the central 60% to be extra-safe.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from skimage.filters import gaussian
from skimage.io import imread


@dataclass(frozen=True)
class WellAlignment:
    offsets: tuple[tuple[int, int], ...]      # cumulative (dy, dx) of each frame vs frame 0
    placements: tuple[tuple[int, int], ...]   # (y0, x0) top-left for each frame on the canvas
    canvas_shape: tuple[int, int]             # (H, W) of the padded canvas


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


_BOTTOM_CROP_FRAC = 0.18   # discard bottom strip (burned-in timer / scale bar)
_TOP_CROP_FRAC = 0.04      # tiny top trim just in case


def _prep(img: np.ndarray) -> np.ndarray:
    """High-pass + crop the annotation strip. The high-pass kills slow
    illumination so the correlation peak is driven by sharp sample features."""
    g = _to_gray(img)
    hp = g - gaussian(g, sigma=30, preserve_range=True)
    h = hp.shape[0]
    y0 = int(h * _TOP_CROP_FRAC)
    y1 = int(h * (1.0 - _BOTTOM_CROP_FRAC))
    return hp[y0:y1, :]


def _to_u8(img: np.ndarray) -> np.ndarray:
    """Normalize a float image to uint8 for cv2.findTransformECC."""
    lo, hi = float(img.min()), float(img.max())
    if hi - lo < 1e-6:
        return np.zeros_like(img, dtype=np.uint8)
    return np.clip((img - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


_FEAT_PARAMS = dict(maxCorners=300, qualityLevel=0.01, minDistance=20, blockSize=9)
_LK_PARAMS = dict(winSize=(31, 31), maxLevel=4,
                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _lk_shift(ref_u8: np.ndarray, mv_u8: np.ndarray) -> tuple[float, float]:
    """Translation between two frames via sparse Lucas-Kanade + RANSAC.

    Pick high-quality corner features in `ref`, track them into `mv` with a
    pyramidal LK tracker, then fit a translation via RANSAC. RANSAC's inlier
    consensus is much more robust than plain median when many tracks have
    drifted onto growing organoid edges or matched the wrong feature.
    """
    p0 = cv2.goodFeaturesToTrack(ref_u8, mask=None, **_FEAT_PARAMS)
    if p0 is None or len(p0) < 5:
        return 0.0, 0.0
    p1, st, _ = cv2.calcOpticalFlowPyrLK(ref_u8, mv_u8, p0, None, **_LK_PARAMS)
    if p1 is None:
        return 0.0, 0.0
    good = st.flatten() == 1
    if good.sum() < 5:
        return 0.0, 0.0
    src = p0[good].reshape(-1, 2)
    dst = p1[good].reshape(-1, 2)

    # RANSAC over a similarity model (translation + rotation + scale). We
    # only read the translation back out — but allowing rotation/scale lets
    # RANSAC find the dominant rigid-body consensus instead of getting
    # confused by features near growing organoids that translate AND scale.
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0, maxIters=2000,
    )
    if M is None or inliers is None or int(inliers.sum()) < 5:
        # Fallback: plain median over tracked points.
        flow = dst - src
        return float(np.median(flow[:, 1])), float(np.median(flow[:, 0]))

    # Recompute translation as the median of (dst - src) over inliers only,
    # ignoring the rotation/scale terms RANSAC fit. We want translation only.
    keep = inliers.flatten().astype(bool)
    flow = (dst - src)[keep]
    return float(np.median(flow[:, 1])), float(np.median(flow[:, 0]))


def compute_alignment(paths: list[Path]) -> WellAlignment:
    if not paths:
        return WellAlignment((), (), (0, 0))

    ref_img = imread(paths[0])
    ref_shape = _to_gray(ref_img).shape[:2]

    # Pairwise pyramidal ECC, then accumulate. Pairwise means consecutive
    # frames always look similar regardless of how much frame 19 has changed
    # from frame 0; pyramidal ECC handles large stage drifts without falling
    # into the aliased peaks that plague single-scale phase correlation.
    prev_u8 = _to_u8(_prep(ref_img))
    cum_dy, cum_dx = 0.0, 0.0
    offsets: list[tuple[int, int]] = [(0, 0)]
    for p in paths[1:]:
        curr_u8 = _to_u8(_prep(imread(p)))
        ddy, ddx = _lk_shift(prev_u8, curr_u8)
        cum_dy += ddy
        cum_dx += ddx
        offsets.append((int(round(cum_dy)), int(round(cum_dx))))
        prev_u8 = curr_u8

    h, w = ref_shape
    dys = [o[0] for o in offsets]
    dxs = [o[1] for o in offsets]
    dy_max, dy_min = max(dys), min(dys)
    dx_max, dx_min = max(dxs), min(dxs)
    canvas_h = h + (dy_max - dy_min)
    canvas_w = w + (dx_max - dx_min)
    placements = tuple(((dy_max - dy), (dx_max - dx)) for dy, dx in offsets)
    return WellAlignment(tuple(offsets), placements, (canvas_h, canvas_w))


def paste_onto_canvas(img: np.ndarray, placement: tuple[int, int],
                      canvas_shape: tuple[int, int], fill: int = 0) -> np.ndarray:
    y0, x0 = placement
    H, W = canvas_shape
    h, w = img.shape[:2]
    if img.ndim == 2:
        out = np.full((H, W), fill, dtype=img.dtype)
    else:
        out = np.full((H, W, img.shape[2]), fill, dtype=img.dtype)
    out[y0:y0 + h, x0:x0 + w] = img
    return out
