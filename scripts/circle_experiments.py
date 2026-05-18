"""Batch experiments to find a circle-detection variant that beats the baseline.

Runs several detector variants against every annotated frame, scores each
against the ground truth, and prints a comparison table.

Variants in play:
  baseline       : current defaults (r_min=30, nms=0.9, score=0.20)
  tuned-A        : r_min=20, nms_factor=0.5  (catch small + overlapping)
  tuned-B        : tuned-A + score_frac=0.10 (recall-leaning)
  tuned-C        : tuned-A + contrast_floor=10 (precision-leaning)
  persist-min    : run detector on per-pixel min flat-field across time
  persist-count  : run detector on "fraction-of-frames-dark" map
  persist+verify : centres from persist-min; each frame must agree
  track-filter   : generous per-frame detection, keep only tracks present in
                   >= K frames (proxy for "temporal persistence")
"""
from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter
from skimage.io import imread

from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import compute_alignment_cached, paste_onto_canvas
from cellyoulite.pipeline.circle_methods import detect_circles, DEFAULT_PARAMS


MATCH_CENTER_FRAC = 0.5
MATCH_R_RATIO = 1.6


def _to_gray(img):
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


def _flat(img, illum_sigma):
    g = _to_gray(img)
    bg = cv2.GaussianBlur(g, (0, 0), float(illum_sigma))
    return g - bg


def _aligned_stack(well):
    paths = [tp.path for tp in well.timepoints]
    align = compute_alignment_cached(paths)
    frames = []
    for i, p in enumerate(paths):
        raw = imread(p)
        if raw.ndim == 2:
            raw = np.stack([raw] * 3, axis=-1)
        if raw.dtype != np.uint8:
            raw = np.clip(raw[..., :3], 0, 255).astype(np.uint8)
        frames.append(paste_onto_canvas(raw[..., :3], align.placements[i],
                                        align.canvas_shape, fill=0))
    return frames


# ---------------------- detector variants ----------------------

def _detect_on_image(img, params):
    return detect_circles(img, params=params)


def _detect_on_2d_map(flat_map, params):
    """Run the LoG+peak pipeline directly on a precomputed scalar map.
    Bypasses the illumination-flatten step (the map IS flat already)."""
    p = dict(DEFAULT_PARAMS)
    p.update({k: v for k, v in (params or {}).items() if v is not None})

    radii = list(range(int(p["r_min"]), int(p["r_max"]) + 1, int(p["r_step"])))
    if not radii:
        radii = [int(p["r_min"])]
    best_resp = np.full(flat_map.shape, -np.inf, dtype=np.float32)
    best_r = np.zeros(flat_map.shape, dtype=np.int16)
    for r in radii:
        sigma = r / np.sqrt(2.0)
        log_resp = gaussian_laplace(flat_map.astype(np.float32), sigma=sigma) * (sigma ** 2)
        update = log_resp > best_resp
        best_resp = np.where(update, log_resp, best_resp)
        best_r = np.where(update, np.int16(r), best_r)

    nms_size = max(3, int(p["r_min"] * float(p["nms_factor"])) | 1)
    local_max = maximum_filter(best_resp, size=nms_size)
    peaks = (best_resp == local_max) & (best_resp > 0)
    if not peaks.any():
        return []
    score_thr = float(best_resp[peaks].max()) * float(p["score_frac"])
    peaks &= best_resp >= score_thr
    ys, xs = np.nonzero(peaks)
    scores = best_resp[ys, xs]
    radii_pick = best_r[ys, xs]
    order = np.argsort(-scores)
    h, w = flat_map.shape
    out = []
    for i in order:
        cx, cy, r = float(xs[i]), float(ys[i]), float(radii_pick[i])
        margin = r + 2
        if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
            continue
        out.append((cx, cy, r))
    return out


# Cellpose — heavy import, lazy-loaded once.
_cellpose_model = None


def _get_cellpose():
    """Load Cellpose-SAM (the v4 default). Cellpose 4 ships only the cpsam
    model — there is no cyto3 in v4."""
    global _cellpose_model
    if _cellpose_model is None:
        from cellpose import models
        _cellpose_model = models.CellposeModel(gpu=False, pretrained_model="cpsam")
    return _cellpose_model


def _masks_to_circles(masks: np.ndarray):
    """Convert a per-instance label mask to (cx, cy, r) circles. Centre =
    centroid, radius = area-equivalent radius (sqrt(area/π)). Drops tiny
    masks below 18 px radius."""
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
        out.append((cx, cy, r))
    return out


def variant_cellpose_sam(stack, frame_idx):
    """Cellpose 4 / Cellpose-SAM, the cpsam model. Modern API: pass the
    grayscale image and let it normalize internally."""
    model = _get_cellpose()
    img = _to_gray(stack[frame_idx]).astype(np.float32)
    masks, _, _ = model.eval(img, diameter=None)
    return _masks_to_circles(masks)


def variant_cellpose_clean(stack, frame_idx):
    cand = variant_cellpose_sam(stack, frame_idx)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=20.0)
    cand = _filter_circularity(cand, flat, min_circ=0.50)
    return cand


def variant_baseline(stack, frame_idx):
    return _detect_on_image(stack[frame_idx], None)


def variant_tuned_A(stack, frame_idx):
    return _detect_on_image(stack[frame_idx],
                            {"r_min": 20, "nms_factor": 0.5})


def variant_tuned_B(stack, frame_idx):
    return _detect_on_image(stack[frame_idx],
                            {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.10})


def variant_tuned_C(stack, frame_idx):
    return _detect_on_image(stack[frame_idx],
                            {"r_min": 20, "nms_factor": 0.5, "contrast_floor": 10.0})


# Persistence-map detectors. Cache the persistence map per well.
_persist_cache: dict[str, dict] = {}


def _compute_persistence(stack):
    """Build per-well persistence maps: minimum flat-field over time
    (organoids = most-negative pixels that PERSIST being dark), and a
    count-based map (number of frames a pixel is significantly dark)."""
    flats = np.stack([_flat(f, DEFAULT_PARAMS["illum_sigma"]) for f in stack],
                     axis=0)  # (T, H, W)
    # Minimum over time: organoid pixels are most-negative.
    min_map = flats.min(axis=0)
    # Negate so "darkness" is positive — matches what the LoG wants.
    persist_min = -min_map
    # Count map: fraction of frames where pixel < some negative threshold.
    thr = float(np.percentile(flats, 5))  # 5th-percentile across the cube
    persist_count = (flats < thr).sum(axis=0).astype(np.float32)
    return persist_min, persist_count


def _ensure_persistence(well_key, stack):
    if well_key in _persist_cache:
        return _persist_cache[well_key]
    persist_min, persist_count = _compute_persistence(stack)
    centres_min = _detect_on_2d_map(persist_min, {"r_min": 18, "nms_factor": 0.5,
                                                   "score_frac": 0.15})
    centres_count = _detect_on_2d_map(persist_count, {"r_min": 18, "nms_factor": 0.5,
                                                       "score_frac": 0.20})
    _persist_cache[well_key] = {
        "persist_min": persist_min,
        "persist_count": persist_count,
        "centres_min": centres_min,
        "centres_count": centres_count,
    }
    return _persist_cache[well_key]


def variant_persist_min(stack, frame_idx, *, well_key):
    return _ensure_persistence(well_key, stack)["centres_min"]


def variant_persist_count(stack, frame_idx, *, well_key):
    return _ensure_persistence(well_key, stack)["centres_count"]


def variant_persist_verify(stack, frame_idx, *, well_key):
    """Centres come from persist-min; each centre must ALSO be locally dark
    in the specific frame (radial darkness check). Rejects centres that have
    moved off-organoid or aren't real."""
    centres = _ensure_persistence(well_key, stack)["centres_min"]
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    out = []
    h, w = flat.shape
    for cx, cy, r in centres:
        # Average flat-field intensity inside the inner disc of radius 0.6r.
        yy, xx = np.ogrid[max(0,int(cy-r)):min(h,int(cy+r)+1),
                          max(0,int(cx-r)):min(w,int(cx+r)+1)]
        if yy.size == 0 or xx.size == 0:
            continue
        local_y = yy - cy; local_x = xx - cx
        mask = (local_y * local_y + local_x * local_x) <= (0.6 * r) ** 2
        patch = flat[yy[0,0]:yy[-1,0]+1, xx[0,0]:xx[0,-1]+1]
        if patch.size == 0:
            continue
        inner = patch[mask] if mask.shape == patch.shape else patch[mask[:patch.shape[0], :patch.shape[1]]]
        if inner.size == 0:
            continue
        if float(inner.mean()) < -3.0:  # darker than ~3 intensity units below local bg
            out.append((cx, cy, r))
    return out


def variant_track_filter(stack, frame_idx, *, well_key, persist_k=12):
    """Detect generously per frame; keep only candidates that appear in
    >= persist_k frames at roughly the same aligned-canvas location."""
    key = (well_key, "track_filter")
    if key not in _persist_cache:
        per_frame = []
        for f in stack:
            per_frame.append(_detect_on_image(
                f, {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.08,
                    "contrast_floor": 2.0}))
        # Cluster across frames by centre proximity.
        all_pts = []  # (frame_idx, cx, cy, r)
        for fi, dets in enumerate(per_frame):
            for cx, cy, r in dets:
                all_pts.append((fi, cx, cy, r))
        # Greedy clustering: assign each point to the closest existing cluster
        # whose representative centre is within cluster_r pixels.
        cluster_r = 20.0
        clusters: list[dict] = []
        for fi, cx, cy, r in all_pts:
            best = None
            for ci, c in enumerate(clusters):
                d = np.hypot(c["cx"] - cx, c["cy"] - cy)
                if d <= cluster_r and (best is None or d < best[1]):
                    best = (ci, d)
            if best is None:
                clusters.append({"cx": cx, "cy": cy, "r": r,
                                 "frames": {fi}, "rs": [r]})
            else:
                ci, _ = best
                c = clusters[ci]
                n = len(c["rs"])
                c["cx"] = (c["cx"] * n + cx) / (n + 1)
                c["cy"] = (c["cy"] * n + cy) / (n + 1)
                c["rs"].append(r)
                c["frames"].add(fi)
        kept = [(c["cx"], c["cy"], float(np.median(c["rs"])))
                for c in clusters if len(c["frames"]) >= persist_k]
        _persist_cache[key] = kept
    return _persist_cache[key]


# --- new variants ----------------------------------------------------------

def variant_baseline_illum_100(stack, frame_idx):
    return _detect_on_image(stack[frame_idx], {"illum_sigma": 100})


def variant_baseline_illum_500(stack, frame_idx):
    return _detect_on_image(stack[frame_idx], {"illum_sigma": 500})


def variant_baseline_no_contrast(stack, frame_idx):
    return _detect_on_image(stack[frame_idx], {"contrast_floor": 0.0})


def variant_baseline_loose_contrast(stack, frame_idx):
    return _detect_on_image(stack[frame_idx], {"contrast_floor": 3.0})


# Correct persistence: aggregator that highlights "always at least somewhat
# dark", not "dark at any moment". Use max-over-time of the flat-field
# (which is signed; negative = dark). A pixel that is dark even in its
# brightest moment is a persistent organoid.

def _persistence_max_map(stack):
    """For each pixel, the LEAST-dark moment over time. Persistent organoid →
    still negative. Transient debris → near zero. Returned in flat-image
    convention (negative = dark) so it can be passed straight to the LoG
    detector."""
    flats = np.stack([_flat(f, DEFAULT_PARAMS["illum_sigma"]) for f in stack],
                     axis=0)
    return flats.max(axis=0)


def _persistence_median_map(stack):
    flats = np.stack([_flat(f, DEFAULT_PARAMS["illum_sigma"]) for f in stack],
                     axis=0)
    return np.median(flats, axis=0)


def variant_persist_max(stack, frame_idx, *, well_key):
    key = (well_key, "persist_max")
    if key not in _persist_cache:
        m = _persistence_max_map(stack)
        _persist_cache[key] = _detect_on_2d_map(
            m, {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.15,
                "contrast_floor": 0.0})
    return _persist_cache[key]


def variant_persist_median(stack, frame_idx, *, well_key):
    key = (well_key, "persist_median")
    if key not in _persist_cache:
        m = _persistence_median_map(stack)
        _persist_cache[key] = _detect_on_2d_map(
            m, {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.15,
                "contrast_floor": 0.0})
    return _persist_cache[key]


def variant_tophat(stack, frame_idx):
    """Morphological black-tophat: enhances dark regions of scale ~= kernel.
    Tophat is HIGH at dark spots; we negate it to keep the dark-blob convention."""
    g = _to_gray(stack[frame_idx]).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101))
    th = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    return _detect_on_2d_map(-th.astype(np.float32),
                             {"r_min": 20, "nms_factor": 0.5,
                              "score_frac": 0.20, "contrast_floor": 0.0})


def variant_dog(stack, frame_idx):
    """Difference-of-Gaussians: bandpass for dark blobs in a specific scale
    range. (small - large) is naturally negative at dark blobs, matching the
    flat-image convention."""
    g = _to_gray(stack[frame_idx])
    small = cv2.GaussianBlur(g, (0, 0), 8)
    large = cv2.GaussianBlur(g, (0, 0), 40)
    dog = small - large  # negative at dark blobs
    return _detect_on_2d_map(dog, {"r_min": 20, "nms_factor": 0.5,
                                    "score_frac": 0.15, "contrast_floor": 0.0})


def variant_local_min(stack, frame_idx):
    """Blur the flat-field at organoid scale, pass it through the LoG
    detector. Already in dark-blob convention (negative at organoid centres)."""
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    blurred = cv2.GaussianBlur(flat, (0, 0), 25)
    return _detect_on_2d_map(blurred, {"r_min": 25, "nms_factor": 0.5,
                                        "score_frac": 0.10, "contrast_floor": 0.0})


def variant_dist_transform(stack, frame_idx):
    """Threshold the flat-field at the dark tail, run distance transform on
    the dark mask, peaks of the DT are centres of large compact dark regions
    and the DT value at the peak gives the inscribed radius."""
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    thr = np.percentile(flat, 8.0)  # dark 8% tail
    dark = (flat < thr).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    dist = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    # Local maxima in dist → centres; dist value at the max → inscribed radius.
    nms_size = 41
    local_max = maximum_filter(dist, size=nms_size)
    peaks = (dist == local_max) & (dist > 15.0)
    ys, xs = np.nonzero(peaks)
    out = []
    h, w = flat.shape
    for x, y in zip(xs, ys):
        r = float(dist[y, x])
        if r < 18 or r > 150:
            continue
        if x < r + 2 or y < r + 2 or x > w - r - 2 or y > h - r - 2:
            continue
        out.append((float(x), float(y), r))
    return out


def variant_baseline_temporal_gate(stack, frame_idx, *, well_key):
    """Baseline detections, but keep only those whose centre is also dark in
    the temporal MEDIAN — i.e., the spot was dark in most frames, not just
    this one. A pure FP filter on top of the baseline."""
    cand = _detect_on_image(stack[frame_idx], None)
    key = (well_key, "median_map")
    if key not in _persist_cache:
        _persist_cache[key] = _persistence_median_map(stack)
    med = _persist_cache[key]
    out = []
    for cx, cy, r in cand:
        ix, iy = int(cx), int(cy)
        if 0 <= ix < med.shape[1] and 0 <= iy < med.shape[0]:
            # average over a small disc of radius r*0.4
            rr = max(3, int(r * 0.4))
            patch = med[max(0, iy-rr):iy+rr+1, max(0, ix-rr):ix+rr+1]
            if patch.size and float(patch.mean()) < -2.0:  # persistently dark
                out.append((cx, cy, r))
    return out


def variant_baseline_intensity_gate(stack, frame_idx):
    """Baseline detections, but keep only those whose CENTRE region is
    actually dark in the flat-field of THIS frame. Sometimes peaks land on
    bright spots due to surround context; this nukes those."""
    cand = _detect_on_image(stack[frame_idx], None)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    out = []
    for cx, cy, r in cand:
        ix, iy = int(cx), int(cy)
        rr = max(3, int(r * 0.5))
        patch = flat[max(0, iy-rr):iy+rr+1, max(0, ix-rr):ix+rr+1]
        if patch.size and float(patch.mean()) < -3.0:
            out.append((cx, cy, r))
    return out


def variant_union_pm_baseline(stack, frame_idx, *, well_key):
    """Union of baseline single-frame detections and persist-median centres,
    deduplicated by NMS at 0.5 × min-radius."""
    a = _detect_on_image(stack[frame_idx], None)
    key = (well_key, "persist_median")
    if key not in _persist_cache:
        m = _persistence_median_map(stack)
        _persist_cache[key] = _detect_on_2d_map(
            m, {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.15,
                "contrast_floor": 0.0})
    b = _persist_cache[key]
    all_c = list(a) + list(b)
    # Greedy NMS on (cx, cy): drop circles within 25 px of an earlier one.
    kept = []
    for cx, cy, r in all_c:
        if any(np.hypot(cx - kx, cy - ky) < 25 for kx, ky, kr in kept):
            continue
        kept.append((cx, cy, r))
    return kept


# --- post-filter helpers for union-style candidates ------------------------

def _filter_contrast(cands, flat, ring_band=0.18, halo_factor=1.35, floor=6.0):
    h, w = flat.shape
    out = []
    for cx, cy, r in cands:
        margin = r * halo_factor + 2
        if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
            continue
        n = max(48, int(2 * np.pi * r))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        def ring_mean(radius):
            sx = np.clip(np.round(cx + radius * cos_t).astype(int), 0, w - 1)
            sy = np.clip(np.round(cy + radius * sin_t).astype(int), 0, h - 1)
            return float(flat[sy, sx].mean())
        perim = 0.5 * (ring_mean(r * (1 - ring_band)) + ring_mean(r * (1 + ring_band)))
        halo = ring_mean(r * halo_factor)
        if (halo - perim) >= floor:
            out.append((cx, cy, r))
    return out


def _filter_interior_variance(cands, gray, max_std=22.0):
    """Drop candidates whose interior is too noisy (debris clumps)."""
    h, w = gray.shape
    out = []
    for cx, cy, r in cands:
        rr = max(3, int(r * 0.6))
        y0, y1 = max(0, int(cy) - rr), min(h, int(cy) + rr + 1)
        x0, x1 = max(0, int(cx) - rr), min(w, int(cx) + rr + 1)
        patch = gray[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        if float(patch.std()) < max_std:
            out.append((cx, cy, r))
    return out


def _filter_edge_support(cands, gray, min_edge_frac=0.30):
    """Drop candidates whose perimeter doesn't show a real edge response.
    Computes Canny once and asks each circle: what fraction of my circumference
    falls within 2 px of an edge pixel?"""
    g8 = gray.astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (5, 5), 1.2), 40, 120)
    edges_d = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    h, w = edges.shape
    out = []
    for cx, cy, r in cands:
        n = max(48, int(2 * np.pi * r))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        sx = np.clip(np.round(cx + r * np.cos(theta)).astype(int), 0, w - 1)
        sy = np.clip(np.round(cy + r * np.sin(theta)).astype(int), 0, h - 1)
        near = (edges_d[sy, sx] <= 2.0).mean()
        if near >= min_edge_frac:
            out.append((cx, cy, r))
    return out


def variant_union_strict(stack, frame_idx, *, well_key):
    """Union pool, then strict 3-stage filter: contrast + interior variance
    + edge support. The expensive parts are computed once per frame."""
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    g = _to_gray(stack[frame_idx])
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_contrast(cand, flat, floor=6.0)
    cand = _filter_interior_variance(cand, g, max_std=22.0)
    cand = _filter_edge_support(cand, g, min_edge_frac=0.30)
    return cand


def variant_union_contrast_only(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    return _filter_contrast(cand, flat, floor=6.0)


def variant_union_contrast_strong(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    return _filter_contrast(cand, flat, floor=10.0)


def variant_union_edge_only(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    g = _to_gray(stack[frame_idx])
    return _filter_edge_support(cand, g, min_edge_frac=0.35)


def variant_union_variance_only(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    g = _to_gray(stack[frame_idx])
    return _filter_interior_variance(cand, g, max_std=22.0)


def _filter_min_radius(cands, min_r=25.0):
    return [(cx, cy, r) for cx, cy, r in cands if r >= min_r]


def _filter_circularity(cands, flat, min_circ=0.55, dark_z=-1.0):
    """Threshold a local patch around each candidate, find the dark blob
    that contains the centre, compute (4π·area / perimeter²). Drop irregular
    or split-up blobs."""
    h, w = flat.shape
    out = []
    for cx, cy, r in cands:
        crop_r = max(8, int(r * 2.5))
        y0, y1 = max(0, int(cy) - crop_r), min(h, int(cy) + crop_r + 1)
        x0, x1 = max(0, int(cx) - crop_r), min(w, int(cx) + crop_r + 1)
        patch = flat[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        # Threshold at darker-than-mean-by-stdev (robust to lighting variation).
        thr = float(patch.mean()) + dark_z * float(patch.std() + 1e-6)
        mask = (patch < thr).astype(np.uint8)
        if not mask.any():
            continue
        # Connected component containing the local centre.
        n_lbl, lbls, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        local_cy, local_cx = int(cy) - y0, int(cx) - x0
        local_cy = max(0, min(lbls.shape[0] - 1, local_cy))
        local_cx = max(0, min(lbls.shape[1] - 1, local_cx))
        label = lbls[local_cy, local_cx]
        if label == 0:
            # centre not in the dark region — search a small neighbourhood
            ny, nx = np.where(lbls > 0)
            if ny.size == 0:
                continue
            d = (ny - local_cy) ** 2 + (nx - local_cx) ** 2
            label = lbls[ny[d.argmin()], nx[d.argmin()]]
        comp = (lbls == label).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        # Largest contour (in case of holes / noise).
        ct = max(contours, key=cv2.contourArea)
        area = float(cv2.contourArea(ct))
        perim = float(cv2.arcLength(ct, closed=True))
        if perim < 1.0 or area < (np.pi * 0.3 * r * r):  # must fill at least 30% of expected
            continue
        circ = 4.0 * np.pi * area / (perim * perim)
        if circ >= min_circ:
            out.append((cx, cy, r))
    return out


def variant_tophat_clean(stack, frame_idx):
    cand = variant_tophat(stack, frame_idx)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_dog_clean(stack, frame_idx):
    cand = variant_dog(stack, frame_idx)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_union_clean(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_persist_med_clean(stack, frame_idx, *, well_key):
    cand = variant_persist_median(stack, frame_idx, well_key=well_key)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_hough_alt(stack, frame_idx):
    """OpenCV Hough circle transform (HOUGH_GRADIENT_ALT). Edge-direction
    voting in (cx, cy, r) space — built for circles with internal bright."""
    g = _to_gray(stack[frame_idx]).astype(np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    out = cv2.HoughCircles(
        g, cv2.HOUGH_GRADIENT_ALT,
        dp=1.5, minDist=40,
        param1=200, param2=0.55,
        minRadius=20, maxRadius=140,
    )
    if out is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in out[0]]


def variant_hough_alt_clean(stack, frame_idx):
    cand = variant_hough_alt(stack, frame_idx)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_contour_fit(stack, frame_idx):
    """Canny edges → findContours → fit ellipse → keep round, well-filled,
    appropriately-sized ellipses. Each kept ellipse becomes a circle at the
    mean of its semi-axes."""
    g = _to_gray(stack[frame_idx]).astype(np.uint8)
    g = cv2.GaussianBlur(g, (5, 5), 1.2)
    edges = cv2.Canny(g, 40, 120)
    # Close small gaps so a broken ring becomes one contour.
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for ct in contours:
        if len(ct) < 25:
            continue
        try:
            (ex, ey), (ma, mi), _ = cv2.fitEllipse(ct)
        except cv2.error:
            continue
        if ma < 2 or mi < 2:
            continue
        a, b = ma / 2.0, mi / 2.0  # semi-axes
        # axis ratio close to 1 = circle
        ratio = b / a if a > 0 else 0
        if ratio < 0.55:
            continue
        r = 0.5 * (a + b)
        if r < 20 or r > 150:
            continue
        # Filled-ness: how much of the contour's perimeter is actually drawn
        # by the underlying edges. Discourages random arc-like contours.
        perim_est = float(cv2.arcLength(ct, closed=True))
        if perim_est < 0.7 * (2 * np.pi * r):
            continue
        out.append((float(ex), float(ey), float(r)))
    # NMS within ~25 px / r_min.
    kept = []
    for cx, cy, r in sorted(out, key=lambda c: -c[2]):
        if any(np.hypot(cx - kx, cy - ky) < 25 for kx, ky, _ in kept):
            continue
        kept.append((cx, cy, r))
    return kept


def variant_contour_fit_clean(stack, frame_idx):
    cand = variant_contour_fit(stack, frame_idx)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_grad_blob(stack, frame_idx):
    """Gradient magnitude → multi-scale LoG-peak detection. Organoid
    boundaries form a high-gradient annulus; the LoG on |∇I| has a bright
    response in that annulus and a peak just inside (since the annulus
    looks like a thick bright ring → LoG positive near centre)."""
    g = _to_gray(stack[frame_idx])
    gx = cv2.Scharr(g, cv2.CV_32F, 1, 0)
    gy = cv2.Scharr(g, cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx * gx + gy * gy)
    # The detector expects "dark blob convention" (negative at centre), so negate.
    return _detect_on_2d_map(-mag, {"r_min": 25, "nms_factor": 0.5,
                                     "score_frac": 0.15, "contrast_floor": 0.0})


def variant_grad_blob_clean(stack, frame_idx):
    cand = variant_grad_blob(stack, frame_idx)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_baseline_clean(stack, frame_idx):
    cand = _detect_on_image(stack[frame_idx], None)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    cand = _filter_min_radius(cand, min_r=25.0)
    cand = _filter_circularity(cand, flat, min_circ=0.55)
    return cand


def variant_union_var_loose(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    g = _to_gray(stack[frame_idx])
    return _filter_interior_variance(cand, g, max_std=80.0)


def variant_union_ctr_var(stack, frame_idx, *, well_key):
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    g = _to_gray(stack[frame_idx])
    cand = _filter_contrast(cand, flat, floor=8.0)
    return _filter_interior_variance(cand, g, max_std=60.0)


def variant_union_top_k_scorer(stack, frame_idx, *, well_key):
    """For each candidate in the union pool, build a composite score from
    multiple cheap features, then keep the top K. K is chosen to be a few
    standard deviations above the expected per-frame GT count."""
    cand = variant_union_pm_baseline(stack, frame_idx, well_key=well_key)
    if not cand:
        return []
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    g = _to_gray(stack[frame_idx])
    g8 = g.astype(np.uint8)
    edges = cv2.Canny(cv2.GaussianBlur(g8, (5, 5), 1.2), 40, 120)
    edges_d = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    key = (well_key, "persist_median")
    if key not in _persist_cache:
        m = _persistence_median_map(stack)
        _persist_cache[key] = _detect_on_2d_map(
            m, {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.15,
                "contrast_floor": 0.0})
    med_map = _persistence_median_map(stack)

    h, w = flat.shape
    scored = []
    for cx, cy, r in cand:
        margin = r * 1.35 + 2
        if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
            continue
        # contrast
        n = max(48, int(2 * np.pi * r))
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        def rm(rad):
            sx = np.clip(np.round(cx + rad * cos_t).astype(int), 0, w - 1)
            sy = np.clip(np.round(cy + rad * sin_t).astype(int), 0, h - 1)
            return float(flat[sy, sx].mean())
        perim = 0.5 * (rm(r * 0.82) + rm(r * 1.18))
        halo = rm(r * 1.35)
        contrast = max(0.0, halo - perim)
        # edge support
        sx = np.clip(np.round(cx + r * cos_t).astype(int), 0, w - 1)
        sy = np.clip(np.round(cy + r * sin_t).astype(int), 0, h - 1)
        edge_frac = float((edges_d[sy, sx] <= 2.0).mean())
        # interior smoothness (1 / std)
        rr = max(3, int(r * 0.6))
        y0, y1 = max(0, int(cy) - rr), min(h, int(cy) + rr + 1)
        x0, x1 = max(0, int(cx) - rr), min(w, int(cx) + rr + 1)
        std = float(g[y0:y1, x0:x1].std())
        smoothness = 1.0 / (1.0 + std / 30.0)
        # temporal persistence: median flat at centre
        persist = -float(med_map[int(cy), int(cx)])  # positive when persistently dark
        # combine (rough hand-weights — equal-ish)
        score = (contrast / 10.0) + (edge_frac * 1.5) + smoothness + (persist / 10.0)
        scored.append((score, cx, cy, r))
    if not scored:
        return []
    scored.sort(reverse=True)
    # Keep top K (per frame); pick K based on observed annotation density.
    K = min(28, len(scored))
    return [(cx, cy, r) for _, cx, cy, r in scored[:K]]


def variant_baseline_plus_temporal_pool(stack, frame_idx, *, well_key):
    """Baseline + persist-median pool, then baseline's normal filter (contrast).
    Lighter-touch than the union-strict family."""
    a = _detect_on_image(stack[frame_idx], None)
    key = (well_key, "persist_median")
    if key not in _persist_cache:
        m = _persistence_median_map(stack)
        _persist_cache[key] = _detect_on_2d_map(
            m, {"r_min": 20, "nms_factor": 0.5, "score_frac": 0.15,
                "contrast_floor": 0.0})
    b = _persist_cache[key]
    # Add persist-median centres that have no baseline match within 25 px,
    # but only if they pass the baseline-style contrast check.
    flat = _flat(stack[frame_idx], DEFAULT_PARAMS["illum_sigma"])
    new_b = []
    for cx, cy, r in b:
        if any(np.hypot(cx - kx, cy - ky) < 25 for kx, ky, kr in a):
            continue
        new_b.append((cx, cy, r))
    new_b = _filter_contrast(new_b, flat, floor=6.0)
    return list(a) + new_b


VARIANTS = [
    ("baseline",         variant_baseline),
    ("baseline-il100",   variant_baseline_illum_100),
    ("baseline-il500",   variant_baseline_illum_500),
    ("baseline-noctr",   variant_baseline_no_contrast),
    ("baseline-ctr3",    variant_baseline_loose_contrast),
    ("tuned-A",          variant_tuned_A),
    ("tuned-C",          variant_tuned_C),
    ("persist-max",      variant_persist_max),
    ("persist-median",   variant_persist_median),
    ("tophat",           variant_tophat),
    ("dog",              variant_dog),
    ("local-min",        variant_local_min),
    ("dist-transform",   variant_dist_transform),
    ("union-pm+base",    variant_union_pm_baseline),
    ("base+temporal",    variant_baseline_temporal_gate),
    ("base+intensity",   variant_baseline_intensity_gate),
    ("union+contrast",   variant_union_contrast_only),
    ("union+ctr-strong", variant_union_contrast_strong),
    ("union+edge",       variant_union_edge_only),
    ("union+variance",   variant_union_variance_only),
    ("union+strict",     variant_union_strict),
    ("base+temp-pool",   variant_baseline_plus_temporal_pool),
    ("union+var-loose",  variant_union_var_loose),
    ("union+ctr+var",    variant_union_ctr_var),
    ("union+top-k",      variant_union_top_k_scorer),
    ("baseline-clean",   variant_baseline_clean),
    ("tophat-clean",     variant_tophat_clean),
    ("dog-clean",        variant_dog_clean),
    ("persist-med-clean", variant_persist_med_clean),
    ("union-clean",      variant_union_clean),
    ("hough-alt",        variant_hough_alt),
    ("hough-alt-clean",  variant_hough_alt_clean),
    ("contour-fit",      variant_contour_fit),
    ("contour-fit-clean", variant_contour_fit_clean),
    ("grad-blob",        variant_grad_blob),
    ("grad-blob-clean",  variant_grad_blob_clean),
    ("cellpose-sam",     variant_cellpose_sam),
    ("cellpose-clean",   variant_cellpose_clean),
]


# ---------------------- scoring ----------------------

def _match(gt, pred):
    used = set()
    matches = []
    gt_order = sorted(range(len(gt)), key=lambda i: -gt[i]["r"])
    for gi in gt_order:
        g = gt[gi]
        best = None
        for pi, (px, py, pr) in enumerate(pred):
            if pi in used:
                continue
            d = float(np.hypot(px - g["cx"], py - g["cy"]))
            if d > MATCH_CENTER_FRAC * g["r"]:
                continue
            ratio = pr / g["r"]
            if ratio > MATCH_R_RATIO or ratio < 1.0 / MATCH_R_RATIO:
                continue
            r_err = abs(pr - g["r"])
            if best is None or d < best[0]:
                best = (d, pi, r_err)
        if best is not None:
            _, pi, r_err = best
            used.add(pi)
            matches.append((gi, pi, best[0], r_err))
    return matches, used


def _to_rgb_u8(img):
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[..., :3]
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _render_gt(frame_rgb, gt):
    out = frame_rgb.copy()
    for g in gt:
        color = (90, 200, 255) if g.get("star") else (130, 255, 130)  # gold-ish / green
        cv2.circle(out, (int(g["cx"]), int(g["cy"])), int(g["r"]),
                   color, thickness=2, lineType=cv2.LINE_AA)
    return out


def _render_pred(frame_rgb, gt, pred, matches):
    """GT in faint white outline; matched predictions in green; FP in red.
    Missed GT is highlighted in magenta."""
    out = frame_rgb.copy()
    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}
    # GT outlines (thin, white, with magenta for misses)
    for i, g in enumerate(gt):
        color = (255, 255, 255) if i in matched_gt else (255, 90, 255)
        cv2.circle(out, (int(g["cx"]), int(g["cy"])), int(g["r"]),
                   color, thickness=1, lineType=cv2.LINE_AA)
    # Predictions
    for i, (cx, cy, r) in enumerate(pred):
        color = (90, 220, 90) if i in matched_pred else (90, 90, 255)
        cv2.circle(out, (int(cx), int(cy)), int(r),
                   color, thickness=2, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (int(cx), int(cy)), color,
                       cv2.MARKER_CROSS, markerSize=10, thickness=2)
    return out


def _save_jpg(path: Path, bgr_or_rgb: np.ndarray, *, max_w: int = 900):
    """Write a JPEG; downscale to max_w if larger (keeps the report light)."""
    img = bgr_or_rgb
    if img.shape[1] > max_w:
        scale = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * scale)),
                         interpolation=cv2.INTER_AREA)
    # _render_* returns RGB; convert before encoding
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 78])


REPORT_VARIANTS = [
    "baseline",
    "baseline-clean",
    "tophat-clean",
    "dog-clean",
    "hough-alt-clean",
    "contour-fit-clean",
    "grad-blob-clean",
    "union-clean",
    "cellpose-sam",
    "cellpose-clean",
]


def build_report(report_dir: Path, frames: list, per_frame_results: dict,
                 agg_rows: list[dict]):
    """`frames` is [(frame_id, gt, gt_jpg_path), ...]
    `per_frame_results` is {variant: {frame_id: {pred, matches, jpg_path, metrics}}}
    `agg_rows` is the aggregate metric rows we already print to stdout."""
    img_rel = lambda p: p.relative_to(report_dir).as_posix()

    # Build the metric table
    metric_html = ["<table class='metrics'><thead><tr>"
                   "<th>variant</th><th>TP</th><th>FP</th><th>FN</th>"
                   "<th>prec</th><th>rec</th><th>★rec</th><th>F1</th></tr></thead><tbody>"]
    for r in agg_rows:
        cls = "best" if r["best"] else ""
        metric_html.append(
            f"<tr class='{cls}'><td>{html.escape(r['variant'])}</td>"
            f"<td>{r['tp']}</td><td>{r['fp']}</td><td>{r['fn']}</td>"
            f"<td>{r['prec']:.2f}</td><td>{r['rec']:.2f}</td>"
            f"<td>{r['srec']:.2f}</td><td>{r['f1']:.2f}</td></tr>"
        )
    metric_html.append("</tbody></table>")

    # Image grid: rows = frames, cols = [GT, baseline, …]
    cols = ["GT"] + [v for v in REPORT_VARIANTS]
    grid = ["<table class='grid'><thead><tr><th>frame</th>"]
    for c in cols:
        grid.append(f"<th>{html.escape(c)}</th>")
    grid.append("</tr></thead><tbody>")
    for frame_id, gt, gt_jpg in frames:
        grid.append(f"<tr><th class='rowhdr'>{html.escape(frame_id)}<br>"
                    f"<small>GT={len(gt)} (★{sum(1 for g in gt if g.get('star'))})</small></th>")
        grid.append(f"<td><a href='{img_rel(gt_jpg)}' target='_blank'>"
                    f"<img src='{img_rel(gt_jpg)}'/></a></td>")
        for v in REPORT_VARIANTS:
            cell = per_frame_results.get(v, {}).get(frame_id)
            if cell is None:
                grid.append("<td>—</td>")
                continue
            m = cell["metrics"]
            jpg = cell["jpg_path"]
            grid.append(
                f"<td><a href='{img_rel(jpg)}' target='_blank'>"
                f"<img src='{img_rel(jpg)}'/></a>"
                f"<div class='cell-metrics'>TP {m['tp']} · FP {m['fp']} · FN {m['fn']}<br>"
                f"prec {m['prec']:.2f} · rec {m['rec']:.2f} · ★ {m['srec']:.2f}</div></td>"
            )
        grid.append("</tr>")
    grid.append("</tbody></table>")

    page = f"""<!doctype html>
<html><head>
<meta charset='utf-8'>
<title>circle-detection experiments</title>
<style>
  body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 16px; background:#fafafa; color:#222; }}
  h1 {{ margin-top: 0; }}
  .legend {{ font-size: 12px; color:#666; margin-bottom: 16px; }}
  table {{ border-collapse: collapse; }}
  table.metrics {{ margin-bottom: 24px; }}
  table.metrics th, table.metrics td {{ padding: 4px 10px; border-bottom: 1px solid #ddd; font-size: 13px; text-align:right; }}
  table.metrics th:first-child, table.metrics td:first-child {{ text-align:left; }}
  table.metrics tr.best {{ background:#f6ffe8; font-weight:600; }}
  table.grid th, table.grid td {{ padding: 4px; vertical-align: top; border:1px solid #ccc; }}
  table.grid th {{ background:#eee; font-size: 12px; }}
  table.grid img {{ max-width: 260px; display:block; }}
  .cell-metrics {{ font-size: 11px; color:#444; margin-top:4px; line-height:1.3; font-variant-numeric: tabular-nums; }}
  .rowhdr {{ text-align:left; vertical-align: middle; white-space: nowrap; }}
  small {{ color:#888; font-weight: normal; }}
</style>
</head><body>
<h1>circle-detection experiments</h1>
<p class='legend'>
  Annotated GT in <span style='color:#2a9'>green</span> (normal) / <span style='color:#48f'>gold-ish</span> (starred).
  In prediction columns: GT in <span style='color:#999'>white outline</span> (matched) / <span style='color:#e3e'>magenta</span> (missed);
  predictions in <span style='color:#2a4'>green</span> (matched TP) / <span style='color:#44e'>blue</span> (FP).
</p>
{''.join(metric_html)}
{''.join(grid)}
</body></html>
"""
    (report_dir / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None,
                    help="if set, emit an HTML report with overlay images to this dir")
    args = ap.parse_args()

    spec = discover_grid("data")
    wells_by_name = {w.folder_name: w for w in spec.wells}

    ann_files = sorted(Path("annotations").rglob("*.json"))
    if not ann_files:
        raise SystemExit("no annotations found")

    # Pre-load each annotated well's aligned stack once.
    stack_cache: dict[str, list[np.ndarray]] = {}
    annotations: list[tuple] = []  # (well_name, t_idx, ann dict)
    for f in ann_files:
        ann = json.loads(f.read_text())
        w = wells_by_name.get(ann["well"])
        if w is None:
            continue
        t_idx = next((i for i, tp in enumerate(w.timepoints) if tp.label == ann["label"]), None)
        if t_idx is None:
            continue
        if ann["well"] not in stack_cache:
            print(f"loading + aligning {ann['well']}...", flush=True)
            stack_cache[ann["well"]] = _aligned_stack(w)
        annotations.append((ann["well"], t_idx, ann))

    if not annotations:
        raise SystemExit("no usable annotations")

    print()
    header = f"{'variant':16s} {'frame':24s} {'GT':>3s} {'P':>3s} {'TP':>3s} {'FP':>3s} {'FN':>3s} {'prec':>5s} {'rec':>5s} {'★rec':>5s}"
    print(header)
    print("-" * len(header))

    agg: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0,
                                                "star_tp": 0, "star_fn": 0})

    # Report-building state.
    report_dir: Path | None = None
    images_dir: Path | None = None
    per_frame_results: dict = defaultdict(dict)
    report_frames: list = []
    if args.report:
        report_dir = Path(args.report)
        images_dir = report_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        # Emit GT-only images for each annotated frame.
        for well_name, t_idx, ann in annotations:
            frame = stack_cache[well_name][t_idx]
            frame_rgb = _to_rgb_u8(frame)
            gt_img = _render_gt(frame_rgb, ann["circles"])
            frame_id = f"{well_name} {ann['label']}"
            safe = (well_name + "__" + ann["label"]).replace(" ", "_").replace("+", "p").replace("/", "_")
            gt_path = images_dir / f"{safe}__GT.jpg"
            _save_jpg(gt_path, gt_img)
            report_frames.append((frame_id, ann["circles"], gt_path))
    for vname, variant_fn in VARIANTS:
        for well_name, t_idx, ann in annotations:
            stack = stack_cache[well_name]
            kwargs = {}
            if (vname.startswith("persist") or vname == "track-filter"
                    or vname.startswith("union") or vname.startswith("base+temp")
                    or vname == "persist-med-clean"):
                kwargs["well_key"] = well_name
            pred = variant_fn(stack, t_idx, **kwargs)
            gt = ann["circles"]
            matches, used = _match(gt, pred)
            matched_gt = {m[0] for m in matches}
            star_tp = sum(1 for m in matches if gt[m[0]].get("star"))
            star_fn = sum(1 for i, g in enumerate(gt) if g.get("star") and i not in matched_gt)
            tp = len(matches)
            fp = len(pred) - len(used)
            fn_count = len(gt) - tp
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn_count) if (tp + fn_count) else 0.0
            srec = star_tp / (star_tp + star_fn) if (star_tp + star_fn) else 1.0
            frame_tag = f"{well_name} {ann['label']}"
            print(f"{vname:16s} {frame_tag:24s} {len(gt):3d} {len(pred):3d} {tp:3d} {fp:3d} {fn_count:3d} "
                  f"{prec:5.2f} {rec:5.2f} {srec:5.2f}")
            a = agg[vname]
            a["tp"] += tp; a["fp"] += fp; a["fn"] += fn_count

            # Save overlay JPEG for the report if requested + this variant is featured.
            if images_dir is not None and vname in REPORT_VARIANTS:
                frame_rgb = _to_rgb_u8(stack_cache[well_name][t_idx])
                overlay = _render_pred(frame_rgb, gt, pred, matches)
                safe = (well_name + "__" + ann["label"] + "__" + vname).replace(" ", "_").replace("+", "p").replace("/", "_")
                jpg_path = images_dir / f"{safe}.jpg"
                _save_jpg(jpg_path, overlay)
                per_frame_results[vname][f"{well_name} {ann['label']}"] = {
                    "metrics": {"tp": tp, "fp": fp, "fn": fn_count,
                                 "prec": prec, "rec": rec, "srec": srec},
                    "jpg_path": jpg_path,
                }
            a["star_tp"] += star_tp; a["star_fn"] += star_fn
        # aggregate row for this variant
        a = agg[vname]
        prec = a["tp"] / (a["tp"] + a["fp"]) if (a["tp"] + a["fp"]) else 0.0
        rec = a["tp"] / (a["tp"] + a["fn"]) if (a["tp"] + a["fn"]) else 0.0
        srec = a["star_tp"] / (a["star_tp"] + a["star_fn"]) if (a["star_tp"] + a["star_fn"]) else 1.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        print(f"{vname:16s} {'== AGGREGATE ==':24s} {'':3s} {'':3s} "
              f"{a['tp']:3d} {a['fp']:3d} {a['fn']:3d} {prec:5.2f} {rec:5.2f} {srec:5.2f}  F1={f1:.2f}")
        print()
        a["_summary"] = {"variant": vname, "tp": a["tp"], "fp": a["fp"],
                          "fn": a["fn"], "prec": prec, "rec": rec, "srec": srec,
                          "f1": f1, "best": False}

    if report_dir is not None:
        agg_rows = [a["_summary"] for a in agg.values()]
        if agg_rows:
            best_f1 = max(r["f1"] for r in agg_rows)
            for r in agg_rows:
                r["best"] = r["f1"] >= best_f1 - 1e-9
        agg_rows.sort(key=lambda r: -r["f1"])
        build_report(report_dir, report_frames, per_frame_results, agg_rows)
        print(f"\nreport: {(report_dir / 'index.html').resolve()}")


if __name__ == "__main__":
    main()
