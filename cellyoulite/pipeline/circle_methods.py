"""Single-method circle detector targeted at the actual organoid signature.

Discriminator the user named: organoids are *dark* at their own spatial scale,
in a way that survives illumination correction and beats out noise/debris.

Pipeline:
  1. Flatten illumination — subtract a heavy Gaussian blur (sigma ≫ organoid
     radius) so the bright vertical illumination band stops biasing scores.
  2. Multi-scale dark-blob response — scale-normalised Laplacian of Gaussian
     at a stack of radii. Each scale strongly responds to dark patches of
     that radius. We keep the per-pixel best (max response, argmax radius).
  3. Local-maxima peak picking with non-max suppression at organoid scale.
  4. Per-candidate radial-profile check — sample mean intensity on the
     proposed perimeter vs. a slightly larger annulus. A real organoid has a
     dark boundary (low perimeter intensity) sitting on a lighter halo, so
     `outer - perimeter` must clear a contrast floor.

This is one method on purpose: it directly encodes the discriminator
(darkness at scale) rather than trying generic edge/Hough heuristics that
fire on every speck of debris.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_laplace, maximum_filter

# Organoid scale at native resolution (pixels).
_R_MIN = 30
_R_MAX = 140
_R_STEP = 10

# Illumination-flattening blur. Must be MUCH larger than any organoid so
# the high-pass doesn't eat its own signal.
_ILLUM_SIGMA = 250.0

# Peak picking.
_NMS_FACTOR = 0.9          # NMS window = factor * candidate radius
_SCORE_FRAC_OF_MAX = 0.20  # absolute peak threshold, relative to best response

# Boundary check.
_RING_BAND = 0.18  # +/- fraction of r used as the perimeter band
_HALO_FACTOR = 1.35  # outer halo annulus radius (× r)
_CONTRAST_FLOOR = 6.0  # halo_mean - perimeter_mean, in flattened-intensity units


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    return img.astype(np.float32)


def _flatten_illumination(g: np.ndarray) -> np.ndarray:
    """Subtract a heavy Gaussian to kill the slow illumination gradient.
    The result is centred near zero with dark features negative."""
    bg = cv2.GaussianBlur(g, (0, 0), _ILLUM_SIGMA)
    return g - bg


def _radial_means(flat: np.ndarray, cx: float, cy: float, r: float
                  ) -> tuple[float, float]:
    """Mean flattened intensity on the proposed perimeter band, and on a
    slightly larger halo annulus. Returns (perimeter_mean, halo_mean)."""
    h, w = flat.shape
    # Sample dense points on each ring.
    n = max(48, int(2 * np.pi * r))
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cos_t = np.cos(theta); sin_t = np.sin(theta)

    def _ring_mean(radius: float) -> float:
        sx = np.clip(np.round(cx + radius * cos_t).astype(np.int32), 0, w - 1)
        sy = np.clip(np.round(cy + radius * sin_t).astype(np.int32), 0, h - 1)
        return float(flat[sy, sx].mean())

    perim = 0.5 * (_ring_mean(r * (1.0 - _RING_BAND))
                   + _ring_mean(r * (1.0 + _RING_BAND)))
    halo = _ring_mean(r * _HALO_FACTOR)
    return perim, halo


def detect_circles(img: np.ndarray) -> list[tuple[float, float, float]]:
    g = _to_gray(img)
    flat = _flatten_illumination(g)

    # Multi-scale scale-normalised LoG. We want DARK blobs, so the response
    # of interest is +LoG (LoG is positive at dark centres). Take the
    # per-pixel best across scales.
    radii = list(range(_R_MIN, _R_MAX + 1, _R_STEP))
    best_resp = np.full(g.shape, -np.inf, dtype=np.float32)
    best_r = np.zeros(g.shape, dtype=np.int16)
    for r in radii:
        sigma = r / np.sqrt(2.0)  # LoG peaks at sigma = r / sqrt(2) for radius-r blob
        # gaussian_laplace returns 2nd derivative; multiply by sigma^2 for scale-norm.
        log_resp = gaussian_laplace(flat, sigma=sigma) * (sigma ** 2)
        # Dark blob in flat (negative-valued centre) → LoG is positive there. Good.
        update = log_resp > best_resp
        best_resp = np.where(update, log_resp, best_resp)
        best_r = np.where(update, np.int16(r), best_r)

    # Non-max suppression across the score map.
    nms_size = int(_R_MIN * _NMS_FACTOR) | 1
    local_max = maximum_filter(best_resp, size=nms_size)
    peaks = (best_resp == local_max) & (best_resp > 0)

    if not peaks.any():
        return []

    # Absolute threshold relative to the strongest response.
    score_thr = float(best_resp[peaks].max()) * _SCORE_FRAC_OF_MAX
    peaks &= best_resp >= score_thr

    ys, xs = np.nonzero(peaks)
    scores = best_resp[ys, xs]
    radii_pick = best_r[ys, xs]

    # Sort strongest first.
    order = np.argsort(-scores)
    out: list[tuple[float, float, float]] = []
    h, w = g.shape
    for i in order:
        cx, cy, r = float(xs[i]), float(ys[i]), float(radii_pick[i])
        # Stay clear of borders so radial samples are valid.
        margin = r * _HALO_FACTOR + 2
        if cx < margin or cy < margin or cx > w - margin or cy > h - margin:
            continue
        perim, halo = _radial_means(flat, cx, cy, r)
        if (halo - perim) < _CONTRAST_FLOOR:
            continue
        out.append((cx, cy, r))
    return out


# Back-compat shim for any caller still importing the old names.
method_a_hough = detect_circles
method_b_frst = detect_circles
method_c_ransac = detect_circles
