from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import filters, measure, morphology

# Defaults for masks_to_circles. Tuned on the CA1 T1 plate, where organoids
# are ~125 um across; MIN_RADIUS_PX corresponds to ~102 um at 2.83 um/px. A
# plate imaged with smaller objects needs a lower floor — see --min-radius.
MIN_RADIUS_PX = 18.0
MAX_RADIUS_PX = 200.0
MIN_AREA_PX = 50


def masks_to_circles(masks: np.ndarray, *,
                     min_radius_px: float = MIN_RADIUS_PX,
                     max_radius_px: float = MAX_RADIUS_PX,
                     min_area_px: int = MIN_AREA_PX) -> list[dict]:
    """Reduce a Cellpose per-instance label mask to circle detections.

    Each instance becomes its centroid plus an area-equivalent radius. The
    size gates decide what counts as an organoid rather than debris, and they
    are the single biggest determinant of yield: on a plate of small objects
    the default floor can reject well over 90% of what Cellpose found.

    This is pure post-processing — the mask is what the GPU produced, so
    re-running it at a different threshold needs no segmentation.
    """
    out = []
    for label in np.unique(masks):
        if label == 0:
            continue
        ys, xs = np.where(masks == label)
        if ys.size < min_area_px:
            continue
        r = float(np.sqrt(ys.size / np.pi))
        if r < min_radius_px or r > max_radius_px:
            continue
        out.append({"cx": float(xs.mean()), "cy": float(ys.mean()),
                    "r": r, "area_px": int(ys.size)})
    return out


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray            # bool, same shape as input
    labels: np.ndarray          # int labels, same shape as input
    regions: list               # skimage RegionProperties, one per organoid


def segment_image(
    image: np.ndarray,
    *,
    min_area_px: int = 200,
    max_area_px: int = 200_000,
) -> SegmentationResult:
    """Classical segmentation: organoids are darker than background.

    Pipeline: Gaussian denoise -> Otsu on inverted image -> morphological
    cleanup -> connected components -> area filter. Returns a strict subset
    of objects that pass size filters; downstream QC handles shape filters.
    """
    if image.ndim == 3:
        # Collapse RGB(A) to greyscale. Brightfield channels are near-identical;
        # a simple mean is fine and avoids a colour-science detour.
        image = image[..., :3].mean(axis=-1)
    if image.ndim != 2:
        raise ValueError(f"expected 2D greyscale, got shape {image.shape}")

    smoothed = filters.gaussian(image, sigma=1.5, preserve_range=True)
    inverted = smoothed.max() - smoothed
    threshold = filters.threshold_otsu(inverted)
    binary = inverted > threshold

    binary = morphology.remove_small_holes(binary, 64)
    binary = morphology.remove_small_objects(binary, min_area_px - 1)
    binary = morphology.opening(binary, morphology.disk(2))

    labels = measure.label(binary)
    regions = [
        r for r in measure.regionprops(labels, intensity_image=image)
        if min_area_px <= r.area <= max_area_px
    ]
    keep = np.zeros_like(labels)
    for new_id, r in enumerate(regions, start=1):
        keep[labels == r.label] = new_id

    return SegmentationResult(mask=keep > 0, labels=keep, regions=regions)
