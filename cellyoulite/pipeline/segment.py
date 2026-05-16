from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage import filters, measure, morphology


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
