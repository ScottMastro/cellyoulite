from __future__ import annotations

import numpy as np
from skimage.draw import disk

from cellyoulite.pipeline import fit_circle, segment_image
from cellyoulite.pipeline.segment import masks_to_circles


def _synthetic_image(shape=(256, 256), centres=((80, 80), (180, 170)),
                     radii=(20, 25), bg=200, fg=80, noise=5, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full(shape, bg, dtype=np.float32)
    for (cy, cx), r in zip(centres, radii):
        rr, cc = disk((cy, cx), r, shape=shape)
        img[rr, cc] = fg
    img += rng.normal(0, noise, shape)
    return img


def test_segment_finds_two_organoids():
    img = _synthetic_image()
    result = segment_image(img, min_area_px=100)
    assert len(result.regions) == 2


def test_fit_circle_radius_matches_area():
    img = _synthetic_image(centres=((128, 128),), radii=(30,))
    result = segment_image(img, min_area_px=100)
    assert len(result.regions) == 1
    fit = fit_circle(result.mask)
    assert abs(fit.radius - 30) < 2.0


def _label_mask(radii_px, shape=(400, 400)):
    """A label mask with one disc per requested radius, laid out in a row."""
    masks = np.zeros(shape, dtype=np.uint16)
    x = 40
    for i, r in enumerate(radii_px, start=1):
        rr, cc = disk((200, x), r, shape=shape)
        masks[rr, cc] = i
        x += 2 * int(max(radii_px)) + 10
    return masks


def test_masks_to_circles_size_gates():
    """The radius floor is what decides yield — on a plate of small objects
    it can reject nearly everything Cellpose found, so pin its behaviour."""
    masks = _label_mask([5, 10, 20, 30])

    # Default floor keeps only the two large discs.
    kept = masks_to_circles(masks)
    assert sorted(round(c["r"]) for c in kept) == [20, 30]

    # Lowering it recovers the smaller ones, in area-equivalent radius.
    kept = masks_to_circles(masks, min_radius_px=8)
    assert sorted(round(c["r"]) for c in kept) == [10, 20, 30]

    # The ceiling drops the largest.
    kept = masks_to_circles(masks, min_radius_px=8, max_radius_px=25)
    assert sorted(round(c["r"]) for c in kept) == [10, 20]

    # The area gate is independent of the radius one.
    kept = masks_to_circles(masks, min_radius_px=0, min_area_px=2000)
    assert sorted(round(c["r"]) for c in kept) == [30]


def test_masks_to_circles_reports_centroid_and_area():
    masks = _label_mask([20])
    (c,) = masks_to_circles(masks)
    assert round(c["cy"]) == 200
    assert c["area_px"] > 0
    # r is the area-equivalent radius, so it round-trips through the area.
    assert abs(c["r"] - np.sqrt(c["area_px"] / np.pi)) < 1e-9


def test_masks_to_circles_ignores_background():
    assert masks_to_circles(np.zeros((50, 50), dtype=np.uint16)) == []
