from __future__ import annotations

import numpy as np
from skimage.draw import disk

from cellyoulite.pipeline import fit_circle, segment_image


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
