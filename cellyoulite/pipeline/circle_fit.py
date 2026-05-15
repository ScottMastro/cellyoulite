from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CircleFit:
    cx: float
    cy: float
    radius: float

    @property
    def area(self) -> float:
        return math.pi * self.radius ** 2


def fit_circle(mask: np.ndarray) -> CircleFit:
    """Best-fit circle for a binary mask.

    Uses the mask centroid as the centre and the equivalent-area radius
    (r = sqrt(area / pi)). This is the conservative choice — it matches the
    pixel-volume readout exactly when the object is circular and degrades
    gracefully on irregular shapes without overfitting.
    """
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if not mask.any():
        return CircleFit(cx=float("nan"), cy=float("nan"), radius=0.0)

    ys, xs = np.nonzero(mask)
    area = float(mask.sum())
    return CircleFit(cx=float(xs.mean()), cy=float(ys.mean()),
                     radius=math.sqrt(area / math.pi))
