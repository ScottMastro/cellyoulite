from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QCThresholds:
    min_solidity: float = 0.85
    max_eccentricity: float = 0.85
    min_area_px: int = 200
    max_area_px: int = 200_000


def passes_qc(region, thresholds: QCThresholds = QCThresholds()) -> bool:
    """Precision-over-recall filter for a single regionprops object."""
    if not (thresholds.min_area_px <= region.area <= thresholds.max_area_px):
        return False
    if region.solidity < thresholds.min_solidity:
        return False
    if region.eccentricity > thresholds.max_eccentricity:
        return False
    return True
