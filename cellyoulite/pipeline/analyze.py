from __future__ import annotations

from pathlib import Path

import pandas as pd
from skimage.io import imread

from cellyoulite.io.grid import GridSpec, discover_grid
from cellyoulite.pipeline.circle_fit import fit_circle
from cellyoulite.pipeline.qc import QCThresholds, passes_qc
from cellyoulite.pipeline.segment import segment_image


def analyze_folder(folder: str | Path,
                   qc: QCThresholds = QCThresholds()) -> pd.DataFrame:
    """Run the full pipeline on every image in a folder and return long-format results."""
    spec: GridSpec = discover_grid(folder)
    rows: list[dict] = []

    for cell in spec.cells:
        image = imread(cell.path, as_gray=True).astype(float)
        seg = segment_image(image, min_area_px=qc.min_area_px, max_area_px=qc.max_area_px)
        for idx, region in enumerate(seg.regions):
            if not passes_qc(region, qc):
                continue
            obj_mask = seg.labels == (idx + 1)
            circle = fit_circle(obj_mask)
            rows.append({
                "treatment": cell.treatment,
                "timepoint": cell.timepoint,
                "organoid_id": f"{cell.treatment}{cell.timepoint:02d}-{idx}",
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "area_px": int(region.area),
                "circle_radius_px": circle.radius,
                "circle_area_px": circle.area,
                "solidity": float(region.solidity),
                "eccentricity": float(region.eccentricity),
            })

    return pd.DataFrame.from_records(rows)
