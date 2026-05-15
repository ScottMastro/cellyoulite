from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
_PATTERN = re.compile(r"(?P<treatment>[A-H])[_-]?(?P<timepoint>\d{1,2})", re.IGNORECASE)


@dataclass(frozen=True)
class GridCell:
    treatment: str
    timepoint: int
    path: Path


@dataclass(frozen=True)
class GridSpec:
    treatments: list[str]
    timepoints: list[int]
    cells: list[GridCell]


def discover_grid(folder: str | Path) -> GridSpec:
    """Walk a folder and group images into a treatment x timepoint grid.

    Convention: filenames contain a token like "A3", "B_07", "h-12" where the
    letter is treatment and the digits are timepoint. Files not matching are
    skipped — callers should warn the user about skipped files.
    """
    folder = Path(folder)
    cells: list[GridCell] = []
    for p in sorted(folder.iterdir()):
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        m = _PATTERN.search(p.stem)
        if not m:
            continue
        cells.append(GridCell(
            treatment=m.group("treatment").upper(),
            timepoint=int(m.group("timepoint")),
            path=p,
        ))
    treatments = sorted({c.treatment for c in cells})
    timepoints = sorted({c.timepoint for c in cells})
    return GridSpec(treatments=treatments, timepoints=timepoints, cells=cells)
