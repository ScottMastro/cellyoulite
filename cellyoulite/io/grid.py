from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Folder name: "<treatment> r<replicate>" — treatment may contain "+" / spaces.
_FOLDER_RE = re.compile(r"^(?P<treatment>.+?)\s+r(?P<replicate>\d+)\s*$", re.IGNORECASE)
# Filename timestamp: e.g. "00d04h15m"
_TIME_RE = re.compile(r"(?P<d>\d+)d(?P<h>\d+)h(?P<m>\d+)m", re.IGNORECASE)


def _normalize_treatment(name: str) -> str:
    # Collapse internal whitespace and tidy stray spaces around "+".
    return re.sub(r"\s*\+\s*", "+", " ".join(name.split())).strip()


def is_experiment_folder(name: str) -> bool:
    """True if `name` looks like an experiment folder ("<treatment> r<rep>")."""
    return bool(_FOLDER_RE.match(name.strip()))


def is_image_name(name: str) -> bool:
    """True if `name` has a recognised image extension."""
    return Path(name).suffix.lower() in _IMAGE_EXTS


@dataclass(frozen=True)
class Timepoint:
    minutes: int          # total minutes since start of series
    label: str            # raw "00d04h15m" token from filename
    key: str              # path relative to the root folder ("<subdir>/<file>")
    path: Path


@dataclass(frozen=True)
class Well:
    treatment: str
    replicate: int
    folder_name: str
    timepoints: list[Timepoint]


@dataclass(frozen=True)
class GridSpec:
    treatments: list[str]
    replicates: list[int]
    wells: list[Well]

    @property
    def n_images(self) -> int:
        return sum(len(w.timepoints) for w in self.wells)


def discover_grid(folder: str | Path) -> GridSpec:
    """Walk a root folder. Each immediate subdirectory is one well.

    Subfolder naming: "<treatment> r<replicate>", e.g. "012+S9A13 r2".
    Timepoints are parsed from any "\\d+d\\d+h\\d+m" token in the filename.
    """
    root = Path(folder)
    wells: list[Well] = []

    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        m = _FOLDER_RE.match(sub.name)
        if not m:
            continue
        treatment = _normalize_treatment(m.group("treatment"))
        replicate = int(m.group("replicate"))

        tps: list[Timepoint] = []
        for p in sorted(sub.iterdir()):
            if p.suffix.lower() not in _IMAGE_EXTS:
                continue
            tm = _TIME_RE.search(p.stem)
            if not tm:
                continue
            minutes = int(tm.group("d")) * 1440 + int(tm.group("h")) * 60 + int(tm.group("m"))
            tps.append(Timepoint(
                minutes=minutes,
                label=tm.group(0),
                key=f"{sub.name}/{p.name}",
                path=p,
            ))
        tps.sort(key=lambda t: t.minutes)
        if tps:
            wells.append(Well(
                treatment=treatment,
                replicate=replicate,
                folder_name=sub.name,
                timepoints=tps,
            ))

    treatments = sorted({w.treatment for w in wells})
    replicates = sorted({w.replicate for w in wells})
    return GridSpec(treatments=treatments, replicates=replicates, wells=wells)
