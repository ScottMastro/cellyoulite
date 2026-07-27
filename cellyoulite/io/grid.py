from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_IMAGE_EXTS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# Folder name: "<treatment> r<replicate>" — treatment may contain "+" / spaces.
_FOLDER_RE = re.compile(r"^(?P<treatment>.+?)\s+r(?P<replicate>\d+)\s*$", re.IGNORECASE)
# Filename timestamp: e.g. "00d04h15m"
_TIME_RE = re.compile(r"(?P<d>\d+)d(?P<h>\d+)h(?P<m>\d+)m", re.IGNORECASE)
# Filename stem: "<position>_<field>_<timestamp>", e.g. "CA1_B9_1_00d00h00m" or
# "2026_7_22_E2_1_00d03h15m". The position is the plate coordinate prefix.
_STEM_RE = re.compile(
    r"^(?P<position>.+?)_(?P<field>\d+)_(?P<time>\d+d\d+h\d+m)$", re.IGNORECASE)


# Pipeline output, not experiment images — never mistake these for wells.
_NON_EXPERIMENT_DIRS = {".align_cache", ".cellpose_cache", "tracks", "annotations"}


def _normalize_treatment(name: str) -> str:
    # Collapse internal whitespace and tidy stray spaces around "+".
    return re.sub(r"\s*\+\s*", "+", " ".join(name.split())).strip()


def is_experiment_folder(name: str) -> bool:
    """True if `name` looks like an experiment folder ("<treatment> r<rep>")."""
    return bool(_FOLDER_RE.match(name.strip()))


def is_experiment_dir(path: Path) -> bool:
    """True if `path` is a directory holding experiment images.

    Name-based `is_experiment_folder` only recognises the "<treatment> r<rep>"
    layout. Newer plate exports name the folder after the condition alone
    ("008", "FSK"), so fall back to looking for timestamped images inside.
    """
    if not path.is_dir() or path.name in _NON_EXPERIMENT_DIRS \
            or path.name.startswith("."):
        return False
    if is_experiment_folder(path.name):
        return True
    return any(p.suffix.lower() in _IMAGE_EXTS and _TIME_RE.search(p.stem)
               for p in path.iterdir() if p.is_file())


def _position_of(stem: str) -> str | None:
    """Plate position from a filename stem, e.g. "CA1_B9_1_00d00h00m" -> "CA1_B9"
    and "2026_7_22_E2_1_00d03h15m" -> "2026_7_22_E2". None if unrecognised."""
    m = _STEM_RE.match(stem)
    return m.group("position") if m else None


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
    folder_name: str          # identity within a batch; synthetic when a
                              # directory holds more than one plate position
    timepoints: list[Timepoint]
    source_dir: str = ""      # the directory on disk this well came from
    position: str | None = None   # plate position, when the layout names one


@dataclass(frozen=True)
class GridSpec:
    treatments: list[str]
    replicates: list[int]
    wells: list[Well]

    @property
    def n_images(self) -> int:
        return sum(len(w.timepoints) for w in self.wells)


def _timepoints_by_position(sub: Path) -> dict[str | None, list[Timepoint]]:
    """Group a directory's timestamped images by plate position, in name order.

    Files whose stem doesn't carry a position group under None, so a directory
    with no recognisable positions still yields a single series.
    """
    groups: dict[str | None, list[Timepoint]] = {}
    for p in sorted(sub.iterdir()):
        if p.suffix.lower() not in _IMAGE_EXTS:
            continue
        tm = _TIME_RE.search(p.stem)
        if not tm:
            continue
        minutes = int(tm.group("d")) * 1440 + int(tm.group("h")) * 60 + int(tm.group("m"))
        groups.setdefault(_position_of(p.stem), []).append(Timepoint(
            minutes=minutes,
            label=tm.group(0),
            key=f"{sub.name}/{p.name}",
            path=p,
        ))
    for tps in groups.values():
        tps.sort(key=lambda t: t.minutes)
    return groups


def discover_grid(folder: str | Path) -> GridSpec:
    """Walk a root folder and return the wells it contains.

    Two subfolder layouts are recognised:

    - "<treatment> r<replicate>", e.g. "012+S9A13 r2" — one well per folder.
    - "<treatment>", e.g. "008" — a plate export holding several positions
      ("..._E2_1_...", "..._E3_1_..."). Each position becomes its own well,
      numbered as a replicate in position order, so "008" with E2/E3/E4 yields
      "008 r1", "008 r2" and "008 r3".

    Timepoints are parsed from any "\\d+d\\d+h\\d+m" token in the filename.
    """
    root = Path(folder)
    wells: list[Well] = []

    for sub in sorted(root.iterdir()):
        if not is_experiment_dir(sub):
            continue
        groups = _timepoints_by_position(sub)
        if not groups:
            continue

        m = _FOLDER_RE.match(sub.name)
        if m:
            # Folder names the replicate: one well, however it's split on disk.
            tps = sorted((t for g in groups.values() for t in g),
                         key=lambda t: t.minutes)
            positions = [p for p in groups if p is not None]
            wells.append(Well(
                treatment=_normalize_treatment(m.group("treatment")),
                replicate=int(m.group("replicate")),
                folder_name=sub.name,
                timepoints=tps,
                source_dir=sub.name,
                position=positions[0] if len(positions) == 1 else None,
            ))
            continue

        # Folder names the condition only — each position is a replicate.
        treatment = _normalize_treatment(sub.name)
        for i, position in enumerate(sorted(groups, key=lambda p: (p is None, p)), 1):
            wells.append(Well(
                treatment=treatment,
                replicate=i,
                folder_name=f"{treatment} r{i}",
                timepoints=groups[position],
                source_dir=sub.name,
                position=position,
            ))

    treatments = sorted({w.treatment for w in wells})
    replicates = sorted({w.replicate for w in wells})
    return GridSpec(treatments=treatments, replicates=replicates, wells=wells)
