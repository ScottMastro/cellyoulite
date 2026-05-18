"""Link Cellpose detections across time into per-organoid tracks.

Reads `.cellpose_cache/<well>/<label>.json` for every timepoint in each well,
then matches detections frame-to-frame by centroid proximity (allowing for
small residual drift after alignment + a tolerance for radius change as
organoids grow). Writes one tracks file per well to `tracks/<well>.json`.

Match rule (per consecutive frame pair):
  - Build cost = centroid distance + λ·|Δr|
  - Hungarian assignment, then drop matches whose cost exceeds MAX_COST
  - Unmatched detections start new tracks; tracks unmatched for >MAX_GAP
    consecutive frames are retired (won't pick up later reappearances —
    they'd start a new ID)

Usage:
    python scripts/cellpose_track.py                       # all cached wells
    python scripts/cellpose_track.py --wells "DMSO r1"
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


_CACHE_ROOT = Path.cwd() / ".cellpose_cache"
_TRACKS_ROOT = Path.cwd() / "tracks"

# Match tolerances. Aligned canvas → expected residual drift is small.
MAX_CENTER_DIST = 35.0   # pixels
RADIUS_LAMBDA = 0.5      # weight of |Δr| in cost (1 px Δr ~= 0.5 px Δcenter)
MAX_COST = 40.0          # reject matches above this
MAX_GAP = 2              # frames a track can disappear and still continue

# Validity: a track must appear in at least this fraction of frames to be
# considered "good" — i.e. usable for growth analysis. Sparse tracks are
# kept in the file but flagged invalid so they can be greyed out / dropped.
DEFAULT_MIN_FRAC = 0.75


_LABEL_RE = re.compile(r"^(\d+)d(\d+)h(\d+)m$")


def _label_to_minutes(label: str) -> int:
    m = _LABEL_RE.match(label)
    if not m:
        raise ValueError(f"unparseable label: {label!r}")
    d, h, mn = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return ((d * 24) + h) * 60 + mn


@dataclass
class Track:
    id: int
    detections: list[dict] = field(default_factory=list)  # [{t_idx, label, cx, cy, r}, ...]
    last_seen_t: int = -1

    @property
    def latest(self) -> dict | None:
        return self.detections[-1] if self.detections else None


def link_well(well_dir: Path, min_frac: float = DEFAULT_MIN_FRAC) -> dict:
    files = sorted(well_dir.glob("*.json"),
                   key=lambda p: _label_to_minutes(p.stem))
    frames: list[tuple[int, str, list[dict]]] = []
    for i, f in enumerate(files):
        data = json.loads(f.read_text())
        frames.append((i, data["label"], data.get("circles", [])))

    tracks: list[Track] = []
    next_id = 0
    for t_idx, label, dets in frames:
        # Tracks that are still eligible to match (last seen within MAX_GAP).
        active = [t for t in tracks if (t_idx - t.last_seen_t) <= MAX_GAP + 1
                   and t.detections]
        if active and dets:
            cost = np.full((len(active), len(dets)), 1e9, dtype=np.float64)
            for ti, tr in enumerate(active):
                last = tr.latest
                for di, d in enumerate(dets):
                    cd = float(np.hypot(d["cx"] - last["cx"], d["cy"] - last["cy"]))
                    if cd > MAX_CENTER_DIST:
                        continue
                    c = cd + RADIUS_LAMBDA * abs(d["r"] - last["r"])
                    cost[ti, di] = c
            r_ix, c_ix = linear_sum_assignment(cost)
            matched_dets = set()
            matched_tracks = set()
            for ri, ci in zip(r_ix, c_ix):
                if cost[ri, ci] <= MAX_COST:
                    tr = active[ri]
                    d = dets[ci]
                    tr.detections.append({
                        "t_idx": t_idx, "label": label,
                        "cx": d["cx"], "cy": d["cy"], "r": d["r"],
                        "area_px": d.get("area_px"),
                    })
                    tr.last_seen_t = t_idx
                    matched_dets.add(ci)
                    matched_tracks.add(id(tr))
        else:
            matched_dets = set()

        # Unmatched detections start fresh tracks.
        for di, d in enumerate(dets):
            if di in matched_dets:
                continue
            tracks.append(Track(
                id=next_id,
                detections=[{
                    "t_idx": t_idx, "label": label,
                    "cx": d["cx"], "cy": d["cy"], "r": d["r"],
                    "area_px": d.get("area_px"),
                }],
                last_seen_t=t_idx,
            ))
            next_id += 1

    n_frames = len(frames)
    min_detections = max(1, int(round(min_frac * n_frames)))
    return {
        "n_frames": n_frames,
        "min_frac": min_frac,
        "min_detections": min_detections,
        "frame_labels": [lbl for _, lbl, _ in frames],
        "tracks": [
            {
                "id": t.id,
                "n_detections": len(t.detections),
                "first_t": t.detections[0]["t_idx"],
                "last_t": t.detections[-1]["t_idx"],
                "valid": len(t.detections) >= min_detections,
                "detections": t.detections,
            }
            for t in sorted(tracks, key=lambda x: -len(x.detections))
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", nargs="*", default=None)
    ap.add_argument("--out", default=str(_TRACKS_ROOT))
    ap.add_argument("--min-frac", type=float, default=DEFAULT_MIN_FRAC,
                    help="minimum fraction of frames a track must appear in "
                         "to be considered valid (default %(default)s)")
    args = ap.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    well_dirs = [d for d in sorted(_CACHE_ROOT.iterdir())
                  if d.is_dir() and (not args.wells or d.name in args.wells)]
    if not well_dirs:
        raise SystemExit(f"no cached wells under {_CACHE_ROOT}/"
                          + (f" matching {args.wells}" if args.wells else ""))

    print(f"tracking {len(well_dirs)} well(s)  min_frac={args.min_frac}", flush=True)
    for i, d in enumerate(well_dirs, 1):
        result = link_well(d, min_frac=args.min_frac)
        out_path = out_root / f"{d.name}.json"
        out_path.write_text(json.dumps(result, indent=2))
        n_total = len(result["tracks"])
        n_good = sum(1 for t in result["tracks"] if t["valid"])
        n_bad = n_total - n_good
        n_full = sum(1 for t in result["tracks"]
                      if t["n_detections"] == result["n_frames"])
        # Per-well progress line — matches the [N/M] format the UI parses.
        print(f"[{i:3d}/{len(well_dirs):3d}] {d.name:24s}  "
              f"tracks={n_total:3d}  good={n_good:3d}  bad={n_bad:3d}  "
              f"full={n_full:3d}", flush=True)
    print(f"\ndone — wrote tracks for {len(well_dirs)} well(s)", flush=True)


if __name__ == "__main__":
    main()
