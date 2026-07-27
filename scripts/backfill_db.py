"""One-shot importer: existing JSON files -> SQLite. Idempotent / re-runnable.

Reads the pre-batch *flat* layout (tracks/<well>.json, data/<well>/), so it
imports one batch at a time and needs to be told which:

    python scripts/backfill_db.py --batch "CA1 (May 2026)"

Reproducible caches (alignment, organoids, detections) are upserted. Authored
curation (organoid filters, well sign-off, users) is inserted only if absent,
so re-running never clobbers decisions made after the first import.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from cellyoulite.db.connection import connect
from cellyoulite.db.migrate import migrate
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import _CACHE_VERSION, _fingerprint, _load_cached

_TRACKS = Path("tracks")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _data_dir() -> Path:
    env = os.environ.get("CELLYOULITE_DATA")
    if env and Path(env).is_dir():
        return Path(env)
    return Path("data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True,
                    help='batch these wells belong to, e.g. "CA1 (May 2026)"')
    args = ap.parse_args()
    batch = args.batch

    migrate()
    conn = connect()
    try:
        conn.execute("INSERT INTO batch(name,created_at) VALUES(?,?) "
                     "ON CONFLICT(name) DO NOTHING", (batch, _now()))
        batch_id = conn.execute("SELECT id FROM batch WHERE name=?",
                                (batch,)).fetchone()["id"]

        # ---- wells (treatment/replicate from the grid where available) ----
        meta: dict[str, tuple] = {}  # folder_name -> (treatment, replicate, paths)
        ddir = _data_dir()
        if ddir.is_dir():
            for w in discover_grid(ddir).wells:
                meta[w.folder_name] = (w.treatment, w.replicate, [tp.path for tp in w.timepoints])
        names = set(meta)
        if _TRACKS.is_dir():
            for p in _TRACKS.glob("*.json"):
                names.add(p.stem[:-len("__validation")]
                          if p.stem.endswith("__validation") else p.stem)
        for name in sorted(names):
            tr, rep, _ = meta.get(name, (None, None, None))
            conn.execute(
                "INSERT INTO well(batch_id,folder_name,treatment,replicate,first_seen) "
                "VALUES(?,?,?,?,?) "
                "ON CONFLICT(batch_id,folder_name) DO UPDATE SET "
                "treatment=excluded.treatment, replicate=excluded.replicate",
                (batch_id, name, tr, rep, _now()))
        conn.commit()
        well_id = {r["folder_name"]: r["id"] for r in conn.execute(
            "SELECT id, folder_name FROM well WHERE batch_id=?", (batch_id,))}

        # ---- alignment (per well, only if a cache file is present) ----
        for name, (_, _, paths) in meta.items():
            if not paths:
                continue
            al = _load_cached(paths)
            if al is None:
                continue
            conn.execute(
                "INSERT INTO alignment(well_id,fingerprint,cache_version,canvas_h,canvas_w,"
                "offsets,placements,created_at) VALUES(?,?,?,?,?,?,?,?) "
                "ON CONFLICT(well_id) DO UPDATE SET fingerprint=excluded.fingerprint,"
                "cache_version=excluded.cache_version,canvas_h=excluded.canvas_h,"
                "canvas_w=excluded.canvas_w,offsets=excluded.offsets,"
                "placements=excluded.placements,created_at=excluded.created_at",
                (well_id[name], _fingerprint(paths), _CACHE_VERSION,
                 al.canvas_shape[0], al.canvas_shape[1],
                 json.dumps([list(o) for o in al.offsets]),
                 json.dumps([list(p) for p in al.placements]), _now()))
        conn.commit()

        # ---- tracks + detections ----
        if _TRACKS.is_dir():
            for p in sorted(_TRACKS.glob("*.json")):
                if p.stem.endswith("__validation"):
                    continue
                wid = well_id.get(p.stem)
                if wid is None:
                    continue
                try:
                    data = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                src_fp = (_fingerprint(meta[p.stem][2])
                          if p.stem in meta and meta[p.stem][2] else None)
                for t in data.get("tracks", []):
                    dets = t.get("detections", [])
                    first = min(dets, key=lambda d: d["t_idx"]) if dets else None
                    anchor_cx = first["cx"] if first else None
                    anchor_cy = first["cy"] if first else None
                    conn.execute(
                        "INSERT INTO track(well_id,track_num,n_detections,first_t,last_t,"
                        "auto_valid,edge_clipped,source_fingerprint,anchor_cx,anchor_cy) "
                        "VALUES(?,?,?,?,?,?,?,?,?,?) "
                        "ON CONFLICT(well_id,track_num) DO UPDATE SET "
                        "n_detections=excluded.n_detections,first_t=excluded.first_t,"
                        "last_t=excluded.last_t,auto_valid=excluded.auto_valid,"
                        "edge_clipped=excluded.edge_clipped,source_fingerprint=excluded.source_fingerprint,"
                        "anchor_cx=excluded.anchor_cx,anchor_cy=excluded.anchor_cy",
                        (wid, t["id"], t.get("n_detections", 0), t.get("first_t", 0),
                         t.get("last_t", 0), int(bool(t.get("valid"))),
                         int(bool(t.get("edge_clipped"))), src_fp, anchor_cx, anchor_cy))
                    tid = conn.execute("SELECT id FROM track WHERE well_id=? AND track_num=?",
                                       (wid, t["id"])).fetchone()["id"]
                    conn.execute("DELETE FROM detection WHERE track_id=?", (tid,))
                    conn.executemany(
                        "INSERT INTO detection(track_id,t_idx,label,cx,cy,r,area_px) "
                        "VALUES(?,?,?,?,?,?,?)",
                        [(tid, d["t_idx"], d.get("label", ""), d["cx"], d["cy"], d["r"],
                          d.get("area_px")) for d in t.get("detections", [])])
            conn.commit()

        # ---- validation -> track_filter + well_validation (insert-if-absent) ----
        if _TRACKS.is_dir():
            for p in sorted(_TRACKS.glob("*__validation.json")):
                wid = well_id.get(p.stem[:-len("__validation")])
                if wid is None:
                    continue
                try:
                    data = json.loads(p.read_text())
                except (OSError, ValueError):
                    continue
                for k, v in (data.get("overrides") or {}).items():
                    try:
                        tnum = int(k)
                    except (TypeError, ValueError):
                        continue
                    row = conn.execute("SELECT id FROM track WHERE well_id=? AND track_num=?",
                                       (wid, tnum)).fetchone()
                    if not row:
                        continue
                    conn.execute(
                        "INSERT INTO track_filter(track_id,accepted,decided_by,decided_at,reason) "
                        "VALUES(?,?,?,?,?) ON CONFLICT(track_id) DO NOTHING",
                        (row["id"], int(bool(v)), "auto", _now(), "legacy-import"))
                if "human_validated" in data:
                    conn.execute(
                        "INSERT INTO well_validation(well_id,validated,validated_by,validated_at) "
                        "VALUES(?,?,?,?) ON CONFLICT(well_id) DO NOTHING",
                        (wid, int(bool(data["human_validated"])), None, _now()))
            conn.commit()

        # ---- users ----
        up = Path("users.json")
        if up.is_file():
            try:
                udata = json.loads(up.read_text())
            except (OSError, ValueError):
                udata = {}
            users = udata.get("users", []) if isinstance(udata, dict) else udata
            for u in (users or []):
                if isinstance(u, str) and u.strip():
                    conn.execute(
                        "INSERT INTO app_user(username,created_at) VALUES(?,?) "
                        "ON CONFLICT(username) DO NOTHING", (u.strip(), _now()))
            conn.commit()

        # ---- report ----
        tables = ["well", "alignment", "track", "detection", "track_filter",
                  "well_validation", "app_user"]
        counts = {t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables}
        print("backfill complete:", counts)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
