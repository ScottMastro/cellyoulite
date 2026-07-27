"""Typed read/write helpers. Route handlers call these — no SQL in app.py.
Each call uses a short-lived connection (connect() sets WAL + busy_timeout)."""
from __future__ import annotations

from .connection import connect
from .connection import now_iso as _now

# ---------------------------- batches ----------------------------

def ensure_batch(conn, name: str) -> int:
    """Get a batch's id, creating the batch if this is the first well from it."""
    conn.execute("INSERT INTO batch(name,created_at) VALUES(?,?) "
                 "ON CONFLICT(name) DO NOTHING", (name, _now()))
    return conn.execute("SELECT id FROM batch WHERE name=?", (name,)).fetchone()["id"]


def list_batches() -> list[str]:
    conn = connect()
    try:
        return [r["name"] for r in conn.execute(
            "SELECT name FROM batch ORDER BY name COLLATE NOCASE")]
    finally:
        conn.close()


# ---------------------------- wells / organoids ----------------------------

def ensure_well(conn, batch: str, folder_name: str,
                treatment=None, replicate=None) -> int:
    bid = ensure_batch(conn, batch)
    # COALESCE keeps any existing treatment/replicate when the caller passes None.
    conn.execute(
        "INSERT INTO well(batch_id,folder_name,treatment,replicate,first_seen) "
        "VALUES(?,?,?,?,?) "
        "ON CONFLICT(batch_id,folder_name) DO UPDATE SET "
        "treatment=COALESCE(excluded.treatment, well.treatment), "
        "replicate=COALESCE(excluded.replicate, well.replicate)",
        (bid, folder_name, treatment, replicate, _now()))
    return conn.execute("SELECT id FROM well WHERE batch_id=? AND folder_name=?",
                        (bid, folder_name)).fetchone()["id"]


def _well_id(conn, batch: str, folder_name: str) -> int | None:
    """Look up a well without creating it (or its batch). None if unknown."""
    r = conn.execute(
        "SELECT w.id FROM well w JOIN batch b ON b.id=w.batch_id "
        "WHERE b.name=? AND w.folder_name=?", (batch, folder_name)).fetchone()
    return r["id"] if r else None


# ---------------------------- ingestion (analysis results -> rows) ----------------------------

def set_alignment(batch: str, folder_name: str, fingerprint: str,
                  cache_version: int, canvas_h: int, canvas_w: int,
                  offsets, placements, treatment=None, replicate=None) -> None:
    conn = connect()
    try:
        import json
        wid = ensure_well(conn, batch, folder_name, treatment, replicate)
        conn.execute(
            "INSERT INTO alignment(well_id,fingerprint,cache_version,canvas_h,canvas_w,"
            "offsets,placements,created_at) VALUES(?,?,?,?,?,?,?,?) "
            "ON CONFLICT(well_id) DO UPDATE SET fingerprint=excluded.fingerprint,"
            "cache_version=excluded.cache_version,canvas_h=excluded.canvas_h,"
            "canvas_w=excluded.canvas_w,offsets=excluded.offsets,"
            "placements=excluded.placements,created_at=excluded.created_at",
            (wid, fingerprint, cache_version, canvas_h, canvas_w,
             json.dumps(offsets), json.dumps(placements), _now()))
        conn.commit()
    finally:
        conn.close()


def set_tracks(batch: str, folder_name: str, tracks_data: dict,
               source_fingerprint=None, treatment=None, replicate=None) -> int:
    """Ingest a well's organoids + detections (the tracks JSON shape). Upserts
    cache rows and replaces their detections; authored track_filter and stars
    are preserved (only cache columns change)."""
    conn = connect()
    try:
        wid = ensure_well(conn, batch, folder_name, treatment, replicate)
        n = 0
        for t in tracks_data.get("tracks", []):
            # Fixed-id anchor = the organoid's first-frame centroid.
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
                 int(bool(t.get("edge_clipped"))), source_fingerprint,
                 anchor_cx, anchor_cy))
            tid = conn.execute("SELECT id FROM track WHERE well_id=? AND track_num=?",
                               (wid, t["id"])).fetchone()["id"]
            conn.execute("DELETE FROM detection WHERE track_id=?", (tid,))
            conn.executemany(
                "INSERT INTO detection(track_id,t_idx,label,cx,cy,r,area_px) "
                "VALUES(?,?,?,?,?,?,?)",
                [(tid, d["t_idx"], d.get("label", ""), d["cx"], d["cy"], d["r"],
                  d.get("area_px")) for d in t.get("detections", [])])
            n += 1
        conn.commit()
        return n
    finally:
        conn.close()


def get_alignment(batch: str, folder_name: str) -> dict | None:
    """The alignment row for a well (fingerprint, cache_version, canvas, and
    JSON offsets/placements), or None."""
    conn = connect()
    try:
        r = conn.execute(
            "SELECT a.fingerprint, a.cache_version, a.canvas_h, a.canvas_w, "
            "a.offsets, a.placements FROM alignment a JOIN well w ON w.id=a.well_id "
            "JOIN batch b ON b.id=w.batch_id "
            "WHERE b.name=? AND w.folder_name=?", (batch, folder_name)).fetchone()
        return dict(r) if r else None
    finally:
        conn.close()


def get_tracks(batch: str, folder_name: str) -> dict | None:
    """Reconstruct the tracks payload (matching the JSON shape) from the DB,
    or None if the well has no organoids. Includes per-organoid `starred`."""
    conn = connect()
    try:
        wid = _well_id(conn, batch, folder_name)
        if wid is None:
            return None
        trows = conn.execute(
            "SELECT id,track_num,n_detections,first_t,last_t,auto_valid,edge_clipped,"
            "starred,anchor_cx,anchor_cy "
            "FROM track WHERE well_id=? ORDER BY track_num", (wid,)).fetchall()
        if not trows:
            return None
        tracks, n_frames = [], 0
        for tr in trows:
            dets = conn.execute(
                "SELECT t_idx,label,cx,cy,r,area_px FROM detection WHERE track_id=? "
                "ORDER BY t_idx", (tr["id"],)).fetchall()
            for d in dets:
                n_frames = max(n_frames, d["t_idx"] + 1)
            tracks.append({
                "id": tr["track_num"], "n_detections": tr["n_detections"],
                "first_t": tr["first_t"], "last_t": tr["last_t"],
                "valid": bool(tr["auto_valid"]), "edge_clipped": bool(tr["edge_clipped"]),
                "starred": bool(tr["starred"]),
                "anchor_cx": tr["anchor_cx"], "anchor_cy": tr["anchor_cy"],
                "detections": [dict(d) for d in dets],
            })
        return {"n_frames": n_frames, "tracks": tracks}
    finally:
        conn.close()


def _ensure_track(conn, well_id: int, track_num: int) -> int:
    """Get a track row's id, creating a placeholder if the track isn't in the
    DB yet (e.g. a freshly-uploaded well not yet backfilled). Cache fields get
    filled in later by the backfill/read-through UPSERT; the filter we attach
    here survives that (it only updates cache columns)."""
    conn.execute(
        "INSERT INTO track(well_id,track_num,n_detections,first_t,last_t,auto_valid,edge_clipped) "
        "VALUES(?,?,0,0,0,0,0) ON CONFLICT(well_id,track_num) DO NOTHING",
        (well_id, track_num))
    return conn.execute("SELECT id FROM track WHERE well_id=? AND track_num=?",
                        (well_id, track_num)).fetchone()["id"]


# ---------------------------- validation ----------------------------

def get_validation(batch: str, folder_name: str) -> dict:
    """{"overrides": {track_num:int -> bool}, "human_validated": bool}."""
    conn = connect()
    try:
        wid = _well_id(conn, batch, folder_name)
        if wid is None:
            return {"overrides": {}, "human_validated": False}
        rows = conn.execute(
            "SELECT t.track_num AS num, f.accepted AS acc FROM track_filter f "
            "JOIN track t ON t.id = f.track_id WHERE t.well_id=?", (wid,)).fetchall()
        overrides = {r["num"]: bool(r["acc"]) for r in rows}
        wv = conn.execute("SELECT validated FROM well_validation WHERE well_id=?",
                          (wid,)).fetchone()
        return {"overrides": overrides,
                "human_validated": bool(wv["validated"]) if wv else False}
    finally:
        conn.close()


def set_overrides(batch: str, folder_name: str, overrides: dict, decided_by: str,
                  reason: str = "manual") -> int:
    """Replace the per-organoid filter decisions for a well (full-replace,
    matching the POST semantics). Records who decided and when."""
    conn = connect()
    try:
        wid = ensure_well(conn, batch, folder_name)
        conn.execute(
            "DELETE FROM track_filter WHERE track_id IN (SELECT id FROM track WHERE well_id=?)",
            (wid,))
        now = _now()
        for tnum, accepted in overrides.items():
            tid = _ensure_track(conn, wid, int(tnum))
            conn.execute(
                "INSERT INTO track_filter(track_id,accepted,decided_by,decided_at,reason) "
                "VALUES(?,?,?,?,?)",
                (tid, int(bool(accepted)), decided_by or "unknown", now, reason))
        conn.commit()
        return len(overrides)
    finally:
        conn.close()


def set_human_validated(batch: str, folder_name: str, validated: bool, by: str) -> bool:
    conn = connect()
    try:
        wid = ensure_well(conn, batch, folder_name)
        conn.execute(
            "INSERT INTO well_validation(well_id,validated,validated_by,validated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(well_id) DO UPDATE SET validated=excluded.validated,"
            "validated_by=excluded.validated_by, validated_at=excluded.validated_at",
            (wid, int(bool(validated)), by, _now()))
        conn.commit()
        return bool(validated)
    finally:
        conn.close()


def human_validated_map(batch: str) -> dict:
    """folder_name -> bool for every signed-off well in a batch (status polls)."""
    conn = connect()
    try:
        return {r["folder_name"]: bool(r["validated"]) for r in conn.execute(
            "SELECT w.folder_name, v.validated FROM well_validation v "
            "JOIN well w ON w.id = v.well_id JOIN batch b ON b.id = w.batch_id "
            "WHERE b.name=?", (batch,))}
    finally:
        conn.close()


# ---------------------------- organoid stars ----------------------------

def get_stars(batch: str, folder_name: str) -> dict:
    """track_num -> True for every starred organoid in the well."""
    conn = connect()
    try:
        wid = _well_id(conn, batch, folder_name)
        if wid is None:
            return {}
        return {r["track_num"]: True for r in conn.execute(
            "SELECT track_num FROM track WHERE well_id=? AND starred=1", (wid,))}
    finally:
        conn.close()


def set_star(batch: str, folder_name: str, track_num: int, starred: bool,
             by: str) -> bool:
    conn = connect()
    try:
        wid = ensure_well(conn, batch, folder_name)
        tid = _ensure_track(conn, wid, int(track_num))
        if starred:
            conn.execute("UPDATE track SET starred=1, starred_by=?, starred_at=? WHERE id=?",
                         (by or "unknown", _now(), tid))
        else:
            conn.execute("UPDATE track SET starred=0, starred_by=NULL, starred_at=NULL WHERE id=?",
                         (tid,))
        conn.commit()
        return bool(starred)
    finally:
        conn.close()


# ---------------------------- frame annotations ----------------------------
# Hand-drawn ground truth, used to score the detectors. Authored and
# irreplaceable, so writes replace one frame at a time and never cascade.

def get_annotation(batch: str, well: str, label: str) -> dict | None:
    """One frame's annotation, or None if it was never drawn."""
    conn = connect()
    try:
        wid = _well_id(conn, batch, well)
        if wid is None:
            return None
        a = conn.execute(
            "SELECT id, aligned, source_key, image_w, image_h FROM annotation "
            "WHERE well_id=? AND label=?", (wid, label)).fetchone()
        if not a:
            return None
        circles = [
            {"cx": r["cx"], "cy": r["cy"], "r": r["r"], "star": bool(r["star"])}
            for r in conn.execute(
                "SELECT cx, cy, r, star FROM annotation_circle "
                "WHERE annotation_id=? ORDER BY idx", (a["id"],))]
        return {"batch": batch, "well": well, "label": label,
                "aligned": bool(a["aligned"]), "key": a["source_key"],
                "image_w": a["image_w"], "image_h": a["image_h"],
                "circles": circles}
    finally:
        conn.close()


def set_annotation(batch: str, well: str, label: str, circles: list,
                   aligned: bool = False, source_key=None,
                   image_w=None, image_h=None) -> int:
    """Replace one frame's circles. Full-replace, matching the POST semantics
    (the editor always sends the complete set for that frame)."""
    conn = connect()
    try:
        wid = ensure_well(conn, batch, well)
        conn.execute(
            "INSERT INTO annotation(well_id,label,aligned,source_key,image_w,"
            "image_h,updated_at) VALUES(?,?,?,?,?,?,?) "
            "ON CONFLICT(well_id,label) DO UPDATE SET aligned=excluded.aligned,"
            "source_key=excluded.source_key,image_w=excluded.image_w,"
            "image_h=excluded.image_h,updated_at=excluded.updated_at",
            (wid, label, int(bool(aligned)), source_key, image_w, image_h, _now()))
        aid = conn.execute("SELECT id FROM annotation WHERE well_id=? AND label=?",
                           (wid, label)).fetchone()["id"]
        conn.execute("DELETE FROM annotation_circle WHERE annotation_id=?", (aid,))
        conn.executemany(
            "INSERT INTO annotation_circle(annotation_id,idx,cx,cy,r,star) "
            "VALUES(?,?,?,?,?,?)",
            [(aid, i, c["cx"], c["cy"], c["r"], int(bool(c.get("star"))))
             for i, c in enumerate(circles)])
        conn.commit()
        return len(circles)
    finally:
        conn.close()


def list_annotations(batch: str | None = None) -> list[dict]:
    """Every annotated frame, newest schema shape. Used by the offline scoring
    scripts, which compare detector output against this ground truth."""
    conn = connect()
    try:
        sql = ("SELECT a.id, b.name AS batch, w.folder_name AS well, a.label, "
               "a.aligned, a.source_key, a.image_w, a.image_h "
               "FROM annotation a JOIN well w ON w.id=a.well_id "
               "JOIN batch b ON b.id=w.batch_id")
        params: tuple = ()
        if batch is not None:
            sql += " WHERE b.name=?"
            params = (batch,)
        sql += " ORDER BY b.name, w.folder_name, a.label"
        out = []
        for a in conn.execute(sql, params).fetchall():
            out.append({
                "batch": a["batch"], "well": a["well"], "label": a["label"],
                "aligned": bool(a["aligned"]), "key": a["source_key"],
                "image_w": a["image_w"], "image_h": a["image_h"],
                "circles": [
                    {"cx": r["cx"], "cy": r["cy"], "r": r["r"],
                     "star": bool(r["star"])}
                    for r in conn.execute(
                        "SELECT cx, cy, r, star FROM annotation_circle "
                        "WHERE annotation_id=? ORDER BY idx", (a["id"],))],
            })
        return out
    finally:
        conn.close()


# ---------------------------- users ----------------------------

def list_users() -> list[str]:
    conn = connect()
    try:
        return [r["username"] for r in conn.execute(
            "SELECT username FROM app_user ORDER BY username COLLATE NOCASE")]
    finally:
        conn.close()


def create_user(name: str) -> tuple[str, bool]:
    """Returns (canonical_username, created?). Case-insensitive de-dup."""
    conn = connect()
    try:
        existing = conn.execute(
            "SELECT username FROM app_user WHERE username = ? COLLATE NOCASE", (name,)).fetchone()
        if existing:
            return existing["username"], False
        conn.execute("INSERT INTO app_user(username,created_at) VALUES(?,?)", (name, _now()))
        conn.commit()
        return name, True
    finally:
        conn.close()
