"""Typed read/write helpers. Route handlers call these — no SQL in app.py.
Each call uses a short-lived connection (connect() sets WAL + busy_timeout)."""
from __future__ import annotations

from datetime import datetime, timezone

from .connection import connect


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------- wells / tracks ----------------------------

def ensure_well(conn, folder_name: str, treatment=None, replicate=None) -> int:
    conn.execute(
        "INSERT INTO well(folder_name,treatment,replicate,first_seen) VALUES(?,?,?,?) "
        "ON CONFLICT(folder_name) DO NOTHING",
        (folder_name, treatment, replicate, _now()))
    return conn.execute("SELECT id FROM well WHERE folder_name=?", (folder_name,)).fetchone()["id"]


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

def get_validation(folder_name: str) -> dict:
    """{"overrides": {track_num:int -> bool}, "human_validated": bool}."""
    conn = connect()
    try:
        w = conn.execute("SELECT id FROM well WHERE folder_name=?", (folder_name,)).fetchone()
        if not w:
            return {"overrides": {}, "human_validated": False}
        rows = conn.execute(
            "SELECT t.track_num AS num, f.accepted AS acc FROM track_filter f "
            "JOIN track t ON t.id = f.track_id WHERE t.well_id=?", (w["id"],)).fetchall()
        overrides = {r["num"]: bool(r["acc"]) for r in rows}
        wv = conn.execute("SELECT validated FROM well_validation WHERE well_id=?",
                          (w["id"],)).fetchone()
        return {"overrides": overrides,
                "human_validated": bool(wv["validated"]) if wv else False}
    finally:
        conn.close()


def set_overrides(folder_name: str, overrides: dict, decided_by: str,
                  reason: str = "manual") -> int:
    """Replace the per-track filter decisions for a well (full-replace, matching
    the POST semantics). Records who decided and when."""
    conn = connect()
    try:
        wid = ensure_well(conn, folder_name)
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


def set_human_validated(folder_name: str, validated: bool, by: str) -> bool:
    conn = connect()
    try:
        wid = ensure_well(conn, folder_name)
        conn.execute(
            "INSERT INTO well_validation(well_id,validated,validated_by,validated_at) "
            "VALUES(?,?,?,?) ON CONFLICT(well_id) DO UPDATE SET validated=excluded.validated,"
            "validated_by=excluded.validated_by, validated_at=excluded.validated_at",
            (wid, int(bool(validated)), by, _now()))
        conn.commit()
        return bool(validated)
    finally:
        conn.close()


def human_validated_map() -> dict:
    """folder_name -> bool for every well with a sign-off row (for status polls)."""
    conn = connect()
    try:
        return {r["folder_name"]: bool(r["validated"]) for r in conn.execute(
            "SELECT w.folder_name, v.validated FROM well_validation v "
            "JOIN well w ON w.id = v.well_id")}
    finally:
        conn.close()


# ---------------------------- track stars ----------------------------

def get_stars(folder_name: str) -> dict:
    """track_num -> True for every starred track in the well."""
    conn = connect()
    try:
        w = conn.execute("SELECT id FROM well WHERE folder_name=?", (folder_name,)).fetchone()
        if not w:
            return {}
        return {r["track_num"]: True for r in conn.execute(
            "SELECT track_num FROM track WHERE well_id=? AND starred=1", (w["id"],))}
    finally:
        conn.close()


def set_star(folder_name: str, track_num: int, starred: bool, by: str) -> bool:
    conn = connect()
    try:
        wid = ensure_well(conn, folder_name)
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
