"""Schema migrations, tracked via PRAGMA user_version — an ordered list of
steps applied on startup. No external migration framework."""
from __future__ import annotations

from pathlib import Path

from .connection import connect

_SCHEMA = Path(__file__).resolve().parent / "schema.sql"


def _v1(conn) -> None:
    conn.executescript(_SCHEMA.read_text())


def _v2(conn) -> None:
    """Per-organoid fixed-id anchor: the organoid's first-frame centroid, so the
    playback id label stays put instead of tracking the moving organoid each
    frame. Additive columns on `track`; backfilled from the earliest detection."""
    conn.execute("ALTER TABLE track ADD COLUMN anchor_cx REAL")
    conn.execute("ALTER TABLE track ADD COLUMN anchor_cy REAL")
    conn.execute(
        "UPDATE track SET "
        "anchor_cx = (SELECT cx FROM detection d WHERE d.track_id = track.id "
        "             ORDER BY d.t_idx LIMIT 1), "
        "anchor_cy = (SELECT cy FROM detection d WHERE d.track_id = track.id "
        "             ORDER BY d.t_idx LIMIT 1)")


# (target_version, step). Each runs once, in order, when user_version < target.
_MIGRATIONS = [
    (1, _v1),
    (2, _v2),
]


def migrate() -> None:
    """Bring the DB up to the latest schema version (creating the file if
    absent). Safe to call on every startup — already-applied steps are skipped."""
    conn = connect()
    try:
        current = conn.execute("PRAGMA user_version").fetchone()[0]
        for version, step in _MIGRATIONS:
            if current < version:
                step(conn)
                # PRAGMA can't be parameterised; version is a trusted int literal.
                conn.execute(f"PRAGMA user_version = {version}")
                conn.commit()
                current = version
    finally:
        conn.close()
