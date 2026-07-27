"""Schema migrations, tracked via PRAGMA user_version — an ordered list of
steps applied on startup. No external migration framework."""
from __future__ import annotations

from pathlib import Path

from .connection import connect
from .connection import now_iso as _now

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


# Tables whose row counts must survive the `well` rebuild in _v3 untouched.
# alignment/track/well_validation cascade on well.id, and detection/track_filter
# cascade on track.id — a dropped well row would silently take all of them.
_GUARDED_TABLES = (
    "well", "alignment", "track", "detection", "track_filter",
    "well_validation", "app_user",
)


def _row_counts(conn) -> dict[str, int]:
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in _GUARDED_TABLES}


def _v3(conn) -> None:
    """Batch scoping: wells belong to a batch, so two datasets can reuse the
    same folder name ("DMSO", "012") without colliding.

    The key moves from UNIQUE(folder_name) to UNIQUE(batch_id, folder_name).
    SQLite cannot alter a UNIQUE constraint in place, so `well` is rebuilt via
    the documented 12-step procedure. This is the one destructive step in the
    migration list: alignment/track/well_validation cascade on well.id, so the
    rebuild preserves id values explicitly, runs with foreign_keys OFF so the
    DROP cannot cascade, and verifies every guarded row count (plus
    foreign_key_check) before committing.
    """
    before = _row_counts(conn)

    # PRAGMAs are no-ops inside a transaction — leave any implicit one first.
    conn.commit()
    conn.execute("PRAGMA foreign_keys=OFF")
    # Stops RENAME from rewriting the child tables' REFERENCES clauses.
    conn.execute("PRAGMA legacy_alter_table=ON")
    try:
        conn.execute("BEGIN")
        conn.execute("""
            CREATE TABLE batch (
              id         INTEGER PRIMARY KEY,
              name       TEXT NOT NULL UNIQUE,   -- e.g. "CA1 (May 2026)"
              created_at TEXT NOT NULL
            )""")
        # Every pre-batch well came from the one May 2026 CA1 plate.
        cur = conn.execute(
            "INSERT INTO batch(name, created_at) VALUES(?, ?)",
            ("CA1 (May 2026)", _now()))
        batch_id = cur.lastrowid

        conn.execute("""
            CREATE TABLE well_new (
              id          INTEGER PRIMARY KEY,
              batch_id    INTEGER NOT NULL REFERENCES batch(id) ON DELETE CASCADE,
              folder_name TEXT NOT NULL,
              treatment   TEXT,
              replicate   INTEGER,
              first_seen  TEXT NOT NULL,
              UNIQUE (batch_id, folder_name)
            )""")
        # id is carried over verbatim — children reference it.
        conn.execute(
            "INSERT INTO well_new(id, batch_id, folder_name, treatment, "
            "replicate, first_seen) "
            "SELECT id, ?, folder_name, treatment, replicate, first_seen "
            "FROM well", (batch_id,))
        conn.execute("DROP TABLE well")
        conn.execute("ALTER TABLE well_new RENAME TO well")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_well_batch ON well(batch_id)")

        after = _row_counts(conn)
        if after != before:
            raise RuntimeError(
                f"v3 would have changed row counts: {before} -> {after}")
        orphans = conn.execute("PRAGMA foreign_key_check").fetchall()
        if orphans:
            raise RuntimeError(f"v3 left {len(orphans)} orphaned row(s)")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("PRAGMA legacy_alter_table=OFF")
        conn.execute("PRAGMA foreign_keys=ON")


# (target_version, step). Each runs once, in order, when user_version < target.
_MIGRATIONS = [
    (1, _v1),
    (2, _v2),
    (3, _v3),
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
