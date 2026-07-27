"""Migration tests.

Most of these cover v3 — the one step that rebuilds a table rather than adding
to it. `well` is rebuilt to move its key from UNIQUE(folder_name) to
UNIQUE(batch_id, folder_name), and alignment/track/well_validation all cascade
on well.id, so a mistake there silently destroys authored curation. These tests
pin that it doesn't.

The rest cover v4, which moves hand-drawn ground truth into the database so it
is batch-scoped and backed up with everything else."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from cellyoulite.db import migrate as mig
from cellyoulite.db import repo

_LATEST = mig._MIGRATIONS[-1][0]

# Row counts the fixture below creates, per guarded table.
_EXPECTED = {
    "well": 2, "alignment": 2, "track": 4, "detection": 6,
    "track_filter": 3, "well_validation": 1, "app_user": 2,
}


def _build_v2(path: Path) -> None:
    """A database at user_version 2, populated the way production is: two
    wells, four organoids, and authored curation hanging off both."""
    conn = sqlite3.connect(path)
    mig._v1(conn)
    mig._v2(conn)
    conn.execute("PRAGMA user_version = 2")

    conn.executemany(
        "INSERT INTO well(id,folder_name,treatment,replicate,first_seen) "
        "VALUES(?,?,?,?,?)",
        [(1, "DMSO r1", "DMSO", 1, "2026-05-16T00:00:00+00:00"),
         (2, "FSK r1", "FSK", 1, "2026-05-16T00:00:00+00:00")])
    conn.executemany(
        "INSERT INTO alignment(well_id,fingerprint,cache_version,canvas_h,"
        "canvas_w,offsets,placements,created_at) VALUES(?,?,?,?,?,?,?,?)",
        [(1, "fp1", 1, 100, 100, "[]", "[]", "2026-05-16T00:00:00+00:00"),
         (2, "fp2", 1, 100, 100, "[]", "[]", "2026-05-16T00:00:00+00:00")])
    conn.executemany(
        "INSERT INTO track(id,well_id,track_num,n_detections,first_t,last_t,"
        "auto_valid,edge_clipped,starred) VALUES(?,?,?,?,?,?,?,?,?)",
        [(1, 1, 1, 2, 0, 1, 1, 0, 1),   # starred
         (2, 1, 2, 1, 0, 0, 1, 0, 0),
         (3, 2, 1, 2, 0, 1, 0, 1, 0),
         (4, 2, 2, 1, 0, 0, 1, 0, 1)])  # starred
    conn.executemany(
        "INSERT INTO detection(track_id,t_idx,label,cx,cy,r) VALUES(?,?,?,?,?,?)",
        [(1, 0, "00d00h00m", 10.0, 10.0, 5.0),
         (1, 1, "00d00h15m", 11.0, 10.0, 5.0),
         (2, 0, "00d00h00m", 20.0, 20.0, 6.0),
         (3, 0, "00d00h00m", 30.0, 30.0, 7.0),
         (3, 1, "00d00h15m", 31.0, 30.0, 7.0),
         (4, 0, "00d00h00m", 40.0, 40.0, 8.0)])
    conn.executemany(
        "INSERT INTO track_filter(track_id,accepted,decided_by,decided_at,reason) "
        "VALUES(?,?,?,?,?)",
        [(1, 1, "scott", "2026-07-20T00:00:00+00:00", "manual"),
         (2, 0, "scott", "2026-07-20T00:00:00+00:00", "manual"),
         (3, 0, "auto", "2026-07-20T00:00:00+00:00", "min_frac")])
    conn.execute(
        "INSERT INTO well_validation(well_id,validated,validated_by,validated_at) "
        "VALUES(1,1,'scott','2026-06-03T00:00:00+00:00')")
    conn.executemany("INSERT INTO app_user(username,created_at) VALUES(?,?)",
                     [("scott", "2026-05-16T00:00:00+00:00"),
                      ("kim", "2026-05-16T00:00:00+00:00")])
    conn.commit()
    conn.close()


@pytest.fixture
def v2_db(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "cellyoulite.db"
    _build_v2(path)
    monkeypatch.setenv("CELLYOULITE_DB", str(path))
    return path


def _counts(path: Path) -> dict[str, int]:
    conn = sqlite3.connect(path)
    try:
        return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                for t in mig._GUARDED_TABLES}
    finally:
        conn.close()


def _open(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def test_v3_preserves_every_row(v2_db):
    assert _counts(v2_db) == _EXPECTED
    mig.migrate()
    assert _counts(v2_db) == _EXPECTED

    conn = _open(v2_db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == _LATEST
    assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not conn.execute("PRAGMA foreign_key_check").fetchall()
    conn.close()


def test_v3_preserves_well_ids_so_children_still_resolve(v2_db):
    mig.migrate()
    conn = _open(v2_db)
    assert dict(conn.execute("SELECT id, folder_name FROM well")) == {
        1: "DMSO r1", 2: "FSK r1"}
    # Authored curation must still be reachable through the rebuilt table.
    assert conn.execute(
        "SELECT COUNT(*) FROM track_filter f JOIN track t ON t.id=f.track_id "
        "JOIN well w ON w.id=t.well_id").fetchone()[0] == 3
    assert conn.execute(
        "SELECT COUNT(*) FROM well_validation v JOIN well w ON w.id=v.well_id"
    ).fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM track WHERE starred=1").fetchone()[0] == 2
    conn.close()


def test_v3_puts_existing_wells_in_one_named_batch(v2_db):
    mig.migrate()
    conn = _open(v2_db)
    assert conn.execute("SELECT name FROM batch").fetchall() == [("CA1 (May 2026)",)]
    assert conn.execute(
        "SELECT COUNT(*) FROM well WHERE batch_id=(SELECT id FROM batch)"
    ).fetchone()[0] == 2
    conn.close()


def test_v3_scopes_folder_name_to_its_batch(v2_db):
    """The point of the migration: a second batch may reuse "DMSO r1"."""
    mig.migrate()
    conn = _open(v2_db)
    conn.execute("INSERT INTO batch(name,created_at) VALUES('2026-07-22','t')")
    b2 = conn.execute("SELECT id FROM batch WHERE name='2026-07-22'").fetchone()[0]

    conn.execute("INSERT INTO well(batch_id,folder_name,first_seen) "
                 "VALUES(?,'DMSO r1','t')", (b2,))
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO well(batch_id,folder_name,first_seen) "
                     "VALUES(?,'DMSO r1','t')", (b2,))
    conn.close()


def test_v3_rejects_a_well_with_no_real_batch(v2_db):
    mig.migrate()
    conn = _open(v2_db)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO well(batch_id,folder_name,first_seen) "
                     "VALUES(9999,'ghost','t')")
    conn.close()


def test_v3_keeps_the_well_cascade_wired(v2_db):
    mig.migrate()
    conn = _open(v2_db)
    conn.execute("DELETE FROM well WHERE id=1")
    assert conn.execute(
        "SELECT COUNT(*) FROM track WHERE well_id=1").fetchone()[0] == 0
    assert conn.execute(
        "SELECT COUNT(*) FROM detection WHERE track_id IN (1,2)").fetchone()[0] == 0
    conn.close()


def test_v3_is_idempotent(v2_db):
    mig.migrate()
    mig.migrate()
    mig.migrate()
    conn = _open(v2_db)
    assert conn.execute("SELECT COUNT(*) FROM batch").fetchone()[0] == 1
    conn.close()
    assert _counts(v2_db) == _EXPECTED


def test_v4_stores_annotations_against_the_well(v2_db, monkeypatch):
    """Ground truth used to be files keyed by well name alone. It now hangs off
    well_id, so the same well name in two batches keeps its own circles."""
    monkeypatch.setenv("CELLYOULITE_DB", str(v2_db))
    mig.migrate()

    repo.set_annotation("CA1 (May 2026)", "DMSO r1", "00d00h00m",
                        [{"cx": 10.0, "cy": 20.0, "r": 5.0, "star": True},
                         {"cx": 30.0, "cy": 40.0, "r": 6.0}],
                        aligned=True, source_key="m_x/b/DMSO r1/a.tif",
                        image_w=100, image_h=80)
    repo.set_annotation("2026-07-22", "DMSO r1", "00d00h00m",
                        [{"cx": 99.0, "cy": 99.0, "r": 9.0}])

    may = repo.get_annotation("CA1 (May 2026)", "DMSO r1", "00d00h00m")
    jul = repo.get_annotation("2026-07-22", "DMSO r1", "00d00h00m")
    assert [c["cx"] for c in may["circles"]] == [10.0, 30.0]   # draw order kept
    assert may["circles"][0]["star"] is True
    assert may["aligned"] is True and may["image_w"] == 100
    assert [c["cx"] for c in jul["circles"]] == [99.0]
    assert repo.get_annotation("CA1 (May 2026)", "FSK r1", "00d00h00m") is None


def test_v4_annotation_write_replaces_only_that_frame(v2_db, monkeypatch):
    monkeypatch.setenv("CELLYOULITE_DB", str(v2_db))
    mig.migrate()
    b = "CA1 (May 2026)"
    repo.set_annotation(b, "DMSO r1", "00d00h00m", [{"cx": 1.0, "cy": 1.0, "r": 1.0}])
    repo.set_annotation(b, "DMSO r1", "00d00h15m", [{"cx": 2.0, "cy": 2.0, "r": 2.0}])
    # Re-drawing one frame must not disturb the other.
    repo.set_annotation(b, "DMSO r1", "00d00h00m", [{"cx": 7.0, "cy": 7.0, "r": 7.0},
                                                    {"cx": 8.0, "cy": 8.0, "r": 8.0}])
    assert len(repo.get_annotation(b, "DMSO r1", "00d00h00m")["circles"]) == 2
    assert repo.get_annotation(b, "DMSO r1", "00d00h15m")["circles"][0]["cx"] == 2.0
    assert len(repo.list_annotations(b)) == 2
    assert len(repo.list_annotations("2026-07-22")) == 0


def test_v4_annotations_cascade_with_their_well(v2_db, monkeypatch):
    monkeypatch.setenv("CELLYOULITE_DB", str(v2_db))
    mig.migrate()
    repo.set_annotation("CA1 (May 2026)", "DMSO r1", "00d00h00m",
                        [{"cx": 1.0, "cy": 1.0, "r": 1.0}])
    conn = _open(v2_db)
    conn.execute("DELETE FROM well WHERE folder_name='DMSO r1'")
    assert conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM annotation_circle").fetchone()[0] == 0
    conn.close()


def test_v3_rolls_back_if_the_row_count_guard_trips(v2_db, monkeypatch):
    """The guard is the whole safety story — prove it actually aborts, and
    that a failed migration leaves the database untouched at v2."""
    calls = {"n": 0}
    real = mig._row_counts

    def flaky(conn):
        calls["n"] += 1
        counts = real(conn)
        if calls["n"] > 1:          # pretend the rebuild lost a well
            counts["well"] -= 1
        return counts

    monkeypatch.setattr(mig, "_row_counts", flaky)
    with pytest.raises(RuntimeError, match="row counts"):
        mig.migrate()

    conn = _open(v2_db)
    assert conn.execute("PRAGMA user_version").fetchone()[0] == 2
    assert not conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE name='batch'").fetchone()[0]
    conn.close()
    assert _counts(v2_db) == _EXPECTED
