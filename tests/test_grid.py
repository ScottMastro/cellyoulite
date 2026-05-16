from __future__ import annotations

from pathlib import Path

from cellyoulite.io.grid import discover_grid


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def test_discover_grid_subfolders(tmp_path: Path):
    _touch(tmp_path / "DMSO r1" / "CA1_B9_1_00d00h00m.tif")
    _touch(tmp_path / "DMSO r1" / "CA1_B9_1_00d00h15m.tif")
    _touch(tmp_path / "DMSO r2" / "CA1_B10_1_00d01h30m.tif")
    _touch(tmp_path / "012+S9A13 r1" / "CA1_D1_1_00d00h00m.tif")
    _touch(tmp_path / "012 +S9A13 r2" / "CA1_D2_1_00d00h00m.tif")  # stray space before +
    _touch(tmp_path / "DMSO r1" / "notes.txt")  # ignored

    spec = discover_grid(tmp_path)
    assert spec.treatments == ["012+S9A13", "DMSO"]
    assert spec.replicates == [1, 2]
    assert spec.n_images == 5

    dmso_r1 = next(w for w in spec.wells if w.treatment == "DMSO" and w.replicate == 1)
    assert [tp.minutes for tp in dmso_r1.timepoints] == [0, 15]
    assert dmso_r1.timepoints[0].key == "DMSO r1/CA1_B9_1_00d00h00m.tif"
