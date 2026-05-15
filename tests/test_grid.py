from __future__ import annotations

from pathlib import Path

from cellyoulite.io.grid import discover_grid


def test_discover_grid(tmp_path: Path):
    for name in ["A01.tif", "A02.tif", "B_01.png", "h-12.jpeg", "notes.txt"]:
        (tmp_path / name).write_bytes(b"")
    spec = discover_grid(tmp_path)
    assert spec.treatments == ["A", "B", "H"]
    assert set(spec.timepoints) == {1, 2, 12}
    assert len(spec.cells) == 4
