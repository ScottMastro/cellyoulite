from __future__ import annotations

from pathlib import Path

from cellyoulite.io.grid import discover_grid, is_experiment_dir


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"")


def _plate(root: Path, folder: str, positions: list[str], labels: list[str]) -> None:
    """A plate-export folder: one condition, several positions, shared clock."""
    for pos in positions:
        for label in labels:
            _touch(root / folder / f"2026_7_22_{pos}_1_{label}.tif")


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
    # The folder names the replicate, so it stays a single well.
    assert dmso_r1.source_dir == "DMSO r1"
    assert dmso_r1.position == "CA1_B9"


def test_plate_folder_splits_positions_into_replicates(tmp_path: Path):
    """A folder named for the condition alone holds several plate positions;
    each becomes its own well, numbered in position order."""
    _plate(tmp_path, "008", ["E2", "E3", "E4"], ["00d00h00m", "00d00h15m"])

    spec = discover_grid(tmp_path)
    assert spec.treatments == ["008"]
    assert spec.replicates == [1, 2, 3]
    assert spec.n_images == 6

    assert [(w.folder_name, w.position) for w in spec.wells] == [
        ("008 r1", "2026_7_22_E2"),
        ("008 r2", "2026_7_22_E3"),
        ("008 r3", "2026_7_22_E4"),
    ]
    # folder_name is a label; the real directory is still on source_dir, and
    # every key remains a usable path relative to the root.
    for w in spec.wells:
        assert w.source_dir == "008"
        assert len(w.timepoints) == 2
        for tp in w.timepoints:
            assert tp.key.startswith("008/")
            assert (tmp_path / tp.key).exists()


def test_plate_folders_yield_unique_well_names(tmp_path: Path):
    """Uneven position counts still number cleanly, and nothing collides."""
    _plate(tmp_path, "DMSO", ["B2", "B3"], ["00d00h00m"])
    _plate(tmp_path, "FSK", ["B4", "B5", "B6", "B7"], ["00d00h00m"])

    spec = discover_grid(tmp_path)
    names = [w.folder_name for w in spec.wells]
    assert names == ["DMSO r1", "DMSO r2", "FSK r1", "FSK r2", "FSK r3", "FSK r4"]
    assert len(set(names)) == len(names)


def test_both_layouts_coexist_in_one_root(tmp_path: Path):
    _touch(tmp_path / "DMSO r1" / "CA1_B9_1_00d00h00m.tif")
    _plate(tmp_path, "008", ["E2", "E3"], ["00d00h00m"])

    spec = discover_grid(tmp_path)
    assert [(w.folder_name, w.source_dir) for w in spec.wells] == [
        ("008 r1", "008"), ("008 r2", "008"), ("DMSO r1", "DMSO r1"),
    ]


def test_result_dirs_are_not_mistaken_for_wells(tmp_path: Path):
    """Pipeline output sits beside the data and must never register as a well
    — .cellpose_cache in particular holds timestamp-named PNGs."""
    _plate(tmp_path, "008", ["E2"], ["00d00h00m"])
    _touch(tmp_path / ".cellpose_cache" / "008" / "00d00h00m.mask.png")
    _touch(tmp_path / "tracks" / "008.json")
    _touch(tmp_path / "annotations" / "008" / "00d00h00m.json")

    assert not is_experiment_dir(tmp_path / ".cellpose_cache")
    assert not is_experiment_dir(tmp_path / "tracks")
    assert not is_experiment_dir(tmp_path / "annotations")
    assert [w.folder_name for w in discover_grid(tmp_path).wells] == ["008 r1"]


def test_unrecognised_stems_still_yield_one_well(tmp_path: Path):
    """No parseable position — fall back to a single series for the folder."""
    _touch(tmp_path / "sim" / "sim_00d00h00m.png")
    _touch(tmp_path / "sim" / "sim_00d00h15m.png")

    spec = discover_grid(tmp_path)
    assert [(w.folder_name, w.replicate, w.position) for w in spec.wells] == [
        ("sim r1", 1, None)]
    assert spec.wells[0].source_dir == "sim"
    assert spec.n_images == 2
