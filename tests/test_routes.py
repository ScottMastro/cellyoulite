"""Smoke-test every GET route against a small two-batch dataset.

These exist because batch scoping threaded a new argument through most of the
server, and a missed call site (`_load_tracks(well.folder_name)`) only showed
up as a 500 in the browser. Nothing here checks rendering quality — the point
is that every endpoint answers, for both batches, with a well name that exists
in both."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from skimage.io import imsave

MAY, JUL = "CA1 T1", "CA1 T2"
# "DMSO r1" exists in both batches — the collision the whole feature is about.
WELL, LABELS = "DMSO r1", ["00d00h00m", "00d00h15m"]


def _frame(seed: int) -> np.ndarray:
    """A small image with a couple of blobs, so alignment has something to
    correlate on and the pipeline isn't handed uniform noise."""
    rng = np.random.default_rng(seed)
    img = np.full((64, 80), 200, dtype=np.uint8)
    img[rng.integers(0, 40):][:8, 10:18] = 40
    img[20:30, 40:50] = 60
    return img


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / "data"
    # May names its own replicate; July is a plate folder holding positions.
    for i, label in enumerate(LABELS):
        p = data / MAY / "DMSO r1"
        p.mkdir(parents=True, exist_ok=True)
        imsave(p / f"CA1_B9_1_{label}.tif", _frame(i), check_contrast=False)
        p = data / JUL / "DMSO"
        p.mkdir(parents=True, exist_ok=True)
        imsave(p / f"2026_7_22_B2_1_{label}.tif", _frame(i + 10), check_contrast=False)

    monkeypatch.setenv("CELLYOULITE_DATA", str(data))
    monkeypatch.setenv("CELLYOULITE_DB", str(tmp_path / "cellyoulite.db"))
    monkeypatch.chdir(tmp_path)          # tracks/ and caches are cwd-relative

    import importlib

    from cellyoulite.db.migrate import migrate
    migrate()
    # app.py resolves several cache roots at import time against the cwd.
    from cellyoulite.server import app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


def _mount(client) -> str:
    return client.get("/api/grid").json()["mounts"][0]["id"]


def test_both_batches_are_discovered(client):
    body = client.get("/api/grid").json()
    assert sorted(body["batches"]) == sorted([JUL, MAY])
    assert body["batch_counts"] == {MAY: 1, JUL: 1}
    # Same well name in both, kept apart by batch.
    assert sorted((w["batch"], w["folder_name"]) for w in body["wells"]) == sorted(
        [(MAY, WELL), (JUL, WELL)])


@pytest.mark.parametrize("batch", [MAY, JUL])
@pytest.mark.parametrize("path", [
    "/api/well", "/api/well-align", "/api/tracks", "/api/track-validation",
    "/api/well-growth",
])
def test_per_well_routes_answer_for_each_batch(client, path, batch):
    r = client.get(path, params={"mount_id": _mount(client), "batch": batch,
                                 "folder_name": WELL})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("path", [
    "/api/batches", "/api/grid", "/api/version", "/api/data-dir", "/api/users",
    "/api/track-status", "/api/align-status", "/api/cellpose-status",
    "/api/boxplot-data", "/api/growth-csv", "/healthz",
])
def test_global_routes_answer(client, path):
    assert client.get(path).status_code == 200


@pytest.mark.parametrize("batch", [MAY, JUL])
def test_batch_filtered_routes_answer(client, batch):
    for path in ("/api/grid", "/api/boxplot-data", "/api/growth-csv"):
        r = client.get(path, params={"batch": batch})
        assert r.status_code == 200, f"{path}: {r.text}"


@pytest.mark.parametrize("batch", [MAY, JUL])
def test_key_based_routes_answer_for_each_batch(client, batch):
    """The overlay routes parse the batch back out of the image key. This is
    where the missed argument hid: they only fail once actually requested."""
    body = client.get("/api/well", params={
        "mount_id": _mount(client), "batch": batch, "folder_name": WELL}).json()
    key = body["timepoints"][0]["key"]
    assert key.split("/")[1] == batch          # <mount>/<batch>/<folder>/<file>

    assert client.get("/api/image", params={"key": key}).status_code == 200
    assert client.get("/api/image", params={"key": key, "aligned": 1}).status_code == 200
    assert client.get("/api/image", params={"key": key, "thumb": 64}).status_code == 200
    # No cellpose cache in this fixture, so these answer with empty results
    # rather than data — a 500 is the failure being guarded against.
    assert client.get("/api/cellpose-edges", params={"key": key}).status_code == 200
    assert client.get("/api/cellpose-circles", params={"key": key}).status_code == 200


@pytest.mark.parametrize("batch", [MAY, JUL])
def test_annotations_round_trip_per_batch(client, batch):
    q = {"batch": batch, "well": WELL, "label": LABELS[0]}
    assert client.get("/api/annotations", params=q).json()["exists"] is False
    r = client.post("/api/annotations", params=q,
                    json={"circles": [{"cx": 1.0, "cy": 2.0, "r": 3.0, "star": True}]})
    assert r.status_code == 200 and r.json()["n"] == 1
    got = client.get("/api/annotations", params=q).json()
    assert got["exists"] is True and got["circles"][0]["star"] is True


def test_annotations_do_not_leak_between_batches(client):
    q = {"well": WELL, "label": LABELS[0]}
    client.post("/api/annotations", params={**q, "batch": MAY},
                json={"circles": [{"cx": 1.0, "cy": 2.0, "r": 3.0}]})
    assert client.get("/api/annotations",
                      params={**q, "batch": JUL}).json()["exists"] is False


def test_unknown_batch_is_a_404_not_a_crash(client):
    r = client.get("/api/well", params={"mount_id": _mount(client),
                                        "batch": "nope", "folder_name": WELL})
    assert r.status_code == 404


def _seed_organoid(client, batch):
    """One organoid with a detection in each frame, plus a mask sidecar so the
    stitch renderer has something to crop."""
    import cv2
    import numpy as np

    from cellyoulite.db import repo
    dets = [{"t_idx": i, "label": lab, "cx": 40.0, "cy": 32.0, "r": 9.0,
             "area_px": 254} for i, lab in enumerate(LABELS)]
    repo.set_tracks(batch, WELL, {"n_frames": len(LABELS), "tracks": [
        {"id": 1, "n_detections": len(dets), "first_t": 0,
         "last_t": len(dets) - 1, "valid": True, "detections": dets}]})
    cache = Path(".cellpose_cache") / batch / WELL
    cache.mkdir(parents=True, exist_ok=True)
    mask = np.zeros((64, 80), dtype=np.uint16)
    mask[24:41, 32:49] = 1
    for lab in LABELS:
        cv2.imwrite(str(cache / f"{lab}.mask.png"), mask)


@pytest.mark.parametrize("batch", [MAY, JUL])
def test_stitch_thumbnail_is_small_and_cached(client, batch):
    """A miss used to ignore `thumb` and ship the full-resolution strip, then
    re-render it on every view. Both are load-bearing for the organoid list,
    which renders one row per organoid."""
    _seed_organoid(client, batch)
    q = {"mount_id": _mount(client), "batch": batch, "folder_name": WELL,
         "track_id": 1, "variant": "both"}

    thumbed = client.get("/api/track-stitch", params={**q, "thumb": 128})
    assert thumbed.status_code == 200, thumbed.text
    full = client.get("/api/track-stitch", params=q)
    assert full.status_code == 200

    # The thumbnail must not be the full-resolution strip.
    assert len(thumbed.content) < len(full.content)
    # And the render must have been persisted, not repeated.
    strip = Path("tracks") / batch / WELL / "track_1_both.png"
    assert strip.is_file()
    again = client.get("/api/track-stitch", params={**q, "thumb": 128})
    assert again.content == thumbed.content
