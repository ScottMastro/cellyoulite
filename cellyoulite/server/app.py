from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from skimage.io import imread
from skimage.segmentation import find_boundaries
from starlette.requests import Request

from cellyoulite.__version__ import __version__
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline import analyze_folder, fit_circle, passes_qc, segment_image
from cellyoulite.pipeline.circle_fit import fit_circle as _fit_circle
from cellyoulite.pipeline.qc import QCThresholds, passes_qc as _passes_qc
from cellyoulite.pipeline.segment import segment_image as _segment
from cellyoulite.pipeline.align import (
    WellAlignment, compute_alignment_cached, is_alignment_cached, paste_onto_canvas,
)
from cellyoulite.plotting.boxplot import boxplot_by_timepoint, well_lineplot

_WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(_WEB / "templates"))

app = FastAPI(title="CellxYou Lite", version=__version__)
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")


# ---------------------- data folder ----------------------
# Single rooted folder: `$CELLYOULITE_DATA` or `./data` (relative to cwd).
# Internal "mount" abstraction is kept (so keys stay <mount_id>/<rest>) but
# there is now only ever one mount — the data folder.

def _mount_id(path: Path) -> str:
    h = hashlib.sha1(str(path.resolve()).encode()).hexdigest()[:8]
    return f"m_{h}"


def _data_dir() -> Path:
    env = os.environ.get("CELLYOULITE_DATA")
    if env:
        p = Path(env).expanduser()
        if p.is_dir():
            return p.resolve()
    return (Path.cwd() / "data").resolve()


def _mounts() -> list[dict]:
    p = _data_dir()
    if not p.is_dir():
        return []
    return [{"id": _mount_id(p), "path": str(p), "alias": p.name}]


def _mount_by_id(mid: str) -> dict:
    for m in _mounts():
        if m["id"] == mid:
            return m
    raise HTTPException(status_code=404, detail=f"no such mount: {mid}")


# ---------------------- key plumbing ----------------------
# A "key" is "<mount_id>/<well_folder>/<filename>". The mount id is the first
# segment; the remainder is the path within the mount.

def _split_key(key: str) -> tuple[dict, str]:
    if "/" not in key:
        raise HTTPException(status_code=400, detail="malformed key")
    mid, rest = key.split("/", 1)
    return _mount_by_id(mid), rest


def _safe_image_path(key: str) -> Path:
    mount, rest = _split_key(key)
    base = Path(mount["path"]).resolve()
    target = (base / rest).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="key escapes mount")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"no such file: {key}")
    return target


def _wells_for_mount(mount: dict):
    spec = discover_grid(Path(mount["path"]))
    return spec


def _find_well(mount_id: str, folder_name: str):
    mount = _mount_by_id(mount_id)
    spec = _wells_for_mount(mount)
    for w in spec.wells:
        if w.folder_name == folder_name:
            return mount, w
    raise HTTPException(status_code=404, detail="well not found")


_align_cache: dict[tuple[str, str], WellAlignment] = {}


def _well_alignment(mount: dict, well) -> WellAlignment:
    cache_key = (mount["id"], well.folder_name)
    if cache_key not in _align_cache:
        _align_cache[cache_key] = compute_alignment_cached([tp.path for tp in well.timepoints])
    return _align_cache[cache_key]


def _read_aligned(key: str) -> np.ndarray:
    """Read the image and pad it onto its well's union canvas."""
    img = imread(_safe_image_path(key))
    mount, rest = _split_key(key)
    well_folder = rest.split("/", 1)[0] if "/" in rest else None
    if not well_folder:
        return img
    spec = _wells_for_mount(mount)
    well = next((w for w in spec.wells if w.folder_name == well_folder), None)
    if well is None:
        return img
    align = _well_alignment(mount, well)
    inner_key = rest  # well_folder/filename — matches Timepoint.key from grid
    idx = next((i for i, tp in enumerate(well.timepoints) if tp.key == inner_key), None)
    if idx is None or not align.placements:
        return img
    return paste_onto_canvas(img, align.placements[idx], align.canvas_shape)


def _prefixed_key(mount: dict, inner_key: str) -> str:
    return f"{mount['id']}/{inner_key}"


# ---------------------- routes ----------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    resp = templates.TemplateResponse(
        request, "index.html", {"version": __version__}
    )
    resp.headers["Cache-Control"] = "no-store, must-revalidate"
    return resp


@app.get("/api/version")
def version() -> dict:
    return {"version": __version__}


@app.get("/api/data-dir")
def data_dir() -> dict:
    p = _data_dir()
    return {"path": str(p), "exists": p.is_dir()}


@app.post("/api/reveal")
def reveal() -> dict:
    """Open the data folder in the OS file manager."""
    p = _data_dir()
    if not p.is_dir():
        raise HTTPException(status_code=404, detail=f"data folder not found: {p}")
    try:
        if sys.platform.startswith("darwin"):
            subprocess.Popen(["open", str(p)])
        elif os.name == "nt":
            os.startfile(str(p))  # type: ignore[attr-defined]
        else:
            subprocess.Popen(["xdg-open", str(p)])
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=f"no file manager available: {e}")
    return {"ok": True, "path": str(p)}


@app.get("/api/grid")
def grid() -> dict:
    """Walk every mount; return wells from all of them, merged."""
    mounts = _mounts()
    if not mounts:
        return {"treatments": [], "replicates": [], "n_wells": 0, "n_images": 0,
                "wells": [], "mounts": []}

    wells: list[dict] = []
    treatments: set[str] = set()
    replicates: set[int] = set()
    n_images = 0
    for mount in mounts:
        spec = _wells_for_mount(mount)
        treatments.update(spec.treatments)
        replicates.update(spec.replicates)
        n_images += spec.n_images
        for w in spec.wells:
            thumb = w.timepoints[-1] if w.timepoints else None
            wells.append({
                "mount_id": mount["id"],
                "mount_alias": mount["alias"],
                "treatment": w.treatment,
                "replicate": w.replicate,
                "folder_name": w.folder_name,
                "n_timepoints": len(w.timepoints),
                "thumb_key": _prefixed_key(mount, thumb.key) if thumb else None,
                "thumb_label": thumb.label if thumb else None,
            })

    return {
        "treatments": sorted(treatments),
        "replicates": sorted(replicates),
        "n_wells": len(wells),
        "n_images": n_images,
        "wells": wells,
        "mounts": mounts,
    }


@app.get("/api/well")
def well(mount_id: str, folder_name: str) -> dict:
    mount, w = _find_well(mount_id, folder_name)
    return {
        "mount_id": mount["id"],
        "mount_alias": mount["alias"],
        "treatment": w.treatment,
        "replicate": w.replicate,
        "folder_name": w.folder_name,
        "timepoints": [
            {"t_idx": i, "minutes": tp.minutes, "label": tp.label,
             "key": _prefixed_key(mount, tp.key)}
            for i, tp in enumerate(w.timepoints)
        ],
    }


_BROWSER_NATIVE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@app.get("/api/image")
def image(key: str, thumb: int = 0, aligned: int = 0) -> Response:
    """Serve an image. TIFFs (and any non-browser-native format) are
    re-encoded to JPEG. `thumb=<px>` downscales the longest side.
    `aligned=1` pads the frame onto its well's registered canvas."""
    path = _safe_image_path(key)
    if path.suffix.lower() in _BROWSER_NATIVE and not thumb and not aligned:
        return FileResponse(path)

    if aligned:
        img = _read_aligned(key)
    else:
        img = imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)

    if thumb and thumb > 0:
        h, w = bgr.shape[:2]
        scale = thumb / max(h, w)
        if scale < 1.0:
            bgr = cv2.resize(bgr, (max(1, int(w * scale)), max(1, int(h * scale))),
                             interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="jpeg encode failed")
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@app.get("/api/overlay")
def overlay(key: str, aligned: int = 0) -> dict:
    img = _read_aligned(key) if aligned else imread(_safe_image_path(key))
    seg = segment_image(img)

    circles: list[dict] = []
    for idx, region in enumerate(seg.regions):
        if not passes_qc(region):
            continue
        obj_mask = seg.labels == (idx + 1)
        fit = fit_circle(obj_mask)
        circles.append({"cx": fit.cx, "cy": fit.cy, "r": fit.radius})

    h, w = seg.mask.shape
    return {"width": int(w), "height": int(h),
            "circles": circles, "n_passed": len(circles)}


@app.get("/api/edges")
def edges(key: str, aligned: int = 0) -> Response:
    img = _read_aligned(key) if aligned else imread(_safe_image_path(key))
    seg = segment_image(img)

    qc_mask = np.zeros_like(seg.mask, dtype=bool)
    for idx, region in enumerate(seg.regions):
        if passes_qc(region):
            qc_mask |= seg.labels == (idx + 1)

    boundary = find_boundaries(qc_mask, mode="thick")
    h, w = boundary.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[boundary] = [72, 86, 255, 255]
    ok, png = cv2.imencode(".png", bgra)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return Response(content=png.tobytes(), media_type="image/png")


@app.get("/api/well-align")
def well_align(mount_id: str, folder_name: str) -> dict:
    mount, w = _find_well(mount_id, folder_name)
    align = _well_alignment(mount, w)
    return {
        "canvas": list(align.canvas_shape),
        "offsets": [list(o) for o in align.offsets],
        "placements": [list(p) for p in align.placements],
    }


@app.get("/api/well-analyze")
def well_analyze(mount_id: str, folder_name: str) -> JSONResponse:
    import pandas as pd
    _, w = _find_well(mount_id, folder_name)

    qc = QCThresholds()
    rows: list[dict] = []
    for t_idx, tp in enumerate(w.timepoints):
        img = imread(tp.path)
        seg = _segment(img, min_area_px=qc.min_area_px, max_area_px=qc.max_area_px)
        for idx, region in enumerate(seg.regions):
            if not _passes_qc(region, qc):
                continue
            obj_mask = seg.labels == (idx + 1)
            circle = _fit_circle(obj_mask)
            rows.append({
                "t_idx": t_idx,
                "minutes": tp.minutes,
                "label": tp.label,
                "area_px": int(region.area),
                "circle_area_px": circle.area,
            })
    df = pd.DataFrame.from_records(rows)
    return JSONResponse({
        "n_organoids": int(len(df)),
        "plot_html": well_lineplot(df) if len(df) else "",
    })


@app.get("/api/analyze")
def analyze() -> JSONResponse:
    import pandas as pd
    frames = []
    for mount in _mounts():
        df = analyze_folder(Path(mount["path"]))
        if len(df):
            df = df.copy()
            df["mount"] = mount["alias"]
            frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    plot_html = boxplot_by_timepoint(df) if len(df) else ""
    return JSONResponse({
        "n_organoids": int(len(df)),
        "treatments": sorted(df["treatment"].unique().tolist()) if len(df) else [],
        "n_timepoints": int(df["t_idx"].nunique()) if len(df) else 0,
        "median_area_px": float(df["area_px"].median()) if len(df) else None,
        "plot_html": plot_html,
        "rows": df.head(20).to_dict(orient="records"),
    })


# ---------------------- alignment warm-up ----------------------

@app.post("/api/warm-alignments")
def warm_alignments() -> dict:
    """Walk every well in every mount; for each, ensure an up-to-date
    alignment cache file exists on disk. Already-cached wells are O(1)
    (just a file-exists check on the fingerprinted path)."""
    import time
    out: list[dict] = []
    n_recomputed = 0
    for mount in _mounts():
        try:
            spec = _wells_for_mount(mount)
        except Exception as e:  # noqa: BLE001
            out.append({"mount": mount["alias"], "error": str(e)})
            continue
        for w in spec.wells:
            paths = [tp.path for tp in w.timepoints]
            cached = is_alignment_cached(paths)
            t0 = time.perf_counter()
            align = compute_alignment_cached(paths)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            _align_cache[(mount["id"], w.folder_name)] = align
            if not cached:
                n_recomputed += 1
            out.append({
                "mount": mount["alias"],
                "well": w.folder_name,
                "n_timepoints": len(paths),
                "was_cached": cached,
                "ms": elapsed_ms,
            })
    return {"n_wells": len(out), "n_recomputed": n_recomputed, "wells": out}


# ---------------------- cellpose cache status ----------------------

_CELLPOSE_CACHE_ROOT = Path.cwd() / ".cellpose_cache"


def _cellpose_done_labels(well_folder: str) -> set[str]:
    """A frame is 'done' only when BOTH the circles JSON and the mask PNG
    sidecar are present. Older runs (JSON-only) get re-processed by the
    batch script and shouldn't be reported as complete."""
    safe = well_folder.replace("/", "_")
    d = _CELLPOSE_CACHE_ROOT / safe
    if not d.is_dir():
        return set()
    done = set()
    for p in d.glob("*.json"):
        if (d / f"{p.stem}.mask.png").is_file():
            done.add(p.stem)
    return done


# ---------------------- tracking ----------------------

_TRACKS_ROOT = Path.cwd() / "tracks"
_track_job: dict = {
    "proc": None, "started_at": None, "stopped_at": None,
    "returncode": None, "last_log": [], "last_line": "",
}
_track_job_lock = None


@app.get("/api/track-status")
def track_status() -> dict:
    """Per-well: does tracks/<well>.json exist and is it newer than every
    .cellpose_cache/<well>/*.json file under it? If a frame is segmented
    after the tracks file was written, the well counts as stale."""
    rows = []
    n_total = 0; n_done = 0
    for mount in _mounts():
        try:
            spec = _wells_for_mount(mount)
        except Exception:  # noqa: BLE001
            continue
        for w in spec.wells:
            safe = w.folder_name.replace("/", "_")
            track_path = _TRACKS_ROOT / f"{safe}.json"
            cache_dir = _CELLPOSE_CACHE_ROOT / safe
            track_mtime = track_path.stat().st_mtime if track_path.is_file() else None
            stale = False
            if track_mtime is not None and cache_dir.is_dir():
                for f in cache_dir.glob("*.json"):
                    if f.stat().st_mtime > track_mtime:
                        stale = True; break
            done = track_mtime is not None and not stale
            rows.append({
                "mount_id": mount["id"],
                "folder_name": w.folder_name,
                "done": done,
                "stale": stale,
            })
            n_total += 1
            n_done += 1 if done else 0
    return {"wells": rows, "n_total": n_total, "n_done": n_done}


@app.post("/api/track-run")
def track_run(min_frac: float = 0.75) -> dict:
    import subprocess, sys, threading, time as _t
    global _track_job_lock
    if _track_job_lock is None:
        _track_job_lock = threading.Lock()
    with _track_job_lock:
        proc = _track_job["proc"]
        if proc is not None and proc.poll() is None:
            return {"started": False, "running": True, "pid": proc.pid}
        cmd = [sys.executable, "-u", "scripts/cellpose_track.py",
               "--min-frac", str(min_frac)]
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(Path.cwd()),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"spawn failed: {e}")
        _track_job.update({
            "proc": proc, "started_at": _t.time(), "stopped_at": None,
            "returncode": None, "last_log": [], "last_line": "starting…",
        })
        threading.Thread(target=_make_reader(_track_job, _track_job_lock),
                         args=(proc,), daemon=True).start()
    return {"started": True, "running": True, "pid": proc.pid}


@app.post("/api/track-cancel")
def track_cancel() -> dict:
    if _track_job_lock is None:
        return {"running": False, "stopped": False}
    with _track_job_lock:
        proc = _track_job["proc"]
        if proc is None or proc.poll() is not None:
            return {"running": False, "stopped": False}
        try:
            proc.terminate()
        except OSError:
            pass
    return {"running": False, "stopped": True}


@app.get("/api/track-job")
def track_job() -> dict:
    if _track_job_lock is None:
        return {"running": False, "pid": None, "started_at": None,
                "stopped_at": None, "returncode": None,
                "last_line": "", "last_log": []}
    with _track_job_lock:
        proc = _track_job["proc"]
        running = bool(proc and proc.poll() is None)
        return {
            "running": running,
            "pid": proc.pid if proc else None,
            "started_at": _track_job["started_at"],
            "stopped_at": _track_job["stopped_at"],
            "returncode": _track_job["returncode"],
            "last_line": _track_job["last_line"],
            "last_log": list(_track_job["last_log"]),
        }


@app.get("/api/tracks")
def tracks_for_well(mount_id: str, folder_name: str) -> dict:
    """The full tracks JSON for one well, used by the viewer to colour
    overlays and to plot growth curves."""
    safe = folder_name.replace("/", "_")
    path = _TRACKS_ROOT / f"{safe}.json"
    if not path.is_file():
        return {"available": False, "tracks": [], "n_frames": 0}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"available": False, "tracks": [], "n_frames": 0}
    data["available"] = True
    return data


@app.get("/api/track-stitch")
def track_stitch(mount_id: str, folder_name: str, track_id: int,
                  pad: float = 1.4) -> Response:
    """For one track, crop a square region around its centre in every frame
    where it appears and concatenate horizontally into a wide PNG. The crop
    side equals 2 * (largest_r in this track) * pad, so the organoid stays
    fully visible even in its biggest frame. Coordinates are aligned-canvas."""
    _, well = _find_well(mount_id, folder_name)
    safe = folder_name.replace("/", "_")
    path = _TRACKS_ROOT / f"{safe}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no tracks file for well")
    data = json.loads(path.read_text())
    track = next((t for t in data.get("tracks", []) if t["id"] == track_id), None)
    if track is None:
        raise HTTPException(status_code=404, detail=f"no track {track_id}")
    dets = track.get("detections", [])
    if not dets:
        raise HTTPException(status_code=404, detail="track has no detections")

    # Crop size — anchored on the largest detection in this track so the
    # organoid never gets clipped, with `pad` of slack for context.
    largest = max(dets, key=lambda d: d.get("r", 0))
    side = int(round(2 * largest["r"] * pad))
    side = max(side, 32)  # minimum crop so tiny early-life tracks are visible

    # Load the aligned frame for every detection and crop.
    align = _well_alignment(*_find_well(mount_id, folder_name))
    panels: list[np.ndarray] = []
    label_strip_h = 18
    sep_w = 2
    for d in dets:
        tp = next((t for t in well.timepoints if t.label == d["label"]), None)
        if tp is None:
            continue
        # Use the cached _read_aligned helper via building the key.
        key = f"{mount_id}/{well.folder_name}/{tp.path.name}"
        try:
            frame = _read_aligned(key)
        except HTTPException:
            continue
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        H, W = frame.shape[:2]
        cx, cy = int(round(d["cx"])), int(round(d["cy"]))
        x0 = max(0, cx - side // 2); x1 = min(W, x0 + side)
        y0 = max(0, cy - side // 2); y1 = min(H, y0 + side)
        crop = frame[y0:y1, x0:x1, :3]
        # Pad to (side, side) if we hit a border.
        if crop.shape[0] != side or crop.shape[1] != side:
            full = np.zeros((side, side, 3), dtype=np.uint8)
            full[:crop.shape[0], :crop.shape[1]] = crop
            crop = full
        # Label strip on top.
        strip = np.zeros((label_strip_h, side, 3), dtype=np.uint8)
        cv2.putText(strip, d["label"], (4, 13), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
        panel = np.vstack([strip, crop])
        panels.append(panel)

    if not panels:
        raise HTTPException(status_code=404, detail="no panels rendered")

    sep = np.full((panels[0].shape[0], sep_w, 3), 36, dtype=np.uint8)
    pieces = []
    for i, p in enumerate(panels):
        if i > 0:
            pieces.append(sep)
        pieces.append(p)
    big = np.concatenate(pieces, axis=1)
    # Header strip with track meta.
    hdr = np.zeros((22, big.shape[1], 3), dtype=np.uint8)
    text = (f"track {track_id}  ·  n={len(dets)}  ·  "
            f"crop={side}px (largest r={largest['r']:.0f})")
    cv2.putText(hdr, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                (220, 220, 220), 1, cv2.LINE_AA)
    big = np.vstack([hdr, big])

    bgr = cv2.cvtColor(big, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return Response(content=buf.tobytes(), media_type="image/png")


@app.get("/api/well-growth")
def well_growth(mount_id: str, folder_name: str, valid_only: int = 1) -> dict:
    """Per-track normalised area-over-time, ready for plotting.
    Returns {tracks: [{id, valid, color_index, points: [{minutes, area_px,
    norm}, ...]}, ...]}. norm = area_px / area_px_at_first_detection."""
    _, well = _find_well(mount_id, folder_name)
    safe = folder_name.replace("/", "_")
    path = _TRACKS_ROOT / f"{safe}.json"
    if not path.is_file():
        return {"available": False, "tracks": []}
    data = json.loads(path.read_text())
    tp_minutes = {tp.label: tp.minutes for tp in well.timepoints}
    out: list[dict] = []
    for t in data.get("tracks", []):
        if valid_only and not t.get("valid"):
            continue
        dets = t.get("detections", [])
        if not dets:
            continue
        a0 = dets[0].get("area_px")
        if not a0:
            continue
        points = []
        for d in dets:
            mn = tp_minutes.get(d["label"])
            ap = d.get("area_px")
            if mn is None or ap is None:
                continue
            points.append({"minutes": int(mn), "area_px": int(ap),
                            "norm": float(ap) / float(a0)})
        if not points:
            continue
        out.append({
            "id": t["id"], "valid": bool(t.get("valid")),
            "n_detections": t.get("n_detections", len(points)),
            "points": points,
        })
    return {"available": True, "tracks": out,
            "n_frames": data.get("n_frames", 0)}


# ---------------------- alignment run / cancel ----------------------

_align_job: dict = {
    "proc": None, "started_at": None, "stopped_at": None,
    "returncode": None, "last_log": [], "last_line": "",
}
_align_job_lock = None  # set after threading import below


def _make_reader(state: dict, lock):
    def _reader(proc):
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode(errors="replace").rstrip()
            if not line:
                continue
            with lock:
                state["last_line"] = line
                state["last_log"].append(line)
                if len(state["last_log"]) > 30:
                    state["last_log"].pop(0)
        proc.stdout.close(); proc.wait()
        with lock:
            state["returncode"] = proc.returncode
            import time as _t
            state["stopped_at"] = _t.time()
    return _reader


@app.get("/api/align-status")
def align_status() -> dict:
    """Per-well: is the alignment cache currently valid? Drives the UI bar."""
    rows = []
    n_total = 0; n_done = 0
    for mount in _mounts():
        try:
            spec = _wells_for_mount(mount)
        except Exception:  # noqa: BLE001
            continue
        for w in spec.wells:
            paths = [tp.path for tp in w.timepoints]
            cached = is_alignment_cached(paths)
            rows.append({
                "mount_id": mount["id"],
                "folder_name": w.folder_name,
                "cached": cached,
            })
            n_total += 1
            n_done += 1 if cached else 0
    return {"wells": rows, "n_total": n_total, "n_done": n_done}


@app.post("/api/align-run")
def align_run(force: bool = False) -> dict:
    import subprocess, sys, threading, time as _t
    global _align_job_lock
    if _align_job_lock is None:
        _align_job_lock = threading.Lock()
    with _align_job_lock:
        proc = _align_job["proc"]
        if proc is not None and proc.poll() is None:
            return {"started": False, "running": True, "pid": proc.pid}
        cmd = [sys.executable, "-u", "scripts/align_batch.py"]
        if force:
            cmd.append("--force")
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(Path.cwd()),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"spawn failed: {e}")
        _align_job.update({
            "proc": proc, "started_at": _t.time(), "stopped_at": None,
            "returncode": None, "last_log": [], "last_line": "starting…",
        })
        threading.Thread(target=_make_reader(_align_job, _align_job_lock),
                         args=(proc,), daemon=True).start()
    return {"started": True, "running": True, "pid": proc.pid}


@app.post("/api/align-cancel")
def align_cancel() -> dict:
    if _align_job_lock is None:
        return {"running": False, "stopped": False}
    with _align_job_lock:
        proc = _align_job["proc"]
        if proc is None or proc.poll() is not None:
            return {"running": False, "stopped": False}
        try:
            proc.terminate()
        except OSError:
            pass
    return {"running": False, "stopped": True}


@app.get("/api/align-job")
def align_job() -> dict:
    if _align_job_lock is None:
        return {"running": False, "pid": None, "started_at": None,
                "stopped_at": None, "returncode": None,
                "last_line": "", "last_log": []}
    with _align_job_lock:
        proc = _align_job["proc"]
        running = bool(proc and proc.poll() is None)
        return {
            "running": running,
            "pid": proc.pid if proc else None,
            "started_at": _align_job["started_at"],
            "stopped_at": _align_job["stopped_at"],
            "returncode": _align_job["returncode"],
            "last_line": _align_job["last_line"],
            "last_log": list(_align_job["last_log"]),
        }


# ---------------------- cellpose run / cancel ----------------------

import subprocess
import sys
import threading
import time as _time

_cellpose_job: dict = {
    "proc": None,         # type: subprocess.Popen | None
    "started_at": None,   # epoch seconds
    "stopped_at": None,
    "returncode": None,
    "last_log": [],       # last ~30 lines of stdout, for the UI
    "last_line": "",
}
_cellpose_lock = threading.Lock()


def _reader_thread(proc):
    """Drain the subprocess stdout into _cellpose_job['last_log'] so the UI
    can show the most recent line and a small tail buffer."""
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode(errors="replace").rstrip()
        if not line:
            continue
        with _cellpose_lock:
            _cellpose_job["last_line"] = line
            _cellpose_job["last_log"].append(line)
            if len(_cellpose_job["last_log"]) > 30:
                _cellpose_job["last_log"].pop(0)
    proc.stdout.close()
    proc.wait()
    with _cellpose_lock:
        _cellpose_job["returncode"] = proc.returncode
        _cellpose_job["stopped_at"] = _time.time()


@app.post("/api/cellpose-run")
def cellpose_run(gpu: bool = True, workers: int = 4, force: bool = False) -> dict:
    """Kick off the cellpose batch as a subprocess. Returns immediately;
    the UI polls /api/cellpose-job for progress."""
    with _cellpose_lock:
        proc = _cellpose_job["proc"]
        if proc is not None and proc.poll() is None:
            return {"started": False, "running": True,
                    "pid": proc.pid, "reason": "already running"}

        cmd = [sys.executable, "-u", "scripts/cellpose_batch.py"]
        if gpu:
            cmd.append("--gpu")
        else:
            cmd += ["--workers", str(workers)]
        if force:
            cmd.append("--force")
        try:
            proc = subprocess.Popen(
                cmd, cwd=str(Path.cwd()),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except OSError as e:
            raise HTTPException(status_code=500, detail=f"spawn failed: {e}")
        _cellpose_job.update({
            "proc": proc, "started_at": _time.time(),
            "stopped_at": None, "returncode": None,
            "last_log": [], "last_line": "starting…",
        })
        threading.Thread(target=_reader_thread, args=(proc,), daemon=True).start()
        return {"started": True, "running": True, "pid": proc.pid,
                "cmd": " ".join(cmd)}


@app.post("/api/cellpose-cancel")
def cellpose_cancel() -> dict:
    with _cellpose_lock:
        proc = _cellpose_job["proc"]
        if proc is None or proc.poll() is not None:
            return {"running": False, "stopped": False}
        try:
            proc.terminate()
        except OSError:
            pass
    return {"running": False, "stopped": True}


@app.get("/api/cellpose-job")
def cellpose_job() -> dict:
    with _cellpose_lock:
        proc = _cellpose_job["proc"]
        running = bool(proc and proc.poll() is None)
        return {
            "running": running,
            "pid": proc.pid if proc else None,
            "started_at": _cellpose_job["started_at"],
            "stopped_at": _cellpose_job["stopped_at"],
            "returncode": _cellpose_job["returncode"],
            "last_line": _cellpose_job["last_line"],
            "last_log": list(_cellpose_job["last_log"]),
        }


@app.get("/api/cellpose-edges")
def cellpose_edges(key: str, aligned: int = 1) -> Response:
    """Boundaries of the cached cellpose mask, returned as a transparent
    PNG. aligned=1 returns the full canvas-sized boundary image;
    aligned=0 returns the boundary cropped back to the raw frame so it
    lines up with /api/image (no aligned padding)."""
    mount, rest = _split_key(key)
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="bad key")
    well_folder, filename = parts
    spec = _wells_for_mount(mount)
    well = next((w for w in spec.wells if w.folder_name == well_folder), None)
    if well is None:
        raise HTTPException(status_code=404, detail="well not found")
    t_idx, tp = _well_tp_index(mount, well, filename)
    if tp is None:
        raise HTTPException(status_code=404, detail="timepoint not found")
    safe_well = well_folder.replace("/", "_")
    mask_path = _CELLPOSE_CACHE_ROOT / safe_well / f"{tp.label}.mask.png"
    if not mask_path.is_file():
        bgra = np.zeros((1, 1, 4), dtype=np.uint8)
        ok, png = cv2.imencode(".png", bgra)
        return Response(content=png.tobytes(), media_type="image/png")
    masks = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if masks is None:
        raise HTTPException(status_code=500, detail="bad mask png")
    boundary = find_boundaries(masks, mode="thick")

    if not aligned:
        align = _well_alignment(mount, well)
        if align.placements:
            y0, x0 = align.placements[t_idx]
            img = imread(_safe_image_path(key))
            raw_h, raw_w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else boundary.shape
            boundary = boundary[y0:y0 + raw_h, x0:x0 + raw_w]

    h, w = boundary.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    bgra[boundary] = [72, 86, 255, 255]
    ok, png = cv2.imencode(".png", bgra)
    if not ok:
        raise HTTPException(status_code=500, detail="png encode failed")
    return Response(content=png.tobytes(), media_type="image/png")


def _well_tp_index(mount, well, filename):
    for i, t in enumerate(well.timepoints):
        if t.path.name == filename:
            return i, t
    return None, None


@app.get("/api/cellpose-circles")
def cellpose_circles(key: str, aligned: int = 1) -> dict:
    """Return cached Cellpose circles for one frame.
    aligned=1 → coordinates in the aligned-canvas space (matches the
    aligned image served by /api/image?aligned=1).
    aligned=0 → coordinates shifted into the raw image space, and
    circles whose centre falls outside the raw image are dropped."""
    mount, rest = _split_key(key)
    parts = rest.split("/", 1)
    if len(parts) != 2:
        raise HTTPException(status_code=400, detail="bad key")
    well_folder, filename = parts
    spec = _wells_for_mount(mount)
    well = next((w for w in spec.wells if w.folder_name == well_folder), None)
    if well is None:
        raise HTTPException(status_code=404, detail="well not found")
    t_idx, tp = _well_tp_index(mount, well, filename)
    if tp is None:
        raise HTTPException(status_code=404, detail="timepoint not found")
    safe_well = well_folder.replace("/", "_")
    path = _CELLPOSE_CACHE_ROOT / safe_well / f"{tp.label}.json"
    if not path.is_file():
        return {"cached": False, "circles": []}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"cached": False, "circles": []}
    align = _well_alignment(mount, well)
    h, w = align.canvas_shape if align.canvas_shape else (0, 0)
    circles = data.get("circles", [])

    if not aligned and align.placements:
        y0, x0 = align.placements[t_idx]
        # Raw image dimensions — we need them to clip.
        img = imread(_safe_image_path(key))
        raw_h, raw_w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else (0, 0)
        shifted = []
        for c in circles:
            cx2, cy2 = c["cx"] - x0, c["cy"] - y0
            if 0 <= cx2 < raw_w and 0 <= cy2 < raw_h:
                shifted.append({**c, "cx": cx2, "cy": cy2})
        circles = shifted
        w, h = raw_w, raw_h

    return {
        "cached": True,
        "model": data.get("model", "cpsam"),
        "circles": circles,
        "width": int(w),
        "height": int(h),
    }


@app.get("/api/cellpose-status")
def cellpose_status() -> dict:
    """Per-well counts of how many timepoints have a Cellpose cache file on
    disk. The viewer uses this to draw 'X/Y processed' badges."""
    rows = []
    n_total_all = 0
    n_done_all = 0
    for mount in _mounts():
        try:
            spec = _wells_for_mount(mount)
        except Exception:  # noqa: BLE001
            continue
        for w in spec.wells:
            done = _cellpose_done_labels(w.folder_name)
            total_labels = {tp.label for tp in w.timepoints}
            n_done = len(done & total_labels)
            n_total = len(total_labels)
            rows.append({
                "mount_id": mount["id"],
                "folder_name": w.folder_name,
                "n_total": n_total,
                "n_done": n_done,
                "labels_done": sorted(done & total_labels),
            })
            n_total_all += n_total
            n_done_all += n_done
    return {
        "wells": rows,
        "n_total": n_total_all,
        "n_done": n_done_all,
    }


# ---------------------- detector debug page ----------------------

@app.get("/debug", response_class=HTMLResponse)
def debug_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "debug.html", {"version": __version__}
    )


def _colorize_signed(arr: np.ndarray) -> np.ndarray:
    """Render a signed float map as a centred BGR colormap: negative=cool,
    positive=hot. Used for LoG response / flat-field visualisation."""
    a = arr.astype(np.float32)
    finite = a[np.isfinite(a)]
    if finite.size == 0:
        return np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    vmax = max(abs(float(finite.min())), abs(float(finite.max())), 1e-6)
    norm = np.clip((a / vmax + 1.0) * 0.5, 0, 1)
    u8 = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_TURBO)


def _grayscale_rgb(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(np.float32)
    lo, hi = float(np.percentile(a, 1)), float(np.percentile(a, 99))
    if hi - lo < 1e-6:
        u8 = np.zeros_like(a, dtype=np.uint8)
    else:
        u8 = np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)


@app.get("/api/detect-debug")
def detect_debug(
    key: str,
    stage: str = "final",
    r_min: int | None = None,
    r_max: int | None = None,
    r_step: int | None = None,
    illum_sigma: float | None = None,
    nms_factor: float | None = None,
    score_frac: float | None = None,
    ring_band: float | None = None,
    halo_factor: float | None = None,
    contrast_floor: float | None = None,
) -> Response:
    from cellyoulite.pipeline.circle_methods import detect_circles_debug

    params = {
        "r_min": r_min, "r_max": r_max, "r_step": r_step,
        "illum_sigma": illum_sigma, "nms_factor": nms_factor,
        "score_frac": score_frac, "ring_band": ring_band,
        "halo_factor": halo_factor, "contrast_floor": contrast_floor,
    }
    img = _read_aligned(key)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img[..., :3], 0, 255).astype(np.uint8)
    # Early-exit the pipeline by stage so the page is responsive when only
    # looking at flat-field / LoG (don't run downstream work).
    stop_at = stage if stage in ("gray", "flat", "log", "final") else "final"
    d = detect_circles_debug(img[..., :3], params, stop_at=stop_at)

    if stage == "gray":
        base = _grayscale_rgb(d["gray"])
    elif stage == "flat":
        base = _colorize_signed(d["flat"])
    elif stage == "log":
        base = _colorize_signed(d["log_response"])
    else:  # "final"
        base = cv2.cvtColor(img[..., :3], cv2.COLOR_RGB2BGR)

    # Only draw circles when explicitly looking at the final stage. The user
    # wants the transformed image to be visible underneath; circles would
    # just obscure what they're trying to inspect.
    if stage == "final":
        for cx, cy, r in d["rejected_by_contrast"]:
            cv2.circle(base, (int(cx), int(cy)), int(r), (60, 60, 220), 1, cv2.LINE_AA)
        for cx, cy, r in d["circles"]:
            cv2.circle(base, (int(cx), int(cy)), int(r), (90, 220, 90), 2, cv2.LINE_AA)
            cv2.drawMarker(base, (int(cx), int(cy)), (90, 220, 90),
                           cv2.MARKER_CROSS, markerSize=10, thickness=2)

    # Header strip with the count summary.
    h_strip = 28
    strip = np.zeros((h_strip, base.shape[1], 3), dtype=np.uint8)
    summary = (f"stage={stage}  accepted={len(d['circles'])}  "
               f"rejected={len(d['rejected_by_contrast'])}  "
               f"pre={len(d['pre_contrast'])}")
    cv2.putText(strip, summary, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    base = np.vstack([strip, base])

    ok, buf = cv2.imencode(".jpg", base, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        raise HTTPException(status_code=500, detail="jpeg encode failed")
    resp = Response(content=buf.tobytes(), media_type="image/jpeg")
    resp.headers["X-Accepted"] = str(len(d["circles"]))
    resp.headers["X-Rejected"] = str(len(d["rejected_by_contrast"]))
    resp.headers["X-PreContrast"] = str(len(d["pre_contrast"]))
    return resp


# ---------------------- annotation tool ----------------------

_ANN_ROOT = Path.cwd() / "annotations"


def _ann_dir(well: str) -> Path:
    safe_well = well.replace("/", "_")
    p = _ANN_ROOT / safe_well
    p.mkdir(parents=True, exist_ok=True)
    return p


def _ann_path(well: str, label: str) -> Path:
    safe_label = label.replace("/", "_")
    return _ann_dir(well) / f"{safe_label}.json"


@app.get("/annotate", response_class=HTMLResponse)
def annotate_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "annotate.html", {"version": __version__}
    )


@app.get("/api/annotations")
def get_annotations(well: str, label: str) -> dict:
    path = _ann_path(well, label)
    if not path.is_file():
        return {"circles": [], "exists": False}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {"circles": [], "exists": False}
    return {"circles": data.get("circles", []), "exists": True,
            "meta": {k: v for k, v in data.items() if k != "circles"}}


@app.post("/api/annotations")
def save_annotations(well: str, label: str, aligned: int = 0,
                     body: dict = Body(...)) -> dict:
    circles = body.get("circles", [])
    # Normalise + validate.
    clean: list[dict] = []
    for c in circles:
        try:
            clean.append({
                "cx": float(c["cx"]),
                "cy": float(c["cy"]),
                "r": float(c["r"]),
                "star": bool(c.get("star", False)),
            })
        except (KeyError, TypeError, ValueError):
            raise HTTPException(status_code=400, detail="bad circle in payload")
    payload = {
        "well": well,
        "label": label,
        "aligned": bool(aligned),
        "key": body.get("key"),
        "image_w": body.get("image_w"),
        "image_h": body.get("image_h"),
        "circles": clean,
    }
    _ann_path(well, label).write_text(json.dumps(payload, indent=2))
    return {"ok": True, "n": len(clean)}


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})
