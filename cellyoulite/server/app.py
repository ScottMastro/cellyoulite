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
