from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
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
    WellAlignment, compute_alignment, paste_onto_canvas,
)
from cellyoulite.plotting.boxplot import boxplot_by_timepoint, well_lineplot
from cellyoulite.simulator import SimConfig, generate_dataset

_WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(_WEB / "templates"))

app = FastAPI(title="CellxYou Lite", version=__version__)
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")


_align_cache: dict[tuple[str, str], WellAlignment] = {}


def _well_for_key(folder: Path, key: str):
    """Locate the Well whose subfolder is the first segment of `key`."""
    sub = key.split("/", 1)[0] if "/" in key else None
    if not sub:
        return None
    for w in discover_grid(folder).wells:
        if w.folder_name == sub:
            return w
    return None


def _well_alignment(folder: Path, well) -> WellAlignment:
    cache_key = (str(folder.resolve()), well.folder_name)
    if cache_key not in _align_cache:
        _align_cache[cache_key] = compute_alignment([tp.path for tp in well.timepoints])
    return _align_cache[cache_key]


def _read_aligned(folder: Path, key: str) -> np.ndarray:
    """Read the image and pad it onto its well's union canvas."""
    img = imread(_safe_image_path(str(folder), key))
    well = _well_for_key(folder, key)
    if well is None:
        return img
    align = _well_alignment(folder, well)
    idx = next((i for i, tp in enumerate(well.timepoints) if tp.key == key), None)
    if idx is None or not align.placements:
        return img
    return paste_onto_canvas(img, align.placements[idx], align.canvas_shape)


def _safe_image_path(folder: str, key: str) -> Path:
    """Resolve `folder/key` and confirm it stays inside `folder` and exists."""
    base = Path(folder).expanduser().resolve()
    if not base.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {folder}")
    target = (base / key).resolve()
    if base not in target.parents and target != base:
        raise HTTPException(status_code=400, detail="key escapes folder")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"no such file: {key}")
    return target


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "index.html", {"version": __version__}
    )


@app.get("/api/version")
def version() -> dict:
    return {"version": __version__}


@app.get("/api/default-folder")
def default_folder() -> dict:
    """Return a folder to auto-load, if one is obvious.

    Order: $CELLYOULITE_DATA env var, then ./data relative to cwd.
    """
    import os
    candidates = []
    env = os.environ.get("CELLYOULITE_DATA")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(Path.cwd() / "data")
    for c in candidates:
        if c.is_dir() and any(p.is_dir() for p in c.iterdir()):
            return {"folder": str(c.resolve())}
    return {"folder": None}


@app.get("/api/grid")
def grid(folder: str) -> dict:
    """Return the well grid: one entry per (treatment, replicate)."""
    path = Path(folder).expanduser()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {folder}")
    spec = discover_grid(path)
    wells = []
    for w in spec.wells:
        # Thumbnail = last timepoint (max-growth frame, most informative).
        thumb = w.timepoints[-1] if w.timepoints else None
        wells.append({
            "treatment": w.treatment,
            "replicate": w.replicate,
            "folder_name": w.folder_name,
            "n_timepoints": len(w.timepoints),
            "thumb_key": thumb.key if thumb else None,
            "thumb_label": thumb.label if thumb else None,
        })
    return {
        "treatments": spec.treatments,
        "replicates": spec.replicates,
        "n_wells": len(spec.wells),
        "n_images": spec.n_images,
        "wells": wells,
    }


@app.get("/api/well")
def well(folder: str, treatment: str, replicate: int) -> dict:
    """Return the ordered timepoint list for a single well."""
    spec = discover_grid(Path(folder).expanduser())
    for w in spec.wells:
        if w.treatment == treatment and w.replicate == replicate:
            return {
                "treatment": w.treatment,
                "replicate": w.replicate,
                "folder_name": w.folder_name,
                "timepoints": [
                    {"t_idx": i, "minutes": tp.minutes, "label": tp.label, "key": tp.key}
                    for i, tp in enumerate(w.timepoints)
                ],
            }
    raise HTTPException(status_code=404, detail="well not found")


@app.post("/api/simulate")
def simulate() -> dict:
    out = Path(tempfile.gettempdir()) / "cellyoulite-sim"
    generate_dataset(out, SimConfig())
    n_files = sum(1 for p in out.rglob("*") if p.is_file())
    return {"folder": str(out), "n_files": n_files}


_BROWSER_NATIVE = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


@app.get("/api/image")
def image(folder: str, key: str, thumb: int = 0, aligned: int = 0) -> Response:
    """Serve an image. TIFFs (and any non-browser-native format) are
    re-encoded to JPEG. `thumb=<px>` downscales the longest side.
    `aligned=1` pads the frame onto its well's registered canvas."""
    path = _safe_image_path(folder, key)
    if path.suffix.lower() in _BROWSER_NATIVE and not thumb and not aligned:
        return FileResponse(path)

    if aligned:
        img = _read_aligned(Path(folder).expanduser(), key)
    else:
        img = imread(path)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # skimage delivers RGB; cv2 wants BGR.
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
def overlay(folder: str, key: str, aligned: int = 0) -> dict:
    if aligned:
        img = _read_aligned(Path(folder).expanduser(), key)
    else:
        img = imread(_safe_image_path(folder, key))
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
def edges(folder: str, key: str, aligned: int = 0) -> Response:
    if aligned:
        img = _read_aligned(Path(folder).expanduser(), key)
    else:
        img = imread(_safe_image_path(folder, key))
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
def well_align(folder: str, treatment: str, replicate: int) -> dict:
    """Compute (or fetch cached) phase-correlation alignment for one well."""
    root = Path(folder).expanduser()
    well = next((w for w in discover_grid(root).wells
                 if w.treatment == treatment and w.replicate == replicate), None)
    if well is None:
        raise HTTPException(status_code=404, detail="well not found")
    align = _well_alignment(root, well)
    return {
        "canvas": list(align.canvas_shape),       # [H, W]
        "offsets": [list(o) for o in align.offsets],
        "placements": [list(p) for p in align.placements],
    }


@app.get("/api/well-analyze")
def well_analyze(folder: str, treatment: str, replicate: int) -> JSONResponse:
    """Run the pipeline on a single well's timeseries and return a line plot."""
    import pandas as pd
    spec = discover_grid(Path(folder).expanduser())
    well = next((w for w in spec.wells
                 if w.treatment == treatment and w.replicate == replicate), None)
    if well is None:
        raise HTTPException(status_code=404, detail="well not found")

    qc = QCThresholds()
    rows: list[dict] = []
    for t_idx, tp in enumerate(well.timepoints):
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
def analyze(folder: str) -> JSONResponse:
    path = Path(folder).expanduser()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {folder}")
    df = analyze_folder(path)
    plot_html = boxplot_by_timepoint(df) if len(df) else ""
    return JSONResponse({
        "n_organoids": int(len(df)),
        "treatments": sorted(df["treatment"].unique().tolist()) if len(df) else [],
        "n_timepoints": int(df["t_idx"].nunique()) if len(df) else 0,
        "median_area_px": float(df["area_px"].median()) if len(df) else None,
        "plot_html": plot_html,
        "rows": df.head(20).to_dict(orient="records"),
    })


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})
