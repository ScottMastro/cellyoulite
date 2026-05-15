from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from skimage import measure
from skimage.io import imread
from starlette.requests import Request

from cellyoulite.__version__ import __version__
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline import analyze_folder, fit_circle, passes_qc, segment_image
from cellyoulite.plotting.boxplot import boxplot_by_timepoint
from cellyoulite.simulator import SimConfig, generate_dataset

_WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(_WEB / "templates"))

app = FastAPI(title="CellxYou Lite", version=__version__)
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")


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


@app.get("/api/grid")
def grid(folder: str) -> dict:
    path = Path(folder).expanduser()
    if not path.exists() or not path.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {folder}")
    spec = discover_grid(path)
    return {
        "treatments": spec.treatments,
        "timepoints": spec.timepoints,
        "n_cells": len(spec.cells),
        "cells": [
            {"treatment": c.treatment, "timepoint": c.timepoint, "key": c.path.name}
            for c in spec.cells
        ],
    }


@app.post("/api/simulate")
def simulate() -> dict:
    out = Path(tempfile.gettempdir()) / "cellyoulite-sim"
    generate_dataset(out, SimConfig())
    return {"folder": str(out), "n_files": sum(1 for _ in out.iterdir())}


@app.get("/api/image")
def image(folder: str, key: str) -> FileResponse:
    return FileResponse(_safe_image_path(folder, key))


@app.get("/api/overlay")
def overlay(folder: str, key: str) -> dict:
    """Per-image segmentation: contours (pixel mask) + best-fit circles."""
    path = _safe_image_path(folder, key)
    img = imread(path, as_gray=True).astype(float)
    seg = segment_image(img)

    contours: list[list[list[float]]] = []
    circles: list[dict] = []

    for idx, region in enumerate(seg.regions):
        if not passes_qc(region):
            continue
        obj_mask = seg.labels == (idx + 1)
        for c in measure.find_contours(obj_mask.astype(np.uint8), 0.5):
            # find_contours returns (row, col) — flip to (x, y) for SVG.
            contours.append([[float(c[i, 1]), float(c[i, 0])] for i in range(len(c))])
        fit = fit_circle(obj_mask)
        circles.append({"cx": fit.cx, "cy": fit.cy, "r": fit.radius})

    h, w = img.shape
    return {"width": int(w), "height": int(h),
            "contours": contours, "circles": circles,
            "n_passed": len(circles)}


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
        "timepoints": sorted(df["timepoint"].unique().tolist()) if len(df) else [],
        "median_area_px": float(df["area_px"].median()) if len(df) else None,
        "plot_html": plot_html,
        "rows": df.head(20).to_dict(orient="records"),
    })


@app.get("/healthz")
def healthz() -> JSONResponse:
    return JSONResponse({"ok": True})
