from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from cellyoulite.__version__ import __version__
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline import analyze_folder
from cellyoulite.plotting.boxplot import boxplot_by_timepoint
from cellyoulite.simulator import SimConfig, generate_dataset

_WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(_WEB / "templates"))

app = FastAPI(title="CellYouLite", version=__version__)
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")


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
    }


@app.post("/api/simulate")
def simulate() -> dict:
    out = Path(tempfile.gettempdir()) / "cellyoulite-sim"
    generate_dataset(out, SimConfig())
    return {"folder": str(out), "n_files": sum(1 for _ in out.iterdir())}


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
