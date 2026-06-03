from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from skimage.io import imread
from skimage.segmentation import find_boundaries
from starlette.requests import Request

from cellyoulite.__version__ import __version__
from cellyoulite.db import repo
from cellyoulite.io.grid import (
    discover_grid, is_experiment_folder, is_image_name,
)
from cellyoulite.pipeline.align import (
    WellAlignment, compute_alignment_cached, is_alignment_cached, paste_onto_canvas,
    _CACHE_VERSION, _fingerprint,
)

_WEB = Path(__file__).resolve().parent.parent / "web"
templates = Jinja2Templates(directory=str(_WEB / "templates"))

app = FastAPI(title="CellxYou Lite", version=__version__)
app.mount("/static", StaticFiles(directory=str(_WEB / "static")), name="static")


@app.middleware("http")
async def _revalidate_static(request: Request, call_next):
    """Make browsers revalidate static assets (JS modules, CSS) on every load.
    The ES-module entry carries ?v=<version>, but that query doesn't propagate
    to its relative imports — so without this a deploy could serve a fresh
    main.js against stale cached feature modules. `no-cache` keeps responses
    cacheable but forces an If-None-Match/304 check, so it's cheap and correct.
    """
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


# Create/upgrade the SQLite schema on startup (no endpoint reads it yet).
try:
    from cellyoulite.db.migrate import migrate as _db_migrate
    _db_migrate()
except Exception as _db_err:  # noqa: BLE001 — DB is not yet on any request path
    import sys as _sys
    print(f"[cellyoulite] DB init skipped: {_db_err}", file=_sys.stderr)


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
    """Resolve a well's alignment, read-through the DB:
      in-memory cache → DB row (matching fingerprint+version) → compute.
    A compute (which uses the JSON disk cache when present) is persisted back
    to the DB so the ingested rows are the source of truth going forward.
    align.py stays DB-free, so the offline analysis tool is unaffected."""
    cache_key = (mount["id"], well.folder_name)
    if cache_key in _align_cache:
        return _align_cache[cache_key]
    paths = [tp.path for tp in well.timepoints]
    fp = _fingerprint(paths)
    row = repo.get_alignment(well.folder_name)
    if row and row["fingerprint"] == fp and row["cache_version"] == _CACHE_VERSION:
        align = WellAlignment(
            offsets=tuple(tuple(o) for o in json.loads(row["offsets"])),
            placements=tuple(tuple(p) for p in json.loads(row["placements"])),
            canvas_shape=(row["canvas_h"], row["canvas_w"]),
        )
    else:
        align = compute_alignment_cached(paths)
        repo.set_alignment(
            well.folder_name, fp, _CACHE_VERSION,
            align.canvas_shape[0], align.canvas_shape[1],
            [list(o) for o in align.offsets], [list(p) for p in align.placements],
            treatment=well.treatment, replicate=well.replicate)
    _align_cache[cache_key] = align
    return align


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
        request, "index.html",
        {"version": __version__,
         # Pre-hide the profile gate server-side when a user cookie is
         # already present, so returning visitors don't see it flash.
         "has_user": bool(request.cookies.get("cyl_user"))},
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


# ---------------------- user accounts ----------------------
# Password-less profiles, stored in the DB (app_user). The deployment sits
# behind an upstream HTTP basic-auth gate; once past that, a user just picks
# (or creates) a username — there are no per-account credentials.


@app.get("/api/users")
def list_users() -> dict:
    return {"users": repo.list_users()}


@app.post("/api/users")
def create_user(body: dict = Body(...)) -> dict:
    name = (body.get("username") or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="username required")
    if len(name) > 40:
        raise HTTPException(status_code=400, detail="username too long (max 40)")
    username, created = repo.create_user(name)
    return {"ok": True, "username": username, "users": repo.list_users(), "created": created}


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


# Decoded+resized JPEGs are cached on disk so we don't re-decode the (LZW)
# TIFF and re-encode on every request — the dominant cost for grid thumbs,
# the viewer, and especially frame-by-frame playback. Invalidated by source
# mtime (raw) and cleared on bundle-import (alignment may change).
_IMG_CACHE_ROOT = Path.cwd() / ".img_cache"


def _render_image_jpeg(key: str, thumb: int, aligned: int) -> bytes:
    path = _safe_image_path(key)
    img = _read_aligned(key) if aligned else imread(path)
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
    return buf.tobytes()


@app.get("/api/image")
def image(key: str, thumb: int = 0, aligned: int = 0) -> Response:
    """Serve an image. TIFFs (and any non-browser-native format) are
    re-encoded to JPEG. `thumb=<px>` downscales the longest side.
    `aligned=1` pads the frame onto its well's registered canvas.
    Results are cached on disk and the browser is told to cache them."""
    path = _safe_image_path(key)
    if path.suffix.lower() in _BROWSER_NATIVE and not thumb and not aligned:
        return FileResponse(path)

    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        raise HTTPException(status_code=404, detail=f"no such image: {key}")
    tag = hashlib.sha1(f"{key}|{thumb}|{aligned}|{mtime_ns}".encode()).hexdigest()
    cache_path = _IMG_CACHE_ROOT / f"{tag}.jpg"
    if cache_path.is_file():
        data = cache_path.read_bytes()
    else:
        data = _render_image_jpeg(key, thumb, aligned)
        _IMG_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".jpg.tmp")
        tmp.write_bytes(data)
        tmp.replace(cache_path)   # atomic publish
    return Response(content=data, media_type="image/jpeg",
                    headers={"Cache-Control": "public, max-age=86400",
                             "ETag": f'"{tag}"'})


@app.get("/api/well-align")
def well_align(mount_id: str, folder_name: str) -> dict:
    mount, w = _find_well(mount_id, folder_name)
    align = _well_alignment(mount, w)
    return {
        "canvas": list(align.canvas_shape),
        "offsets": [list(o) for o in align.offsets],
        "placements": [list(p) for p in align.placements],
    }




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
    hv = repo.human_validated_map()   # one query for all wells (polled often)
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
                "human_validated": hv.get(w.folder_name, False),
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


# Validation (track filters + well sign-off) lives in the DB; this helper reads
# it so every consumer (grid, growth, boxplot, CSV) sees the same source.
def _load_validation(well_folder: str) -> dict[int, bool]:
    return repo.get_validation(well_folder)["overrides"]


def _load_tracks(folder_name: str) -> dict | None:
    """A well's tracks + detections from the DB (the source of truth since the
    analysis-result ingestion). Returns the tracks-JSON shape, or None when the
    well has no tracks in the DB."""
    return repo.get_tracks(folder_name)


@app.get("/api/track-validation")
def get_track_validation(mount_id: str, folder_name: str) -> dict:
    """Manual user overrides on whether a track is valid for analysis.
    Maps track_id -> bool (true=accepted, false=rejected). Auto-classified
    tracks fall back to their script-assigned `valid` field if no override.
    Also returns `human_validated` — whether a human has signed off on the
    whole well via the Validate button."""
    state = repo.get_validation(folder_name)
    return {"well": folder_name,
            "overrides": {str(k): v for k, v in state["overrides"].items()},
            "human_validated": state["human_validated"]}


@app.post("/api/track-validation")
def set_track_validation(mount_id: str, folder_name: str,
                          body: dict = Body(...)) -> dict:
    """Update the validation state for a well. Body may contain:
      - overrides: {id: bool}  → replaces the per-track override map
      - human_validated: bool   → flips the well-level sign-off
      - user: str               → who is making the decision (provenance)

    Either field may be omitted; existing values are preserved when so."""
    decided_by = (body.get("user") or "").strip() or "unknown"
    if "overrides" in body:
        cleaned: dict[int, bool] = {}
        for k, v in (body.get("overrides") or {}).items():
            try:
                cleaned[int(k)] = bool(v)
            except (TypeError, ValueError):
                continue
        repo.set_overrides(folder_name, cleaned, decided_by)
    if "human_validated" in body:
        repo.set_human_validated(folder_name, bool(body["human_validated"]), decided_by)
    state = repo.get_validation(folder_name)
    return {"ok": True,
            "n_overrides": len(state["overrides"]),
            "human_validated": state["human_validated"]}


@app.get("/api/tracks")
def tracks_for_well(mount_id: str, folder_name: str) -> dict:
    """The full tracks payload for one well (DB-backed — the ingested analysis
    results), used by the viewer to colour overlays. Includes per-track stars."""
    data = _load_tracks(folder_name)
    if data is None:
        return {"available": False, "tracks": [], "n_frames": 0}
    data["available"] = True
    return data


@app.post("/api/track-star")
def set_track_star(mount_id: str, folder_name: str, track_id: int,
                   body: dict = Body(...)) -> dict:
    """Toggle a track-level star ("a really good exemplar"), recording who
    starred it and when."""
    by = (body.get("user") or "").strip() or "unknown"
    starred = bool(body.get("starred"))
    repo.set_star(folder_name, track_id, starred, by)
    return {"ok": True, "track_id": track_id, "starred": starred}


@app.get("/api/track-stitch")
def track_stitch(mount_id: str, folder_name: str, track_id: int,
                  variant: str = "both", pad: float = 1.4) -> Response:
    """For one track, crop a square region around its centre in every frame
    where it appears and concatenate horizontally into a wide PNG.
    variant=raw → just the raw row; variant=seg → just the segmented row;
    variant=both → two rows stacked (default). Cached versions from the
    tracking script are served when available."""
    if variant not in ("raw", "seg", "both"):
        raise HTTPException(status_code=400, detail="variant must be raw/seg/both")
    safe = folder_name.replace("/", "_")
    cached = _TRACKS_ROOT / safe / f"track_{track_id}_{variant}.png"
    if cached.is_file() and variant in ("raw", "seg"):
        return FileResponse(cached, media_type="image/png")

    _, well = _find_well(mount_id, folder_name)
    data = _load_tracks(folder_name)
    if data is None:
        raise HTTPException(status_code=404, detail="no tracks for well")
    track = next((t for t in data.get("tracks", []) if t["id"] == track_id), None)
    if track is None:
        raise HTTPException(status_code=404, detail=f"no track {track_id}")
    dets = track.get("detections", [])
    if not dets:
        raise HTTPException(status_code=404, detail="track has no detections")

    # Use the shared renderer (same code path the tracking script uses to
    # cache stitches). Replaces the inline implementation below.
    safe_well = well.folder_name.replace("/", "_")

    def _load_frame(label: str):
        tp = next((t for t in well.timepoints if t.label == label), None)
        if tp is None:
            return None
        key = f"{mount_id}/{well.folder_name}/{tp.path.name}"
        try:
            return _read_aligned(key)
        except HTTPException:
            return None

    def _load_mask(label: str):
        p = _CELLPOSE_CACHE_ROOT / safe_well / f"{label}.mask.png"
        if not p.is_file():
            return None
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    from cellyoulite.pipeline.track_stitch import render_track_strips
    strips = render_track_strips(
        track_id=track_id, detections=dets,
        load_frame=_load_frame, load_mask=_load_mask, pad=pad,
    )
    if not strips:
        raise HTTPException(status_code=404, detail="no panels rendered")
    return Response(content=strips[variant], media_type="image/png")


@app.get("/api/track-gif")
def track_gif(mount_id: str, folder_name: str, track_id: int,
               variant: str = "seg", fps: int = 4, pad: float = 1.4) -> Response:
    """Animated GIF of one track: each detection is a single frame, cropped
    to the track's largest size and (when variant=seg) tinted with the
    track's hue."""
    if variant not in ("raw", "seg"):
        raise HTTPException(status_code=400, detail="variant must be raw or seg")
    _, well = _find_well(mount_id, folder_name)
    safe = folder_name.replace("/", "_")
    data = _load_tracks(folder_name)
    if data is None:
        raise HTTPException(status_code=404, detail="no tracks for well")
    track = next((t for t in data.get("tracks", []) if t["id"] == track_id), None)
    if track is None:
        raise HTTPException(status_code=404, detail=f"no track {track_id}")
    dets = track.get("detections", [])
    if not dets:
        raise HTTPException(status_code=404, detail="track has no detections")

    def _load_frame(label: str):
        tp = next((t for t in well.timepoints if t.label == label), None)
        if tp is None:
            return None
        key = f"{mount_id}/{well.folder_name}/{tp.path.name}"
        try:
            return _read_aligned(key)
        except HTTPException:
            return None

    def _load_mask(label: str):
        p = _CELLPOSE_CACHE_ROOT / safe / f"{label}.mask.png"
        if not p.is_file():
            return None
        return cv2.imread(str(p), cv2.IMREAD_UNCHANGED)

    from cellyoulite.pipeline.track_stitch import render_track_gif as _render_gif
    buf = _render_gif(
        track_id=track_id, detections=dets,
        load_frame=_load_frame, load_mask=_load_mask,
        well_name=well.folder_name,
        variant=variant, pad=pad, fps=fps,
    )
    if not buf:
        raise HTTPException(status_code=404, detail="no frames")
    headers = {
        "Content-Disposition":
            f'attachment; filename="{safe}_track_{track_id}_{variant}.gif"'
    }
    return Response(content=buf, media_type="image/gif", headers=headers)


@app.get("/api/growth-csv")
def growth_csv(treatment: str | None = None) -> Response:
    """Long-format CSV of every accepted track-frame observation, suitable
    for loading into R as `read.csv(...)`.

    Columns:
      treatment, replicate, well, track_id, valid_auto, accepted,
      t_idx, minutes, label, area_px, area_px_t0, log2_fold
    """
    import csv
    import io
    import math

    spec_wells: list = []
    for mount in _mounts():
        try:
            spec = _wells_for_mount(mount)
        except Exception:  # noqa: BLE001
            continue
        for w in spec.wells:
            if treatment is None or w.treatment == treatment:
                spec_wells.append(w)

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow([
        "treatment", "replicate", "well", "track_id",
        "valid_auto", "accepted",
        "t_idx", "minutes", "label",
        "area_px", "area_px_t0", "log2_fold",
    ])
    for w in spec_wells:
        data = _load_tracks(w.folder_name)
        if data is None:
            continue
        overrides = _load_validation(w.folder_name)
        tp_minutes = {tp.label: tp.minutes for tp in w.timepoints}
        for tr in data.get("tracks", []):
            valid_auto = bool(tr.get("valid"))
            accepted = overrides.get(tr["id"], valid_auto)
            # Only export accepted tracks — rejected ones aren't useful for
            # downstream analysis and clutter the data frame.
            if not accepted:
                continue
            dets = tr.get("detections", [])
            if not dets:
                continue
            a0 = dets[0].get("area_px")
            if not a0:
                continue
            for d in dets:
                ap = d.get("area_px")
                mn = tp_minutes.get(d["label"])
                if mn is None or not ap:
                    continue
                writer.writerow([
                    w.treatment, w.replicate, w.folder_name, tr["id"],
                    "TRUE" if valid_auto else "FALSE",
                    "TRUE" if accepted else "FALSE",
                    d.get("t_idx", ""), int(mn), d["label"],
                    int(ap), int(a0),
                    f"{math.log2(ap / a0):.6f}",
                ])

    name = "growth" + (f"_{treatment.replace(' ', '_')}" if treatment else "_all") + ".csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


@app.get("/api/boxplot-data")
def boxplot_data(treatment: str | None = None, by_replicate: int = 0) -> dict:
    """Pool every accepted track from every replicate of each requested
    treatment, normalise to log2(area / area@t0), and return per-treatment
    per-timepoint box statistics. When by_replicate=1, replicates are kept
    as separate series within each treatment."""
    import math

    spec_wells: list = []
    for mount in _mounts():
        try:
            spec = _wells_for_mount(mount)
        except Exception:  # noqa: BLE001
            continue
        for w in spec.wells:
            if treatment is None or w.treatment == treatment:
                spec_wells.append(w)
    if not spec_wells:
        return {"treatments": [], "boxes": [], "minutes": [], "replicates": []}

    # (treatment, replicate_or_None) -> minutes -> [log2 values]
    bucket: dict[tuple, dict[int, list[float]]] = {}
    minutes_seen: set[int] = set()

    for w in spec_wells:
        data = _load_tracks(w.folder_name)
        if data is None:
            continue
        overrides = _load_validation(w.folder_name)
        tp_minutes = {tp.label: tp.minutes for tp in w.timepoints}
        key = (w.treatment, w.replicate if by_replicate else None)
        for tr in data.get("tracks", []):
            accepted = overrides.get(tr["id"], bool(tr.get("valid")))
            if not accepted:
                continue
            dets = tr.get("detections", [])
            if not dets:
                continue
            a0 = dets[0].get("area_px")
            if not a0:
                continue
            for d in dets:
                ap = d.get("area_px")
                mn = tp_minutes.get(d["label"])
                if mn is None or not ap:
                    continue
                fold = math.log2(ap / a0)
                bucket.setdefault(key, {}).setdefault(int(mn), []).append(fold)
                minutes_seen.add(int(mn))

    def _box(values: list[float]) -> dict:
        if not values:
            return {}
        vs = sorted(values)
        def pct(p):
            if len(vs) == 1: return vs[0]
            k = (len(vs) - 1) * p
            f = int(k); c = f + 1
            if c >= len(vs): return vs[-1]
            return vs[f] + (vs[c] - vs[f]) * (k - f)
        q1, q2, q3 = pct(0.25), pct(0.50), pct(0.75)
        iqr = q3 - q1
        lo_whisker = max(vs[0], q1 - 1.5 * iqr)
        hi_whisker = min(vs[-1], q3 + 1.5 * iqr)
        outliers = [v for v in vs if v < lo_whisker or v > hi_whisker]
        return {"n": len(vs), "median": q2, "q1": q1, "q3": q3,
                "lo": lo_whisker, "hi": hi_whisker,
                "outliers": outliers[:50]}  # cap outliers so payload stays light

    boxes = []
    treatments_set: set[str] = set()
    replicates_set: set[int] = set()
    for key, by_min in sorted(bucket.items()):
        tx, rep = key
        treatments_set.add(tx)
        if rep is not None:
            replicates_set.add(rep)
        for mn in sorted(by_min):
            stats = _box(by_min[mn])
            if not stats:
                continue
            stats["treatment"] = tx
            stats["minutes"] = mn
            if rep is not None:
                stats["replicate"] = rep
            boxes.append(stats)
    return {
        "treatments": sorted(treatments_set),
        "replicates": sorted(replicates_set),
        "minutes": sorted(minutes_seen),
        "boxes": boxes,
        "by_replicate": bool(by_replicate),
    }


@app.get("/api/well-gif")
def well_gif(mount_id: str, folder_name: str,
              max_width: int = 800, fps: int = 4,
              accepted_only: int = 1, labels: int = 1) -> Response:
    """Animated GIF of every aligned frame in a well, with accepted tracks
    coloured (boundary + semi-transparent fill in each track's hue). The
    aligned canvas naturally pads with black where frames have shifted, so
    that shows up as borders in the GIF."""
    import io
    import imageio.v2 as imageio

    mount, well = _find_well(mount_id, folder_name)
    safe = folder_name.replace("/", "_")

    # Tracks + validation overrides → which (track_id, valid) per frame.
    tdata = _load_tracks(folder_name)
    if tdata is None:
        raise HTTPException(status_code=404, detail="no tracks for well")
    overrides = _load_validation(folder_name)

    # Per-frame label→(track_id, hue_rgb). Build once.
    align = _well_alignment(mount, well)
    H_full, W_full = align.canvas_shape
    fill_alpha = 0.32
    # Compute per-frame index of (track_id, accepted, hue) for fast lookup.
    per_frame_tracks: dict[int, list[dict]] = {}
    for tr in tdata.get("tracks", []):
        accepted = overrides.get(tr["id"], bool(tr.get("valid")))
        if accepted_only and not accepted:
            continue
        rb, rg, r_ = _hue_for_track(tr["id"])  # returns bgr
        # _hue_for_track in this file returns (b, g, r); convert to rgb
        hue_rgb = np.array([r_, rg, rb], dtype=np.float32)
        for det in tr.get("detections", []):
            per_frame_tracks.setdefault(det["t_idx"], []).append({
                "track_id": tr["id"], "cx": det["cx"], "cy": det["cy"],
                "hue": hue_rgb,
            })

    from cellyoulite.pipeline.track_stitch import make_label_fmt
    fmt_label = make_label_fmt([tp.label for tp in well.timepoints])
    frames_out: list[np.ndarray] = []
    for i, tp in enumerate(well.timepoints):
        key = f"{mount_id}/{well.folder_name}/{tp.path.name}"
        try:
            frame = _read_aligned(key)
        except HTTPException:
            continue
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame[..., :3], 0, 255).astype(np.uint8)
        canvas = frame[..., :3].astype(np.float32)

        # Tint each accepted track's instance in this frame; remember the
        # (cx, cy, track_id, hue, r) of each so we can burn labels after.
        label_jobs: list[tuple[int, int, int, np.ndarray, int]] = []
        mask_path = _CELLPOSE_CACHE_ROOT / safe / f"{tp.label}.mask.png"
        if mask_path.is_file() and i in per_frame_tracks:
            masks = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if masks is not None and masks.shape == canvas.shape[:2]:
                for entry in per_frame_tracks[i]:
                    cx, cy = int(round(entry["cx"])), int(round(entry["cy"]))
                    if not (0 <= cx < masks.shape[1] and 0 <= cy < masks.shape[0]):
                        continue
                    lab = int(masks[cy, cx])
                    if lab == 0:
                        continue
                    inst = masks == lab
                    if not inst.any():
                        continue
                    canvas[inst] = canvas[inst] * (1 - fill_alpha) + entry["hue"] * fill_alpha
                    bnd = find_boundaries(inst, mode="thick")
                    canvas[bnd] = entry["hue"]
                    # Equivalent radius for label sizing.
                    n_inst = int(inst.sum())
                    r_eq = int(round((n_inst / 3.14159) ** 0.5))
                    label_jobs.append((cx, cy, entry["track_id"], entry["hue"], r_eq))

        # Timestamp strip on the bottom-right corner.
        out = np.clip(canvas, 0, 255).astype(np.uint8)

        # Burn ID labels (after the canvas is uint8 so OpenCV likes it).
        if labels:
            for cx, cy, tid, hue, r_eq in label_jobs:
                font_scale = max(0.35, min(1.4, r_eq / 38.0))
                thickness = 1 if font_scale < 0.6 else 2
                text = str(tid)
                size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                          font_scale, thickness)
                tx = cx - size[0] // 2
                ty = cy + size[1] // 2
                # Black outline first, then coloured fill on top.
                cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
                col = (int(hue[0]), int(hue[1]), int(hue[2]))
                cv2.putText(out, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale, col, thickness, cv2.LINE_AA)
        disp = fmt_label(tp.label)
        cv2.rectangle(out, (0, out.shape[0] - 18), (140, out.shape[0]),
                      (0, 0, 0), -1)
        cv2.putText(out, disp, (4, out.shape[0] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        # Downscale if requested.
        if max_width and out.shape[1] > max_width:
            scale = max_width / out.shape[1]
            new_h = int(round(out.shape[0] * scale))
            out = cv2.resize(out, (max_width, new_h), interpolation=cv2.INTER_AREA)
        frames_out.append(out)

    if not frames_out:
        raise HTTPException(status_code=404, detail="no frames")

    buf = io.BytesIO()
    imageio.mimsave(buf, frames_out, format="GIF", duration=1.0 / max(1, fps), loop=0)
    headers = {
        "Content-Disposition":
            f'attachment; filename="{safe}_tracked.gif"'
    }
    return Response(content=buf.getvalue(), media_type="image/gif",
                    headers=headers)


@app.get("/api/well-growth")
def well_growth(mount_id: str, folder_name: str, valid_only: int = 1) -> dict:
    """Per-track normalised area-over-time, ready for plotting.
    Returns {tracks: [{id, valid, color_index, points: [{minutes, area_px,
    norm}, ...]}, ...]}. norm = area_px / area_px_at_first_detection."""
    _, well = _find_well(mount_id, folder_name)
    data = _load_tracks(folder_name)
    if data is None:
        return {"available": False, "tracks": []}
    overrides = _load_validation(folder_name)
    tp_minutes = {tp.label: tp.minutes for tp in well.timepoints}
    out: list[dict] = []
    for t in data.get("tracks", []):
        accepted = overrides.get(t["id"], bool(t.get("valid")))
        if valid_only and not accepted:
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


# ---------------------- dataset bundle ----------------------

@app.get("/api/bundle-export")
def bundle_export(include_raw: int = 0) -> Response:
    """Stream a .tar.gz containing the current processed dataset:
    .align_cache, .cellpose_cache, tracks, annotations. With include_raw=1
    the raw data/ folder is bundled too (large)."""
    import io
    import tarfile
    import time as _t
    dirs = [".align_cache", ".cellpose_cache", "tracks", "annotations"]
    if include_raw:
        dirs.append("data")

    root = Path.cwd()
    contents = []
    for d in dirs:
        p = root / d
        if not p.exists():
            continue
        n_files = sum(1 for x in p.rglob("*") if x.is_file())
        n_bytes = sum(x.stat().st_size for x in p.rglob("*") if x.is_file())
        contents.append({"name": d, "files": n_files, "bytes": n_bytes})

    if not contents:
        raise HTTPException(status_code=404, detail="nothing to bundle")

    manifest = {
        "bundle_version": 1,
        "created_at": _t.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(root.resolve()),
        "include_raw": bool(include_raw),
        "contents": contents,
    }

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        man_bytes = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(man_bytes); info.mtime = int(_t.time())
        tar.addfile(info, io.BytesIO(man_bytes))
        for c in contents:
            tar.add(root / c["name"], arcname=c["name"])
    buf.seek(0)
    stamp = _t.strftime("%Y%m%d-%H%M%S")
    return Response(
        content=buf.getvalue(),
        media_type="application/gzip",
        headers={"Content-Disposition":
                  f'attachment; filename="cellyoulite_{stamp}.tar.gz"'},
    )


@app.post("/api/bundle-import")
async def bundle_import(request: Request) -> dict:
    """Restore a .tar.gz bundle into the working directory. Expects the
    raw bundle bytes as the request body (Content-Type: application/gzip)."""
    import io
    import tarfile
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty body")
    try:
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
            try:
                m = tar.getmember("manifest.json")
                manifest = json.loads(tar.extractfile(m).read())
            except KeyError:
                raise HTTPException(status_code=400,
                                     detail="not a cellyoulite bundle (no manifest.json)")
            root = Path.cwd()
            updated: set[str] = set()
            for member in tar.getmembers():
                target = (root / member.name).resolve()
                if root.resolve() not in target.parents and target != root.resolve():
                    continue  # refuse paths that escape the project root
                # Figure out which experiments this bundle touches, by name:
                # result files are keyed by the "<treatment> r<rep>" folder
                # (as a dir segment, or a "<folder>[ __validation].json" stem).
                for seg in Path(member.name).parts:
                    stem = seg[:-5] if seg.endswith(".json") else seg
                    stem = stem.split("__", 1)[0]
                    if is_experiment_folder(stem):
                        updated.add(stem)
            tar.extractall(root, filter="data")
    except tarfile.TarError as e:
        raise HTTPException(status_code=400, detail=f"bad tar: {e}")
    # Reset in-memory caches so the just-restored disk state is picked up.
    _align_cache.clear()
    # Imported results may change alignment → drop the rendered-image cache.
    import shutil
    shutil.rmtree(_IMG_CACHE_ROOT, ignore_errors=True)
    # Translate the just-extracted analysis results into DB rows.
    ingested = _ingest_results_to_db(sorted(updated))
    return {"ok": True, "manifest": manifest,
            "updated_experiments": sorted(updated), **ingested}


def _ingest_results_to_db(experiments: list[str]) -> dict:
    """After a results bundle is extracted to disk, translate its alignment +
    tracks/detections into DB rows (the cellpose mask PNGs stay on disk). This
    is what makes "Add segmentation data" an upload to the database."""
    ddir = _data_dir()
    meta: dict[str, tuple] = {}
    if ddir.is_dir():
        for w in discover_grid(ddir).wells:
            meta[w.folder_name] = (w.treatment, w.replicate, [tp.path for tp in w.timepoints])
    n_align = n_tracks = 0
    for name in experiments:
        tr, rep, paths = meta.get(name, (None, None, None))
        if paths and is_alignment_cached(paths):
            al = compute_alignment_cached(paths)
            repo.set_alignment(
                name, _fingerprint(paths), _CACHE_VERSION,
                al.canvas_shape[0], al.canvas_shape[1],
                [list(o) for o in al.offsets], [list(p) for p in al.placements],
                treatment=tr, replicate=rep)
            n_align += 1
        tpath = _TRACKS_ROOT / f"{name.replace('/', '_')}.json"
        if tpath.is_file():
            try:
                tdata = json.loads(tpath.read_text())
            except (OSError, ValueError):
                tdata = None
            if tdata:
                src = _fingerprint(paths) if paths else None
                n_tracks += repo.set_tracks(name, tdata, src, treatment=tr, replicate=rep)
    return {"db_alignments": n_align, "db_tracks": n_tracks}


@app.post("/api/upload-images")
async def upload_images(files: list[UploadFile] = File(...)) -> dict:
    """Add raw images. Files arrive from a folder picker, each carrying a
    relative path; the experiment folder ("<treatment> r<rep>") is detected
    from that path by name, and the image is dropped into data/<folder>/.
    Non-images and anything without a recognisable experiment folder are
    skipped — so the caller can hand us a whole tree and we figure it out."""
    data_root = _data_dir()
    data_root.mkdir(parents=True, exist_ok=True)
    droot = data_root.resolve()
    added: dict[str, int] = {}
    skipped = 0
    for up in files:
        rel = (up.filename or "").replace("\\", "/")
        parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
        if not parts or not is_image_name(parts[-1]):
            skipped += 1
            continue
        exp = next((seg for seg in parts[:-1] if is_experiment_folder(seg)), None)
        if exp is None:
            skipped += 1
            continue
        dest_dir = (data_root / exp).resolve()
        if droot not in dest_dir.parents and dest_dir != droot:
            skipped += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        (dest_dir / parts[-1]).write_bytes(await up.read())
        added[exp] = added.get(exp, 0) + 1
    return {"ok": True, "experiments": dict(sorted(added.items())),
            "n_files": sum(added.values()), "n_experiments": len(added),
            "skipped": skipped}


@app.get("/api/download-images")
def download_images(exp: list[str] = Query(default=[])) -> Response:
    """Stream a .tar.gz of raw images for the chosen experiments (all if
    none are specified). Arcnames are data/<folder>/... so the archive can
    be re-imported as-is."""
    import io
    import tarfile
    import time as _t
    data_root = _data_dir()
    if not data_root.is_dir():
        raise HTTPException(status_code=404, detail="no data folder")
    available = sorted(
        p.name for p in data_root.iterdir()
        if p.is_dir() and is_experiment_folder(p.name)
    )
    chosen = [e for e in exp if e in available] if exp else available
    if not chosen:
        raise HTTPException(status_code=404, detail="no matching experiments")

    manifest = {
        "bundle_version": 1,
        "kind": "images",
        "created_at": _t.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiments": chosen,
    }
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        man_bytes = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(man_bytes); info.mtime = int(_t.time())
        tar.addfile(info, io.BytesIO(man_bytes))
        for e in chosen:
            tar.add(data_root / e, arcname=f"data/{e}")
    buf.seek(0)
    stamp = _t.strftime("%Y%m%d-%H%M%S")
    tag = "all" if len(chosen) == len(available) else f"{len(chosen)}exp"
    return Response(
        content=buf.getvalue(),
        media_type="application/gzip",
        headers={"Content-Disposition":
                  f'attachment; filename="cellyoulite_images_{tag}_{stamp}.tar.gz"'},
    )


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


def _hue_for_track(track_id: int) -> tuple[int, int, int]:
    """Golden-angle HSL → BGR, matching the SVG colours used in the viewer.
    Lightness 60%, saturation 85%."""
    h = (track_id * 137.508) % 360
    s, l = 0.85, 0.60
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(((h / 60) % 2) - 1))
    m = l - c / 2
    if   h < 60:  r, g, b = c, x, 0
    elif h < 120: r, g, b = x, c, 0
    elif h < 180: r, g, b = 0, c, x
    elif h < 240: r, g, b = 0, x, c
    elif h < 300: r, g, b = x, 0, c
    else:         r, g, b = c, 0, x
    return int((b + m) * 255), int((g + m) * 255), int((r + m) * 255)


def _label_to_track(mount, well, t_idx, masks):
    """Map mask instance label → (track_id, valid) for the given frame, by
    looking up which track owns each instance based on its centroid."""
    data = _load_tracks(well.folder_name)
    if data is None:
        return {}
    overrides = _load_validation(well.folder_name)
    # Pull this frame's detections by t_idx for fast lookup.
    frame_dets = []  # [(cx, cy, track_id, accepted)]
    for t in data.get("tracks", []):
        accepted = overrides.get(t["id"], bool(t.get("valid")))
        for d in t.get("detections", []):
            if d.get("t_idx") == t_idx:
                frame_dets.append((d["cx"], d["cy"], t["id"], accepted))
                break
    if not frame_dets:
        return {}
    # For each instance label, find its centroid and match to the closest
    # track-detection centroid (cheap; aligned-canvas coords match).
    out: dict[int, tuple[int, bool]] = {}
    labels = np.unique(masks)
    for lab in labels:
        if lab == 0:
            continue
        ys, xs = np.where(masks == lab)
        if ys.size == 0:
            continue
        cx, cy = float(xs.mean()), float(ys.mean())
        best = None
        for fcx, fcy, tid, valid in frame_dets:
            d = (fcx - cx) ** 2 + (fcy - cy) ** 2
            if best is None or d < best[0]:
                best = (d, tid, valid)
        if best and best[0] < 9.0 ** 2:  # within 9 px
            out[int(lab)] = (best[1], best[2])
    return out


@app.get("/api/cellpose-edges")
def cellpose_edges(key: str, aligned: int = 1, by_track: int = 1,
                    fill: int = 1, fill_alpha: int = 70,
                    hide_invalid: int = 0) -> Response:
    """Boundaries of the cached cellpose mask, returned as a transparent
    PNG. by_track=1 (default) colours each instance's boundary by its
    track's hue (matches the SVG colour); by_track=0 falls back to a
    single colour. aligned=0 crops the result back to the raw frame.
    hide_invalid=1 skips drawing the segmentation of deactivated (rejected)
    tracks entirely, so hiding deactivated organoids hides their masks too."""
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

    h, w = masks.shape
    bgra = np.zeros((h, w, 4), dtype=np.uint8)

    if by_track:
        lab2tr = _label_to_track(mount, well, t_idx, masks)
        fill_a = max(0, min(255, int(fill_alpha)))
        for lab in np.unique(masks):
            if lab == 0:
                continue
            inst = masks == lab
            inst_boundary = inst & boundary
            tr = lab2tr.get(int(lab))
            if tr is None:
                bcol = (255, 255, 255)  # untracked → white
            elif not tr[1]:
                if hide_invalid:
                    continue            # deactivated → don't draw its segmentation
                bcol = (138, 138, 138)  # invalid → desaturated grey
            else:
                bcol = _hue_for_track(tr[0])
            if fill:
                # Semi-transparent interior in the same hue.
                bgra[inst] = (bcol[0], bcol[1], bcol[2], fill_a)
            # Boundary on top, fully opaque.
            bgra[inst_boundary] = (bcol[0], bcol[1], bcol[2], 255)
    else:
        if fill:
            inst_any = masks > 0
            bgra[inst_any] = [72, 86, 255, max(0, min(255, int(fill_alpha)))]
        bgra[boundary] = [72, 86, 255, 255]

    if not aligned:
        align = _well_alignment(mount, well)
        if align.placements:
            y0, x0 = align.placements[t_idx]
            img = imread(_safe_image_path(key))
            raw_h, raw_w = (img.shape[0], img.shape[1]) if img.ndim >= 2 else bgra.shape[:2]
            bgra = bgra[y0:y0 + raw_h, x0:x0 + raw_w]

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
