"""cellyoulite-analyze — standalone analysis pipeline.

Runs the full organoid pipeline on raw images and emits an uploadable
results bundle:

    align  →  segment (Cellpose-SAM)  →  track

This is the "offshored" half of the workflow: download a subset of raw
images from the web app, run this tool on a machine with Cellpose/torch
(GPU ideal), then upload the produced ``results_*.tar.gz`` back via
"Add segmentation data". Everything is keyed by experiment folder name
("<treatment> r<rep>"), so subsets just work.

Input may be a folder of experiment subfolders, a folder containing a
``data/`` dir, or an ``images_*.tar.gz`` produced by the web app. The
pipeline runs in a working directory (``--workdir``); its caches make
re-runs cheap and resumable.

Heavy deps live in the ``analyze`` extra:  pip install -e ".[analyze]"
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from pathlib import Path

from cellyoulite.io.grid import is_experiment_folder

_PKG_ROOT = Path(__file__).resolve().parent          # .../cellyoulite
_REPO_ROOT = _PKG_ROOT.parent                        # repo root (holds scripts/)
_SCRIPTS = _REPO_ROOT / "scripts"

# Result dirs that make up an uploadable bundle (matches the web exporter).
_RESULT_DIRS = [".align_cache", ".cellpose_cache", "tracks", "annotations"]
# "[  12/ 420]" style progress emitted by every stage script.
_PROGRESS_RE = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")


# ----------------------------- helpers -----------------------------

def _fail(console, msg: str, code: int = 1):
    console.print(f"[bold red]✗[/] {msg}")
    raise SystemExit(code)


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
    except Exception:  # noqa: BLE001 — torch missing or broken → CPU
        pass
    return "cpu"


def _setup_workdir(console, input_arg, workdir: Path) -> list[str]:
    """Ensure ``workdir/data`` exists and points at the experiments.
    Returns the sorted list of experiment folder names found."""
    workdir.mkdir(parents=True, exist_ok=True)
    data = workdir / "data"

    if input_arg is not None:
        p = Path(input_arg).expanduser().resolve()
        if not p.exists():
            _fail(console, f"input not found: {p}")
        if p.is_file() and p.name.endswith((".tar.gz", ".tgz", ".gz")):
            with tarfile.open(p, "r:gz") as tar:
                tar.extractall(workdir, filter="data")
            if not data.is_dir():
                _fail(console, "archive has no data/ — is this an images bundle?")
        elif p.is_dir():
            if any(c.is_dir() and is_experiment_folder(c.name) for c in p.iterdir()):
                src = p
            elif (p / "data").is_dir():
                src = p / "data"
            else:
                _fail(console, f"no '<treatment> r<rep>' experiment folders in {p}")
            if data.is_symlink():
                data.unlink()
            elif data.exists() and data.resolve() != src.resolve():
                _fail(console, f"{data} already exists and isn't the given input; "
                               "remove it or pick another --workdir")
            if not data.exists():
                data.symlink_to(src)
        else:
            _fail(console, f"don't know how to read input: {p}")

    if not data.is_dir():
        _fail(console, f"no data in workdir ({data}); pass an input folder/archive")

    exps = sorted(c.name for c in data.iterdir()
                  if c.is_dir() and is_experiment_folder(c.name))
    if not exps:
        _fail(console, f"no experiment folders under {data}")
    return exps


def _run_stage(console, title: str, cmd: list[str], cwd: Path) -> float:
    """Run one stage subprocess, rendering a live rich progress bar by
    parsing its "[i/n]" output. Returns elapsed seconds; aborts on error."""
    from rich.progress import (
        Progress, SpinnerColumn, BarColumn, TextColumn,
        MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn,
    )
    console.rule(f"[bold cyan]{title}")
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    t0 = time.perf_counter()
    tail: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), env=env, text=True, bufsize=1,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    total = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task(title, total=None)
        for line in proc.stdout:
            line = line.rstrip()
            if not line:
                continue
            tail.append(line)
            if len(tail) > 60:
                tail.pop(0)
            m = _PROGRESS_RE.search(line)
            if m:
                done, total = int(m.group(1)), int(m.group(2))
                # show the trailing label (well name etc.) after the counter
                desc = line[m.end():].strip() or title
                prog.update(task, total=total, completed=done,
                            description=desc[:48])
        proc.wait()
        if proc.returncode == 0 and total is not None:
            prog.update(task, completed=total, description=f"{title} ✓")
    if proc.returncode != 0:
        console.print(f"[bold red]✗ {title} failed (exit {proc.returncode})[/]")
        console.print("[dim]" + "\n".join(tail[-25:]) + "[/dim]")
        raise SystemExit(proc.returncode)
    return time.perf_counter() - t0


def _pack_results(console, workdir: Path, out_path: Path,
                  experiments: list[str]) -> int:
    present = [d for d in _RESULT_DIRS if (workdir / d).exists()]
    if not present:
        _fail(console, "no results produced — nothing to bundle")
    manifest = {
        "bundle_version": 1,
        "kind": "results",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "experiments": experiments,
        "contents": present,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out_path, "w:gz") as tar:
        man = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(man)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(man))
        for d in present:
            tar.add(workdir / d, arcname=d)
    return out_path.stat().st_size


def _summary(console, workdir: Path, experiments: list[str],
             timings: dict[str, float], out_path: Path, out_bytes: int):
    from rich.table import Table
    from rich.panel import Panel

    tracks_dir = workdir / "tracks"
    n_tracks_files = len(list(tracks_dir.glob("*.json"))) if tracks_dir.is_dir() else 0
    cp_dir = workdir / ".cellpose_cache"
    n_cp_wells = sum(1 for d in cp_dir.iterdir() if d.is_dir()) if cp_dir.is_dir() else 0

    t = Table(show_header=False, box=None, pad_edge=False)
    t.add_column(style="bold")
    t.add_column()
    t.add_row("experiments", str(len(experiments)))
    t.add_row("segmented wells", str(n_cp_wells))
    t.add_row("track files", str(n_tracks_files))
    for stage, secs in timings.items():
        t.add_row(f"{stage} time", f"{secs:.1f}s")
    t.add_row("results bundle", f"{out_path}  ({out_bytes/1e6:.1f} MB)")
    console.print(Panel(t, title="[bold green]Analysis complete",
                        border_style="green"))
    console.print("[dim]Upload this bundle via 'Add segmentation data'.[/dim]")


# ------------------------------- main ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        prog="cellyoulite-analyze",
        description="Run align → segment → track on raw images and emit an "
                    "uploadable results bundle.")
    ap.add_argument("input", nargs="?",
                    help="folder of experiments, a folder with data/, or an "
                         "images_*.tar.gz. Omit to resume an existing --workdir.")
    ap.add_argument("--workdir", default="./cyl-analysis",
                    help="where the pipeline runs / caches live (default %(default)s)")
    ap.add_argument("-e", "--experiments", nargs="*", default=None,
                    help="subset of experiment folder names (default: all)")
    ap.add_argument("-o", "--out", default=None,
                    help="results bundle path (default results_<stamp>.tar.gz)")
    ap.add_argument("--workers", type=int, default=4,
                    help="CPU worker processes for segmentation (default %(default)s)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gpu", action="store_true", help="force GPU segmentation")
    g.add_argument("--cpu", action="store_true", help="force CPU segmentation")
    ap.add_argument("--force", action="store_true",
                    help="recompute everything, ignoring caches")
    ap.add_argument("--no-bundle", action="store_true",
                    help="leave results in the workdir; don't pack a .tar.gz")
    args = ap.parse_args()

    try:
        from rich.console import Console
    except ModuleNotFoundError:
        sys.stderr.write(
            "cellyoulite-analyze needs the 'analyze' extra:\n"
            "    pip install -e \".[analyze]\"\n")
        raise SystemExit(2)
    console = Console()

    if not _SCRIPTS.is_dir():
        _fail(console, f"analysis scripts not found at {_SCRIPTS} "
                       "(run from a source/editable install)")

    workdir = Path(args.workdir).expanduser().resolve()
    experiments = _setup_workdir(console, args.input, workdir)
    if args.experiments:
        wanted = set(args.experiments)
        unknown = wanted - set(experiments)
        if unknown:
            console.print(f"[yellow]warning:[/] not found, skipping: {sorted(unknown)}")
        experiments = [e for e in experiments if e in wanted]
        if not experiments:
            _fail(console, "no matching experiments to run")

    if args.cpu:
        device = "cpu"
    elif args.gpu:
        device = "cuda"
    else:
        device = _detect_device()
    use_gpu = device != "cpu"

    console.print(
        f"[bold]CellxYou Lite[/] · analyzing [cyan]{len(experiments)}[/] "
        f"experiment(s) · device [cyan]{device}[/] · workdir [dim]{workdir}[/]")
    if not use_gpu and _detect_device() == "cpu":
        # torch absent → only cache hits will succeed in segmentation
        try:
            import torch  # noqa: F401
        except ModuleNotFoundError:
            console.print("[yellow]note:[/] torch not installed — segmentation "
                          "will only succeed for already-cached frames.")

    py = sys.executable
    wells = (["--wells", *experiments]) if args.experiments else []
    force = ["--force"] if args.force else []

    align_cmd = [py, "-u", str(_SCRIPTS / "align_batch.py"),
                 "--folder", "data", *wells, *force]
    seg_cmd = [py, "-u", str(_SCRIPTS / "cellpose_batch.py"),
               "--folder", "data", *wells, *force]
    seg_cmd += ["--gpu"] if use_gpu else ["--workers", str(args.workers)]
    track_cmd = [py, "-u", str(_SCRIPTS / "cellpose_track.py"), *wells]

    timings: dict[str, float] = {}
    timings["align"] = _run_stage(console, "Align", align_cmd, workdir)
    timings["segment"] = _run_stage(console, "Segment (Cellpose-SAM)", seg_cmd, workdir)
    timings["track"] = _run_stage(console, "Track", track_cmd, workdir)

    out_path = Path(args.out).expanduser().resolve() if args.out else \
        (Path.cwd() / f"results_{time.strftime('%Y%m%d-%H%M%S')}.tar.gz")
    out_bytes = 0
    if not args.no_bundle:
        out_bytes = _pack_results(console, workdir, out_path, experiments)

    _summary(console, workdir, experiments, timings,
             out_path if not args.no_bundle else Path("(none — --no-bundle)"),
             out_bytes)


if __name__ == "__main__":
    main()
