<div align="center">

<img src="https://raw.githubusercontent.com/ScottMastro/cellyoulite/main/cellyoulite/web/static/logo.svg" width="72" alt="Cell+You Lite"/>

# Cell+You Lite

**Organoid segmentation and time-series analysis for wet-lab biologists.**

A local web app that turns a folder of brightfield time-lapse images into clean per-organoid size measurements and treatment-vs-time boxplots — no command line, no cloud, your images never leave your machine.

</div>

---

## What it does

Given a grid of greyscale images organised by **treatment × timepoint**, Cell+You Lite:

1. **Segments** each organoid with a classical pipeline (Gaussian → Otsu → morphology → connected components).
2. Reports **two size models** per organoid:
   - **Pixel-volume** — exact mask area.
   - **Best-fit circle** — the equivalent-area circle that best approximates the mask.
3. Filters aggressively for confident detections (**precision over recall**) — solidity, eccentricity, and area thresholds drop ambiguous objects.
4. Produces a faceted boxplot of organoid size over time, one panel per treatment, plus a long-format CSV.

## Why two segmentation models?

- The **pixel mask** is the ground truth for area; it captures the real, possibly irregular shape.
- The **best-fit circle** is the idealised summary biologists already use mentally ("an organoid of radius *r*"). It's the more robust readout when masks are noisy.

Reporting both lets you sanity-check one against the other: when they diverge, the segmentation is probably misbehaving.

## Install

Requires Python 3.10+. From source for now:

```bash
git clone git@github.com:ScottMastro/cellyoulite.git
cd cellyoulite
pip install -e .
```

Once published, the user-facing install will be a single double-click installer that drops a desktop shortcut. The shortcut does `pipx upgrade cellyoulite && cellyoulite serve`, so updates are silent and automatic.

## Use

```bash
cellyoulite serve            # open browser → http://localhost:8765
cellyoulite simulate         # generate synthetic dataset (./sim-data)
cellyoulite analyze --folder ./sim-data
cellyoulite version
```

### In the UI

1. **Load** — paste a folder path, or click *Generate demo* to create a synthetic dataset.
2. **Segment** — click *Analyze folder*. Each image is segmented and QC-filtered.
3. **Plot** — boxplots appear inline, one panel per treatment, x-axis = timepoint.

### Filename convention

Each image filename must contain a treatment letter (A–H) followed by a timepoint number, e.g. `A01.tif`, `B_07.png`, `h-12.jpeg`. Files that don't match are skipped.

## Project layout

```
cellyoulite/
├─ pipeline/          # segmentation, circle fit, QC — pure functions, testable
│  ├─ segment.py
│  ├─ circle_fit.py
│  ├─ qc.py
│  └─ analyze.py
├─ io/                # grid discovery, CSV export
├─ plotting/          # Plotly boxplots
├─ server/            # FastAPI app
├─ web/               # templates + static assets
├─ simulator.py       # synthetic time-series generator
└─ launcher.py        # CLI entry point
tests/
CONTEXT.md            # problem statement, scope, design rationale
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

The pipeline is decoupled from the UI — every module under `cellyoulite/pipeline/` is a pure function over numpy arrays and can be tested without booting the server.

## Distribution

- **Updates:** `git tag v0.x.y && git push --tags` triggers `.github/workflows/publish.yml`, which builds and publishes to PyPI. The launcher runs `pipx upgrade cellyoulite` on every start, so users get new versions on next launch.
- **Cross-platform:** pure-Python install, no compiled binaries to ship. Same package on Windows and macOS.

## Status

Early scaffold. Pipeline works end-to-end on synthetic data; on a 96-image demo grid, ~83% of seeded organoids pass QC. Real-data tuning, per-image overlay viewer, and background job + progress streaming are next.

See [`CONTEXT.md`](./CONTEXT.md) for the full problem statement and design rationale.
