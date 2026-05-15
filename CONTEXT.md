# CellxYou Lite — Project Context

## One-line
A desktop app that segments organoids from a grid of brightfield/greyscale time-lapse images, measures their size across time, and produces simple plots — designed to be launched by a wet-lab biologist on Windows or macOS with no command line.

## Domain
- **Sample:** organoids growing in plates, imaged repeatedly over time.
- **Input data shape:** a 2D grid of images.
  - Columns = timepoints (e.g. T1–T12).
  - Rows = treatment conditions (e.g. A–H).
  - Each cell of the grid is one image (one well / one field of view at one time).
- **Image properties:**
  - Greyscale (black-and-white).
  - Organoids are roughly **round**, generally **darker** than the background.
  - Spatial registration is loose-but-stable: an organoid at position (x, y) at T1 is *approximately* at the same (x, y) at T2…T12. This is a strong prior we should exploit.
  - Background is **noisy** (debris, well edges, lighting artefacts, out-of-focus material).

## Goal
For every organoid we are confident about, track its size across time, and report distributions per treatment.

## Two segmentation models (per organoid, per timepoint)
1. **Pixel-volume model** — exact segmentation mask. Area in pixels (or µm² if calibration is provided) is the primary readout.
2. **Best-fit circle model** — the circle that best approximates the segmentation. Reported as radius / area. Useful as a denoised/idealised summary that matches biologists' mental model of organoids as spheres.

Both readouts are computed for every confident organoid at every timepoint.

## Precision over recall (explicit)
The user has prioritised **precision over recall**: it is better to drop ambiguous organoids than to report false positives. Downstream filters should:
- Discard objects that fail roundness / size / contrast thresholds.
- Discard tracks that don't appear consistently across the time series.
- Prefer "no measurement" to "wrong measurement".

## Outputs
- **Per-image overlays** — original image with segmentation contour + fitted circle, for QC.
- **Per-organoid time series** — area_px and circle_area_px per timepoint.
- **Per-treatment summary plot** — boxplot of organoid sizes at each timepoint, one panel per treatment (or grouped).
- **CSV export** — long-format table (treatment, timepoint, organoid_id, area_px, circle_radius_px, circle_area_px, qc_flags).

## User & deployment constraints
- **User:** wet-lab biologist. Will not open a terminal. Will not `pip install`.
- **Platform:** Windows or macOS (unknown which — must support both).
- **Launch experience:** double-click an app icon → GUI opens.
- **Workflow inside the app:**
  1. Point at a folder of images (or load a grid).
  2. Confirm the grid layout (rows = treatments, cols = timepoints), with auto-detection where possible.
  3. Run segmentation (progress bar, no console).
  4. Review per-image overlays, manually exclude bad wells.
  5. Generate plots and export CSV.

## Out of scope (for v1)
- 3D / z-stack handling.
- Fluorescence / multi-channel images.
- Live acquisition — we only process pre-acquired images.
- Cloud / multi-user features.

## Open questions (to revisit when real images arrive)
- Image file format (TIFF? PNG? proprietary microscope format?).
- Whether each grid cell is one image file or whether one big stitched image must be split.
- Pixel-to-micron calibration.
- Typical organoid size range in pixels (sets the scale for filters).
