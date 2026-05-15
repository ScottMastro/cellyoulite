<div align="center">

<img src="https://raw.githubusercontent.com/ScottMastro/cellyoulite/main/cellyoulite/web/static/logo.svg" width="64" alt="CellxYou Lite"/>

# CellxYou Lite

**Computer-vision tool for measuring organoid size over time.**

</div>

CellxYou Lite takes a time series of brightfield images, segments each organoid, and plots its size over time. Two segmentation models are produced per organoid: an exact pixel-mask area, and a best-fit circle. Results are grouped by treatment and rendered as boxplots across timepoints.

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:ScottMastro/cellyoulite.git
cd cellyoulite
pip install -e .
cellyoulite serve
```

The app opens in your browser at `http://localhost:8765`. Click **Generate demo** to try it on synthetic data.

Filenames must encode treatment + timepoint, e.g. `A01.tif`, `H_12.png`.
