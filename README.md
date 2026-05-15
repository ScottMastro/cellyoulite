<div align="center">

<img src="https://raw.githubusercontent.com/ScottMastro/cellyoulite/main/cellyoulite/web/static/logo.svg" width="64" alt="CellxYou Lite"/>

# CellxYou Lite

**Organoid segmentation and time-series analysis for wet-lab biologists.**

</div>

Drop a folder of brightfield time-lapse images into the app. CellxYou Lite segments each organoid, measures its size two ways (exact pixel mask and best-fit circle), and produces a boxplot of size over time per treatment. Runs locally — your images never leave your machine.

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
