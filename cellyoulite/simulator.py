"""Synthetic organoid time-series generator.

Produces one subdirectory per (treatment, replicate) — matching the layout
that `discover_grid` walks — with PNGs named `sim_<DDdHHhMMm>.png`.

Model:
- Each (treatment, well) seeds N organoids at random positions, stable across time.
- Each organoid has an initial radius and a per-organoid growth rate; the
  growth rate is drawn from a per-treatment distribution so treatments
  separate in the boxplots.
- Background is a gentle illumination gradient + Gaussian noise + a few
  dark speckles to imitate debris (the precision-vs-recall pressure test).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.draw import disk
from skimage.io import imsave


@dataclass(frozen=True)
class SimConfig:
    treatments: tuple[str, ...] = ("DMSO", "FSK", "FSK+CFTRinh", "012", "026", "012+S9A13", "026+S9A13")
    replicates: tuple[int, ...] = (1, 2, 3)
    n_timepoints: int = 12
    minutes_per_step: int = 15
    image_shape: tuple[int, int] = (512, 512)
    organoids_per_image: int = 6
    init_radius_range: tuple[float, float] = (12.0, 22.0)
    bg_intensity: int = 210
    fg_intensity: int = 70
    noise_sigma: float = 6.0
    debris_per_image: int = 8
    seed: int = 0


# Per-treatment growth rate per timestep (additive radius increase, in px).
_TREATMENT_RATES = {
    "DMSO": 0.3,
    "FSK": 2.6,
    "FSK+CFTRinh": 0.6,
    "012": 1.4,
    "026": 1.8,
    "012+S9A13": 2.1,
    "026+S9A13": 2.4,
}


def _minutes_label(minutes: int) -> str:
    d, rem = divmod(minutes, 1440)
    h, m = divmod(rem, 60)
    return f"{d:02d}d{h:02d}h{m:02d}m"


def _draw_circle(img: np.ndarray, cy: float, cx: float, r: float, fg: float) -> None:
    rr, cc = disk((cy, cx), r, shape=img.shape)
    img[rr, cc] = fg


def _illumination(shape, rng) -> np.ndarray:
    y = np.linspace(-1, 1, shape[0])[:, None]
    x = np.linspace(-1, 1, shape[1])[None, :]
    tilt_y, tilt_x = rng.uniform(-6, 6, size=2)
    return tilt_y * y + tilt_x * x


def generate_dataset(out_dir: str | Path, config: SimConfig = SimConfig()) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    h, w = config.image_shape

    for tx in config.treatments:
        base_rate = _TREATMENT_RATES.get(tx, 1.0)
        for rep in config.replicates:
            well_dir = out_dir / f"{tx} r{rep}"
            well_dir.mkdir(parents=True, exist_ok=True)
            positions = rng.uniform(
                low=[40, 40], high=[h - 40, w - 40],
                size=(config.organoids_per_image, 2),
            )
            init_radii = rng.uniform(*config.init_radius_range, size=config.organoids_per_image)
            per_organoid_rate = rng.normal(base_rate, base_rate * 0.15, size=config.organoids_per_image)
            per_organoid_rate = np.clip(per_organoid_rate, 0.05, None)

            for t in range(config.n_timepoints):
                img = np.full(config.image_shape, float(config.bg_intensity), dtype=np.float32)
                img += _illumination(config.image_shape, rng)

                for (cy, cx), r0, rate in zip(positions, init_radii, per_organoid_rate):
                    jitter_y, jitter_x = rng.normal(0, 0.5, size=2)
                    r = r0 + rate * t
                    _draw_circle(img, cy + jitter_y, cx + jitter_x, r, config.fg_intensity)

                for _ in range(config.debris_per_image):
                    dy = rng.integers(0, h)
                    dx = rng.integers(0, w)
                    dr = rng.uniform(1.5, 4.0)
                    _draw_circle(img, dy, dx, dr, rng.uniform(80, 140))

                img += rng.normal(0, config.noise_sigma, size=config.image_shape)
                img = np.clip(img, 0, 255).astype(np.uint8)

                label = _minutes_label(t * config.minutes_per_step)
                imsave(well_dir / f"sim_{label}.png", img, check_contrast=False)

    return out_dir
