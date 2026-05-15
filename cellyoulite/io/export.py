from __future__ import annotations

from pathlib import Path

import pandas as pd


def export_csv(records: list[dict], out_path: str | Path) -> Path:
    """Long-format CSV: one row per (treatment, timepoint, organoid_id)."""
    out_path = Path(out_path)
    df = pd.DataFrame.from_records(records)
    df.to_csv(out_path, index=False)
    return out_path
