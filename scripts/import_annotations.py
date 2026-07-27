"""One-shot importer: annotations/*.json -> SQLite (migration v4). Idempotent.

Hand-drawn ground-truth circles used to live only on disk, keyed by well name
with no batch scoping. This loads them into the `annotation` tables so they sit
alongside every other authored decision — and inside the database backup.

Two on-disk layouts are accepted, since the reorganisation into batches may or
may not have happened yet:

    annotations/<batch>/<well>/<label>.json     (batched)
    annotations/<well>/<label>.json             (flat, pre-batch)

A flat tree needs --batch to say which batch its wells belong to.

    python -m scripts.import_annotations --batch "CA1 (May 2026)"
    python -m scripts.import_annotations            # batched tree

Existing rows are left alone unless --replace is given: these are authored and
irreplaceable, so a re-run must never quietly overwrite newer editing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cellyoulite.db import repo
from cellyoulite.db.migrate import migrate

_ANN_ROOT = Path("annotations")


def _find(root: Path, batch: str | None) -> list[tuple[str, str, str, Path]]:
    """[(batch, well, label, path)] for every annotation JSON under `root`."""
    out: list[tuple[str, str, str, Path]] = []
    for p in sorted(root.rglob("*.json")):
        rel = p.relative_to(root).parts
        if len(rel) == 3:                      # <batch>/<well>/<label>.json
            out.append((rel[0], rel[1], p.stem, p))
        elif len(rel) == 2:                    # <well>/<label>.json
            if batch is None:
                raise SystemExit(
                    f"{p} is in the flat pre-batch layout — pass --batch to say "
                    "which batch these wells belong to")
            out.append((batch, rel[0], p.stem, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", default=None,
                    help='batch for a flat annotations/<well>/ tree')
    ap.add_argument("--replace", action="store_true",
                    help="overwrite frames already in the DB (default: skip)")
    args = ap.parse_args()

    migrate()
    if not _ANN_ROOT.is_dir():
        print(f"no {_ANN_ROOT}/ — nothing to import")
        return

    found = _find(_ANN_ROOT, args.batch)
    if not found:
        print(f"no annotation files under {_ANN_ROOT}/")
        return

    n_new = n_skipped = n_bad = 0
    for batch, well, label, path in found:
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as e:
            print(f"  skip {path}: {e}")
            n_bad += 1
            continue
        circles = data.get("circles") or []
        if repo.get_annotation(batch, well, label) is not None and not args.replace:
            n_skipped += 1
            continue
        repo.set_annotation(
            batch, well, label, circles,
            aligned=bool(data.get("aligned")), source_key=data.get("key"),
            image_w=data.get("image_w"), image_h=data.get("image_h"))
        print(f"  {batch} · {well} · {label}: {len(circles)} circle(s)")
        n_new += 1

    print(f"imported {n_new} frame(s); skipped {n_skipped} already present"
          + (f"; {n_bad} unreadable" if n_bad else ""))
    print(f"database now holds {len(repo.list_annotations())} annotated frame(s)")


if __name__ == "__main__":
    main()
