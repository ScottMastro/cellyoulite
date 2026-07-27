"""Move a pre-batch working directory into the batched layout. Idempotent.

Before batching, everything was keyed by well folder name alone:

    data/<well>/<images>          tracks/<well>.json
    .cellpose_cache/<well>/       tracks/<well>/<stitch PNGs>
    annotations/<well>/

Well names repeat across batches ("DMSO r1" exists in more than one), so each
of those roots gains a batch directory:

    data/<batch>/<well>/...       tracks/<batch>/<well>.json      etc.

Run once per existing dataset, with the service stopped:

    python -m scripts.reorganize_batches --batch "CA1 T1"   # dry run
    python -m scripts.reorganize_batches --batch "CA1 T1" --apply

Alignment caches are NOT moved: .align_cache entries are named
<well>__<fingerprint>, and the fingerprint already differs between batches, so
they cannot collide. Moving files within a filesystem is a rename — the images
themselves are never copied, and alignment fingerprints hash the immediate
parent directory name, so the cache stays valid across the move.
"""
from __future__ import annotations

import argparse
from pathlib import Path

# Roots that become <root>/<batch>/... . .align_cache is deliberately absent.
_ROOTS = ["data", "tracks", ".cellpose_cache", "annotations"]


def _safe(name: str) -> str:
    return name.replace("/", "_")


def _pending(root: Path, batch: str) -> list[Path]:
    """Entries still sitting at the top level, i.e. not yet under a batch."""
    if not root.is_dir():
        return []
    dest = root / _safe(batch)
    return sorted(p for p in root.iterdir() if p.resolve() != dest.resolve())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", required=True,
                    help='batch to move the existing data under, e.g. "CA1 T1"')
    ap.add_argument("--apply", action="store_true",
                    help="actually move (default: report what would move)")
    args = ap.parse_args()
    batch = _safe(args.batch)

    plan: list[tuple[Path, Path]] = []
    for name in _ROOTS:
        root = Path(name)
        for src in _pending(root, args.batch):
            plan.append((src, root / batch / src.name))

    if not plan:
        print(f"nothing to move — already organised under '{args.batch}'")
        return

    by_root: dict[str, int] = {}
    for src, _ in plan:
        by_root[src.parts[0]] = by_root.get(src.parts[0], 0) + 1
    print(f"{'moving' if args.apply else 'would move'} {len(plan)} entries "
          f"into '{batch}':")
    for name, n in sorted(by_root.items()):
        print(f"  {name}/ → {name}/{batch}/   ({n} entries)")

    if not args.apply:
        print("\ndry run — pass --apply to perform the move")
        return

    # Refuse to clobber: a destination that already exists means a partial or
    # repeated run against different data, which needs a human.
    clashes = [dst for _, dst in plan if dst.exists()]
    if clashes:
        raise SystemExit(
            f"refusing to move: {len(clashes)} destination(s) already exist, "
            f"first is {clashes[0]}")

    for src, dst in plan:
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)
    print(f"moved {len(plan)} entries")
    print("next: restart the app (migrations run on startup), then\n"
          f'      python -m scripts.import_annotations --batch "{args.batch}"')


if __name__ == "__main__":
    main()
