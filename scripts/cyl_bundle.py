"""Package or restore a fully-processed CellxYou Lite dataset.

The bundle bakes the artifacts we generate around a dataset into a single
self-contained .tar.gz file:

  .align_cache/        per-well alignment cache
  .cellpose_cache/     per-frame circles JSON + 16-bit mask PNG sidecar
  tracks/              per-well track JSON, validation overrides, cached
                       per-track stitch PNGs
  annotations/         hand-drawn ground truth (validation set)
  manifest.json        bundle metadata (created, version, contents)
  data/                (only with --include-raw)

The raw TIFFs in data/ are large and usually already on the machine you're
moving to, so they're excluded by default. Pass --include-raw to ship a
fully self-contained archive.

Usage:
    python scripts/cyl_bundle.py export out/dataset.tar.gz
    python scripts/cyl_bundle.py export out/dataset.tar.gz --include-raw
    python scripts/cyl_bundle.py inspect out/dataset.tar.gz
    python scripts/cyl_bundle.py import out/dataset.tar.gz [target_dir]
"""
from __future__ import annotations

import argparse
import json
import tarfile
import time
from pathlib import Path


_BUNDLE_VERSION = 1

# Directories included by default. Order matters for the manifest report.
_DEFAULT_DIRS = [
    ".align_cache",
    ".cellpose_cache",
    "tracks",
    "annotations",
]


def _du_bytes(path: Path) -> int:
    """Sum file sizes under `path`."""
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file():
                total += p.stat().st_size
        except OSError:
            continue
    return total


def _count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for p in path.rglob("*") if p.is_file())


def _human(bytes_: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if bytes_ < 1024:
            return f"{bytes_:.1f} {unit}"
        bytes_ /= 1024
    return f"{bytes_:.1f} TB"


def export(out_path: Path, root: Path, include_raw: bool) -> None:
    dirs = list(_DEFAULT_DIRS)
    if include_raw:
        dirs.append("data")

    contents = []
    for d in dirs:
        p = root / d
        if not p.exists():
            continue
        contents.append({
            "name": d,
            "files": _count_files(p),
            "bytes": _du_bytes(p),
        })

    if not contents:
        raise SystemExit("nothing to bundle — none of the expected dirs exist "
                          f"under {root}/")

    manifest = {
        "bundle_version": _BUNDLE_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(root.resolve()),
        "include_raw": include_raw,
        "contents": contents,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    total = sum(c["bytes"] for c in contents)
    print(f"bundling {len(contents)} dirs · {_human(total)} of source data → {out_path}")
    print(f"  ({'including' if include_raw else 'excluding'} data/)")

    with tarfile.open(out_path, "w:gz") as tar:
        # Manifest first so `inspect` can be cheap.
        import io
        man_bytes = json.dumps(manifest, indent=2).encode()
        info = tarfile.TarInfo("manifest.json")
        info.size = len(man_bytes)
        info.mtime = int(time.time())
        tar.addfile(info, io.BytesIO(man_bytes))
        for c in contents:
            src = root / c["name"]
            print(f"  + {c['name']:18s}  {c['files']:5d} files  {_human(c['bytes']):>9s}")
            tar.add(src, arcname=c["name"])

    print(f"\ndone → {out_path}  ({_human(out_path.stat().st_size)} on disk)")


def inspect(bundle: Path) -> None:
    with tarfile.open(bundle, "r:gz") as tar:
        m = tar.getmember("manifest.json")
        f = tar.extractfile(m)
        manifest = json.loads(f.read())
    print(f"bundle: {bundle}  ({_human(bundle.stat().st_size)})")
    print(f"  version    : {manifest.get('bundle_version')}")
    print(f"  created_at : {manifest.get('created_at')}")
    print(f"  include_raw: {manifest.get('include_raw')}")
    print(f"  source     : {manifest.get('source_root')}")
    print("  contents:")
    for c in manifest.get("contents", []):
        print(f"    {c['name']:18s}  {c['files']:5d} files  {_human(c['bytes']):>9s}")


def import_(bundle: Path, target: Path) -> None:
    target = Path(target).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    print(f"extracting {bundle} → {target}/")
    with tarfile.open(bundle, "r:gz") as tar:
        manifest_member = tar.getmember("manifest.json")
        manifest = json.loads(tar.extractfile(manifest_member).read())
        print(f"  bundle created {manifest.get('created_at')}, "
              f"include_raw={manifest.get('include_raw')}")
        # Safe extraction — refuse paths that escape the target.
        for m in tar.getmembers():
            target_path = (target / m.name).resolve()
            if target.resolve() not in target_path.parents and target.resolve() != target_path:
                print(f"  SKIP (escapes target): {m.name}")
                continue
        # Python 3.12+ exposes the filter API. Use 'data' filter for safety.
        tar.extractall(target, filter="data")
    for c in manifest.get("contents", []):
        p = target / c["name"]
        print(f"  ✓ {c['name']:18s}  {c['files']:5d} files  "
              f"{_human(_du_bytes(p)):>9s}")
    print("\ndone.")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_exp = sub.add_parser("export", help="package the current dataset")
    p_exp.add_argument("out", help="path to write the .tar.gz")
    p_exp.add_argument("--root", default=".", help="project root (default cwd)")
    p_exp.add_argument("--include-raw", action="store_true",
                        help="also bundle the raw data/ TIFFs (large)")

    p_ins = sub.add_parser("inspect", help="show a bundle's manifest")
    p_ins.add_argument("bundle", help="bundle .tar.gz to inspect")

    p_imp = sub.add_parser("import", help="restore a bundle to a directory")
    p_imp.add_argument("bundle", help="bundle .tar.gz to restore")
    p_imp.add_argument("target", nargs="?", default=".",
                        help="target dir (default cwd)")

    args = ap.parse_args()
    if args.cmd == "export":
        export(Path(args.out), Path(args.root), args.include_raw)
    elif args.cmd == "inspect":
        inspect(Path(args.bundle))
    elif args.cmd == "import":
        import_(Path(args.bundle), Path(args.target))


if __name__ == "__main__":
    main()
