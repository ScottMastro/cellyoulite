"""Score Cellpose cache vs manual annotations.

Reads .cellpose_cache/<well>/<label>.json (predictions in aligned-canvas
coords) and the hand-drawn ground truth in the database (in the same
coords) and prints precision / recall / ★recall / F1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cellyoulite.db import repo
from cellyoulite.db.migrate import migrate

MATCH_CENTER_FRAC = 0.5
MATCH_R_RATIO = 1.6


def _match(gt, pred):
    used = set()
    matches = []
    gt_order = sorted(range(len(gt)), key=lambda i: -gt[i]["r"])
    for gi in gt_order:
        g = gt[gi]
        best = None
        for pi, p in enumerate(pred):
            if pi in used:
                continue
            d = float(np.hypot(p["cx"] - g["cx"], p["cy"] - g["cy"]))
            if d > MATCH_CENTER_FRAC * g["r"]:
                continue
            ratio = p["r"] / g["r"]
            if ratio > MATCH_R_RATIO or ratio < 1.0 / MATCH_R_RATIO:
                continue
            if best is None or d < best[0]:
                best = (d, pi, abs(p["r"] - g["r"]))
        if best is not None:
            _, pi, r_err = best
            used.add(pi)
            matches.append((gi, pi, best[0], r_err))
    return matches, used


def main():
    migrate()
    cache_root = Path(".cellpose_cache")
    agg = {"tp": 0, "fp": 0, "fn": 0, "star_tp": 0, "star_fn": 0,
           "ce": [], "re": []}

    print(f"{'well':22s} {'label':10s} {'GT':>3s} {'P':>3s} {'TP':>3s} {'FP':>3s} {'FN':>3s} "
          f"{'prec':>5s} {'rec':>5s} {'★rec':>5s} {'Δc':>5s} {'Δr':>5s}")
    annotations = repo.list_annotations()
    if not annotations:
        raise SystemExit("no annotations in the database — see "
                         "scripts/import_annotations.py")
    for ann in annotations:
        batch, well, label = ann["batch"], ann["well"], ann["label"]
        cache_path = (cache_root / batch.replace("/", "_")
                      / well.replace("/", "_") / f"{label}.json")
        if not cache_path.is_file():
            print(f"{well:22s} {label:10s}  (no cellpose cache)")
            continue
        cache = json.loads(cache_path.read_text())
        gt = ann["circles"]
        pred = cache.get("circles", [])
        matches, used = _match(gt, pred)
        matched_gt = {m[0] for m in matches}
        star_tp = sum(1 for m in matches if gt[m[0]].get("star"))
        star_fn = sum(1 for i, g in enumerate(gt) if g.get("star") and i not in matched_gt)
        tp, fp, fn = len(matches), len(pred) - len(used), len(gt) - len(matches)
        ce = [m[2] for m in matches]
        re_ = [m[3] for m in matches]
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec = tp / (tp + fn) if (tp + fn) else 0
        srec = star_tp / (star_tp + star_fn) if (star_tp + star_fn) else 1
        mc = float(np.mean(ce)) if ce else 0
        mr = float(np.mean(re_)) if re_ else 0
        print(f"{well:22s} {label:10s} {len(gt):3d} {len(pred):3d} {tp:3d} {fp:3d} {fn:3d} "
              f"{prec:5.2f} {rec:5.2f} {srec:5.2f} {mc:5.1f} {mr:5.1f}")
        agg["tp"] += tp; agg["fp"] += fp; agg["fn"] += fn
        agg["star_tp"] += star_tp; agg["star_fn"] += star_fn
        agg["ce"] += ce; agg["re"] += re_

    TP, FP, FN = agg["tp"], agg["fp"], agg["fn"]
    prec = TP / (TP + FP) if (TP + FP) else 0
    rec = TP / (TP + FN) if (TP + FN) else 0
    srec = agg["star_tp"] / (agg["star_tp"] + agg["star_fn"]) if (agg["star_tp"] + agg["star_fn"]) else 1
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print("-" * 92)
    print(f"AGGREGATE  TP={TP} FP={FP} FN={FN} | prec={prec:.3f} rec={rec:.3f} "
          f"★rec={srec:.3f} | F1={f1:.3f} | Δc̄={np.mean(agg['ce']) if agg['ce'] else 0:.1f}px "
          f"Δr̄={np.mean(agg['re']) if agg['re'] else 0:.1f}px")


if __name__ == "__main__":
    main()
