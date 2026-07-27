"""Score the circle detector against manual annotations.

For each annotated frame in the database, load the corresponding image,
align it, run detect_circles(), and compare to the ground-truth circles.

Match rule: a predicted circle (cx', cy', r') matches a GT circle (cx, cy, r)
if center distance <= MATCH_CENTER_FRAC * r AND r' / r is in [1/MATCH_R_RATIO,
MATCH_R_RATIO]. Greedy matching by smallest center error.

Reports per-frame and aggregate: precision, recall, mean center / radius
error on matched pairs, plus recall on the STARRED subset (the canonical
should-never-miss examples).

Usage:
    python scripts/circle_score.py
    python scripts/circle_score.py --annotated-stills out/scored
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
from skimage.io import imread

from cellyoulite.db import repo
from cellyoulite.db.migrate import migrate
from cellyoulite.io.grid import discover_grid
from cellyoulite.pipeline.align import compute_alignment_cached, paste_onto_canvas
from cellyoulite.pipeline.circle_methods import detect_circles

MATCH_CENTER_FRAC = 0.5   # |Δcenter| <= this × GT radius
MATCH_R_RATIO = 1.6       # GT/pred radius ratio must lie within [1/x, x]


@dataclass
class FrameScore:
    well: str
    label: str
    n_gt: int
    n_pred: int
    n_star: int
    tp: int
    fp: int
    fn: int
    star_tp: int
    star_fn: int
    center_errs: list[float]
    radius_errs: list[float]

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def star_recall(self) -> float:
        denom = self.star_tp + self.star_fn
        return self.star_tp / denom if denom else 1.0


def _match(gt: list[dict], pred: list[tuple[float, float, float]]):
    """Greedy: for each GT, find the closest unmatched pred satisfying both
    center and radius constraints. Returns (matches, gt_matched, pred_matched)
    where matches is a list of (gt_idx, pred_idx, center_err, radius_err)."""
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float, float]] = []
    # Sort GT by descending radius — large/easy ones get first dibs on preds.
    gt_order = sorted(range(len(gt)), key=lambda i: -gt[i]["r"])
    for gi in gt_order:
        g = gt[gi]
        best = None
        for pi, (px, py, pr) in enumerate(pred):
            if pi in used_pred:
                continue
            d = float(np.hypot(px - g["cx"], py - g["cy"]))
            if d > MATCH_CENTER_FRAC * g["r"]:
                continue
            ratio = pr / g["r"]
            if ratio > MATCH_R_RATIO or ratio < 1.0 / MATCH_R_RATIO:
                continue
            r_err = abs(pr - g["r"])
            score = d  # rank by center distance
            if best is None or score < best[0]:
                best = (score, pi, d, r_err)
        if best is not None:
            _, pi, d, r_err = best
            used_pred.add(pi)
            matches.append((gi, pi, d, r_err))
    return matches, used_pred


def score_frame(well_obj, t_idx: int, ann: dict) -> tuple[FrameScore, np.ndarray, list]:
    """Run the detector on this frame and compare to GT. Returns (score,
    aligned RGB frame, list of predicted circles)."""
    paths = [tp.path for tp in well_obj.timepoints]
    align = compute_alignment_cached(paths)
    raw = imread(well_obj.timepoints[t_idx].path)
    if raw.ndim == 2:
        raw = np.stack([raw] * 3, axis=-1)
    if raw.dtype != np.uint8:
        raw = np.clip(raw[..., :3], 0, 255).astype(np.uint8)
    frame = paste_onto_canvas(raw[..., :3], align.placements[t_idx],
                              align.canvas_shape, fill=0)

    pred = detect_circles(frame)
    gt = ann["circles"]
    matches, used = _match(gt, pred)

    matched_gt = {m[0] for m in matches}
    star_tp = sum(1 for m in matches if gt[m[0]].get("star"))
    star_fn = sum(1 for i, g in enumerate(gt) if g.get("star") and i not in matched_gt)

    s = FrameScore(
        well=well_obj.folder_name,
        label=ann["label"],
        n_gt=len(gt),
        n_pred=len(pred),
        n_star=sum(1 for g in gt if g.get("star")),
        tp=len(matches),
        fp=len(pred) - len(used),
        fn=len(gt) - len(matches),
        star_tp=star_tp,
        star_fn=star_fn,
        center_errs=[m[2] for m in matches],
        radius_errs=[m[3] for m in matches],
    )
    return s, frame, pred, matches


def _annotate_frame(frame: np.ndarray, gt: list[dict],
                    pred: list[tuple[float, float, float]],
                    matches: list[tuple[int, int, float, float]]) -> np.ndarray:
    """Draw GT (green/gold), predictions (red unmatched, blue matched)."""
    out = frame.copy()
    matched_gt = {m[0] for m in matches}
    matched_pred = {m[1] for m in matches}

    # GT first.
    for i, g in enumerate(gt):
        color = (130, 255, 130) if not g.get("star") else (90, 200, 255)
        if i not in matched_gt:
            color = (255, 90, 255)  # magenta = missed
        cv2.circle(out, (int(g["cx"]), int(g["cy"])), int(g["r"]),
                   color, thickness=2, lineType=cv2.LINE_AA)
    # Predictions.
    for i, (cx, cy, r) in enumerate(pred):
        color = (90, 180, 255) if i in matched_pred else (255, 90, 90)
        cv2.circle(out, (int(cx), int(cy)), int(r),
                   color, thickness=1, lineType=cv2.LINE_AA)
        cv2.drawMarker(out, (int(cx), int(cy)), color,
                       cv2.MARKER_CROSS, markerSize=10, thickness=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="data")
    ap.add_argument("--batch", default=None,
                    help="only score this batch (default: every batch)")
    ap.add_argument("--annotated-stills", default=None,
                    help="if set, write GT-vs-pred overlay PNGs to this dir")
    args = ap.parse_args()

    migrate()
    spec = discover_grid(args.folder)
    wells_by_name = {w.folder_name: w for w in spec.wells}

    annotations = repo.list_annotations(args.batch)
    if not annotations:
        raise SystemExit("no annotations in the database — see "
                         "scripts/import_annotations.py")

    out_dir = Path(args.annotated_stills) if args.annotated_stills else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    scores: list[FrameScore] = []
    print(f"{'well':22s} {'t':10s} {'GT':>3s} {'P':>3s} {'TP':>3s} {'FP':>3s} {'FN':>3s} "
          f"{'prec':>5s} {'rec':>5s} {'★rec':>5s} {'Δc':>5s} {'Δr':>5s}")
    for ann in annotations:
        well_name = ann["well"]
        label = ann["label"]
        well = wells_by_name.get(well_name)
        if well is None:
            print(f"  skip — no well {well_name!r}")
            continue
        t_idx = next((i for i, tp in enumerate(well.timepoints) if tp.label == label), None)
        if t_idx is None:
            print(f"  skip — no timepoint {label!r} in {well_name}")
            continue
        s, frame, pred, matches = score_frame(well, t_idx, ann)
        scores.append(s)
        mc = float(np.mean(s.center_errs)) if s.center_errs else 0.0
        mr = float(np.mean(s.radius_errs)) if s.radius_errs else 0.0
        print(f"{s.well:22s} {s.label:10s} {s.n_gt:3d} {s.n_pred:3d} "
              f"{s.tp:3d} {s.fp:3d} {s.fn:3d} {s.precision:5.2f} {s.recall:5.2f} "
              f"{s.star_recall:5.2f} {mc:5.1f} {mr:5.1f}")
        if out_dir:
            stamp = _annotate_frame(frame, ann["circles"], pred, matches)
            cv2.rectangle(stamp, (0, 0), (stamp.shape[1], 28), (0, 0, 0), -1)
            txt = (f"{s.well} {s.label}  GT={s.n_gt} P={s.n_pred} "
                   f"TP={s.tp} FP={s.fp} FN={s.fn}  "
                   f"prec={s.precision:.2f} rec={s.recall:.2f} *rec={s.star_recall:.2f}")
            cv2.putText(stamp, txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1, cv2.LINE_AA)
            safe = (s.well + "__" + s.label).replace(" ", "_").replace("+", "p")
            imageio.imwrite(out_dir / f"{safe}.png", stamp)

    if not scores:
        return

    # Aggregate.
    TP = sum(s.tp for s in scores)
    FP = sum(s.fp for s in scores)
    FN = sum(s.fn for s in scores)
    STP = sum(s.star_tp for s in scores)
    SFN = sum(s.star_fn for s in scores)
    all_ce = [e for s in scores for e in s.center_errs]
    all_re = [e for s in scores for e in s.radius_errs]
    prec = TP / (TP + FP) if (TP + FP) else 0.0
    rec = TP / (TP + FN) if (TP + FN) else 0.0
    srec = STP / (STP + SFN) if (STP + SFN) else 1.0

    print("-" * 88)
    print(f"AGGREGATE: TP={TP} FP={FP} FN={FN} | prec={prec:.3f} rec={rec:.3f} "
          f"| ★rec={srec:.3f} (★TP={STP} ★FN={SFN}) "
          f"| Δc̄={np.mean(all_ce) if all_ce else 0:.1f}px "
          f"Δr̄={np.mean(all_re) if all_re else 0:.1f}px")
    if out_dir:
        print(f"overlays in {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
