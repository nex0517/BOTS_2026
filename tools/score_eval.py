#!/usr/bin/env python3
"""Evaluate discovery and daughter geometry against local reference annotations.

The five daughter counts were confirmed by the user as hackathon results.
Spatial labels retain their recorded review/measurement uncertainty. The 6 mm
matching tolerance is a local choice, not a published official scoring rule.
Only non-null seed-radius measurements are scored; missing values stay missing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from detector.branchseed import Config, detect, read_nifti  # noqa: E402


def _match(pred_points: list[np.ndarray], ref_points: list[np.ndarray], tolerance: float):
    if not pred_points or not ref_points:
        return [], np.zeros((len(pred_points), len(ref_points)))
    distances = np.array([[float(np.linalg.norm(p - r)) for r in ref_points] for p in pred_points])
    if tolerance < 0 or not np.isfinite(tolerance):
        raise ValueError("Matching tolerance must be finite and non-negative")
    # A rejected pair must cost more than every valid pair combined, ensuring
    # maximum cardinality first, then minimum total localisation error.
    penalty = (min(distances.shape) + 1) * (tolerance + 1)
    cost = np.where(distances <= tolerance, distances, penalty)
    pred_indices, ref_indices = linear_sum_assignment(cost)
    pairs = [
        (int(i), int(j), float(distances[i, j]))
        for i, j in zip(pred_indices, ref_indices)
        if distances[i, j] <= tolerance
    ]
    return pairs, distances


def _case_files(folder: Path) -> tuple[str, Path, Path, Path | None] | None:
    images = sorted(folder.glob("orig*.nii*"))
    if not images:
        return None
    image = images[0]
    stem = image.name.split(".")[0].replace("orig", "")
    masks = [p for pattern in ("mask*.nii*", "aorta*.nii*") for p in sorted(folder.glob(pattern)) if "daughter" not in p.name]
    if not masks:
        return None
    labels = folder / f"daughters{stem}_draft.nii.gz"
    return stem, image, masks[0], labels if labels.exists() else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="EVAL_SET", help="Folder of case_* subfolders with annotations.json")
    parser.add_argument("--tolerance", type=float, default=6.0, help="Ostium match tolerance in mm")
    parser.add_argument("--report", help="Optional JSON report destination")
    args = parser.parse_args(argv)

    try:
        import SimpleITK as sitk
    except Exception:  # pragma: no cover
        sitk = None

    root = Path(args.data_root)
    cases = sorted(p for p in root.glob("case_*") if (p / "annotations.json").exists())
    if not cases:
        raise SystemExit(f"No annotated cases under {root}")

    totals = {"tp": 0, "fp": 0, "fn": 0}
    ostium_errors: list[float] = []
    direction_cosines: list[float] = []
    radius_errors: list[float] = []
    seed_hits: list[bool] = []
    seed_errors: list[float] = []
    traced_count = 0
    fallback_count = 0
    runtimes: list[float] = []
    per_case = []

    for folder in cases:
        files = _case_files(folder)
        if files is None:
            continue
        stem, image_path, mask_path, label_path = files
        annotations = json.loads((folder / "annotations.json").read_text())
        references = annotations["daughters"]
        started = time.perf_counter()
        ct = read_nifti(image_path)
        mask = read_nifti(mask_path)
        result, diagnostics = detect(ct, mask, Config())
        runtime = time.perf_counter() - started
        runtimes.append(runtime)

        predictions = result["daughters"]
        traced_count += sum(d.get("path_status") == "traced" for d in diagnostics["branches"])
        fallback_count += sum(d.get("path_status") != "traced" for d in diagnostics["branches"])
        pairs, distances = _match(
            [np.asarray(d["ostium_xyz_mm"], dtype=float) for d in predictions],
            [np.asarray(r["ostium_xyz_mm"], dtype=float) for r in references],
            args.tolerance,
        )
        tp = len(pairs)
        fp = len(predictions) - tp
        fn = len(references) - tp
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

        labels = None
        if label_path is not None and sitk is not None:
            labels = sitk.GetArrayFromImage(sitk.ReadImage(str(label_path))).transpose(2, 1, 0)

        matched_detail = []
        for i, j, distance in pairs:
            ostium_errors.append(distance)
            cosine = float(
                np.dot(
                    np.asarray(predictions[i]["direction_xyz"], dtype=float),
                    np.asarray(references[j]["direction_xyz"], dtype=float),
                )
            )
            direction_cosines.append(cosine)
            reference_radius = references[j].get("radius_mm")
            radius_error = None
            if reference_radius is not None and np.isfinite(reference_radius) and reference_radius > 0:
                radius_error = abs(float(predictions[i]["radius_mm"]) - float(reference_radius))
                radius_errors.append(radius_error)
            seed_error = float(np.linalg.norm(
                np.asarray(predictions[i]["seed_xyz_mm"]) - references[j]["seed_xyz_mm"]
            ))
            seed_errors.append(seed_error)
            hit = None
            if labels is not None and ct.sitk_image is not None:
                index = np.asarray(
                    ct.sitk_image.TransformPhysicalPointToIndex(
                        tuple(float(v) for v in predictions[i]["seed_xyz_mm"])
                    )
                )
                hit = False
                if np.all(index >= 0) and np.all(index < np.asarray(labels.shape)):
                    hit = bool(labels[tuple(index)] == references[j]["label_value"])
                seed_hits.append(hit)
            matched_detail.append(
                {
                    "prediction": predictions[i]["instance_id"],
                    "reference": references[j]["instance_id"],
                    "ostium_distance_mm": round(distance, 2),
                    "direction_cosine": round(cosine, 3),
                    "seed_inside_reference_label": hit,
                    "seed_distance_mm": round(seed_error, 3),
                    "seed_radius_error_mm": round(radius_error, 3) if radius_error is not None else None,
                }
            )

        missed = [
            references[j]["instance_id"]
            for j in range(len(references))
            if j not in {p[1] for p in pairs}
        ]
        per_case.append(
            {
                "case": folder.name,
                "references": len(references),
                "predictions": len(predictions),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "missed_references": missed,
                "runtime_s": round(runtime, 2),
                "threshold_hu": diagnostics["threshold"],
                "matched": matched_detail,
                "unmatched_predictions": [
                    {
                        "instance_id": predictions[i]["instance_id"],
                        "ostium_xyz_mm": predictions[i]["ostium_xyz_mm"],
                        "radius_mm": predictions[i]["radius_mm"],
                        "nearest_reference_mm": round(float(distances[i].min()), 1) if len(references) else None,
                    }
                    for i in range(len(predictions))
                    if i not in {p[0] for p in pairs}
                ],
            }
        )
        print(
            f"{folder.name}: ref={len(references)} pred={len(predictions)} TP={tp} FP={fp} FN={fn} "
            f"missed={missed} runtime={runtime:.2f}s"
        )

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    summary = {
        "tolerance_mm": args.tolerance,
        **totals,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_ostium_error_mm": round(float(np.mean(ostium_errors)), 2) if ostium_errors else None,
        "median_ostium_error_mm": round(float(np.median(ostium_errors)), 2) if ostium_errors else None,
        "mean_direction_cosine": round(float(np.mean(direction_cosines)), 3) if direction_cosines else None,
        "radius_measurement_count": len(radius_errors),
        "mean_seed_error_mm": round(float(np.mean(seed_errors)), 3) if seed_errors else None,
        "traced_predictions": traced_count,
        "unresolved_path_predictions": fallback_count,
        "mean_radius_error_mm": round(float(np.mean(radius_errors)), 2) if radius_errors else None,
        "seed_inside_reference_label": f"{sum(1 for h in seed_hits if h)}/{len(seed_hits)}" if seed_hits else None,
        "mean_runtime_s": round(float(np.mean(runtimes)), 2) if runtimes else None,
        "max_runtime_s": round(float(np.max(runtimes)), 2) if runtimes else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    if args.report:
        destination = Path(args.report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "methodology": {"matching": "maximum-cardinality, minimum-distance one-to-one",
                            "seed_hit": "nearest native voxel; no dilation",
                            "radius_reference": "radius_mm at seed; null values excluded",
                            "scope": "development cases; spatial annotations retain review uncertainty"},
            "summary": summary, "cases": per_case}, indent=2) + "\n", encoding="utf-8")
        print(f"report: {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
