#!/usr/bin/env python3
"""Run the official Branchseed submission checks without manual point placement.

With no data path, this script exercises synthetic versions of the important
cases listed in the challenge. With --data-root, it runs every discovered
orig*/mask* pair, writes one prediction per case, creates at least three visual
checks, and reports runtime plus contract compliance.
"""

from __future__ import annotations

import argparse
import json
import math
import time
import tracemalloc
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from BOTS_2026.detector.branchseed import Config, Volume, detect, read_nifti, validate_output


ROOT = Path(__file__).resolve().parent
OFFICIAL_RUN = "python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json"
SYNTHETIC_EXPECTATIONS = {
    "variable_branches": (2, None),
    "close_origins": (2, None),
    "common_trunk": (1, 1),
    "downstream_branch": (1, 1),
    "flat_cap": (0, 0),
    "empty": (0, 0),
}


def _path_points(z0: float, side: int = 1, y_slope: float = 0.0, start: int = 4, stop: int = 22):
    cx = cy = 36.0
    return [(cx + side * (7 + step), cy + y_slope * step, z0) for step in range(start, stop)]


def _paint_path(ct: np.ndarray, points, radius: float, value: float = 240.0) -> None:
    x, y, z = np.indices(ct.shape)
    for px, py, pz in points:
        tube = (x - px) ** 2 + (y - py) ** 2 + (z - pz) ** 2 <= radius ** 2
        ct[tube] = value


def make_scenario(name: str) -> tuple[Volume, Volume]:
    shape = (72, 72, 112)
    rng = np.random.default_rng(20260912)
    x, y, z = np.indices(shape)
    aorta = ((x - 36) ** 2 + (y - 36) ** 2 <= 8 ** 2) & (z >= 10) & (z <= 101)
    ct = rng.normal(32, 6, shape).astype(np.float32)
    ct[aorta] = rng.normal(245, 7, int(aorta.sum()))
    if name == "variable_branches":
        _paint_path(ct, _path_points(34, 1, 0.00), 2.8)
        _paint_path(ct, _path_points(57, -1, 0.16), 2.5)
        _paint_path(ct, _path_points(79, 1, -0.12), 2.2)
    elif name == "close_origins":
        _paint_path(ct, _path_points(44, 1, -0.10, 2, 22), 1.6)
        _paint_path(ct, _path_points(50, 1, 0.12, 2, 22), 1.6)
    elif name == "common_trunk":
        trunk = _path_points(55, 1, 0.0, 4, 16)
        _paint_path(ct, trunk, 2.5)
        junction = trunk[-1]
        arm_a = [(junction[0] + s, junction[1] + 0.55 * s, junction[2]) for s in range(1, 9)]
        arm_b = [(junction[0] + s, junction[1] - 0.55 * s, junction[2]) for s in range(1, 9)]
        _paint_path(ct, arm_a + arm_b, 2.0)
    elif name == "downstream_branch":
        main = _path_points(55, 1, 0.0, 4, 22)
        _paint_path(ct, main, 2.4)
        junction = main[10]
        child = [(junction[0], junction[1] + s, junction[2] + 0.25 * s) for s in range(1, 9)]
        _paint_path(ct, child, 1.7)
    elif name == "flat_cap":
        _paint_path(ct, [(36, 36, zz) for zz in range(99, 112)], 3.0)
    elif name != "empty":
        raise ValueError(f"Unknown scenario: {name}")
    affine = np.eye(4)
    return Volume(ct, affine, np.ones(3)), Volume(aorta.astype(np.uint8), affine, np.ones(3))


def _output_invariants(result: dict) -> list[str]:
    failures = []
    try:
        validate_output(result)
    except Exception as exc:
        failures.append(str(exc))
        return failures
    ids = [d["instance_id"] for d in result["daughters"]]
    if ids != [f"branch_{i:03d}" for i in range(1, len(ids) + 1)]:
        failures.append("daughter IDs are not deterministic branch_### values")
    for daughter in result["daughters"]:
        direction = np.asarray(daughter["direction_xyz"], dtype=float)
        if abs(float(np.linalg.norm(direction)) - 1.0) > 2e-7:
            failures.append(f"{daughter['instance_id']}: direction is not unit length")
        ostium = np.asarray(daughter["ostium_xyz_mm"], dtype=float)
        seed = np.asarray(daughter["seed_xyz_mm"], dtype=float)
        if abs(float(np.linalg.norm(seed - ostium)) - 5.0) > 0.01:
            failures.append(f"{daughter['instance_id']}: seed is not 5 mm from ostium")
    return failures


def _normalized_projection(array: np.ndarray, axis: int) -> np.ndarray:
    view = np.max(array, axis=axis).T
    lo, hi = np.percentile(view, [1, 99.5])
    return np.clip((view - lo) / max(1e-6, hi - lo) * 255, 0, 255).astype(np.uint8)


def _outline(mask: np.ndarray) -> np.ndarray:
    p = np.pad(mask.astype(bool), 1)
    eroded = p[1:-1, 1:-1].copy()
    eroded &= p[:-2, 1:-1]
    eroded &= p[2:, 1:-1]
    eroded &= p[1:-1, :-2]
    eroded &= p[1:-1, 2:]
    return mask.astype(bool) & ~eroded


def _draw_arrow(draw: ImageDraw.ImageDraw, start, end, color=(255, 138, 76), width=3):
    draw.line([start, end], fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    size = 9
    left = (end[0] - size * math.cos(angle - 0.55), end[1] - size * math.sin(angle - 0.55))
    right = (end[0] - size * math.cos(angle + 0.55), end[1] - size * math.sin(angle + 0.55))
    draw.polygon([end, left, right], fill=color)


def write_visual_check(ct: Volume, mask: Volume, result: dict, destination: Path, title: str) -> None:
    panel_size = 300
    header = 72
    canvas = Image.new("RGB", (panel_size * 3, panel_size + header + 34), (7, 20, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 16), f"BRANCHSEED VISUAL CHECK | {title}", fill=(232, 241, 242))
    draw.text((20, 42), "cyan = parent aorta mask   orange = ostium to 5 mm seed", fill=(137, 162, 173))
    specs = [(2, "AXIAL MIP", (0, 1)), (1, "CORONAL MIP", (0, 2)), (0, "SAGITTAL MIP", (1, 2))]
    mask_bool = mask.data > 0
    for panel, (axis, label, dims) in enumerate(specs):
        gray = Image.fromarray(_normalized_projection(ct.data, axis), mode="L").convert("RGB")
        gray = gray.resize((panel_size, panel_size), Image.Resampling.BILINEAR)
        x0 = panel * panel_size
        canvas.paste(gray, (x0, header))
        overlay = ImageDraw.Draw(canvas)
        mask_projection = np.any(mask_bool, axis=axis).T
        edge = _outline(mask_projection)
        rows, cols = edge.shape
        ys, xs = np.where(edge)
        for ex, ey in zip(xs[::2], ys[::2]):
            px = x0 + int(ex / max(1, cols - 1) * (panel_size - 1))
            py = header + int(ey / max(1, rows - 1) * (panel_size - 1))
            overlay.point((px, py), fill=(92, 225, 230))
        for daughter in result["daughters"]:
            points = ct.physical_to_index(np.array([daughter["ostium_xyz_mm"], daughter["seed_xyz_mm"]]))
            projected = []
            for point in points:
                u, v = point[dims[0]], point[dims[1]]
                projected.append((x0 + u / max(1, ct.data.shape[dims[0]] - 1) * (panel_size - 1), header + v / max(1, ct.data.shape[dims[1]] - 1) * (panel_size - 1)))
            _draw_arrow(overlay, projected[0], projected[1])
            overlay.ellipse((projected[0][0]-4, projected[0][1]-4, projected[0][0]+4, projected[0][1]+4), outline=(255,255,255), width=2)
        overlay.rectangle((x0, header, x0 + panel_size - 1, header + panel_size - 1), outline=(32, 61, 73))
        overlay.text((x0 + 10, header + panel_size + 9), label, fill=(232, 241, 242))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def _timed_detect(ct: Volume, mask: Volume, config: Config):
    tracemalloc.start()
    started = time.perf_counter()
    result, diagnostics = detect(ct, mask, config)
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    diagnostics["runtime_s"] = round(elapsed, 4)
    diagnostics["python_tracemalloc_peak_mb"] = round(peak / (1024 * 1024), 2)
    return result, diagnostics


def run_synthetic(output_root: Path) -> dict:
    predictions = output_root / "synthetic_predictions"
    visuals = output_root / "visual_checks"
    cases = []
    all_passed = True
    visual_names = {"variable_branches", "close_origins", "common_trunk"}
    for name, (minimum, maximum) in SYNTHETIC_EXPECTATIONS.items():
        ct, mask = make_scenario(name)
        result, diagnostics = _timed_detect(ct, mask, Config())
        result["case_id"] = f"synthetic_{name}"
        validate_output(result)
        rerun, _ = detect(ct, mask, Config())
        rerun["case_id"] = result["case_id"]
        count = len(result["daughters"])
        failures = _output_invariants(result)
        if json.dumps(result, sort_keys=True) != json.dumps(rerun, sort_keys=True):
            failures.append("rerun was not byte-equivalent after canonical JSON serialization")
        if count < minimum or (maximum is not None and count > maximum):
            failures.append(f"expected branch count in [{minimum}, {maximum}], got {count}")
        case_passed = not failures
        all_passed &= case_passed
        prediction_path = predictions / f"{name}.json"
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        prediction_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if name in visual_names:
            write_visual_check(ct, mask, result, visuals / f"{name}.png", f"SYNTHETIC {name.upper()}")
        cases.append({"case": name, "passed": case_passed, "detected_daughters": count, "failures": failures, **{k: diagnostics[k] for k in ("runtime_s", "python_tracemalloc_peak_mb")}})
    return {"mode": "synthetic", "passed": all_passed, "cases": cases, "visual_check_count": len(list(visuals.glob("*.png")))}


def _find_pairs(data_root: Path):
    for image_path in sorted(data_root.rglob("orig*.nii*")):
        masks = sorted(image_path.parent.glob("mask*.nii*"))
        if len(masks) == 1:
            yield image_path.parent.name, image_path, masks[0]


def run_dataset(data_root: Path, output_root: Path, min_radius_mm: float) -> dict:
    predictions = output_root / "development_predictions"
    visuals = output_root / "visual_checks"
    cases = []
    all_passed = True
    for index, (case_id, image_path, mask_path) in enumerate(_find_pairs(data_root)):
        started = time.perf_counter()
        ct, mask = read_nifti(image_path), read_nifti(mask_path)
        result, diagnostics = _timed_detect(ct, mask, Config(min_radius_mm=min_radius_mm))
        diagnostics["runtime_s_including_io"] = round(time.perf_counter() - started, 4)
        result["case_id"] = case_id
        failures = _output_invariants(result)
        passed = not failures
        all_passed &= passed
        predictions.mkdir(parents=True, exist_ok=True)
        (predictions / f"{case_id}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if index < 3:
            write_visual_check(ct, mask, result, visuals / f"{case_id}.png", case_id.upper())
        cases.append({"case": case_id, "passed": passed, "detected_daughters": len(result["daughters"]), "failures": failures, "runtime_s": diagnostics["runtime_s_including_io"], "python_tracemalloc_peak_mb": diagnostics["python_tracemalloc_peak_mb"]})
    average = round(sum(c["runtime_s"] for c in cases) / len(cases), 3) if cases else None
    if average is not None and average > 60:
        all_passed = False
    return {"mode": "development-data", "passed": all_passed and bool(cases), "case_count": len(cases), "average_runtime_s": average, "runtime_target_s": 60, "visual_check_count": len(list(visuals.glob("*.png"))), "cases": cases}


def submission_inventory(project_root: Path, test_report: dict) -> dict:
    readme = (project_root / "README.md").read_text(encoding="utf-8") if (project_root / "README.md").exists() else ""
    prediction_dir = project_root / "submission" / "development_predictions"
    visual_dir = project_root / "submission" / "visual_checks"
    checks = {
        "source_code": (project_root / "run.py").exists() and (project_root / "detector" / "branchseed.py").exists(),
        "dependency_file": (project_root / "requirements.txt").exists(),
        "readme_with_setup_command": "python -m pip install -r requirements.txt" in readme,
        "readme_with_exact_run_command": OFFICIAL_RUN in readme,
        "development_predictions": prediction_dir.exists() and len(list(prediction_dir.glob("*.json"))) > 0,
        "three_visual_checks": visual_dir.exists() and len(list(visual_dir.glob("*.png"))) >= 3,
        "five_minute_demo": (project_root / "PITCH.md").exists(),
        "self_tests_pass": bool(test_report.get("passed")),
    }
    return {"checks": checks, "submission_ready": all(checks.values()), "note": "Development predictions require the organizers' development cases; synthetic predictions do not satisfy that submission item."}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Branchseed contract, edge-case, visual, runtime and submission checks")
    parser.add_argument("--data-root", type=Path, help="Optional organizer development-set root containing subject folders")
    parser.add_argument("--output-root", type=Path, default=ROOT / "submission", help="Report/prediction/visual destination")
    parser.add_argument("--min-radius-mm", type=float, default=0.0, help="Minimum eligible radius supplied with the final dataset")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = run_dataset(args.data_root, args.output_root, max(0.0, args.min_radius_mm)) if args.data_root else run_synthetic(args.output_root)
    report["submission"] = submission_inventory(ROOT, report)
    report["official_run_command"] = OFFICIAL_RUN
    report_path = args.output_root / "self_test_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "mode": report["mode"], "cases": len(report["cases"]), "visual_checks": report["visual_check_count"], "submission_ready": report["submission"]["submission_ready"], "report": str(report_path)}))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
