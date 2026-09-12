#!/usr/bin/env python3
"""CPU-first Branchseed baseline with a scorer-safe JSON contract.

The implementation is intentionally classical and auditable. It accepts NIfTI-1
CT/mask pairs (including gzip content stored behind a .nii suffix), crops around
the supplied aorta, learns contrast from that mask, proposes connected bright
structures touching the wall, proves short path continuity, and estimates the
required proximal geometry in physical millimetres.

This is a hackathon MVP, not a medical device. Thresholds are configuration,
and outputs must be reviewed on the organizers' development cases.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import struct
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

try:  # Fast path when SciPy is available; the fallback keeps the demo portable.
    from scipy import ndimage as ndi  # type: ignore
except Exception:  # pragma: no cover - exercised by the dependency-light demo
    ndi = None

try:
    import SimpleITK as sitk  # type: ignore
except Exception:  # Synthetic self-tests do not require NIfTI I/O.
    sitk = None


DTYPES = {
    2: np.uint8,
    4: np.int16,
    8: np.int32,
    16: np.float32,
    64: np.float64,
    256: np.int8,
    512: np.uint16,
    768: np.uint32,
}


@dataclass(frozen=True)
class Volume:
    data: np.ndarray
    affine: np.ndarray
    spacing: np.ndarray
    sitk_image: object | None = None
    index_offset: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))

    def index_to_physical(self, xyz: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(xyz).astype(float)
        source_points = points + self.index_offset
        if self.sitk_image is not None:
            converted = []
            for point in source_points:
                if np.allclose(point, np.rint(point), atol=1e-9):
                    converted.append(self.sitk_image.TransformIndexToPhysicalPoint(tuple(int(v) for v in np.rint(point))))
                else:
                    converted.append(self.sitk_image.TransformContinuousIndexToPhysicalPoint(tuple(float(v) for v in point)))
            return np.asarray(converted, dtype=float)
        return source_points @ self.affine[:3, :3].T + self.affine[:3, 3]

    def physical_to_index(self, xyz: np.ndarray) -> np.ndarray:
        points = np.atleast_2d(xyz).astype(float)
        inv = np.linalg.inv(self.affine)
        homogeneous = np.c_[points, np.ones(points.shape[0])]
        return (homogeneous @ inv.T)[:, :3] - self.index_offset


@dataclass(frozen=True)
class Config:
    crop_margin_mm: float = 25.0
    shell_mm: float = 15.0
    cap_exclusion_mm: float = 5.0
    acceptance_path_mm: float = 5.0
    max_path_mm: float = 10.0
    merge_radius_mm: float = 2.7
    min_radius_mm: float = 0.0
    min_component_voxels: int = 8


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        magic = handle.read(2)
    opener = gzip.open if magic == b"\x1f\x8b" else open
    with opener(path, "rb") as handle:
        return handle.read()


def _quaternion_affine(header: bytes, endian: str, spacing: np.ndarray) -> np.ndarray:
    b, c, d = struct.unpack_from(endian + "fff", header, 256)
    qx, qy, qz = struct.unpack_from(endian + "fff", header, 268)
    a = math.sqrt(max(0.0, 1.0 - (b * b + c * c + d * d)))
    r = np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-c*c-b*b],
    ])
    pixdim0 = struct.unpack_from(endian + "f", header, 76)[0]
    scales = spacing.copy()
    scales[2] *= -1.0 if pixdim0 < 0 else 1.0
    affine = np.eye(4)
    affine[:3, :3] = r @ np.diag(scales)
    affine[:3, 3] = [qx, qy, qz]
    return affine


def _read_nifti_raw(path: Path) -> Volume:
    """Minimal NIfTI-1 fallback used only when explicitly allowed in tests."""
    path = Path(path)
    raw = _read_bytes(path)
    if len(raw) < 352:
        raise ValueError(f"{path.name}: file is too small to be NIfTI-1")
    little = struct.unpack_from("<I", raw, 0)[0]
    big = struct.unpack_from(">I", raw, 0)[0]
    endian = "<" if little == 348 else ">" if big == 348 else None
    if endian is None:
        raise ValueError(f"{path.name}: unsupported NIfTI header")
    dims = struct.unpack_from(endian + "8h", raw, 40)
    if dims[0] < 3:
        raise ValueError(f"{path.name}: expected a 3D volume")
    shape = tuple(int(v) for v in dims[1:4])
    datatype = struct.unpack_from(endian + "h", raw, 70)[0]
    if datatype not in DTYPES:
        raise ValueError(f"{path.name}: unsupported NIfTI datatype {datatype}")
    dtype = np.dtype(DTYPES[datatype]).newbyteorder(endian)
    spacing = np.abs(np.array(struct.unpack_from(endian + "3f", raw, 80), dtype=float))
    if np.any(spacing <= 0):
        raise ValueError(f"{path.name}: invalid voxel spacing {spacing.tolist()}")
    offset = max(352, int(round(struct.unpack_from(endian + "f", raw, 108)[0])))
    count = int(np.prod(shape))
    flat = np.frombuffer(raw, dtype=dtype, count=count, offset=offset)
    if flat.size != count:
        raise ValueError(f"{path.name}: incomplete voxel payload")
    data = flat.reshape(shape, order="F")
    slope = struct.unpack_from(endian + "f", raw, 112)[0]
    intercept = struct.unpack_from(endian + "f", raw, 116)[0]
    if slope not in (0.0, 1.0) or intercept != 0.0:
        data = data.astype(np.float32) * (slope if slope else 1.0) + intercept
    sform_code = struct.unpack_from(endian + "h", raw, 254)[0]
    qform_code = struct.unpack_from(endian + "h", raw, 252)[0]
    if sform_code > 0:
        affine = np.eye(4)
        affine[:3] = np.array(struct.unpack_from(endian + "12f", raw, 280)).reshape(3, 4)
    elif qform_code > 0:
        affine = _quaternion_affine(raw, endian, spacing)
    else:
        affine = np.diag([*spacing, 1.0])
    return Volume(np.asarray(data), affine, spacing)


def _orthonormalize_sform(raw: bytearray) -> bytearray:
    """Replace a non-orthonormal sform rotation with its nearest orthonormal fit.

    Some organizer NIfTI files carry sform matrices with tiny floating-point
    drift in their direction cosines. SimpleITK refuses to load these outright,
    so we snap the rotation to the nearest orthogonal matrix (via SVD) while
    preserving voxel spacing and the translation, and leave everything else
    (voxel data, qform, spacing) untouched.
    """
    endian = "<" if struct.unpack_from("<I", raw, 0)[0] == 348 else ">"
    sform_code = struct.unpack_from(endian + "h", raw, 254)[0]
    if sform_code <= 0:
        return raw
    vals = np.array(struct.unpack_from(endian + "12f", raw, 280), dtype=float).reshape(3, 4)
    rotation, translation = vals[:, :3], vals[:, 3]
    col_norms = np.linalg.norm(rotation, axis=0)
    col_norms[col_norms == 0] = 1.0
    unit_rotation = rotation / col_norms
    u, _, vt = np.linalg.svd(unit_rotation)
    fixed_rotation = (u @ vt) * col_norms
    fixed = np.concatenate([fixed_rotation, translation[:, None]], axis=1).astype(np.float32)
    struct.pack_into(endian + "12f", raw, 280, *fixed.flatten().tolist())
    return raw


def read_nifti(path: str | Path, require_simpleitk: bool = True) -> Volume:
    """Read NIfTI through SimpleITK and preserve its physical coordinate system.

    The raw parser remains available for dependency-light format tests, but the
    submission CLI requires SimpleITK as directed by the official challenge.
    """
    path = Path(path)
    if sitk is None:
        if require_simpleitk:
            raise RuntimeError("SimpleITK is required for organizer NIfTI cases. Run: python -m pip install -r requirements.txt")
        return _read_nifti_raw(path)
    temporary: Path | None = None
    try:
        with path.open("rb") as handle:
            gzip_content = handle.read(2) == b"\x1f\x8b"
        source = path
        if gzip_content and not path.name.lower().endswith(".gz"):
            handle = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            temporary = Path(handle.name)
            handle.write(path.read_bytes())
            handle.close()
            source = temporary
        try:
            image = sitk.ReadImage(str(source))
        except RuntimeError as exc:
            if "orthonormal direction cosines" not in str(exc):
                raise
            repaired = _orthonormalize_sform(bytearray(_read_bytes(Path(source))))
            handle = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
            with gzip.open(handle.name, "wb") as gz:
                gz.write(bytes(repaired))
            handle.close()
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            temporary = source = Path(handle.name)
            image = sitk.ReadImage(str(source))
        if image.GetDimension() != 3 or image.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{path.name}: expected one scalar 3D volume")
        data = sitk.GetArrayFromImage(image).transpose(2, 1, 0)
        spacing = np.asarray(image.GetSpacing(), dtype=float)
        direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
        affine = np.eye(4)
        affine[:3, :3] = direction @ np.diag(spacing)
        affine[:3, 3] = np.asarray(image.GetOrigin(), dtype=float)
        return Volume(np.asarray(data), affine, spacing, sitk_image=image)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    if ndi is not None:
        return ndi.binary_dilation(mask, iterations=iterations)
    out = mask.astype(bool, copy=True)
    for _ in range(iterations):
        p = np.pad(out, 1, mode="constant")
        out = p[1:-1, 1:-1, 1:-1].copy()
        out |= p[:-2, 1:-1, 1:-1]
        out |= p[2:, 1:-1, 1:-1]
        out |= p[1:-1, :-2, 1:-1]
        out |= p[1:-1, 2:, 1:-1]
        out |= p[1:-1, 1:-1, :-2]
        out |= p[1:-1, 1:-1, 2:]
    return out


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    if ndi is not None:
        return ndi.label(mask)
    labels = np.zeros(mask.shape, dtype=np.int32)
    points = {tuple(v) for v in np.argwhere(mask)}
    label = 0
    shape = mask.shape
    while points:
        label += 1
        seed = points.pop()
        stack = [seed]
        labels[seed] = label
        while stack:
            x, y, z = stack.pop()
            for q in ((x-1,y,z),(x+1,y,z),(x,y-1,z),(x,y+1,z),(x,y,z-1),(x,y,z+1)):
                if 0 <= q[0] < shape[0] and 0 <= q[1] < shape[1] and 0 <= q[2] < shape[2] and q in points:
                    points.remove(q)
                    labels[q] = label
                    stack.append(q)
    return labels, label


def _crop(ct: Volume, mask: Volume, config: Config) -> tuple[Volume, np.ndarray, np.ndarray]:
    binary = mask.data > 0
    coords = np.argwhere(binary)
    if coords.size == 0:
        raise ValueError("Aorta mask is empty")
    pad = np.ceil(config.crop_margin_mm / ct.spacing).astype(int)
    lo = np.maximum(0, coords.min(axis=0) - pad)
    hi = np.minimum(binary.shape, coords.max(axis=0) + pad + 1)
    sl = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    affine = ct.affine.copy()
    offset = np.zeros(3, dtype=float)
    if ct.sitk_image is not None:
        offset = ct.index_offset + lo
    else:
        affine[:3, 3] = ct.index_to_physical(lo)[0]
    return Volume(
        ct.data[sl].astype(np.float32),
        affine,
        ct.spacing,
        sitk_image=ct.sitk_image,
        index_offset=offset,
    ), binary[sl], lo


def _component_candidate(
    coords: np.ndarray,
    wall: np.ndarray,
    ct: Volume,
    threshold: float,
    contrast_scale: float,
    config: Config,
) -> dict | None:
    if coords.shape[0] < config.min_component_voxels:
        return None
    touching = coords[wall[tuple(coords.T)]]
    if touching.size == 0:
        return None
    physical = ct.index_to_physical(coords)
    ostium_idx = np.median(touching, axis=0)
    ostium = ct.index_to_physical(ostium_idx)[0]
    centered = physical - ostium
    covariance = centered.T @ centered / max(1, centered.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, int(np.argmax(values))]
    centroid_vector = physical.mean(axis=0) - ostium
    if float(np.dot(direction, centroid_vector)) < 0:
        direction *= -1
    projections = centered @ direction
    positive = projections[projections >= 0]
    path_mm = float(np.percentile(positive, 98)) if positive.size else 0.0
    if path_mm < config.acceptance_path_mm:
        return None
    seed = ostium + direction * config.acceptance_path_mm
    proximal = (projections >= 0) & (projections <= min(config.max_path_mm, path_mm))
    voxel_volume = abs(float(np.linalg.det(ct.affine[:3, :3])))
    measured_length = max(config.acceptance_path_mm, min(path_mm, config.max_path_mm))
    radius = math.sqrt(max(1e-6, int(proximal.sum()) * voxel_volume / (math.pi * measured_length)))
    signal = float(np.median(ct.data[tuple(coords.T)]))
    continuity = min(1.0, path_mm / config.max_path_mm)
    contrast = 1.0 / (1.0 + math.exp(-(signal - threshold) / max(contrast_scale, 1.0)))
    confidence = float(np.clip(0.46 + 0.34 * continuity + 0.20 * contrast, 0.0, 0.99))
    return {
        "ostium_xyz_mm": [round(float(v), 3) for v in ostium],
        "seed_xyz_mm": [round(float(v), 3) for v in seed],
        "radius_mm": round(float(radius), 3),
        "direction_xyz": [round(float(v), 8) for v in direction / np.linalg.norm(direction)],
        "path_mm": round(path_mm, 3),
        "confidence": round(confidence, 3),
        "evidence": {
            "median_signal": round(signal, 2),
            "adaptive_threshold": round(threshold, 2),
            "component_voxels": int(coords.shape[0]),
        },
    }


def detect(ct: Volume, mask: Volume, config: Config = Config()) -> tuple[dict, dict]:
    if ct.data.shape != mask.data.shape:
        raise ValueError(f"CT/mask shape mismatch: {ct.data.shape} vs {mask.data.shape}")
    if not np.allclose(ct.affine, mask.affine, atol=1e-4):
        raise ValueError("CT and aorta mask do not share the same physical transform")
    cropped, aorta, crop_origin = _crop(ct, mask, config)
    axis = int(np.argmax(np.ptp(np.argwhere(aorta), axis=0) * cropped.spacing))
    iterations = max(1, int(math.ceil(config.shell_mm / float(np.min(cropped.spacing)))))
    shell = _dilate(aorta, iterations) & ~aorta
    wall = _dilate(aorta, 1) & ~aorta
    coords = np.argwhere(aorta)
    cap_voxels = max(1, int(math.ceil(config.cap_exclusion_mm / cropped.spacing[axis])))
    cap_lo, cap_hi = int(coords[:, axis].min()) + cap_voxels, int(coords[:, axis].max()) - cap_voxels
    grid_axis = np.indices(aorta.shape, sparse=True)[axis]
    valid_arc = (grid_axis >= cap_lo) & (grid_axis <= cap_hi)
    shell &= valid_arc
    wall &= valid_arc
    core_values = cropped.data[aorta]
    shell_values = cropped.data[shell]
    core_median = float(np.median(core_values))
    core_mad = float(np.median(np.abs(core_values - core_median)))
    background = float(np.median(shell_values)) if shell_values.size else core_median - 30
    contrast_scale = max(10.0, 1.4826 * core_mad)
    threshold = max(background + 0.45 * contrast_scale, core_median - 1.35 * contrast_scale)
    candidates = shell & (cropped.data >= threshold)
    labels, count = _label_components(candidates)
    found: list[dict] = []
    for label in range(1, count + 1):
        component = np.argwhere(labels == label)
        item = _component_candidate(component, wall, cropped, threshold, contrast_scale, config)
        if item is not None and item["radius_mm"] >= config.min_radius_mm:
            found.append(item)
    found.sort(key=lambda d: d["confidence"], reverse=True)
    kept: list[dict] = []
    for candidate in found:
        o = np.array(candidate["ostium_xyz_mm"])
        direction = np.array(candidate["direction_xyz"])
        duplicate = any(
            np.linalg.norm(o - np.array(k["ostium_xyz_mm"])) < config.merge_radius_mm
            and float(np.dot(direction, np.array(k["direction_xyz"]))) > 0.65
            for k in kept
        )
        if not duplicate:
            kept.append(candidate)
    kept.sort(key=lambda d: tuple(d["ostium_xyz_mm"][::-1]))
    daughters = []
    diagnostics = []
    for index, item in enumerate(kept, start=1):
        instance_id = f"branch_{index:03d}"
        daughters.append({
            "instance_id": instance_id,
            "parent_instance_id": "aorta",
            "ostium_xyz_mm": item["ostium_xyz_mm"],
            "seed_xyz_mm": item["seed_xyz_mm"],
            "radius_mm": item["radius_mm"],
            "direction_xyz": item["direction_xyz"],
        })
        diagnostics.append({"instance_id": instance_id, **item})
    result = {"case_id": "case", "parent": {"instance_id": "aorta"}, "daughters": daughters}
    sidecar = {
        "pipeline": "adaptive-shell-baseline",
        "physical_transform_backend": "SimpleITK" if ct.sitk_image is not None else "synthetic-or-test-affine",
        "min_radius_mm": config.min_radius_mm,
        "crop_origin_index": crop_origin.tolist(),
        "crop_shape": list(cropped.data.shape),
        "aorta_median": round(core_median, 2),
        "aorta_mad": round(core_mad, 2),
        "background_median": round(background, 2),
        "threshold": round(threshold, 2),
        "raw_components": int(count),
        "accepted_daughters": len(daughters),
        "branches": diagnostics,
    }
    validate_output(result)
    return result, sidecar


def validate_output(result: dict) -> None:
    if set(result) != {"case_id", "parent", "daughters"}:
        raise ValueError("Output must contain exactly case_id, parent, and daughters")
    if not isinstance(result.get("case_id"), str) or not result["case_id"]:
        raise ValueError("case_id must be a non-empty string")
    if result.get("parent") != {"instance_id": "aorta"}:
        raise ValueError("parent must be exactly {'instance_id': 'aorta'}")
    if not isinstance(result.get("daughters"), list):
        raise ValueError("daughters must be a list")
    ids: set[str] = set()
    for d in result.get("daughters", []):
        branch_id = d.get("instance_id")
        if not isinstance(branch_id, str) or branch_id in ids:
            raise ValueError("Every daughter needs a unique instance_id")
        ids.add(branch_id)
        if d.get("parent_instance_id") != "aorta":
            raise ValueError(f"{branch_id}: parent_instance_id must be aorta")
        for field in ("ostium_xyz_mm", "seed_xyz_mm", "direction_xyz"):
            values = np.asarray(d.get(field), dtype=float)
            if values.shape != (3,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{branch_id}: invalid {field}")
        radius = float(d.get("radius_mm", 0))
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"{branch_id}: radius must be positive")
        norm = float(np.linalg.norm(np.asarray(d["direction_xyz"], dtype=float)))
        if abs(norm - 1.0) > 2e-7:
            raise ValueError(f"{branch_id}: direction norm is {norm}")


def synthetic_case(shape: tuple[int, int, int] = (72, 72, 112)) -> tuple[Volume, Volume]:
    rng = np.random.default_rng(42)
    x, y, z = np.indices(shape)
    cx, cy = shape[0] / 2, shape[1] / 2
    aorta = ((x - cx) ** 2 + (y - cy) ** 2 <= 8.0 ** 2) & (z >= 10) & (z <= shape[2] - 11)
    ct = rng.normal(34, 9, shape).astype(np.float32)
    ct[aorta] = rng.normal(245, 10, int(aorta.sum()))
    branch_specs = [(34, 1, 0, 3.0), (55, -1, .25, 2.6), (78, 1, -.2, 2.2)]
    for z0, side, slope, radius in branch_specs:
        for step in range(4, 22):
            bx = cx + side * (7 + step)
            by = cy + slope * step
            tube = ((x - bx) ** 2 + (y - by) ** 2 <= radius ** 2) & (np.abs(z - z0) <= 1)
            ct[tube] = rng.normal(230, 11, int(tube.sum()))
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    return Volume(ct, affine, np.ones(3)), Volume(aorta.astype(np.uint8), affine, np.ones(3))


def _json_dump(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect direct aortic daughter origins from CTA + aorta mask")
    parser.add_argument("--image", help="CT/CTA NIfTI file (.nii or gzip content)")
    parser.add_argument("--aorta-mask", "--mask", dest="aorta_mask", help="Binary parent-aorta NIfTI file")
    parser.add_argument("--output", required=True, help="Strict prediction JSON destination")
    parser.add_argument("--diagnostics", help="Optional diagnostics sidecar JSON")
    parser.add_argument("--case-id", help="Stable case ID; defaults to the image stem")
    parser.add_argument("--min-radius-mm", type=float, default=0.0, help="Organizer-specified minimum eligible radius")
    parser.add_argument("--demo", action="store_true", help="Run a deterministic synthetic smoke-test case")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.demo and (not args.image or not args.aorta_mask):
        raise SystemExit("Provide --image and --aorta-mask, or use --demo")
    started = time.perf_counter()
    if args.demo:
        ct, mask = synthetic_case()
        case_id = args.case_id or "synthetic_demo"
    else:
        ct, mask = read_nifti(args.image), read_nifti(args.aorta_mask)
        case_id = args.case_id or Path(args.image).name.split(".")[0]
    result, diagnostics = detect(ct, mask, Config(min_radius_mm=max(0.0, args.min_radius_mm)))
    result["case_id"] = case_id
    validate_output(result)
    diagnostics["runtime_s"] = round(time.perf_counter() - started, 3)
    _json_dump(Path(args.output), result)
    if args.diagnostics:
        _json_dump(Path(args.diagnostics), diagnostics)
    print(json.dumps({"case_id": case_id, "daughters": len(result["daughters"]), "runtime_s": diagnostics["runtime_s"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
