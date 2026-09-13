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
import os

import numpy as np

from scipy import ndimage as ndi
from .tracking import _basis, refine_candidates

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
            if len(source_points) > 1:
                # Bulk component geometry uses the exact affine built from
                # SimpleITK's origin, direction and spacing. Individual output
                # landmarks below still use the SimpleITK transform directly.
                return source_points @ self.affine[:3, :3].T + self.affine[:3, 3]
            converted = []
            for point in source_points:
                if np.allclose(point, np.rint(point), atol=1e-9, rtol=0):
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
    cap_exclusion_mm: float = 4.0
    acceptance_path_mm: float = 5.0
    max_path_mm: float = 10.0
    merge_radius_mm: float = 2.7
    min_radius_mm: float = 0.0
    min_component_voxels: int = 8
    # Width of the annulus just outside the parent mask that is excluded from
    # the candidate search. A supplied lumen mask is usually a voxel or two
    # inside the true lumen, so the un-masked rim of the parent is bright and
    # forms a sheet that wraps the aorta and fuses every daughter into one
    # component. Cutting that rim out separates the daughters again; each
    # component is then re-anchored to the parent surface for its ostium.
    wall_gap_mm: float = 2.0
    # A component larger than this is not a proximal branch segment; it is the
    # periaortic bright network fused together. Only then is the rim-cutting
    # second pass worth its runtime.
    fused_component_mm3: float = 2000.0
    # Candidates recovered from a fused region are inherently riskier than ones
    # that stood alone at some threshold, so they face a stricter shape test.
    split_min_tubularity: float = 4.0
    # Fraction of the lumen-to-tissue contrast span used as the candidate
    # threshold. Daughter lumens sit well below the parent median because of
    # partial-volume averaging at 1.5 mm voxels, so the threshold has to track
    # the span between periaortic tissue and the parent lumen rather than the
    # parent's own spread.
    lumen_fraction: float = 0.45
    # Candidates are collected over a small ladder of thresholds instead of one.
    # A single level either misses faint daughters or fuses bright ones into a
    # blob with the structures next to them; every daughter is a clean tube at
    # some level of the ladder, so the ladder plus one-to-one suppression keeps
    # the clean version of each.
    lumen_fraction_ladder: tuple[float, ...] = (0.35, 0.40, 0.45, 0.55)
    # Geometric plausibility gates for a proximal daughter segment.
    radius_floor_mm: float = 1.25
    radius_ceiling_mm: float = 6.0
    min_tubularity: float = 2.0
    min_shape_anisotropy: float = 1.5
    # The challenge's eligibility floor: 2 mm estimated diameter at the origin.
    min_origin_diameter_mm: float = 2.0
    # A fused blob (bowel, vein, calcified plaque) measures far wider from its
    # volume than it does on the seed plane; a tube measures about the same.
    max_radius_inconsistency: float = 2.2
    blob_radius_mm: float = 4.0
    blob_tubularity: float = 3.0
    # Off by default. Real daughters that descend alongside the aorta (EVAL_SET
    # case 20 branch_001, case 23 branch_001) touch the wall over a long strip,
    # so any tight contact gate deletes true positives. Set it to ~20 mm only
    # if a particular dataset suffers from long mask leaks.
    max_contact_extent_mm: float = float("inf")
    conservative_radius: bool = True


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
        os.environ.setdefault("ITK_NIFTI_SFORM_PERMISSIVE", "1")
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


def _iter_components(labels: np.ndarray, count: int):
    """Yield the voxel coordinates of each labelled component, once.

    `np.argwhere(labels == label)` rescans the whole volume per component, which
    costs minutes when a large fine-spacing case produces several hundred
    components. One sort over the non-zero voxels replaces all of it.
    """
    if count <= 0:
        return
    flat = labels.ravel()
    nonzero = np.flatnonzero(flat)
    if nonzero.size == 0:
        return
    values = flat[nonzero]
    order = np.argsort(values, kind="stable")
    nonzero = nonzero[order]
    values = values[order]
    coordinates = np.stack(np.unravel_index(nonzero, labels.shape), axis=1)
    bounds = np.searchsorted(values, np.arange(1, count + 2))
    for index in range(count):
        yield coordinates[bounds[index]:bounds[index + 1]]

class _WallAnchor:
    """Maps a candidate component back to the parent-lumen surface.

    The search region starts a couple of millimetres outside the supplied mask
    (see Config.wall_gap_mm), so a component no longer touches the wall by
    construction. Its ostium is the parent-surface point nearest the
    component's proximal end, which is also more stable than the old
    median-of-touching-voxels estimate when a component grazes the wall.
    """

    def __init__(self, aorta: np.ndarray, spacing: np.ndarray, config: Config):
        self.config = config
        self.spacing = np.asarray(spacing, dtype=float)
        self.wall = ndi.binary_dilation(aorta) & ~aorta
        self.distance, self.nearest = ndi.distance_transform_edt(
            ~aorta, sampling=self.spacing, return_indices=True
        )

    def reach_mm(self) -> float:
        return self.config.wall_gap_mm + float(np.max(self.spacing)) * 1.5

    def proximal_surface_points(self, coords: np.ndarray, require_contact: bool = True) -> np.ndarray | None:
        """Parent-surface points under the component's proximal end.

        `require_contact` keeps the original rule - the component must actually
        touch the wall - and is relaxed only for components recovered by the
        rim-cutting pass, which start `wall_gap_mm` away from the mask by
        construction.
        """
        if require_contact:
            # Unchanged from the wall-contact baseline: the touching voxels
            # themselves. Projecting these onto the mask surface shifts the
            # ostium (and with it the seed and the measured radius) by about a
            # voxel, which measured worse on the annotated cases.
            band = coords[self.wall[tuple(coords.T)]]
            return band if band.size else None
        distances = self.distance[tuple(coords.T)]
        closest = float(distances.min())
        if closest > self.reach_mm():
            return None
        band = coords[distances <= closest + float(np.max(self.spacing))]
        # Project onto the parent surface: more stable than the median of the
        # wall-touching voxels, which sit one voxel outside the mask.
        return np.stack(
            [self.nearest[axis][tuple(band.T)] for axis in range(3)], axis=1
        ).astype(float)

def _cross_section_radius(
    ct: Volume,
    point: np.ndarray,
    direction: np.ndarray,
    threshold: float,
    rays: int = 16,
    limit_mm: float = 5.0,
    step_mm: float = 0.25,
    percentile: float = 50.0,
) -> float:
    """Star-shaped lumen radius in the plane perpendicular to `direction`.

    Rays are cast outward from `point` until the sampled CT value drops below
    `threshold`; the chosen percentile of the ray lengths is the radius. This
    measures the lumen where the challenge asks for it - on a plane normal to
    the local path - instead of inferring it from a whole component's volume,
    which inflates the estimate whenever the component is bent or fused with a
    neighbour.
    """
    u, v = _basis(direction / max(np.linalg.norm(direction), 1e-9))
    angles = np.linspace(0, 2 * math.pi, rays, endpoint=False)
    offsets = np.cos(angles)[:, None] * u + np.sin(angles)[:, None] * v
    distances = np.arange(1, int(np.floor(limit_mm / step_mm)) + 1) * step_mm
    points = point + offsets[:, None, :] * distances[None, :, None]
    indices = np.rint(ct.physical_to_index(points.reshape(-1, 3))).astype(int)
    valid = np.all((indices >= 0) & (indices < ct.data.shape), axis=1)
    supported = np.zeros(len(indices), dtype=bool)
    supported[valid] = ct.data[tuple(indices[valid].T)] >= threshold
    lengths = np.cumprod(supported.reshape(rays, -1), axis=1).sum(axis=1) * step_mm
    return float(np.percentile(lengths, percentile))


def _component_candidate(
    coords: np.ndarray,
    anchor: "_WallAnchor",
    ct: Volume,
    threshold: float,
    contrast_scale: float,
    config: Config,
    require_contact: bool = True,
) -> dict | None:
    if coords.shape[0] < config.min_component_voxels:
        return None
    touching = anchor.proximal_surface_points(coords, require_contact)
    if touching is None or touching.size == 0:
        return None
    physical = ct.index_to_physical(coords)
    touching_physical = ct.index_to_physical(touching)
    opening_area = touching.shape[0] * float(np.min(ct.spacing)) ** 2
    # Allow a 10% area discretization margin at the voxelized wall.
    if opening_area < .9 * math.pi * (config.min_origin_diameter_mm / 2) ** 2:
        return None
    # A real ostium is a compact opening. A vein, duodenum or mask-leak blob
    # that merely runs alongside the aorta touches the wall over a long strip,
    # so the spread of the contact patch separates the two cases cheaply.
    if touching_physical.shape[0] > 1:
        spread = float(np.linalg.norm(touching_physical - touching_physical.mean(axis=0), axis=1).max()) * 2.0
        if spread > config.max_contact_extent_mm:
            return None
    else:
        spread = 0.0
    ostium_idx = np.median(touching, axis=0)
    ostium = ct.index_to_physical(ostium_idx)[0]
    centered = physical - ostium
    # Fit the axis only from voxels within the proximal window: a global fit
    # is pulled off the ostium-tangent heading by any distal curve or bend,
    # since squared-distance weighting lets far voxels dominate the line fit.
    local = np.linalg.norm(centered, axis=1) <= config.max_path_mm
    fit_points = centered[local] if int(local.sum()) >= 3 else centered
    covariance = fit_points.T @ fit_points / max(1, fit_points.shape[0])
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    # A proximal branch segment is a tube: one dominant axis. A compact blob
    # (node, calcification, bowel gas rim) has no dominant axis and is rejected.
    tubularity = float(values[0] / max(values[1], 1e-6))
    floor = config.min_tubularity if require_contact else max(config.min_tubularity, config.split_min_tubularity)
    if tubularity < floor:
        return None
    centered_fit = fit_points - fit_points.mean(axis=0)
    shape_values = np.linalg.eigvalsh(centered_fit.T @ centered_fit / len(centered_fit))
    # Reject an approximately isotropic solid; displacement from the wall
    # must not be mistaken for elongation about the component's own center.
    if shape_values[-1] < config.min_shape_anisotropy * max(shape_values[0], 1e-6):
        return None
    direction = vectors[:, 0]
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
    volume_radius = math.sqrt(max(1e-6, int(proximal.sum()) * voxel_volume / (math.pi * measured_length)))
    # Measured on the plane the challenge specifies, with the volume estimate as
    # a fallback when the ray cast degenerates (seed outside the grid).
    seed_radius = _cross_section_radius(ct, seed, direction, threshold)
    if seed_radius <= 0:
        radius = volume_radius
    elif config.conservative_radius:
        # Where the ray cast leaks into the parent lumen or a touching vein it
        # over-reports; the volume estimate is the cross-check.
        radius = min(seed_radius, volume_radius)
    else:
        radius = seed_radius
    origin_radius = _cross_section_radius(
        ct, ostium + direction * 1.5, direction, threshold, percentile=40.0
    )
    origin_diameter = 2.0 * (origin_radius if origin_radius > 0 else radius)
    if origin_diameter < min(1.5, config.min_origin_diameter_mm):
        return None
    if not (config.radius_floor_mm <= radius <= config.radius_ceiling_mm):
        return None
    if volume_radius > config.max_radius_inconsistency * max(radius, 1e-6):
        return None
    needs_trace = radius >= config.blob_radius_mm and tubularity < config.blob_tubularity
    signal = float(np.median(ct.data[tuple(coords.T)]))
    continuity = min(1.0, path_mm / config.max_path_mm)
    contrast = 1.0 / (1.0 + math.exp(-(signal - threshold) / max(contrast_scale, 1.0)))
    shape = min(1.0, math.log10(max(tubularity, 1.0)) / 1.3)
    confidence = float(np.clip(0.30 + 0.26 * continuity + 0.18 * contrast + 0.26 * shape, 0.0, 0.99))
    return {
        "ostium_xyz_mm": [round(float(v), 3) for v in ostium],
        "seed_xyz_mm": [round(float(v), 3) for v in seed],
        "radius_mm": round(float(radius), 3),
        "direction_xyz": [round(float(v), 8) for v in direction / np.linalg.norm(direction)],
        "path_mm": round(path_mm, 3),
        "tubularity": round(tubularity, 2),
        "origin_diameter_mm": round(origin_diameter, 3),
        "volume_radius_mm": round(volume_radius, 3),
        "contact_extent_mm": round(spread, 2),
        "contact_xyz_mm": touching_physical.tolist(),
        "confidence": round(confidence, 3),
        "requires_trace": needs_trace,
        "evidence": {
            "median_signal": round(signal, 2),
            "p90_signal": round(float(np.percentile(ct.data[tuple(coords.T)], 90)), 2),
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
    shell = ndi.binary_dilation(aorta, iterations=iterations) & ~aorta
    anchor = _WallAnchor(aorta, cropped.spacing, config)
    wall = anchor.wall
    gapped = None
    if config.wall_gap_mm > 0:
        # The parent's own un-masked rim, cut out. Used only as a fallback split
        # for components that come back fused (see below).
        gapped = shell & (anchor.distance > config.wall_gap_mm)
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
    # Anchor the threshold inside the tissue-to-lumen span. The parent lumen is
    # tight (MAD ~30 HU) while a 2-3 mm daughter loses 40-55% of that signal to
    # partial-volume averaging, so a parent-spread threshold such as
    # core_median - k * MAD lands above every daughter and finds nothing.
    span = max(contrast_scale, core_median - background)
    fractions = tuple(config.lumen_fraction_ladder or (config.lumen_fraction,))
    threshold = background + config.lumen_fraction * span
    found: list[dict] = []
    total_components = 0
    voxel_volume = abs(float(np.linalg.det(cropped.affine[:3, :3])))
    fused_voxels = max(config.min_component_voxels, int(config.fused_component_mm3 / max(voxel_volume, 1e-6)))
    split_passes = 0
    for fraction in fractions:
        level = background + fraction * span
        bright = cropped.data >= level
        candidates = shell & bright
        labels, count = ndi.label(candidates)
        total_components += int(count)
        if count == 0:
            continue
        fused = np.zeros(count + 1, dtype=bool)
        for label, component in enumerate(_iter_components(labels, count), start=1):
            if component.shape[0] >= fused_voxels:
                fused[label] = True
            item = _component_candidate(component, anchor, cropped, level, contrast_scale, config)
            if item is not None and item["radius_mm"] >= config.min_radius_mm:
                item["threshold_fraction"] = round(fraction, 3)
                found.append(item)
        # A component far larger than a branch segment is the periaortic network
        # fused through the parent's un-masked rim - typical when the supplied
        # mask sits inside the true lumen. Re-label just that region with the rim
        # cut away, which separates the daughters that were bridged by it.
        if gapped is None or not fused.any():
            continue
        region = fused[labels] & gapped
        if not region.any():
            continue
        split_labels, split_count = ndi.label(region)
        split_passes += 1
        total_components += int(split_count)
        for component in _iter_components(split_labels, split_count):
            item = _component_candidate(
                component, anchor, cropped, level, contrast_scale, config, require_contact=False
            )
            if item is not None and item["radius_mm"] >= config.min_radius_mm:
                item["threshold_fraction"] = round(fraction, 3)
                item["rim_split"] = True
                found.append(item)
    found = refine_candidates(cropped, aorta, found, core_median, background, config)
    found.sort(key=lambda d: (d.get("path_status") == "traced", d["confidence"]), reverse=True)
    count = total_components
    kept: list[dict] = []
    for candidate in found:
        o = np.array(candidate["ostium_xyz_mm"])
        direction = np.array(candidate["direction_xyz"])
        duplicate = any(
            np.linalg.norm(o - np.array(k["ostium_xyz_mm"])) < float(np.min(cropped.spacing))
            or (min(np.linalg.norm(o - np.array(k["ostium_xyz_mm"])),
                    np.linalg.norm(np.array(candidate["proposal_ostium_xyz_mm"]) - np.array(k["proposal_ostium_xyz_mm"]))) < config.merge_radius_mm
                and float(np.dot(direction, np.array(k["direction_xyz"]))) > 0.65)
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
        "pipeline": "adaptive-shell-shape-filter",
        "physical_transform_backend": "SimpleITK" if ct.sitk_image is not None else "synthetic-or-test-affine",
        "min_radius_mm": config.min_radius_mm,
        "crop_origin_index": crop_origin.tolist(),
        "crop_shape": list(cropped.data.shape),
        "aorta_median": round(core_median, 2),
        "aorta_mad": round(core_mad, 2),
        "background_median": round(background, 2),
        "threshold": round(threshold, 2),
        "raw_components": int(count),
        "wall_gap_mm": config.wall_gap_mm,
        "rim_split_passes": split_passes,
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
