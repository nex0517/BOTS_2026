"""Refine branch proposals with CT-supported paths in physical millimetres.

Thresholds are development-set heuristics, not calibrated probabilities.
Untraceable legacy proposals remain explicitly marked ``unresolved`` in the
sidecar; they must not be presented as proven centerlines.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi


def sample(ct, points, data=None):
    """Trilinear sampling, including floating-point interpolation of masks."""
    indices = ct.physical_to_index(np.asarray(points).reshape(-1, 3))
    return ndi.map_coordinates(
        ct.data if data is None else data,
        indices.T,
        order=1,
        mode="constant",
        cval=-1024 if data is None else 0,
        prefilter=False,
        output=np.float32,
    )


def _basis(direction):
    reference = np.array([0., 0., 1.]) if abs(direction[2]) < .9 else np.array([1., 0., 0.])
    u = np.cross(direction, reference)
    u /= np.linalg.norm(u)
    return u, np.cross(direction, u)


@lru_cache(maxsize=8)
def _section_grid(extent, step):
    """Reuse the fixed sampling grid across sections and candidates."""
    axis = np.arange(-extent, extent + step / 2, step)
    return np.meshgrid(axis, axis, indexing="ij")


def section(ct, aorta, point, direction, threshold, extent=7., step=.35):
    """Return a nearby enclosed lumen's center, area-equivalent radius and HU.

    Parent-mask pixels are excluded. A section that remains connected to the
    sampling boundary cannot supply a bounded radius and is rejected.
    """
    u, v = _basis(direction)
    xx, yy = _section_grid(extent, step)
    offsets = xx[..., None] * u + yy[..., None] * v
    points = point + offsets
    values = sample(ct, points).reshape(xx.shape)
    parent = sample(ct, points, aorta).reshape(xx.shape) > .5
    labels, count = ndi.label((values >= threshold) & ~parent)
    if not count:
        return None
    nearby = xx ** 2 + yy ** 2 <= 1.5 ** 2
    candidates = np.unique(labels[nearby])
    candidates = candidates[candidates > 0]
    if not len(candidates):
        return None

    def closest_region(label_image, choices):
        label = min(choices, key=lambda k: np.min((xx ** 2 + yy ** 2)[label_image == k]))
        return label_image == label

    def touches_edge(region):
        return np.any(region[[0, -1], :]) or np.any(region[:, [0, -1]])

    region = closest_region(labels, candidates)
    if touches_edge(region):
        # A higher local level can separate a lumen from a touching vein/blob.
        local = xx ** 2 + yy ** 2 <= 2. ** 2
        peak = float(np.max(values[region & local])) if np.any(region & local) else threshold
        adaptive = max(threshold, .7 * peak)
        labels, _ = ndi.label((values >= adaptive) & ~parent)
        candidates = np.unique(labels[nearby])
        candidates = candidates[candidates > 0]
        if not len(candidates):
            return None
        region = closest_region(labels, candidates)
        if touches_edge(region):
            return None
    weights = np.maximum(values - threshold, 0) * region
    if weights.sum() == 0:
        return None
    offset = np.sum(offsets * weights[..., None], axis=(0, 1)) / weights.sum()
    if np.linalg.norm(offset) > 2:
        return None
    center = point + offset
    signal = float(sample(ct, [center])[0])
    if signal < threshold:
        return None
    return center, float(np.sqrt(region.sum() * step ** 2 / np.pi)), signal


def point_at_arc(path, arc, distance):
    """Interpolate a point by path length, never by Euclidean displacement."""
    path, arc = np.asarray(path), np.asarray(arc)
    if not 0 <= distance <= arc[-1]:
        raise ValueError("Requested distance lies outside the traced path")
    return np.array([np.interp(distance, arc, path[:, i]) for i in range(3)])


def validate_trace(branch):
    path = np.asarray(branch["path_xyz_mm"], dtype=float)
    arc = np.asarray(branch["path_arc_mm"], dtype=float)
    if path.ndim != 2 or path.shape[1] != 3 or len(path) < 2 or not np.all(np.isfinite(path)):
        raise ValueError("Invalid traced path")
    expected = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if arc.shape != expected.shape or not np.allclose(arc, expected, atol=1e-6):
        raise ValueError("Trace arc lengths do not match physical geometry")
    if np.any(np.diff(arc) <= 0) or arc[-1] < 5 or arc[-1] > 10.000001:
        raise ValueError("Trace must cover 5 to 10 mm")
    if not np.allclose(branch["ostium_xyz_mm"], path[0], atol=.002):
        raise ValueError("Trace does not begin at the ostium")
    if not np.allclose(branch["seed_xyz_mm"], point_at_arc(path, arc, 5), atol=.002):
        raise ValueError("Seed is not 5 mm along the trace")


def trace(ct, aorta, ostium, direction, threshold, length=10.):
    """Follow recentered lumen sections; stop when support is lost or at 10 mm.

    This local tracker does not guarantee detection of every early bifurcation.
    Failure reasons remain available to the caller rather than fabricating a
    centerline or assigning a radius from unrelated component volume.
    """
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    origin = np.array(ostium, dtype=float)
    initial = section(ct, aorta, origin + direction * 1.5, direction, threshold)
    if initial is None:
        return {"failure": "initial"}
    point, radius, signal = initial

    back = point - np.arange(0, 7.01, .25)[:, None] * direction
    hits = np.flatnonzero(sample(ct, back, aorta) >= .5)
    if not len(hits):
        # An oblique tangent can miss the parent. Test a short connection to
        # its nearest boundary instead, without allowing a remote structure.
        index = ct.physical_to_index([point])[0]
        candidates = np.argwhere(aorta)
        nearest = candidates[np.argmin(np.sum(((candidates - index) * ct.spacing) ** 2, axis=1))]
        target = ct.index_to_physical(nearest)[0]
        if np.linalg.norm(target - point) > 5:
            return {"failure": "no_parent"}
        back = point + np.linspace(0, 1, 30)[:, None] * (target - point)
        hits = np.flatnonzero(sample(ct, back, aorta) >= .5)
        if not len(hits):
            return {"failure": "no_parent"}
    first = hits[0]
    if first == 0:
        return {"failure": "inside_parent"}
    origin = (back[first] + back[first - 1]) / 2
    connection = origin + np.linspace(0, 1, 12)[:, None] * (point - origin)
    if np.min(sample(ct, connection)) < threshold * .75:
        return {"failure": "connection"}

    path = [origin, point]
    radii, signals = [radius, radius], [signal, signal]
    total = float(np.linalg.norm(point - origin))
    for _ in range(24):
        result = section(ct, aorta, point + direction * .75, direction, threshold)
        if result is None:
            break
        next_point, radius, signal = result
        delta = next_point - point
        distance = np.linalg.norm(delta)
        if distance < .1 or distance > 2.5:
            break
        heading = delta / distance
        if np.dot(heading, direction) < .35:
            break
        path.append(next_point)
        radii.append(radius)
        signals.append(signal)
        total += distance
        direction = .6 * direction + .4 * heading
        direction /= np.linalg.norm(direction)
        point = next_point
        if total >= length:
            break
    path = np.array(path)
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
    if arc[-1] < 5:
        return {"failure": "short", "length": float(arc[-1])}
    seed = point_at_arc(path, arc, 5)
    index = np.searchsorted(arc, 5)
    tangent = path[min(index, len(path) - 1)] - path[max(0, index - 1)]
    tangent /= np.linalg.norm(tangent)
    seed_section = section(ct, aorta, seed, tangent, threshold)
    if seed_section is None:
        return {"failure": "seed_section"}
    seed_signal = float(sample(ct, [seed])[0])
    if seed_signal < threshold:
        return {"failure": "seed_signal"}
    direction = seed - origin
    direction /= np.linalg.norm(direction)
    minimum_signal = float(min(np.array(signals)[arc <= 5]))
    if arc[-1] > length:
        endpoint = point_at_arc(path, arc, length)
        keep = arc < length
        path = np.vstack([path[keep], endpoint])
        arc = np.r_[arc[keep], length]
    result = {
        "ostium_xyz_mm": origin.tolist(),
        "seed_xyz_mm": seed.tolist(),
        "direction_xyz": direction.tolist(),
        "radius_mm": float(seed_section[1]),
        "path_xyz_mm": path.tolist(),
        "path_arc_mm": arc.tolist(),
        "path_mm": float(arc[-1]),
        "origin_diameter_mm": float(2 * radii[0]),
        "minimum_center_hu": minimum_signal,
        "median_center_hu": float(np.median(signals)),
        "seed_center_hu": seed_signal,
    }
    validate_trace(result)
    return result


def refine_candidates(ct, aorta, candidates, parent_hu, background_hu, config):
    signed = ndi.distance_transform_edt(~aorta, sampling=ct.spacing) - ndi.distance_transform_edt(aorta, sampling=ct.spacing)
    gradients = np.gradient(ndi.gaussian_filter(signed, 1.), *ct.spacing)
    rotation = ct.affine[:3, :3] / ct.spacing
    span = max(10., parent_hu - background_hu)
    accepted = []
    for candidate in candidates:
        level = candidate["evidence"]["adaptive_threshold"]
        contrast = (candidate["evidence"]["median_signal"] - background_hu) / span
        # Suppress weak periaortic tissue and highly attenuating plaque/bone.
        if contrast < .55 or candidate["evidence"]["p90_signal"] > max(parent_hu * 1.6, parent_hu + 200):
            continue
        origin = np.array(candidate["ostium_xyz_mm"])
        heading = np.array(candidate["direction_xyz"])
        candidate["proposal_ostium_xyz_mm"] = origin.tolist()
        index = ct.physical_to_index(origin)[0]
        normal = rotation @ np.array([ndi.map_coordinates(g, index[:, None], order=1)[0] for g in gradients])
        normal /= max(np.linalg.norm(normal), 1e-9)
        outward = np.dot(normal, heading)
        if outward < 0:
            continue
        if candidate["contact_extent_mm"] > config.max_path_mm and (outward < .5 or candidate.get("requires_trace")):
            # A daughter running beside the parent has a long contact strip.
            # Its upstream end, rather than the strip midpoint, anchors origin.
            contact = np.array(candidate["contact_xyz_mm"])
            contact = contact[np.linalg.norm(contact - origin, axis=1) <= config.max_path_mm]
            if len(contact) > 2:
                projection = contact @ heading
                origin = np.mean(contact[projection <= np.quantile(projection, .2)], axis=0)
        result = trace(ct, aorta, origin, heading, level)
        if candidate.get("requires_trace"):
            # A fused component's PCA axis is unreliable. Search a bounded cone
            # about the physical wall normal, retaining only enclosed lumens.
            u, v = _basis(normal)
            options = []
            for a in [-.7, 0., .7]:
                for b in [-.7, 0., .7]:
                    direction = normal + a * u + b * v
                    direction /= np.linalg.norm(direction)
                    traced = trace(ct, aorta, origin, direction, level)
                    if "failure" not in traced and traced["origin_diameter_mm"] >= config.min_origin_diameter_mm and traced["radius_mm"] >= .75:
                        options.append(traced)
            if not options:
                continue
            result = min(options, key=lambda item: item["radius_mm"])
        if "failure" not in result:
            if result["minimum_center_hu"] < background_hu + .5 * span:
                continue
            candidate.update(result)
            candidate["path_status"] = "traced"
        else:
            candidate["path_status"] = "unresolved"
            candidate["trace_failure"] = result["failure"]
        if candidate["origin_diameter_mm"] < config.min_origin_diameter_mm:
            continue
        accepted.append(candidate)
    return accepted
