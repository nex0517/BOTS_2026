"""Validate and rank native-CT-supported physical daughter traces."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


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


def trace(ct, aorta, ostium, direction, threshold, length=10., section_mask=None, parent_tree=None):
    """Follow recentered lumen sections; stop when support is lost or at 10 mm.

    This local tracker does not guarantee detection of every early bifurcation.
    Failure reasons remain available to the caller rather than fabricating a
    centerline or assigning a radius from unrelated component volume.
    """
    section_mask = aorta if section_mask is None else section_mask
    direction = np.array(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    origin = np.array(ostium, dtype=float)
    initial = None
    for offset in (1.5,2.5,3.5):
        initial = section(ct, section_mask, origin + direction * offset, direction, threshold)
        if initial is not None:break
    if initial is None:
        return {"failure": "initial"}
    point, radius, signal = initial

    back = point - np.arange(0, 7.01, .25)[:, None] * direction
    hits = np.flatnonzero(sample(ct, back, aorta) >= .5)
    if not len(hits):
        # An oblique tangent can miss the parent. Test a short connection to
        # its nearest boundary instead, without allowing a remote structure.
        if parent_tree is None:
            boundary=np.argwhere(aorta & ~ndi.binary_erosion(aorta))
            parent_tree=cKDTree(ct.index_to_physical(boundary))
        _, nearest=parent_tree.query(point)
        target=parent_tree.data[nearest]
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
        result = section(ct, section_mask, point + direction * .75, direction, threshold)
        if result is None:
            break
        next_point, radius, signal = result
        delta = next_point - point
        distance = np.linalg.norm(delta)
        if distance < .1 or distance > 2.5:
            break
        segment = point + np.linspace(0,1,max(3,int(np.ceil(distance/.25))+1))[:,None]*delta
        if np.any(sample(ct,segment)<.85*threshold) or np.any(sample(ct,segment,aorta)>.5):
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
    seed_section = section(ct, section_mask, seed, tangent, threshold)
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
        "radius_cv": float(np.std(radii)/max(np.mean(radii),1e-6)),
        "maximum_radius_mm": float(max(radii)),
    }
    validate_trace(result)
    return result


def refine_candidates(ct, aorta, candidates, parent_hu, background_hu, config):
    distance = ndi.distance_transform_edt(~aorta, sampling=ct.spacing)
    section_mask = aorta
    parent_tree=cKDTree(ct.index_to_physical(np.argwhere(aorta & ~ndi.binary_erosion(aorta))))
    signed = distance - ndi.distance_transform_edt(aorta, sampling=ct.spacing)
    gradients = np.gradient(ndi.gaussian_filter(signed, 1./ct.spacing), *ct.spacing)
    rotation = ct.affine[:3, :3] / ct.spacing
    accepted = []
    for candidate in candidates:
        candidate['path_status']='rejected'
        candidate['rejection_reason']='no_supported_trace'
        candidate['trace_failures']={}
        def reject(reason):
            failures=candidate['trace_failures']
            failures[reason]=failures.get(reason,0)+1
        origin = np.array(candidate["ostium_xyz_mm"])
        level = candidate["evidence"]["adaptive_threshold"]
        if candidate['evidence']['median_signal'] < background_hu+.35*(parent_hu-background_hu) or candidate['evidence']['p90_signal'] > max(parent_hu*1.6,parent_hu+200):
            candidate['rejection_reason']='component_contrast'
            continue
        index = ct.physical_to_index(origin)[0]
        normal = rotation @ np.array([ndi.map_coordinates(g, index[:, None], order=1)[0] for g in gradients])
        normal_length=np.linalg.norm(normal)
        if normal_length<1e-9:
            candidate['rejection_reason']='undefined_surface_normal'
            continue
        normal /= normal_length
        heading=np.array(candidate['direction_xyz'])
        outward=float(np.dot(normal,heading))
        if outward<0:
            candidate['rejection_reason']='inward_proposal'
            continue
        if candidate['contact_extent_mm']>config.max_path_mm and (outward<.5 or candidate.get('requires_trace')):
            contact=np.array(candidate['contact_xyz_mm'])
            contact=contact[np.linalg.norm(contact-origin,axis=1)<=config.max_path_mm]
            if len(contact)>2:
                projection=contact@heading
                origin=np.mean(contact[projection<=np.quantile(projection,.2)],axis=0)
        u, v = _basis(normal)
        options = []
        directions=[np.array(candidate['direction_xyz'])]
        if np.dot(directions[0],normal)<0:directions[0]*=-1
        for a,b in [(0,0),(.7,0),(-.7,0),(0,.7),(0,-.7),(.7,.7),(.7,-.7),(-.7,.7),(-.7,-.7)]:
            direction=normal+a*u+b*v
            directions.append(direction/np.linalg.norm(direction))
        for direction in directions:
            result = trace(ct,aorta,origin,direction,level,section_mask=section_mask,parent_tree=parent_tree)
            if 'failure' in result:
                reject(result['failure'])
                continue
            if result['origin_diameter_mm']<config.min_origin_diameter_mm:
                reject('origin_below_2mm')
                continue
            path=np.array(result['path_xyz_mm']);arc=np.array(result['path_arc_mm'])
            positions=np.arange(0,5.01,.25)
            dense=np.array([np.interp(positions,arc,path[:,k]) for k in range(3)]).T
            distances=sample(ct,dense,distance)
            hu=sample(ct,dense)
            if np.any(sample(ct,dense[positions>=1],aorta)>.5) or hu.min()<.85*level:
                reject('contrast_or_parent_return')
                continue
            if distances[-1]<1 or distances[-1]-distances[4]<.5:
                reject('insufficient_parent_separation')
                continue
            if np.percentile(hu[positions>=2],90)>max(parent_hu*1.6,parent_hu+200):
                reject('excessive_attenuation')
                continue
            if result['radius_cv']>.5 or result['maximum_radius_mm']>6:
                reject('unstable_or_unbounded_radius')
                continue
            result['seed_parent_distance_mm']=float(distances[-1])
            from .tubular import evidence
            tube=evidence(ct,aorta,result,parent_hu,background_hu)
            result['tubular_evidence']=tube
            result['confidence'] = (.2*min(1.,result['path_mm']/10)
                +.2*np.clip((result['minimum_center_hu']-background_hu)/max(10.,parent_hu-background_hu),0,1)
                +.3*tube+.15*(1-min(1.,result['radius_cv']/.5))
                +.15*min(1.,distances[-1]/5))
            options.append(result)
        if options:
            candidate['proposal_ostium_xyz_mm'] = origin.tolist()
            candidate.update(max(options,key=lambda x:x['confidence']))
            path=np.array(candidate['path_xyz_mm'])
            headings=np.diff(path,axis=0)
            headings/=np.linalg.norm(headings,axis=1)[:,None]
            candidate['minimum_direction_cosine']=float(np.min(np.sum(headings[1:]*headings[:-1],axis=1))) if len(headings)>1 else 1.
            competitors=[r['confidence'] for r in options if np.linalg.norm(np.array(r['seed_xyz_mm'])-candidate['seed_xyz_mm'])>max(1.,candidate['radius_mm'])]
            candidate['hypothesis_margin']=float(candidate['confidence']-max(competitors)) if competitors else None
            candidate['valid_hypotheses']=len(options)
            candidate['rejection_reason']=None
            candidate['path_status']='traced' 
            accepted.append(candidate)
    return accepted
