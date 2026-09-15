"""Optional local presentation exporter. Never imports or runs the detector."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.measure import marching_cubes


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def physical_vertices(image, vertices_zyx, offset_zyx):
    return np.array([image.TransformContinuousIndexToPhysicalPoint(tuple(float(x) for x in (v + offset_zyx)[::-1])) for v in vertices_zyx])


def validate_prediction(prediction):
    ids = set()
    if not isinstance(prediction.get('daughters'), list):
        raise ValueError('Expected a daughters list')
    for branch in prediction['daughters']:
        if branch['instance_id'] in ids or branch['parent_instance_id'] != 'aorta':
            raise ValueError('Invalid daughter identity')
        ids.add(branch['instance_id'])
        for key in ('ostium_xyz_mm', 'seed_xyz_mm', 'direction_xyz'):
            point = np.asarray(branch[key], dtype=float)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError('Invalid geometry')
        if not np.isfinite(branch['radius_mm']) or branch['radius_mm'] <= 0:
            raise ValueError('Invalid radius')
        if abs(np.linalg.norm(branch['direction_xyz']) - 1) > .001:
            raise ValueError('Direction must be a unit vector')


def export_case(image_path, mask_path, prediction_path, output, label, vmr=None):
    started = time.perf_counter()
    image, mask = sitk.ReadImage(str(image_path)), sitk.ReadImage(str(mask_path))
    if image.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError('Expected 3D inputs')
    for prop in ('Size', 'Origin', 'Spacing', 'Direction'):
        if not np.allclose(getattr(image, 'Get' + prop)(), getattr(mask, 'Get' + prop)()):
            raise ValueError('Image and mask grids differ: ' + prop)
    prediction = json.loads(prediction_path.read_text())
    validate_prediction(prediction)
    array = sitk.GetArrayFromImage(mask)
    if not np.isin(array, [0, 1]).all() or not array.any():
        raise ValueError('Expected a nonempty binary parent mask')
    coords = np.argwhere(array)
    lo, hi = coords.min(0), coords.max(0) + 1
    region = tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))
    cropped = np.pad(array[region], 1)
    # Coarser sampling is display-only; source grids and measurements are untouched.
    vertices, faces, _, _ = marching_cubes(cropped, .5, step_size=2, allow_degenerate=False)
    if len(faces) > 24000:
        vertices, faces, _, _ = marching_cubes(cropped, .5, step_size=3, allow_degenerate=False)
    physical = physical_vertices(mask, vertices, lo - 1)
    matrix = np.asarray(mask.GetDirection()).reshape(3, 3) @ np.diag(mask.GetSpacing())
    affine_points = ((vertices + lo - 1)[:, ::-1] @ matrix.T) + mask.GetOrigin()
    transform_error = float(np.max(np.abs(physical - affine_points)))
    assert transform_error < 1e-8
    point = tuple(int(v) for v in coords[len(coords)//2][::-1])
    assert np.allclose(mask.TransformIndexToPhysicalPoint(point), np.asarray(mask.GetOrigin()) + matrix @ point)
    for branch in prediction['daughters']:
        for key in ('ostium_xyz_mm', 'seed_xyz_mm'):
            p = branch[key]
            idx = image.TransformPhysicalPointToContinuousIndex(p)
            assert np.allclose(image.TransformContinuousIndexToPhysicalPoint(idx), p)
            if not all(-.5 <= v <= n-.5 for v, n in zip(idx, image.GetSize())):
                raise ValueError('Prediction lies outside image')
    payload = {
        'label': label, 'vmr_case': vmr, 'prediction': prediction,
        'vertices': physical.tolist(), 'faces': faces.tolist(),
        'bounds': [physical.min(0).tolist(), physical.max(0).tolist()],
        'geometry': {'origin': mask.GetOrigin(), 'spacing': mask.GetSpacing(), 'direction': mask.GetDirection(), 'size': mask.GetSize(), 'frame': 'SimpleITK physical millimetres (LPS)', 'transform_error_mm': transform_error},
        'provenance': {'image_file': image_path.name, 'mask_file': mask_path.name, 'prediction_file': prediction_path.name, 'image_sha256': digest(image_path), 'mask_sha256': digest(mask_path), 'prediction_sha256': digest(prediction_path), 'surface': 'Marching cubes at 0.5 on supplied parent mask; display-only coarse sampling; no daughter surfaces'},
        'export_seconds': round(time.perf_counter() - started, 3)
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, separators=(',', ':'), allow_nan=False))
    print(f'{label}: {len(faces)} triangles, {len(prediction["daughters"])} daughters, {payload["export_seconds"]} s')
    return {'id': output.stem, 'label': label, 'file': output.name, 'daughters': len(prediction['daughters']), 'vmr_case': vmr}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation-root', type=Path)
    parser.add_argument('--image', type=Path)
    parser.add_argument('--aorta-mask', type=Path)
    parser.add_argument('--predictions', type=Path)
    parser.add_argument('--label', default='Local case')
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'public' / 'pitch-data')
    args = parser.parse_args()
    manifest = []
    if args.validation_root:
        report = json.loads((args.validation_root / 'comparison_report.json').read_text())
        for case in report['cases']:
            i = case['case']
            manifest.append(export_case(args.validation_root / 'data' / f'patient_{i}_image.nii', args.validation_root / 'data' / f'patient_{i}_binary_mask.nii', args.validation_root / 'predictions' / f'patient_{i}_predictions.json', args.output_dir / f'patient_{i}.json', f'VMR case {i:02d}', case['vmr_case']))
        benchmarks = json.loads((args.validation_root / 'benchmark_metrics.json').read_text())
        source = Path(__file__).resolve().parents[1]
        evidence = {'comparison': report, 'benchmarks': benchmarks, 'source_hashes': {p: digest(source / p) for p in ['detector/branchseed.py', 'detector/tracking.py', 'detector/openings.py', 'detector/tubular.py', 'detector/parent_geometry.py', 'Branchseed Challenge.pdf', 'validation_runs/comparison_report.json', 'validation_runs/benchmark_metrics.json']}, 'note': 'Five development cases informed tuning; expert-derived VMR proxy references, not official hidden-test or clinical validation. Historical timing reports do not record a commit or timestamp; hashes identify inspected files at export, not proof of historical benchmark revision.'}
        (args.output_dir / 'evidence.json').write_text(json.dumps(evidence, indent=2))
    else:
        if not all([args.image, args.aorta_mask, args.predictions]):
            parser.error('Use --validation-root or provide --image, --aorta-mask and --predictions')
        manifest.append(export_case(args.image, args.aorta_mask, args.predictions, args.output_dir / 'local_case.json', args.label))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
