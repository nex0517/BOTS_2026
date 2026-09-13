"""Regressions for physical paths, matching and the verified development cases."""
import json
from pathlib import Path
import unittest

import numpy as np

from detector.branchseed import Config, Volume, detect, read_nifti, sitk
from detector.tracking import point_at_arc, validate_trace
from tools.score_eval import _match

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def test_five_mm_seed_follows_bend_not_straight_chord(self):
        path = np.array([[0., 0., 0.], [3., 0., 0.], [3., 4., 0.]])
        arc = np.array([0., 3., 7.])
        seed = point_at_arc(path, arc, 5)
        np.testing.assert_allclose(seed, [3, 2, 0])
        self.assertLess(np.linalg.norm(seed), 5)
        branch = dict(path_xyz_mm=path, path_arc_mm=arc,
                      ostium_xyz_mm=path[0], seed_xyz_mm=seed)
        validate_trace(branch)
        branch['seed_xyz_mm'] = [3., 4., 0.]
        with self.assertRaises(ValueError):
            validate_trace(branch)

    def test_matching_maximizes_number_before_distance(self):
        predictions = [np.array([0., 0., 0.]), np.array([-2., 0., 0.])]
        references = [np.array([-.9, 0., 0.]), np.array([1.5, 0., 0.])]
        pairs, _ = _match(predictions, references, 2)
        self.assertEqual({(i, j) for i, j, _ in pairs}, {(0, 1), (1, 0)})

    def test_bulk_coordinates_equal_simpleitk_with_rotation_and_crop(self):
        image = sitk.Image([8, 9, 10], sitk.sitkFloat32)
        image.SetOrigin((14., -31., 800.))
        image.SetSpacing((.7, 1.2, 2.4))
        image.SetDirection((0., -1., 0., 1., 0., 0., 0., 0., 1.))
        affine = np.eye(4)
        affine[:3, :3] = np.array(image.GetDirection()).reshape(3, 3) @ np.diag(image.GetSpacing())
        affine[:3, 3] = image.GetOrigin()
        offset = np.array([2., 3., 4.])
        volume = Volume(np.zeros((3, 3, 3)), affine, np.array(image.GetSpacing()), image, offset)
        points = np.array([[0., 0., 0.], [1.25, .5, 1.75]])
        expected = [image.TransformContinuousIndexToPhysicalPoint(tuple(p + offset)) for p in points]
        np.testing.assert_allclose(volume.index_to_physical(points), expected, atol=1e-9)
        np.testing.assert_allclose(volume.physical_to_index(expected), points, atol=1e-9)

    def test_matching_does_not_count_duplicates(self):
        pairs, _ = _match([np.zeros(3), np.zeros(3)], [np.zeros(3)], 2)
        self.assertEqual(len(pairs), 1)


@unittest.skipUnless((ROOT / 'EVAL_SET/case_19/annotations.json').exists(),
                     'Development data is not installed')
class VerifiedDevelopmentTests(unittest.TestCase):
    def test_counts_and_one_to_one_origins(self):
        # Expectations belong in tests only. Inference must never read these
        # counts, annotation files, case IDs or saved prediction JSONs.
        for number, count in [(19, 3), (20, 4), (21, 3), (22, 6), (23, 3)]:
            with self.subTest(case=number):
                folder = ROOT / 'EVAL_SET' / f'case_{number}'
                result, diagnostics = detect(
                    read_nifti(folder / f'orig{number}.nii.gz'),
                    read_nifti(folder / f'aorta{number}.nii.gz'), Config())
                self.assertEqual(len(result['daughters']), count)
                references = json.loads((folder / 'annotations.json').read_text())['daughters']
                pairs, _ = _match(
                    [np.array(d['ostium_xyz_mm']) for d in result['daughters']],
                    [np.array(d['ostium_xyz_mm']) for d in references], 6)
                self.assertEqual(len(pairs), count)
                for branch in diagnostics['branches']:
                    if branch['path_status'] == 'traced':
                        validate_trace(branch)


if __name__ == '__main__':
    unittest.main()
