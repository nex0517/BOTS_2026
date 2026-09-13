"""Image-only negative controls and numerical regressions; no target counts in inference."""
import unittest

import numpy as np

from detector.branchseed import Volume, _cross_section_radius, detect
from detector.tracking import _basis


def phantom(spacing, daughter=False, blob_radius=None, dots=False):
    shape = (64, 64, 64)
    xyz = np.indices(shape) * spacing
    x, y, z = xyz
    c = 32 * spacing
    parent = ((x-c)**2 + (y-c)**2 <= 36) & (z >= 8*spacing) & (z <= 56*spacing)
    rng = np.random.default_rng(4519)
    ct = rng.normal(35, 12, shape).astype(np.float32)
    ct[parent] = rng.normal(300, 12, parent.sum())
    if daughter:
        tube = (x >= c+4) & (x <= c+18) & ((y-c)**2 + (z-c)**2 <= 1.5**2)
        ct[tube] = rng.normal(280, 12, tube.sum())
    if blob_radius:
        # A round bright solid attached to the parent is not a daughter tube.
        blob = (x-c-6-blob_radius+.75)**2 + (y-c)**2 + (z-c)**2 <= blob_radius**2
        ct[blob] = rng.normal(280, 12, blob.sum())
    if dots:
        for point in rng.uniform(5*spacing, 59*spacing, (120, 3)):
            if np.hypot(point[0]-c, point[1]-c) < 11:
                continue
            blob = np.sum((xyz-point[:, None, None, None])**2, axis=0) <= 1.5**2
            ct[blob] = 330
    affine = np.diag([spacing, spacing, spacing, 1.])
    return Volume(ct, affine, np.full(3, spacing)), Volume(parent, affine, np.full(3, spacing))


class GeneralizationTests(unittest.TestCase):
    def test_empty_and_isolated_dots_produce_empty_lists(self):
        for spacing in (1., 1.5):
            for dots in (False, True):
                with self.subTest(spacing=spacing, dots=dots):
                    result, _ = detect(*phantom(spacing, dots=dots))
                    self.assertEqual(result['daughters'], [])

    def test_attached_round_solids_are_rejected(self):
        for spacing in (1., 1.5):
            for radius in (3., 4., 6.):
                with self.subTest(spacing=spacing, radius=radius):
                    result, _ = detect(*phantom(spacing, blob_radius=radius))
                    self.assertEqual(result['daughters'], [])

    def test_three_mm_tube_is_retained_at_both_resolutions(self):
        for spacing in (1., 1.5):
            with self.subTest(spacing=spacing):
                result, _ = detect(*phantom(spacing, daughter=True))
                self.assertEqual(len(result['daughters']), 1)

    def test_vectorized_rays_match_scalar_sampling(self):
        rng = np.random.default_rng(890)
        ct = Volume(rng.normal(120, 40, (12, 13, 14)), np.diag([.7, 1.2, 2., 1.]), np.array([.7, 1.2, 2.]))
        # Cover oblique planes, off-grid centers and multiple ray step sizes.
        for point in ([3.1, 6.2, 10.3], [-.5, 0., 0.], [8., 15., 26.]):
            for step in (.25, .3):
                direction = np.array([1., 2., 3.]); direction /= np.linalg.norm(direction)
                u, v = _basis(direction)
                lengths = []
                for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
                    offset = np.cos(angle)*u + np.sin(angle)*v
                    reach = 0.; distance = step
                    while distance <= 5.:
                        index = np.rint(ct.physical_to_index(np.array(point)+offset*distance)[0]).astype(int)
                        if np.any(index < 0) or np.any(index >= ct.data.shape) or ct.data[tuple(index)] < 100:
                            break
                        reach = distance; distance += step
                    lengths.append(reach)
                actual = _cross_section_radius(ct, np.array(point), direction, 100, step_mm=step)
                self.assertAlmostEqual(actual, np.percentile(lengths, 50), places=9)


if __name__ == '__main__':
    unittest.main()
