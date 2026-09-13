"""Dependency-light contract checks for the Branchseed MVP."""

import json
import gzip
import math
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from detector.branchseed import build_parser, read_nifti, sitk


ROOT = Path(__file__).resolve().parents[1]


class SmokeTest(unittest.TestCase):
    def test_official_cli_signature_is_supported(self):
        args = build_parser().parse_args([
            "--image", "image.nii.gz",
            "--aorta-mask", "aorta_mask.nii.gz",
            "--output", "prediction.json",
        ])
        self.assertEqual(args.aorta_mask, "aorta_mask.nii.gz")

    def test_synthetic_cli_is_deterministic_and_valid(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first.json"
            second = Path(tmp) / "second.json"
            command = [sys.executable, str(ROOT / "run.py"), "--demo", "--output"]
            subprocess.run(command + [str(first)], cwd=ROOT, check=True, capture_output=True, text=True)
            subprocess.run(command + [str(second)], cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())
            data = json.loads(first.read_text(encoding="utf-8"))
            self.assertEqual(set(data), {"case_id", "parent", "daughters"})
            self.assertEqual(data["parent"], {"instance_id": "aorta"})
            self.assertGreaterEqual(len(data["daughters"]), 1)
            for daughter in data["daughters"]:
                self.assertEqual(daughter["parent_instance_id"], "aorta")
                self.assertEqual(len(daughter["direction_xyz"]), 3)
                direction = np.asarray(daughter["direction_xyz"], dtype=float)
                self.assertAlmostEqual(float(np.linalg.norm(direction)), 1.0, places=5)
                ostium = np.asarray(daughter["ostium_xyz_mm"], dtype=float)
                seed = np.asarray(daughter["seed_xyz_mm"], dtype=float)
                self.assertGreater(float(np.linalg.norm(seed - ostium)), 0)
                self.assertLessEqual(float(np.linalg.norm(seed - ostium)), 5.002)

    def test_nifti_reader_sniffs_gzip_content_behind_nii_suffix(self):
        array = np.arange(24, dtype=np.int16).reshape((2, 3, 4), order="F")
        header = bytearray(352)
        struct.pack_into("<I", header, 0, 348)
        struct.pack_into("<8h", header, 40, 3, 2, 3, 4, 1, 1, 1, 1)
        struct.pack_into("<h", header, 70, 4)
        struct.pack_into("<h", header, 72, 16)
        struct.pack_into("<8f", header, 76, 1, 1.5, 2.0, 2.5, 0, 0, 0, 0)
        struct.pack_into("<f", header, 108, 352.0)
        struct.pack_into("<h", header, 254, 1)
        struct.pack_into("<4f", header, 280, 1.5, 0, 0, 10)
        struct.pack_into("<4f", header, 296, 0, 2.0, 0, -4)
        struct.pack_into("<4f", header, 312, 0, 0, 2.5, 8)
        header[344:348] = b"n+1\0"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gzip_content_with_nii_suffix.nii"
            with gzip.open(path, "wb") as handle:
                handle.write(header)
                handle.write(array.tobytes(order="F"))
            volume = read_nifti(path, require_simpleitk=False)
            np.testing.assert_array_equal(volume.data, array)
            np.testing.assert_allclose(volume.spacing, [1.5, 2.0, 2.5])
            # SimpleITK reports physical points in LPS convention; the raw
            # fallback parser reports the stored NIfTI RAS frame directly.
            expected = [-11.5, 2.0, 10.5] if sitk is not None else [11.5, -2.0, 10.5]
            np.testing.assert_allclose(volume.index_to_physical([1, 1, 1])[0], expected)


if __name__ == "__main__":
    unittest.main()
