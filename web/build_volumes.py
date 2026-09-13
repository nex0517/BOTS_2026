#!/usr/bin/env python3
"""Prepare CT/mask slice assets using this project's NIfTI loader and geometry."""
import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from detector.branchseed import read_nifti


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, default=Path(os.environ.get(
        'BRANCHSEED_DATA_ROOT', Path.home() / 'Downloads' / 'TORALIS CHALLENGE ')))
    args = parser.parse_args()
    output = Path(__file__).parent / 'public' / 'volumes'
    output.mkdir(parents=True, exist_ok=True)
    for i in range(1, 26):
        case = f'subject{i:03d}'
        folder = args.data_root / case
        def pair(stem):
            paths = [folder / f'{stem}{i}{ext}' for ext in ('.nii', '.nii.gz')]
            return next((p for p in paths if p.exists()), None)
        image_path, mask_path = pair('orig'), pair('mask')
        if image_path is None or mask_path is None:
            raise FileNotFoundError(f'Missing CT/mask pair in {folder}')
        image, mask = read_nifti(image_path), read_nifti(mask_path)
        if image.data.shape != mask.data.shape or not np.allclose(image.affine, mask.affine):
            raise ValueError(f'CT and mask geometry mismatch in {case}')
        ct = np.asarray(image.data).transpose(2, 1, 0)
        binary = np.asarray(mask.data).transpose(2, 1, 0) > 0
        coords = np.where(binary)
        if not coords[0].size:
            raise ValueError(f'Empty aorta mask in {case}')
        pads = (8, 48, 48)
        bounds = [(max(0, int(q.min())-pad), min(dim, int(q.max())+pad+1))
                  for q, pad, dim in zip(coords, pads, ct.shape)]
        z0,z1 = bounds[0]; y0,y1 = bounds[1]; x0,x1 = bounds[2]
        gray = np.clip((ct[z0:z1,y0:y1,x0:x1].astype(np.float32)+100)/600*255,0,255).astype(np.uint8)
        parent = binary[z0:z1,y0:y1,x0:x1].astype(np.uint8)*255
        packed = np.stack([gray,parent],axis=-1)
        with gzip.open(output/f'{case}.raw.gz','wb',compresslevel=6) as target:
            target.write(packed.tobytes())
        meta = {'case_id':case,'width':x1-x0,'height':y1-y0,'depth':z1-z0,
                'offset_xyz':[x0,y0,z0],
                'origin_xyz_mm':image.affine[:3,3].tolist(),
                'basis_xyz_mm':image.affine[:3,:3].tolist(),
                'source_shape_zyx':list(ct.shape),'window_hu':[-100,500]}
        (output/f'{case}.meta.json').write_text(json.dumps(meta)+'\n')
        print(f'{case}: {packed.shape[0]} slices')


if __name__ == '__main__':
    main()
