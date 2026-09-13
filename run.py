#!/usr/bin/env python3
"""Branchseed Navigator command-line entry point."""
"""Test if push works"""

# Limit native numerical libraries before importing NumPy, SciPy or SimpleITK.
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS',
                 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'):
    os.environ[variable] = '4'
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

from detector.branchseed import main


if __name__ == "__main__":
    raise SystemExit(main())
