# Accuracy fixes and current measurements

## Verified discovery results

The user confirmed the hackathon counts below. Spatial reference annotations
retain the uncertainty recorded in EVAL_SET, and 6 mm is the local matching gate.

| Case | Verified daughters | Current predictions | Matched | Extras | Misses |
|---|---:|---:|---:|---:|---:|
| 19 | 3 | 3 | 3 | 0 | 0 |
| 20 | 4 | 4 | 4 | 0 | 0 |
| 21 | 3 | 3 | 3 | 0 | 0 |
| 22 | 6 | 6 | 6 | 0 | 0 |
| 23 | 3 | 3 | 3 | 0 | 0 |
| Total | 19 | 19 | 19 | 0 | 0 |

The baseline at the start of this fix matched 17/19 references with 7 unmatched
predictions (F1 0.791, mean ostium error 2.03 mm). The current version matches
19/19 with zero extras (F1 1.00, mean ostium error 1.56 mm, median 1.06 mm).
These are development-set results. No case IDs, expected counts, reference
coordinates, annotation masks or saved predictions are read by inference.

## Changes

- Added a threshold level that exposes faint origins before they disappear.
- Reject inward proposals and weak periaortic tissue; use an upper-tail intensity
  gate to suppress highly attenuating plaque/bone components.
- Estimate voxelized opening support with a discretization margin, avoiding
  isolated subminimum contacts. This remains a resolution-dependent heuristic.
- Permit borderline origin proposals to reach refinement; enforce the 2 mm
  origin estimate after refinement instead of prematurely losing small branches.
- Recenter cross-sections in the CT and interpolate seeds 5 mm along stored paths.
  Radius on supported traces is measured from the local perpendicular section.
- Recover enclosed lumens from fused proposals using a bounded cone search about
  the wall normal. Long tangential contact strips use an upstream anchor.
- Deduplicate using both original opening proposals and refined positions,
  preferring image-supported traces over unresolved proposals.
- Vectorize bulk voxel conversion using SimpleITK-derived geometry, with a
  rotation/crop regression test. Individual landmarks still call SimpleITK.

## Corrected evaluation

Matching maximizes valid one-to-one matches before minimizing total distance.
Seed membership uses the nearest native voxel without neighborhood dilation.
Radius error compares against non-null reference radius_mm at the seed.

- Strict seed hits: **17/19**.
- Mean seed landmark distance: **1.463 mm**.
- Mean direction cosine: **0.974**.
- Mean seed-radius error: **0.22 mm, n=3**; 16 references have no usable radius.
- CT-supported paths: **17**; unresolved fallback paths: **2**.

Both strict seed misses are in case 19: current predictions branch_002 and
branch_003 match reference branch_001 and branch_003 respectively. Their seed
errors are 1.211 and 4.585 mm. These are real remaining geometry issues; matching
all branch counts does not establish full challenge compliance. Fallback status
is available in the optional diagnostics sidecar. The tracker does not guarantee
first-bifurcation stopping in every geometry.

## Reproducibility

Eight unittest checks pass, including exact counts and one-to-one matches for
the five verified cases, non-greedy matching, curved arc-length seeds, and bulk
SimpleITK coordinate equivalence. All six synthetic scenarios pass. The variable
count test now requires exactly three branches; its paths begin inside the
parent to supply complete openings rather than narrow voxel tips.

All 25 organizer cases execute with no individual edits: 1.785 s mean and
2.999 s maximum, including I/O and tracing validation with tracemalloc enabled.
ITK and OpenBLAS thread settings were four. A separate successful complete-set run measured whole-process peak RSS of
1292.8 MiB (about 1.26 GiB), including I/O and visual generation. The Python
allocation peak alone is not whole-process RSS. The organizer machine and hidden annotations remain untested.

```sh
python -m unittest discover -s tests -v
python tools/score_eval.py --data-root EVAL_SET --tolerance 6 --report submission/eval_score_report.json
python self_test.py --data-root EVAL_SET
python self_test.py --data-root "TORALIS CHALLENGE" --output-root submission/organizer
```

Current real predictions are in submission/development_predictions (five
reviewed cases) and submission/organizer/development_predictions (all 25).
Historical files under preds/ are not the current generated submission outputs.
