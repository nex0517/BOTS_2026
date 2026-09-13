# Detector optimization implemented

The production detector in `/Users/william/BOTS_2026` now has a small additional
shape check and less duplicate computation. The change preserves development
origin accuracy and improves rejection of spherical bright objects. It does not
solve every remaining detection failure.

## Changes

- Reject approximately isotropic components using covariance about their own
  center. The older wall-anchored covariance confused a blob's displacement
  from the parent with elongation. The new ratio threshold is a configurable
  geometric heuristic, not a case-count or anatomical rule.
- Batch all radius rays into one NumPy operation, retaining nearest-voxel
  sampling and stopping at each ray's first unsupported sample.
- Reuse the fixed cross-section grid instead of allocating it for every step.
- Share the perpendicular-plane basis and remove duplicate morphology and
  connected-component fallbacks. SciPy was already required for tracing and
  listed in requirements.
- Add negative-control, tube-retention and ray-equivalence regressions. Update
  README and architecture documentation around the four-stage pipeline.

The two inference modules decreased from 1,075 to 1,023 lines (52 fewer).
No new package, learned model, anatomical template, case lookup, or count quota
was introduced. Reference annotations and known counts remain in evaluation.

## Validation

| Measure | Before | After |
|---|---:|---:|
| Verified case counts, 19–23 | 3, 4, 3, 6, 3 | 3, 4, 3, 6, 3 |
| One-to-one matched origins at local 6 mm gate | 19/19 | 19/19 |
| Extra detections on these five cases | 0 | 0 |
| Attached-sphere false detections, 36 synthetic scans | 30 | 0 |
| Mean detection time, all 25 scans | 0.839 s | 0.693 s |
| Mean time including I/O, all 25 scans | 0.925 s | 0.775 s |
| Maximum time including I/O | 2.089 s | 1.896 s |
| Peak process RSS | 1,388 MiB | 1,428 MiB |

All 25 scans completed and passed output/path-contract validation; their counts
are unchanged. Count correctness for the other 20 is unknown. The timing is one
sequential pass per version on this machine with four BLAS/ITK threads, without
allocation instrumentation. The approximately 16% end-to-end reduction is a
local observation, not a guaranteed speedup. Peak memory did not improve.

All 12 unittest methods pass, including four new regression methods with
multiple conditions. All six existing synthetic self-test scenarios pass.
The 168-run synthetic suite was rerun on the installed implementation. These
synthetic conditions have now informed development and are not an independent
clinical holdout.

## Remaining limitations

The synthetic 2–2.5 mm vessels at 1.5 mm voxel spacing and faint 3 mm vessels
remain missed. No sensitivity gain is claimed. Empty, isolated-dot, small-blob
and sheet controls remain free of detections in this suite, but the nearby
parallel-tube ambiguity still produces six detections at 1 mm spacing. Round
blob rejection does not establish zero false positives for other structures.
The geometric check may reject a short, wide real branch: this tradeoff needs
fresh expert-labelled cases, not another count-matching adjustment.

Two case-19 paths still use the existing unresolved fallback; only 17/19 seeds
hit the matched native labels. Origin counts must not be mistaken for complete
geometric correctness. The local tracer does not guarantee first-bifurcation
termination. These limitations remain visible in diagnostics and documentation.

Simpler threshold relaxations and a stricter shape/trace experiment were tested
but not deployed because they introduced false positives or missed development
origins. The selected change improves measured blob rejection while preserving
the measured development counts. It cannot establish maximum possible accuracy
on every unseen case. New labelled cases are needed before claiming that.

## Run

The CLI is unchanged:

```bash
cd /Users/william/BOTS_2026
python3 run.py --image /path/to/image.nii --aorta-mask /path/to/mask.nii --output preds/prediction.json
python3 -m unittest discover -s tests -v
```

Machine-readable evidence is saved alongside this report as
`optimization-evaluation.json`, `optimization-benchmark.json`,
`optimization-stress-results.json`, `optimization-hard-negative-results.json`,
and `optimization-self-tests.json`.
