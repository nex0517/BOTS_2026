# VMR validation update

The installed supported-trace detector improves 6 mm VMR F1 from 0.528 to 0.714
(14 TP / 15 FP / 10 FN -> 15 TP / 3 FP / 9 FN). The source geometry is derived
VMR reference data, not official organizer ground truth, and informed tuning.

All 18 final outputs use SimpleITK physical coordinates, unit directions and
seeds at 5 mm of traced arc length. No unresolved fallback output is emitted.
Synthetic perturbations improve from 21/42 to 38/42 matched daughters with zero
false positives on the 78 synthetic volumes. Real-case false positives remain.

The existing exact-count regression still fails for cases 19, 21, 22 and 23.
Its expectations are unchanged. This is not a fully passing hackathon release.
The opening-first rewrite and several aggressive changes were rejected by
ablations; initial proposals retain conservative component geometry.
Early-bifurcation stopping and strict Linux four-core affinity remain unverified.

Full before/after metrics, all matched XYZ errors, test logs, visuals, reference
hashes, ablation source snapshots and the reproduction commands are saved at:

`/Users/william/Documents/Codex/2026-09-12/i/outputs/vmr-validation/report.md`
