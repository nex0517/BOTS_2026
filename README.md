# Branchseed Navigator

An evidence-first MVP for discovering direct abdominal-aortic daughter origins on contrast CT/CTA. It combines a CPU-oriented classical detector with an offline visual review surface that makes each scorer-visible result auditable.

> Hackathon prototype only. Synthetic demo data is not patient data, and the output is not for diagnosis or clinical use.

## Submission quick start

One setup command:

```bash
python -m pip install -r requirements.txt
```

One official run command:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

Score the detector against the annotated review cases in `EVAL_SET/`:

```bash
python tools/score_eval.py --data-root EVAL_SET --tolerance 6 --report submission/eval_score_report.json
```

The five hackathon-verified daughter counts are now reproduced: **3, 4, 3, 6,
3** for cases 19–23. All **19/19 origins match one-to-one at 6 mm**, with no
extra predictions (development precision/recall/F1 **1.00**). Mean ostium error
is **1.56 mm**. This is performance on the five cases used during development,
not an independent hidden-test score.

The corrected evaluator finds **17/19 seeds inside their matched native labels**.
Two case-19 paths remain unresolved and retain the geometric fallback. Local
seed-radius error is **0.22 mm across only 3 available reference measurements**;
null radii are excluded. Never treat origin-radius error as seed-radius accuracy.

All **25 organizer cases execute** without case-specific edits: **1.785 s mean,
2.999 s maximum** including I/O with tracing validation and Python allocation
instrumentation on this machine. Predictions for all 25 cases are in
`submission/organizer/development_predictions/`; reviewed-case outputs are in
`submission/development_predictions/`. See `EVAL_RESULTS.md` for limitations.

Run synthetic regression checks (these do not certify every hackathon requirement):

```bash
python self_test.py
```

Run the supplied complete organizer dataset:

```bash
python self_test.py --data-root "TORALIS CHALLENGE" --output-root submission/organizer
```

That command runs every discovered `orig*/mask*` pair, writes one prediction per case, measures end-to-end runtime, validates the JSON/geometry contract, and generates verification PNGs for the first three cases under `submission/`.

## Simpler detector update

The detector now rejects approximately round bright solids using centered
component geometry. It also batches radius sampling, reuses section grids, and
uses the required SciPy implementation directly instead of duplicate fallback
code. The two inference files contain 52 fewer lines in total.

In the synthetic attached-sphere suite, false detections decreased from 30 to 0
across 36 scans. The five development counts remain 3, 4, 3, 6, 3, and counts on
all 25 supplied scans are unchanged. A paired local run measured 0.775 s mean
including I/O versus 0.925 s before (about 16% lower elapsed time); this benchmark
omits the allocation instrumentation used by the older self-test timing above.
Single-run timings are approximate. Peak process RSS was about 1.39 GiB.

This change improves blob rejection, not the unresolved small/faint-vessel
sensitivity gap. Synthetic 2–2.5 mm vessels at 1.5 mm spacing and faint 3 mm
vessels remain missed. A nearby parallel-tube ambiguity also remains. The shape
check is a heuristic and may reject short, wide true branches; fresh annotated
cases are required to assess that tradeoff. No count or anatomical template is
used to force predictions.

Run all numerical and detection regressions:

```bash
python -m unittest discover -s tests -v
```

## Why this is the right 24-hour project

- It targets the highest-value problem: branch discovery and ostium placement account for 70% of the stated score.
- It treats the supplied aorta mask as a search anchor, so computation stays inside a small periaortic crop.
- It adapts contrast to each scan instead of relying on one brittle HU cutoff.
- It traces supported proximal lumens, reports unresolved paths in diagnostics, suppresses cap surfaces and deduplicates competing proposals.
- It keeps strict prediction JSON separate from confidence, visuals and diagnostics.
- It turns 3D topology into a memorable Aorta Map linked to evidence slices and proximal geometry.

## Real-case TypeScript dashboard

The integrated dashboard in `web/` reads `submission/organizer/development_predictions/` and uses this repository's Python detector for its **Re-run analysis** action. It displays the 25 real organizer cases with axial CT, aorta mask, branch arrows, counts, run times, and exact JSON. Start it with `node web/server.mjs` and open `http://127.0.0.1:3000`. See `web/README.md` for data-root and rebuild instructions.

## Run the visual MVP

Open `dist/index.html` directly, or serve the folder locally:

```bash
python -m http.server 4173 --directory dist
```

Then visit `http://127.0.0.1:4173`. Switch among the three synthetic edge cases, select branches on the Aorta Map or list, inspect the scorer-safe JSON, and export it. You can also open another prediction JSON locally; no file is uploaded.

## Run the detector

Fast smoke test with a deterministic synthetic CTA phantom:

```bash
python run.py --demo --output sample_data/prediction.json --diagnostics sample_data/diagnostics.json
```

Run on an organizer case:

```bash
python run.py --image path/to/orig.nii --aorta-mask path/to/mask.nii --output prediction.json --diagnostics diagnostics.json
```

Organizer NIfTI input is read through SimpleITK, and index conversion uses `TransformIndexToPhysicalPoint` or its continuous-index equivalent. The loader handles `.nii`, `.nii.gz`, and gzip content stored behind a `.nii` suffix. SciPy is required for morphology, connected-component labeling and tracing.

## Output contract

`prediction.json` contains only the conservative scorer fields:

- `case_id`
- `parent: {"instance_id": "aorta"}`
- one daughter object per origin with `instance_id`, `parent_instance_id`, `ostium_xyz_mm`, `seed_xyz_mm`, `radius_mm`, and `direction_xyz`

Confidence, timing and evidence are written only to the optional diagnostics sidecar.

## What is real vs. mocked

Real and runnable:

- custom NIfTI-1 loading, including gzip-content sniffing
- official SimpleITK physical-coordinate conversion for organizer cases
- image/mask geometry validation
- 25 mm crop, per-case robust contrast model and cap exclusion
- connected wall-origin proposals, CT-supported tracing with arc-length seeds, explicit fallback status, physical-space geometry and strict JSON validation
- offline Aorta Map, selection, linked evidence views, scenario switching, local JSON import and JSON export

Synthetic for the current MVP:

- the browser's CT pixels and the three demo case measurements
- benchmark numbers shown in the browser
- clinical validation and organizer-score claims

## Remaining work

The two unresolved case-19 paths need better lumen tracking and seed placement.
The local tracker does not yet guarantee a stop at every early bifurcation.
The thresholds, voxelized opening-area gate and proposal-size filters were
selected using these five cases; evaluate on held-out annotated cases before
claiming generalization. Counts for the other 20 cases are not verified here.

The browser remains a synthetic visualization concept, including its CT pixels,
evidence bars and benchmark values. Use the generated real-case PNGs for actual
verification. A completed demo and clinical usefulness cannot be certified by
the automatic file-inventory check.

See `SUBMISSION_CHECKLIST.md`, `EVAL_RESULTS.md` and `PITCH.md`.
