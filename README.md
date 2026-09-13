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

The supported-trace update improves VMR discovery at 6 mm from **14 TP / 15 FP /
10 FN (F1 0.528)** to **15 TP / 3 FP / 9 FN (F1 0.714)**. The VMR JSONs are
high-quality derived references, not official organizer ground truth. These
cases informed development and are no longer an untouched holdout.

Every emitted daughter now has a supported physical trace with its seed exactly
5 mm along that path. Unresolved fallback proposals are rejected. This changes
the older case counts to **1, 4, 5, 7, 2** for cases 19–23; their existing
exact-count test still fails. Do not treat this version as fully validated for
the hackathon. See `VMR_VALIDATION.md` for improvements, regressions and commands.

Run synthetic regression checks (these do not certify every hackathon requirement):

```bash
python self_test.py
```

Run the supplied complete organizer dataset:

```bash
python self_test.py --data-root "TORALIS CHALLENGE" --output-root submission/organizer
```

That command runs every discovered `orig*/mask*` pair, writes one prediction per case, measures end-to-end runtime, validates the JSON/geometry contract, and generates verification PNGs for the first three cases under `submission/`.

## Supported-trace update

The detector uses a millimetre distance shell, native-CT cross-sections, a small
cone of initial directions, and local tubular contrast at 0.8, 1.5, 2.5 and 4 mm.
Connected wall labels and overlapping proximal paths control deduplication.
A deterministic acceptance score filters weak proposals. Diagnostics include
accepted traces, rejected proposals and failure reasons; strict JSON is unchanged.

Synthetic tests cover close openings, a common trunk, small/faint vessels,
parallel vessels, attached blobs, dots, anisotropic spacing, rotation and caps.
Some faint and barely resolved vessels remain missed. No expected count, patient
ID, reference geometry or named-anatomy template enters inference.

Run all numerical and detection regressions:

```bash
python -m unittest discover -s tests -v
```

## Why this is the right 24-hour project

- It targets the highest-value problem: branch discovery and ostium placement account for 70% of the stated score.
- It treats the supplied aorta mask as a search anchor, so computation stays inside a small periaortic crop.
- It adapts contrast to each scan instead of relying on one brittle HU cutoff.
- It traces supported proximal lumens, rejects unsupported paths and records the reasons in diagnostics, suppresses cap surfaces and deduplicates competing proposals.
- It keeps strict prediction JSON separate from confidence, visuals and diagnostics.
- It turns 3D topology into a memorable Aorta Map linked to evidence slices and proximal geometry.

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
- connected wall-origin proposals, CT-supported tracing with arc-length seeds, explicit rejection reasons, physical-space geometry and strict JSON validation
- offline Aorta Map, selection, linked evidence views, scenario switching, local JSON import and JSON export

Synthetic for the current MVP:

- the browser's CT pixels and the three demo case measurements
- benchmark numbers shown in the browser
- clinical validation and organizer-score claims

## Remaining work

The old count regression remains failing; the algorithm still has real false
positives and false negatives against the available references. The local
tracker does not guarantee stopping at every early bifurcation. The complete
opening-first rewrite was not retained because it substantially increased false
positives; proposal generation still uses conservative component shape checks.
New independently annotated cases and a Linux four-core-affinity benchmark are
needed before claiming generalization or full challenge compliance.

The browser remains a synthetic visualization concept, including its CT pixels,
evidence bars and benchmark values. Use the generated real-case PNGs for actual
verification. A completed demo and clinical usefulness cannot be certified by
the automatic file-inventory check.

See `SUBMISSION_CHECKLIST.md`, `EVAL_RESULTS.md` and `PITCH.md`.
