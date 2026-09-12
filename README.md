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

Run the complete automated requirement check:

```bash
python self_test.py
```

When the organizer development set is available:

```bash
python self_test.py --data-root data
```

That command runs every discovered `orig*/mask*` pair, writes one prediction per case, measures end-to-end runtime, validates the JSON/geometry contract, and generates verification PNGs for the first three cases under `submission/`.

## Why this is the right 24-hour project

- It targets the highest-value problem: branch discovery and ostium placement account for 70% of the stated score.
- It treats the supplied aorta mask as a search anchor, so computation stays inside a small periaortic crop.
- It adapts contrast to each scan instead of relying on one brittle HU cutoff.
- It proves at least 5 mm of proximal path, suppresses cap surfaces and deduplicates competing candidates.
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

Organizer NIfTI input is read through SimpleITK, and index conversion uses `TransformIndexToPhysicalPoint` or its continuous-index equivalent. The loader handles `.nii`, `.nii.gz`, and gzip content stored behind a `.nii` suffix. SciPy accelerates morphology and connected-component labeling.

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
- connected wall-origin proposals, 5 mm path rule, physical-space geometry, deduplication and strict JSON validation
- offline Aorta Map, selection, linked evidence views, scenario switching, local JSON import and JSON export

Synthetic for the current MVP:

- the browser's CT pixels and the three demo case measurements
- benchmark numbers shown in the browser
- clinical validation and organizer-score claims

## Next engineering priority

Run the real 25 cases immediately. If the adaptive-shell baseline is stable, spend remaining model time on surface normals and a short beam tracker. Do not add vessel naming, a full distal tree, disease claims, CFD or a large neural model.

See `TEAM_PLAN.md` for the compressed 24-hour split and `PITCH.md` for the five-minute demo.

See `SUBMISSION_CHECKLIST.md` for the requirement-by-requirement audit. Real development predictions remain the only external deliverable that cannot be produced until the organizer data is placed under `data/`.
