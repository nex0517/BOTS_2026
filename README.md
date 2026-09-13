# Branchseed Navigator

Branchseed Navigator is a CPU-only hackathon prototype for finding arteries that branch directly from the abdominal aorta in contrast-enhanced CT/CTA volumes. It takes a CT image and a binary mask of the parent aorta, detects a variable number of eligible daughter branches, and writes their geometry in physical millimetres.

For every detected daughter, the output includes:

- the centre of the opening at the aortic wall (the ostium);
- a seed point 5 mm along the daughter vessel;
- an estimate of the local lumen radius; and
- a unit direction vector pointing away from the aorta.

The repository also includes local visualization tools for reviewing detections against the CT and aorta mask.

> This is a research and hackathon prototype. It is not a medical device and must not be used for diagnosis, treatment planning, or clinical decision-making.

## Scope

Branchseed Navigator is designed to:

- process previously unseen CT/CTA and parent-aorta mask pairs without manual point placement;
- discover direct aortic daughter vessels rather than search for a fixed list of named arteries;
- preserve the input image's SimpleITK physical coordinate system;
- keep separate wall openings as separate instances while treating a common trunk as one direct origin;
- reject crop caps, downstream branches, unsupported proposals, and duplicate detections; and
- run offline on a standard CPU-only laptop.

The project does **not** segment the parent aorta, assign anatomical vessel names, reconstruct the full distal vascular tree, or infer vessels that are not visible in the supplied scan. The terminal iliac division is outside the core task.

## Quick start

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the detector on one case:

```bash
python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json
```

The image and mask must use the same grid and physical geometry. Both `.nii` and `.nii.gz` inputs are supported.

Optional arguments include:

```bash
python run.py \
  --image image.nii.gz \
  --aorta-mask aorta_mask.nii.gz \
  --output prediction.json \
  --diagnostics diagnostics.json \
  --case-id subject001 \
  --min-radius-mm 1.0
```

`--diagnostics` writes trace evidence, rejected proposals, timing, and failure reasons to a separate sidecar. `--min-radius-mm` applies the organizer-provided minimum eligible radius; it defaults to `0.0` when no threshold is supplied. These fields never enter the strict prediction JSON.

For a deterministic synthetic smoke test:

```bash
python run.py --demo --output sample_data/prediction.json --diagnostics sample_data/diagnostics.json
```

## Output format

The detector writes one JSON object per case:

```json
{
  "case_id": "subject001",
  "parent": {
    "instance_id": "aorta"
  },
  "daughters": [
    {
      "instance_id": "branch_001",
      "parent_instance_id": "aorta",
      "ostium_xyz_mm": [12.4, -31.8, 184.6],
      "seed_xyz_mm": [15.1, -29.7, 181.2],
      "radius_mm": 2.7,
      "direction_xyz": [0.56, 0.43, -0.71]
    }
  ]
}
```

Coordinates are physical millimetres, not voxel indices. If no eligible daughter is detected, `daughters` is an empty list.

## How the detector works

The supplied aorta mask limits the search to a small physical shell around the aortic wall. Within that region, the pipeline:

1. calibrates contrast from the current scan;
2. proposes bright, vessel-like structures connected to the aortic wall;
3. traces several candidate directions through native CT cross-sections;
4. requires at least 5 mm of supported proximal lumen;
5. estimates the ostium, arc-length seed, radius, and direction in physical space; and
6. filters crop caps, weak paths, and overlapping duplicate proposals.

The implementation is deterministic classical image processing built with NumPy, SciPy, and SimpleITK. It does not use case identifiers, expected branch counts, anatomical templates, saved predictions, or reference annotations during inference. See [ARCHITECTURE.md](ARCHITECTURE.md) for implementation details.

## Run the dataset checks

Run the unit and regression tests:

```bash
python -m unittest discover -s tests -v
```

Run the synthetic submission checks:

```bash
python self_test.py
```

Run every discovered CT/mask pair in an organizer-style dataset and generate predictions, a report, and visual checks:

```bash
python self_test.py --data-root "path/to/TORALIS CHALLENGE" --output-root submission/organizer
```

If development references are available in an `EVAL_SET` directory, score them with:

```bash
python tools/score_eval.py --data-root EVAL_SET --tolerance 6 --report submission/eval_score_report.json
```

Development-set measurements and known geometry issues are documented in [EVAL_RESULTS.md](EVAL_RESULTS.md) and [VMR_VALIDATION.md](VMR_VALIDATION.md). They are not hidden-set or clinical-performance claims.

## Local review dashboard

The dashboard in `web/` displays axial CT slices, the parent mask, predicted ostia, daughter-direction arrows, radii, run times, and the exact output JSON. Its prediction API reads generated files from `submission/organizer/development_predictions/`, so generate those files with the dataset command above before starting it.

Start the dashboard with Node.js 18 or newer:

```bash
node web/server.mjs
```

Then open `http://127.0.0.1:3000`. To enable **Re-run analysis**, make the source dataset available at the expected location or set `BRANCHSEED_DATA_ROOT` as described in [web/README.md](web/README.md).

The older `dist/` interface is a standalone synthetic visualization concept. Its simulated images and displayed benchmark values are not real-case validation.

## Repository guide

- `detector/` - detection, tracing, opening association, tubular evidence, and parent-geometry logic
- `run.py` - required single-case command-line entry point
- `tests/` - synthetic, geometry, generalization, and regression tests
- `tools/` - development-set scoring and VMR evaluation utilities
- `web/` - local real-data review dashboard and volume-building tools
- `dist/` - legacy synthetic visualization demo
- `submission/` - generated predictions, reports, and visual checks currently stored in the repository
- `self_test.py` - batch execution, contract validation, runtime reporting, and visual-check generation

## Current limitations

This remains a development prototype based on geometric and intensity heuristics. Very small or faint vessels, unusual wall contact, early bifurcations, calcification, and ambiguous crop geometry can still cause missed branches, false positives, or inaccurate seeds. Passing the included tests or matching the available development references does not establish generalization to hidden cases or clinical data.

For submission status and remaining checks, see [SUBMISSION_CHECKLIST.md](SUBMISSION_CHECKLIST.md).
