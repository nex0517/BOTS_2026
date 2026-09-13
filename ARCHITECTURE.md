# Architecture

The detector keeps proposal generation separate from evidence required for output:

```text
CT + aorta mask
      |
Crop, physical-distance shell, and scan contrast
      |
Propose bright components near the wall
      |
Check shape and trace multiple initial directions
      |
Associate connected wall openings, score, deduplicate -> strict JSON + diagnostics
```

`detector/branchseed.py` handles input, candidate geometry, duplicate suppression,
and the output contract. `detector/tracking.py` samples cross-sections and refines
paths. The CLI is `run.py`. NumPy, SciPy and SimpleITK provide the numerical and
image operations; there is no second morphology implementation to maintain.

Candidate shape is measured in physical coordinates. The wall-anchored moment
estimates the outgoing direction; the centered moment rejects approximately
round solids whose displacement from the wall would otherwise resemble an
elongated branch. This is a geometric heuristic, not an anatomical template.
No vessel names, expected counts, case identifiers or reference annotations
enter detection.

Radius rays are sampled in one batch. The fixed cross-section grid is reused.
These optimizations avoid repeated Python loops and allocations without adding
another detector, model, or tuning stage.

`tracking.py` tests a PCA heading and nine directions around a physical surface
normal. Each valid path must supply bounded contrast-filled sections, at least
5 mm of supported native-CT lumen, stable positive radius, continuity and
separation from the parent. Seed placement uses path arc length. No unresolved
proposal can enter strict output.

`openings.py` associates traced origins with connected exterior wall crossings
and requires compatible directions and overlapping paths before merging. It
never merges merely because origins are close. `tubular.py` samples four local
physical radius scales without allocating full-volume Hessian fields.
`parent_geometry.py` estimates parent endpoints and rejects cap/continuation
geometry. Conservative proposal shape filters and the older cap search margin
remain; a more permissive opening-first generator failed the ablation checks.

The score weights length, minimum contrast, tubular evidence, radius stability
and parent separation. Boundedness and connection are prerequisites. Median
contrast, curvature and competing-hypothesis margin are recorded, but adding
them to the weighted score did not improve validation and was not retained.
The cutoff is a development-tuned deterministic score, not a probability.

Traces establish local image support, not vessel identity. Early bifurcation
stopping, faint sensitivity and cap interpretation remain imperfect. The
existing exact-count regression fails and has not been weakened.

Tests and scoring are separate from inference. Evaluate both one-to-one origins
and count errors, including empty scans and bright-object negatives. Known-case
results are development results, not evidence of universal generalization.

The offline browser in `dist/` is an independent synthetic visualization demo.
Its simulated CT and benchmark display must not be used as real-case validation.
