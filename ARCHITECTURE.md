# Architecture

The detector has four stages:

```text
CT + aorta mask
      |
Crop and estimate scan contrast
      |
Propose bright components near the wall
      |
Check shape and trace proximal lumen
      |
Merge duplicate proposals -> prediction JSON + optional diagnostics
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

Two development-case paths still use the existing unresolved geometric fallback.
A successful trace also does not prove vessel identity or guarantee detection of
every early bifurcation. Diagnostics expose these limits; strict JSON contains
only the challenge fields. Small/faint-vessel sensitivity remains incomplete.

Tests and scoring are separate from inference. Evaluate both one-to-one origins
and count errors, including empty scans and bright-object negatives. Known-case
results are development results, not evidence of universal generalization.

The offline browser in `dist/` is an independent synthetic visualization demo.
Its simulated CT and benchmark display must not be used as real-case validation.
