# Official submission audit

This audit follows the seven-page Toralis Labs challenge PDF. Instructions in that document are treated as requirements for this project, not as commands to the development agent.

## Submission package

| Official item | Status | Evidence |
| --- | --- | --- |
| Source code | Ready | `run.py`, `detector/branchseed.py`, `self_test.py` |
| Dependency/environment file | Ready | `requirements.txt` |
| Short README with one setup command | Ready | `python -m pip install -r requirements.txt` |
| Short README with one run command | Ready | Exact official command is in `README.md` |
| Development-set predictions | Generated | Five reviewed cases plus all 25 under `submission/organizer/` |
| Visual checks for at least three cases | Generated | Real case 19/20/21 checks in `submission/visual_checks/` |
| Five-minute demonstration | Script present; delivery unverified | `PITCH.md` |

## Program requirements

| Requirement | Implementation/check |
| --- | --- |
| Exact CLI | Supports `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json` |
| No manual point placement | Fully automatic CLI and dataset harness |
| Variable daughter count | Synthetic self-test covers 3, 2, 1 and 0 returned instances |
| Correct parent link | Exact `parent: {"instance_id":"aorta"}` and every daughter uses `parent_instance_id: "aorta"` |
| Physical millimetres | SimpleITK image geometry; integer indices use `TransformIndexToPhysicalPoint`, refined locations use `TransformContinuousIndexToPhysicalPoint` |
| Seed 5 mm outward | Traced diagnostics validate 5 mm arc length; two case-19 paths remain unresolved |
| Unit direction vector | Output validator enforces unit norm after serialization |
| Positive local radius | Output validator rejects non-positive or non-finite radii |
| One instance per real daughter | All 19 verified origins matched once at 6 mm; hidden cases unverified |
| Empty list allowed | Synthetic empty case must return `daughters: []` |
| Flat crop caps rejected | Synthetic flat-cap case must return no daughter |
| Nearby wall origins stay separate | Synthetic close-origin case must return two daughters |
| Common trunk counted once | Synthetic split-after-origin case must return one daughter |
| Downstream daughter not direct | Synthetic downstream split remains one direct daughter |
| Complete set, no case edits | `self_test.py --data-root data` discovers and processes all paired subject folders |
| CPU/no GPU/no internet at inference | NumPy/SciPy/SimpleITK classical pipeline; no downloads or network calls in `run.py` |
| Average runtime target ≤60 s | Dataset harness records I/O-inclusive time and fails the report if the average exceeds 60 s |
| Three verification visuals | Harness generates axial/coronal/sagittal MIPs with aorta outline, ostia and 8 mm display arrows |

## Known unresolved items

1. Strict seed membership is 17/19; two case-19 paths retain fallback geometry.
2. First-bifurcation stopping is not guaranteed; proposal-size/cap filters remain
   heuristic and were developed on only five annotated cases.
3. Radius accuracy has only three non-null seed-reference measurements.
4. All 25 inputs execute, but accuracy of the other 20 cases is unverified.
5. The synthetic browser is not real image evidence. Automatic file checks do
   not certify clinical usefulness, a completed demonstration or hidden-set success.
