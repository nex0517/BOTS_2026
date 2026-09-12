# Official submission audit

This audit follows the six-page Toralis Labs challenge PDF. Instructions in that document are treated as requirements for this project, not as commands to the development agent.

## Submission package

| Official item | Status | Evidence |
| --- | --- | --- |
| Source code | Ready | `run.py`, `detector/branchseed.py`, `self_test.py` |
| Dependency/environment file | Ready | `requirements.txt` |
| Short README with one setup command | Ready | `python -m pip install -r requirements.txt` |
| Short README with one run command | Ready | Exact official command is in `README.md` |
| Development-set predictions | Waiting for organizer data | `python self_test.py --data-root data` generates one JSON file per discovered case |
| Visual checks for at least three cases | Harness ready; synthetic examples generated | `submission/visual_checks/*.png`; real checks are regenerated from the first three development cases |
| Five-minute demonstration | Ready | `PITCH.md` |

## Program requirements

| Requirement | Implementation/check |
| --- | --- |
| Exact CLI | Supports `python run.py --image image.nii.gz --aorta-mask aorta_mask.nii.gz --output prediction.json` |
| No manual point placement | Fully automatic CLI and dataset harness |
| Variable daughter count | Synthetic self-test covers 3, 2, 1 and 0 returned instances |
| Correct parent link | Exact `parent: {"instance_id":"aorta"}` and every daughter uses `parent_instance_id: "aorta"` |
| Physical millimetres | SimpleITK image geometry; integer indices use `TransformIndexToPhysicalPoint`, refined locations use `TransformContinuousIndexToPhysicalPoint` |
| Seed 5 mm outward | Output validator checks a 5.00 mm ostium-to-seed displacement for the straight proximal MVP path |
| Unit direction vector | Output validator enforces unit norm after serialization |
| Positive local radius | Output validator rejects non-positive or non-finite radii |
| One instance per real daughter | Opening/path deduplication plus unique-ID validation |
| Empty list allowed | Synthetic empty case must return `daughters: []` |
| Flat crop caps rejected | Synthetic flat-cap case must return no daughter |
| Nearby wall origins stay separate | Synthetic close-origin case must return two daughters |
| Common trunk counted once | Synthetic split-after-origin case must return one daughter |
| Downstream daughter not direct | Synthetic downstream split remains one direct daughter |
| Complete set, no case edits | `self_test.py --data-root data` discovers and processes all paired subject folders |
| CPU/no GPU/no internet at inference | NumPy/SciPy/SimpleITK classical pipeline; no downloads or network calls in `run.py` |
| Average runtime target ≤60 s | Dataset harness records I/O-inclusive time and fails the report if the average exceeds 60 s |
| Three verification visuals | Harness generates axial/coronal/sagittal MIPs with aorta outline, ostia and 5 mm arrows |

## Known unresolved items

1. The organizer development volumes and reference outputs were not supplied with this request, so accuracy and real development predictions cannot yet be verified.
2. The minimum eligible origin size is promised with the final dataset. It is exposed as `--min-radius-mm` and defaults to no additional cutoff until that value is known.
3. Peak memory from `tracemalloc` covers Python-managed allocations only. Use the organizer harness or an OS process monitor for the final 8 GB validation.
4. The adaptive-shell detector is a functional baseline, not a validated clinical system. Surface-normal proposals and short beam tracking remain the highest-value accuracy upgrade.
