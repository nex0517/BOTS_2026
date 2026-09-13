# Branchseed Navigator dashboard

The TypeScript/Node.js dashboard is wired to this repository's Python detector and organizer outputs. It shows all 25 patient pages, axial CT slices and aorta mask, physical-coordinate ostium markers with daughter-direction arrows, candidate-count and radius progress bars, run time, and the exact prediction JSON. Select a patient, scroll over the scan or drag the slice slider, and click a branch to jump to its ostium.

## Run

From the repository root:

```bash
node web/server.mjs
```

Open **http://127.0.0.1:3000**. Node.js 18+ is required; the compiled TypeScript and preprocessed viewer volumes are included. The dashboard reads the current files in `submission/organizer/development_predictions/` and the runtime measurements in `submission/organizer/self_test_report.json`.

Click **Re-run analysis** to execute this repository's `run.py` for the selected patient and update its prediction JSON. The original CT/mask pairs are needed in `~/Downloads/TORALIS CHALLENGE /`, or point the server to them:

```bash
BRANCHSEED_DATA_ROOT="/path/to/TORALIS CHALLENGE" node web/server.mjs
```

Install the Python dependencies with `python3 -m pip install -r requirements.txt` before rerunning. Set `PYTHON=/path/to/python3` if they are in a different environment. The server binds only to `127.0.0.1`; use `PORT=3002` to change its port.

To regenerate viewer assets after changing the CT or mask files:

```bash
python3 web/build_volumes.py --data-root "/path/to/TORALIS CHALLENGE"
```

To edit `web/src/app.ts`, install the frontend compiler with `npm install --prefix web` and run `npm run build --prefix web`. Runtime serving uses Node built-ins and has no npm package dependency.

## Interpretation

The CT data is locally windowed from −100 to 500 HU. The slice viewer projects the 3-D daughter direction into the axial plane and labels a substantial through-plane component with an up/down Z arrow. Progress bars represent the number of predicted daughter candidates relative to the largest displayed case or the measured branch radius; neither represents model accuracy. This is a research prototype, not a diagnostic tool. The old `dist/` page remains a synthetic design demo; `web/` displays actual organizer predictions and CT data.
