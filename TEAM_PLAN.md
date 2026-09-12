# Four-person, 24-hour execution plan

Always keep a frozen, runnable fallback. A feature stays only if it improves branch accuracy, runtime, reliability or demo clarity.

## Person A — I/O and geometry

- Own `run.py`, NIfTI loading, physical transforms, strict JSON and invariant tests.
- First deliverable by hour 2: deterministic JSON on the synthetic case.
- Integration contract: validated CTA/mask in, scorer-safe daughter records out.

## Person B — proposals and tracking

- Own contrast calibration, surface proposals, 5–10 mm tracking, radius and confidence.
- First deliverable by hour 6: high-recall candidates on four representative real cases.
- Integration contract: physical-space polylines with evidence features.

## Person C — evaluation and reliability

- Own the blinded review sheet, proxy matching, ablation notes, runtime and memory.
- First deliverable by hour 7: one-page error gallery covering cap, duplicate, close-origin and common-trunk cases.
- Integration contract: CSV/JSON metrics and concise failure labels.

## Person D — product and demo

- Own the Aorta Map, evidence cards, deterministic screenshots and five-minute story.
- First deliverable by hour 4: one clickable case reading scorer JSON.
- Integration contract: read-only use of prediction JSON plus optional diagnostics.

## Shared clock

| Hour | Goal | Exit condition |
| --- | --- | --- |
| 0–2 | Lock interpretation; ask organizers the minimum branch size and metric tolerances | Synthetic JSON passes invariants |
| 2–6 | Run adaptive-shell baseline on all 25 cases | No crashes; case audit table exists |
| 6–11 | Add surface-normal proposals and short tracking | Four contrast/resolution cohorts reviewed |
| 11–15 | Refine ostium, seed, direction, radius and graph deduplication | Close-origin/common-trunk phantoms pass |
| 15–18 | Benchmark, tune one threshold and freeze the scorer path | Largest cases meet 4-core/8 GB budget |
| 18–21 | Connect real outputs to the Navigator and capture three visual checks | Every displayed number traces to JSON |
| 21–23 | Clean offline run and five-minute rehearsal | Fresh machine, one command, no internet |
| 23–24 | Buffer only | Submit frozen build and keep backup copy |

## Kill criteria

- Drop full-volume vesselness if it consumes more than roughly one quarter of runtime without a visible recall gain.
- Drop learned models unless labeled ostia appear early and a held-out ablation wins.
- Drop isotropic resampling if memory exceeds 2 GB or runtime margin collapses.
- Freeze visualization as soon as one branch can be selected and defended.
