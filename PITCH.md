# Five-minute demo script

## 0:00–0:35 — Reframe the problem

“This is CTA, not MRI, and it is not a full segmentation problem. We must find every direct opening off the supplied aorta, place it in physical millimetres, and prove the first five millimetres on CPU.”

Point to the score: discovery plus ostium placement is 70%. State that the team deliberately optimized for that.

## 0:35–1:15 — Show the pipeline

Use the eight-stage strip. Explain that the parent mask turns a whole-body search into a small surface problem, while per-case calibration handles the observed contrast shift.

## 1:15–2:35 — Defend one hard branch

Open the low-contrast 1.5 mm case. Click a numbered Aorta Map point. Show the linked crosshair, five-millimetre seed, direction, radius plane and evidence stack. Open the scorer-safe JSON so the judge sees that every visual corresponds to the required output.

## 2:35–3:35 — Show topology intelligence

Switch to “close origins” and explain why disconnected wall openings remain separate. Then switch to “common trunk” and show how one shared opening is counted once even if it later splits. This attacks duplicate errors directly.

## 3:35–4:20 — Prove operational fit

Show the runtime, memory and invariant strip. Explain that diagnostics and visualization happen after strict JSON so demo polish cannot corrupt scoring. Mention the clean offline run.

## 4:20–5:00 — Close with honesty

“Our known weakness is very small, low-contrast branches at 1.5 mm. We surface those in a review band and keep a speed-first fallback. Branchseed Navigator is fast, measurable and explainable—and every detection can be challenged.”

End on the Aorta Map, not a slide.
