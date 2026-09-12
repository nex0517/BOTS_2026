# Automatic Discovery of Direct Aortic Daughter Branches — Design Document

**Scope:** CPU-only, ~25 dev cases, ≤60 s/case, no internet, deterministic.
**Verdict up front:** this is a *connectivity + geometry* problem, not a segmentation problem. The single most informative signal is that a daughter lumen is **contrast-contiguous with the parent lumen through the aortic wall**. Almost everything else (vesselness, ML, deep learning) is decoration on top of that observation, and most of it is not worth the time.

---

## Part 0 — The one-paragraph answer

Threshold the CT into a "contrast lumen" mask using statistics derived from the *supplied aorta mask itself*. Restrict everything to a ~20 mm shell around the aorta. Take the 26-connected component of (lumen ∪ aorta-mask) that contains the aorta, subtract the aorta mask, and you are left with exactly the set of structures that are lumen-contiguous with the parent. Each **connected patch of contact with the aortic surface** is one candidate ostium — this definition automatically gives you one instance per common trunk, separate instances for two nearby ostia, and zero granddaughters. Trace each candidate with a **geodesic distance transform** confined to its own component; the geodesic level sets are your cross-sections, their centroids are your centerline, and a level set splitting into two is your bifurcation detector. Reject cropped ends by testing whether the **aortic wall normal** at the contact patch is parallel to the **aortic centerline tangent**. Reject veins and partial-volume bridges by requiring the connection to **persist across a ladder of thresholds**. Score the rest, threshold the score on dev data, emit physical coordinates via SimpleITK.

That is the whole algorithm. The rest of this document is why, and the details that will actually bite you.

---

## Part 1 — Decomposition of the problem

### 1.0 Mathematical structure

Let `I: Ω → ℝ` be the CT (HU) on voxel grid `Ω ⊂ ℤ³`, and `M ⊂ Ω` the parent aortic lumen. Define the *true* full arterial lumen `L ⊂ Ω` (unknown). The problem is:

> Find the connected components of `L \ M` that are 26-adjacent to `M`, and for each **contact interface** `∂ᵢ = N(Cᵢ) ∩ ∂M`, report its centroid, a 5 mm-downstream seed, the local radius, and the initial tangent.

Three observations follow immediately:

1. **Direct-daughter-ness is a topological predicate**, not a geometric one. It is decided by adjacency to `M`, which we know exactly. This is why the aorta mask is such a strong prior: it converts an open-ended detection problem into a constrained graph-connectivity problem on a thin shell.
2. **The instance-defining object is the interface `∂ᵢ`, not the component `Cᵢ`.** This single design choice resolves requirements 6, 7, 8 (granddaughters, nearby ostia, common trunks) for free. A common trunk that divides 8 mm out is one component with one interface → one instance. Two branches whose distal parts touch each other are one component with two interfaces → two instances. A granddaughter never touches `M` → never instantiated.
3. **The only genuinely hard part is estimating `L`** — i.e. deciding which voxels outside `M` are contrast lumen. Everything reduces to the robustness of that estimate and to a handful of hard geometric vetoes.

### 1.1 Obtaining the aortic lumen surface from the binary mask

Do **not** build a triangulated surface (marching cubes / VTK). You do not need one, and it will cost you time and a dependency. Use implicit representations:

- `Dout = distance_transform_edt(~M, sampling=spacing_zyx)` — Euclidean distance in **mm** outside the mask.
- `Din  = distance_transform_edt(M,  sampling=spacing_zyx)` — inradius in mm inside the mask.
- Signed distance `Φ = Dout − Din`. The surface is `{Φ = 0}`.
- Surface normals: `n = ∇Φ / ‖∇Φ‖`, computed with `np.gradient(gaussian_filter(Φ, σ≈1.0 mm), *spacing_zyx)`. This is smooth, cheap, defined everywhere, and orientation-consistent (outward positive) by construction. Smoothing Φ before differentiating is essential — raw EDT gradients are staircased.
- Surface voxel set: `S = M ∧ (Dout_of_M_complement == ...)`; practically `S = M ∧ ~binary_erosion(M)`.

**Aortic core vs. appendages (important, frequently overlooked).** Annotators sometimes include a 2–5 mm stub of a branch inside the "aorta" mask. If you do not handle this, every such ostium is displaced outward by the stub length. Fix:

```
r_ao      = median(Din[M])                  # ≈ aortic radius, mm
core_seed = Din > 0.60 * r_ao
M_core    = binary_propagation(core_seed, mask=M)   # morphological reconstruction
M_append  = M \ M_core
```
`M_append` is the set of thin appendages. Treat `M_core` as the parent for *ostium definition* and treat `M_append` as belonging to the daughter (it becomes the first millimetres of the daughter's traced lumen). Guard: if `|M_append| > 15 %` of `|M|`, the decomposition is untrustworthy (tapering aorta, iliacs included) — fall back to `M_core = M`. Because the aorta tapers craniocaudally, compute `r_ao` **per slice / per centerline station**, not globally.

**Aortic centerline.** Needed only for the cropped-end veto and for the axis tangent. Two options:

- *Cheap (implement first):* per-axial-slice centroid of the largest connected component of `M`, ordered by z, smoothed with a cubic spline (or `scipy.ndimage.uniform_filter1d` over ~10 mm), tangent by finite differences. Cost ~50 ms. Works because the abdominal aorta is within ~25° of the z-axis in essentially every case.
- *Robust (upgrade):* geodesic diameter — pick any voxel in `M`, find the geodesically farthest voxel `a` (MCP with cost `1/Din`), then the farthest from `a` → `b`, then the minimum-cost path `a→b`. Handles tortuosity and iliac limbs. Cost ~0.3 s.

Use the cheap one unless dev cases show tortuosity or the mask includes the bifurcation.

### 1.2 Candidate locations on the wall where another lumen emerges

Candidates are **not** generated by scanning the wall. They are generated by connectivity and then *localized* on the wall. Concretely:

```
ROI        = bbox(M) dilated by 30 mm, clipped to image
Dout       = EDT outside M within ROI (mm)
SHELL      = 0 < Dout ≤ 20 mm
LUMEN_OUT  = SHELL ∧ (T_lo ≤ I ≤ T_hi)
BIG        = label26(LUMEN_OUT ∨ M);  keep the label covering M
BRANCHES   = BIG ∧ ¬M
comps      = label26(BRANCHES)
contact_c  = { v ∈ comps[c] : N26(v) ∩ M_core ≠ ∅ }
patches    = label26(contact_c)      # one patch = one candidate ostium
```

The shell bound of 20 mm is not an accident: you only need 10 mm of trace beyond the wall, so 20 mm gives 2× headroom while cutting the working volume by ~5–10×. It also truncates vein leaks, which is convenient but means volume-based leak tests must be calibrated *within the shell*.

### 1.3 Discriminating true ostia from the seven kinds of impostor

| Impostor | Primary discriminator | Type |
|---|---|---|
| Noise | patch area ≥ `A_min` (≈1.5 mm²) and geodesic extent ≥ 5 mm | hard |
| Calcification | HU ceiling `T_hi` (≈700–800) excludes it from `LUMEN_OUT` entirely; residue fails the 5 mm tubular-extent test | hard |
| Partial volume / thin bridges | **threshold persistence**: the connection must survive raising T by ≥1 ladder step | hard-ish |
| Veins (IVC, renal, lumbar) | HU floor (arterial phase IVC ≈ 80–130 HU); plus persistence; plus radius-growth and axis-parallelism soft scores | hard floor + soft |
| Artifacts (streak, motion) | persistence + centerline smoothness + intensity variance along the trace | soft |
| Cropped superior/inferior ends | **wall-normal ∥ aortic-tangent** test | hard |
| Structures merely passing near | not connected → never enter `BRANCHES` | hard, free |

The cropped-end test deserves emphasis because it is the one place where an obvious-seeming rule is wrong. Do **not** veto on the *branch direction* being axial — real branches (IMA, lumbar arteries) leave at acute angles and run near-parallel to the aorta. Veto on the **surface normal at the contact patch**: a cut face has `|n̄ · t_ao| > cos(35°)`; a genuine ostium sits on the lateral/anterior/posterior wall where `n̄ ⊥ t_ao` regardless of how acutely the daughter then turns. Supplement with: patch centroid within `r_ao + 5 mm` geodesically of a centerline endpoint **and** contact area > 40 % of the local aortic cross-section → veto.

### 1.4 Proving direct connection

Connection is established by construction (the component is 26-connected to `M`). The real question is whether the connection is *real* or a partial-volume artefact. The decisive test is **persistence over the threshold ladder**:

Run candidate generation at `T_k`, k = 0..3 (rising). A genuine ostium is a real lumen, so it stays attached as T rises, until the whole branch disappears. A one-voxel bridge into the IVC or into cancellous bone breaks at the first step. Score:

```
persist(c) = number of ladder levels at which a patch within 2 mm of this
             patch centroid exists AND its component still extends ≥ 5 mm
```
Require `persist ≥ 2` of 4 as a hard rule; use the value as a soft feature. This single test is, in my experience, worth more than any vesselness filter.

### 1.5 Tracing 5–10 mm

Geodesic distance inside the component, seeded on the patch:

```
costs = ones(shape); costs[~cell] = inf
mcp   = skimage.graph.MCP_Geometric(costs, sampling=spacing_zyx)
g, _  = mcp.find_costs(starts=patch_voxels)     # g = geodesic mm from ostium
```
`g` is arclength along the lumen, correct under anisotropy, and it bends with the vessel for free. No stepping, no Kalman filter, no re-seeding, no divergence. Level sets `B_s = {s−δ ≤ g ≤ s+δ}` with δ = 0.4 mm are quasi-perpendicular cross-sections. Centerline `c(s) = centroid(largest CC of B_s)` for s = 0.5, 1.0, …, min(10, s_bif).

If two patches share a component, first partition the component into a **geodesic Voronoi cell** per patch (run MCP once per patch, take argmin) so branch A's trace cannot wander into branch B.

### 1.6 Bifurcation detection

At each `s`, label the CCs of `B_s` within the cell. Declare a bifurcation at `s*` if ≥2 CCs each have equivalent radius ≥ `max(0.8 mm, 0.45·r₀)` and that split persists for ≥1.5 mm of further arclength (guards against noise-induced momentary splits and against a single cross-section grazing a fold). Truncate the trace at `min(10 mm, s* − 0.5 mm)`.

Note the failure case you *will* hit: a celiac trunk dividing at 6–8 mm, or an early-dividing trunk at 3–4 mm. If `s* < 5 mm`, place the seed at `0.85·s*` rather than 5 mm and record a flag. Reporting a seed inside the wrong child is far worse than reporting it slightly proximal.

### 1.7 Two separate nearby ostia

Separate patches → separate instances. The risk is *over*-splitting: a calcified plaque or a noise dropout can bisect one ostium into two patches. Merge patches when **all** of:
- centroid separation < 3.0 mm, **and**
- their traces at s = 3 mm have centerline points within 2.0 mm of each other, **and**
- the angle between their fitted directions < 30°.

Requiring trace convergence (not just proximity) is what stops you from merging genuinely adjacent ostia such as a main + accessory renal artery, which diverge immediately.

### 1.8 Common trunk = one daughter

Free: one interface → one instance, and the bifurcation detector stops the trace before the division. No special handling needed. Do not write any.

### 1.9 Radius — see Part 6. 1.10 Direction — see Part 7.

---

## Part 2 — Candidate approaches

### A. Pure classical: threshold + shell + connected components + morphology

**Algorithm.** Adaptive HU window from aorta statistics → shell → CC → contact patches → EDT radius → chord direction.
**Strengths.** ~8–15 s/case. ~200 lines. Fully deterministic. Directly encodes the task definition. Degrades gracefully.
**Weaknesses.** Single threshold is brittle across contrast phases; leaks into veins/bone; no principled scoring; poor centerline for curved branches.
**Runtime.** 10–15 s. **Difficulty.** Low. **Generalization.** Moderate–good. **Likely F1.** 0.55–0.70.
**Failure modes.** Late-phase cases leak into IVC; low-contrast cases lose small branches entirely; ostium bias where the mask includes stubs.

### B. Vesselness (Frangi/Sato) + centerline graph

**Algorithm.** Multiscale Hessian vesselness over the ROI → ridge extraction / skeletonization → build a graph of centerline segments → attach segments whose endpoints lie near the aortic surface → branch = attached segment.
**Strengths.** In principle recovers branches whose lumen is broken by noise or stenosis; scale-adaptive; radius comes free from the optimal scale.
**Weaknesses — and these are disqualifying for the *detection* role.** Frangi's tubularity model (λ₁≈0, |λ₂|≈|λ₃|≫0) is explicitly violated at junctions. Vesselness is **systematically suppressed exactly at the ostium**, which is the object you are asked to localize. It also responds strongly to the aortic wall/calcium interface and to the aortic lumen itself at large scales, producing wall-hugging ridges. Multiscale over 4–6 scales on even a cropped ROI is 15–45 s in `skimage` on 4 cores, eating most of your budget. And it adds ~6 parameters (α, β, c, scale range, scale count, ridge threshold) you must tune on 25 cases.
**Runtime.** 20–50 s. **Difficulty.** High. **Generalization.** Poor without careful tuning. **Likely F1.** 0.40–0.60, with worse ostium error than A.
**Verdict:** I do not believe this works reliably as a primary detector on this budget. Say no.

### C. Hybrid: connectivity-based candidate generation + geometric/topological validation + scoring (**recommended**)

**Algorithm.** A's connectivity backbone, plus: threshold ladder with persistence, aortic-core/appendage decomposition, geodesic tracing with bifurcation detection, wall-normal cap veto, multi-estimator radius, robust direction fit, explicit candidate score with hard vetoes.
**Strengths.** Keeps the one reliable signal (contiguity) as the generator, and uses geometry only for *rejection* and *measurement*, where geometry is trustworthy. Every component is independently testable. Tunable with a single score threshold. No training required, but trivially upgradable to D.
**Weaknesses.** ~700–900 lines. Cannot recover a branch whose lumen is genuinely disconnected in the image (severe ostial stenosis, calcified ostium occupying the whole opening).
**Runtime.** 12–25 s. **Difficulty.** Medium. **Generalization.** Good — all thresholds are relative to per-case aortic statistics. **Likely F1.** 0.70–0.85.
**Failure modes.** Ostial calcification; very thick slices (≥3 mm) where 1.5 mm branches are unresolved; portal-venous phase cases.

### D. Lightweight ML on candidate features

**Algorithm.** Use C to generate candidates (recall-oriented, loose thresholds), compute 12–20 features per candidate, train gradient-boosted trees / random forest, replace the hand-weighted score with `P(true branch)`.
**Strengths.** 25 cases × (5–12 positives + 20–60 negatives) ≈ 150–250 positives and 500–1500 negatives — genuinely enough for a 15-feature model. Learns the operating point instead of you guessing it. Leave-one-case-out CV gives an honest F1 estimate. Adds ~2 s at inference and one small `.joblib`.
**Weaknesses.** Needs reference labels for dev cases; risks overfitting to 25 patients' contrast protocol; a reproducibility liability if the model file isn't pinned; **useless until C works**.
**Runtime.** +1 s. **Difficulty.** Low *given C*. **Generalization.** Good for the score, risky for the features if hidden cases differ in phase.
**Verdict:** the correct use of ML here, and a strong 6-hour upgrade — but strictly after C is validated.

### E. Deep learning (3D U-Net for branch segmentation / ostium heatmap)

25 cases, CPU-only inference, 60 s, no internet, no pretrained weights available offline. A 3D U-Net on a 20 mm shell would take most of a day to train, would overfit, and would need ~30–60 s CPU inference. **Do not.** If you have a pretrained TotalSegmentator-class model available offline it changes the calculus slightly, but you still have to solve the ostium problem, which is not what those models output.

### Ranking

1. **C (hybrid)** — recommended.
2. **A (pure classical)** — your 1-hour baseline and your fallback; C is a strict superset.
3. **D** — best marginal-value upgrade once C works, purely as a scoring replacement.
4. **B** — optional soft feature only (one scale, as a tubularity prior). Not a detector.
5. **E** — no.

---

## Part 3 — The recommended pipeline, stage by stage

Notation: `spacing_zyx = image.GetSpacing()[::-1]` (SimpleITK returns x,y,z; NumPy arrays are z,y,x).

### Stage 1 — I/O and validation
- **In:** `image.nii.gz`, `aorta_mask.nii.gz`.
- **Out:** `I` (int16, z,y,x), `M` (bool), `spacing_zyx`, the SimpleITK image object (retained for coordinate transforms).
- **Algorithm:** `sitk.ReadImage`. Read the **mask first** (it decompresses fast), compute its bounding box, then read the image. Assert identical size/spacing/origin/direction within 1e-4; if not, resample `M` onto `I`'s grid with nearest-neighbour and warn.
- **Params:** none.
- **Complexity:** O(N). **Cost:** 3–9 s (gzip decompression dominates; it is your single largest fixed cost and it is irreducible for `.nii.gz`).
- **Failure:** mismatched grids; mask empty; multiple disconnected mask components (keep the largest, log the rest).

### Stage 2 — ROI crop and aortic statistics
- **In:** `I`, `M`.
- **Out:** cropped `I_r`, `M_r`, ROI offset; `μ_ao`, `σ_ao`, `r_ao(z)`, background level `μ_bg`.
- **Algorithm:** ROI = bbox(M) + 30 mm margin. `M_ero = binary_erosion(M_r, ball(2 mm))`; `μ_ao = median(I_r[M_ero])`, `σ_ao = 1.4826·MAD`. `μ_bg = median(I_r[SHELL ∧ I_r < 150])`. Per-slice `r_ao(z) = max(Din)` over the slice, smoothed.
- **Params:** ROI margin 30 mm; erosion 2 mm.
- **Why:** every downstream threshold becomes *relative to this patient's contrast*, which is what makes the method generalize across phases and kVp without retuning.
- **Complexity:** O(N) + O(|ROI|). **Cost:** 0.3 s.
- **Failure:** mask includes non-lumen (thrombus) → `μ_ao` depressed → thresholds too low → leaks. Detect via bimodality of `I_r[M_ero]`; if bimodal, use the upper mode.

### Stage 3 — Aortic surface, normals, core/appendage split, centerline
- **In:** `M_r`, `spacing_zyx`.
- **Out:** `Din`, `Dout`, `Φ`, `n(x)`, `M_core`, `M_append`, centerline `A(u)` with tangents `t(u)`, endpoints `e₀,e₁`.
- **Algorithm:** as §1.1.
- **Params:** core fraction 0.60; normal smoothing σ = 1.0 mm; centerline smoothing 10 mm.
- **Complexity:** two EDTs over ROI, O(|ROI|). **Cost:** 1.0–2.5 s.
- **Failure:** tortuous aorta breaks per-slice centroids → switch to geodesic-diameter centerline. Iliacs in mask → per-slice largest component only; or geodesic method.

### Stage 4 — Shell and adaptive lumen thresholding (ladder)
- **In:** `I_r`, `Dout`, aortic stats.
- **Out:** `LUMEN_OUT_k` for k = 0..3.
- **Algorithm:** `SHELL = (0 < Dout ≤ 20 mm)`. `T_hi = max(700, μ_ao + 3.5σ_ao + 150)` (calcium/bone-cortex ceiling). Ladder `T_k = max(150, f_k · μ_ao)` with `f = [0.50, 0.60, 0.70, 0.80]`. `LUMEN_OUT_k = SHELL ∧ (T_k ≤ I_r ≤ T_hi)`.
- **Why fractions, not μ − cσ:** in a uniform contrast lumen σ is just image noise (20–40 HU), so `μ − 2.5σ` barely moves off the aortic mean and loses small branches whose apparent HU is depressed 20–35 % by partial volume. Fractions track partial-volume attenuation, which is the actual mechanism. The absolute floor of 150 HU keeps you above soft tissue (~40–60 HU) and above arterial-phase venous blood (~80–130 HU).
- **Params:** shell 20 mm, `f_k`, floor 150, ceiling formula. These are the top tuning targets.
- **Complexity:** O(|shell|). **Cost:** 0.1 s per level.
- **Failure:** portal-venous or delayed phase → IVC above 150 HU → leaks (persistence and shape tests must catch it). Low cardiac output / poor bolus → `μ_ao` ≈ 180 → `T_0` = 150 floor → thin branches lost.

### Stage 5 — Connectivity and candidate ostium patches
- **In:** `LUMEN_OUT_k`, `M_r`, `M_core`.
- **Out:** per level: components, contact patches with area, centroid, mean normal.
- **Algorithm:** as §1.2, run per ladder level. Keep the level-0 (most permissive) patch set as the candidate list; annotate each with its persistence count from higher levels.
- **Params:** 26-connectivity; `A_min` = 1.5 mm²; patch–patch matching radius 2.0 mm across levels.
- **Complexity:** O(|shell|·α). **Cost:** 0.3 s per level.
- **Failure:** a real branch fused to a vein at every level (shared wall + partial volume) → over-large component; caught by radius-growth veto but may distort the trace.

### Stage 6 — Geodesic tracing
- **In:** component, its patches, `spacing_zyx`.
- **Out:** per candidate: `c(s)` for s ∈ [0.5, L], L ≤ 10 mm; bifurcation arclength `s*`; band volumes.
- **Algorithm:** geodesic Voronoi partition (one MCP run per patch, argmin), then MCP within the cell, level sets at 0.5 mm steps with δ = 0.4 mm, largest-CC centroid, bifurcation rule of §1.6.
- **Params:** step 0.5 mm, δ 0.4 mm, child-radius rule `max(0.8 mm, 0.45 r₀)`, split persistence 1.5 mm, cap 10 mm.
- **Why it works:** geodesic distance inside a tube *is* arclength; its level sets *are* cross-sections; component count of a level set *is* the branching number. You get three needed quantities from one primitive, with no iterative stepping to diverge.
- **Complexity:** O(|cell| log|cell|) per candidate; cells are 1–10 k voxels. **Cost:** 0.5–2 s total for ~30 candidates.
- **Failure:** a branch that immediately hugs the aortic wall and re-touches it ("kissing") → geodesic shortcuts along the contact; mitigate by excluding voxels with `Dout < 0.5 mm` from the cell after the first 1 mm.

### Stage 7 — Ostium localization
- **In:** patch voxels, `Φ`, `c(s)`.
- **Out:** `ostium_xyz_mm`.
- **Algorithm (primary):** area-weighted centroid of the patch's *aortic-side* surface voxels, then snap to the `Φ = 0` isosurface by moving along `−n` a distance `Φ(centroid)` (subvoxel, uses the smoothed `Φ`).
  **Algorithm (alternative, evaluate on dev):** extrapolate the fitted proximal direction back from `c(1.5 mm)` to the `Φ = 0` crossing.
- **Params:** none beyond `Φ` smoothing.
- **Why:** the interface centroid is the anatomically correct "centre of the opening" even for oblique take-offs, where the opening is an elongated ellipse. The extrapolation variant is more accurate when the patch is fragmented by calcium.
- **Complexity:** O(|patch|). **Cost:** negligible.
- **Failure:** annotation dilation/erosion of `M` shifts the wall by ±1 voxel → systematic 0.5–1 mm bias. Measure this bias on dev cases and consider a global correction along `n`.

### Stage 8 — Seed estimation
- **In:** `c(s)`, `s*`.
- **Out:** `seed_xyz_mm`.
- **Algorithm:** `s_seed = min(5.0, 0.85·s*)`; `seed = c(s_seed)` with linear interpolation between 0.5 mm stations. Verify `seed` lies inside the lumen mask; if not (thin/noisy cross-section), snap to the voxel in `B_{s_seed}` with maximal `Din_lumen`.
- **Failure:** branch shorter than 5 mm but above the eligibility bar — should not happen given the 5 mm eligibility rule, but guard anyway.

### Stage 9 — Radius estimation
See Part 6. Output `radius_mm`.

### Stage 10 — Direction estimation
See Part 7. Output unit `direction_xyz`.

### Stage 11 — Duplicate / common-trunk handling
- Merge rule of §1.7. Additionally, deduplicate across ladder levels by patch centroid proximity (2 mm).
- Final global dedup: if two surviving candidates have ostia < 2.5 mm apart **and** directions within 25°, keep the higher-scoring one.

### Stage 12 — Scoring, vetoes, output
See Parts 8–9. Sort deterministically (descending score, tie-break by ostium z then y then x), assign `branch_001…`, write JSON.

---

## Part 4 — The hardest problem: finding ostia (an honest survey)

Evaluating every technique you listed, against the actual constraints:

| Technique | Verdict | Role |
|---|---|---|
| **Intensity outside the mask near its surface** | **Essential.** This is the signal. | Core |
| **Narrow shell around the aorta** | **Essential.** 5–10× compute reduction and it bounds leak extent. | Core |
| **Surface normals** | **Essential, but for rejection not detection.** The wall-normal-vs-axis test is the single cleanest cropped-end veto. Also gives the "radially outward" soft score. | Core (veto/score) |
| **Rays perpendicular to the wall / unrolled wall map** | Useful, but *secondary*. Sampling `max HU` over radial distance 2–8 mm as a function of (arclength, angle) produces a 2D map where ostia are bright blobs. Beautiful for debugging and a decent fallback detector for branches that fail connectivity. But it degrades badly with tortuosity, it double-counts where the aorta bends, and it has no notion of connectivity so it lights up on adjacent veins. | Debug + optional 2nd-pass recall |
| **Vesselness (Frangi/Sato)** | **Weak here.** Suppressed at junctions — precisely the wrong place. Expensive multiscale. Use one scale (σ ≈ 1.0–1.5 mm) evaluated *along the trace at s = 3–5 mm*, not at the ostium, as a soft tubularity feature. That is defensible. Using it to *find* ostia is not. | Optional soft score |
| **3D connected components** | **Essential.** The instance-generating primitive. | Core |
| **Local intensity profiles** | Useful for FWHM radius refinement and for a "contrast continuity" score along the trace. | Refinement |
| **Distance transforms** | **Essential** — shell, core/appendage split, radius, wall snapping. | Core |
| **Geodesic paths** | **Essential** — centerline, arclength, bifurcation, Voronoi partition. One primitive, four jobs. | Core |
| **Vascular graph** | Over-engineering for a 10 mm trace. You need a path, not a graph. Skip. | No |
| **Region growing** | Equivalent to thresholded CC. The *threshold ladder* is a strictly better formulation of the same idea (it gives you persistence for free). Use the ladder. | Subsumed |
| **Tubular structures extending outward** | Yes, via geodesic extent + PCA elongation of the first 8 mm. Cheap, no Hessian needed. | Core score |
| **Local cylinder fitting** | Marginal. RANSAC cylinder fits on 200 voxels are noisy and slow. The geodesic cross-section already gives radius and axis. Skip. | No |
| **Centerline continuity / smoothness** | Yes, as a soft score: RMS residual of a quadratic fit to `c(s)`, and the max turning angle between consecutive 1 mm segments. Discriminates real vessels from noise chains and streak artefacts. | Core score |

**The reliable combination:** *signed-distance shell* + *per-case adaptive threshold ladder* + *26-connectivity to the aortic core* + *contact-patch clustering as the instance primitive* + *geodesic tracing* + *wall-normal cap veto* + *threshold-persistence veto* + *soft score over shape/intensity/continuity*.

What makes this exploit the prior maximally: the aorta mask supplies (a) the connectivity anchor, (b) the intensity reference for *this patient's* contrast, (c) the surface and its normals, (d) the axis for cap rejection, and (e) the radius scale for relative thresholds. Five independent uses of one mask. Any method that does not use all five is leaving information on the table.

---

## Part 5 — Physical coordinates: the bug catalogue

Coordinate bugs are the most likely way to score near zero on ostium localisation while believing your algorithm works. Be paranoid.

### 5.1 The conversion

A voxel index `i = (i_x, i_y, i_z)` maps to physical mm as

```
p = O + D · diag(S) · i
```
where `O = image.GetOrigin()` (3-vector), `S = image.GetSpacing()` (3-vector), `D = image.GetDirection()` reshaped to 3×3 **row-major**. Use SimpleITK rather than doing this by hand:

```python
p = img.TransformIndexToPhysicalPoint((int(ix), int(iy), int(iz)))            # integer index
p = img.TransformContinuousIndexToPhysicalPoint((fx, fy, fz))                 # subvoxel centroid
```

### 5.2 The six ways to get it wrong

1. **Multiplying index by spacing.** This silently drops both `O` and `D`. It is wrong whenever the origin is nonzero (always) or the direction matrix is not identity (very common in clinical NIfTI, where `D` often has −1 entries or an oblique gantry tilt). Errors of 100–400 mm, or mirrored anatomy.
2. **Axis order.** `sitk.GetArrayFromImage(img)` returns `(z, y, x)`. `GetSpacing()`, `GetSize()`, `TransformIndexToPhysicalPoint()` all use `(x, y, z)`. Every conversion between NumPy index and ITK index must reverse. Write exactly two helper functions and never index by hand:
   ```python
   def np_to_itk(idx_zyx):  return tuple(float(v) for v in idx_zyx[::-1])
   def phys(img, idx_zyx):  return np.array(img.TransformContinuousIndexToPhysicalPoint(np_to_itk(idx_zyx)))
   ```
3. **Anisotropic spacing in SciPy.** `distance_transform_edt` and `MCP_Geometric` must be given `sampling=spacing_zyx` (NumPy order). Omitting it makes 1 mm slices count the same as 0.7 mm in-plane steps, corrupting every radius and arclength by up to 40 %.
4. **The ROI offset.** After cropping, local index `+ roi_origin_zyx` before converting. Forgetting this is a clean, constant, large error — which makes it easy to spot and easy to miss.
5. **Mixing nibabel and SimpleITK.** nibabel's affine is **RAS**; SimpleITK's physical space is **LPS**. They differ by sign flips on x and y. If the reference JSON was produced with SimpleITK (the spec says to use it, so assume yes), use SimpleITK for *everything* — reading, transforms, and any resampling. Do not import nibabel at all. If you must, convert explicitly: `p_LPS = diag(-1,-1,1) · p_RAS`.
6. **Direction vectors transformed as points.** A direction in index space maps to physical as `d_phys = D · diag(S) · d_index`, then renormalize — **not** `p(i + d) ` treated as a point. The clean way to avoid this entirely: **convert all centerline points to mm first, then do every geometric operation (PCA, fitting, normalization, distances) in physical space.** Then a direction is just a difference of physical points and cannot be wrong. Do this.

### 5.3 A self-test to run on every case

Assert `‖p(i+ê_x) − p(i)‖ == S_x` to 1e-6 for each axis, and assert that the physical centroid of the aorta mask computed via ITK matches the one computed via `O + D·diag(S)·mean_index` to 1e-6. Both are one-liners and both fail loudly on any of bugs 1–4.

---

## Part 6 — Radius estimation

### Candidate estimators

1. **EDT inradius at the seed.** `r = Din_lumen(seed)` where `Din_lumen = EDT(M ∪ LUMEN_OUT, sampling)`. Cost: already computed. Biased *low* by ~0.5 voxel (EDT measures to the nearest background voxel centre) and biased *high* if the branch touches another bright structure.
2. **EDT maximum over the cross-section.** `r = max(Din_lumen over the CC of B_{s_seed})`. More stable than the point value; still inherits threshold dependence.
3. **Equivalent-area radius from the geodesic band.** `A = V_band / (2δ)`, `r = sqrt(A/π)`. Uses ~hundreds of voxels, so it is much less noisy than a single EDT lookup, and it degrades gracefully for non-circular cross-sections. Cheap.
4. **Ray-cast FWHM on intensity.** From the seed, cast 32 rays in the plane ⊥ `d`, trilinearly sample `I` every 0.2 mm, find the first crossing of `I_half = (I_lumen + I_bg)/2` where `I_lumen` = mean of the inner 1 mm and `I_bg` = 20th percentile over 4–8 mm. `r = median(ray lengths)`. Threshold-independent, subvoxel, and the median makes it robust to 1–2 rays escaping into an adjacent vessel.
5. **Vesselness optimal scale.** Radius ≈ `σ*·√2`. Coarse (quantized by your scale set), expensive, and unreliable near junctions. Skip.

### What I would implement, in order

**First: #3 combined with #2, taking their median along with the value at neighbouring stations.**

```
r_candidates = [r_area(s_seed), r_edtmax(s_seed), r_area(s_seed−1), r_area(s_seed+1)]
radius_mm    = median(r_candidates)
```
Reason: both are already computed as by-products of the geodesic tracing (zero marginal cost), and combining an area-based and a distance-based estimator cancels their opposite biases (area over-estimates for elliptical/partial-volume-blurred cross-sections, EDT under-estimates by the half-voxel offset). Sampling three stations along the vessel suppresses single-slice noise, which matters because a 2 mm vessel in 0.7 × 0.7 × 1.0 mm voxels has only ~12 voxels per cross-section.

**Second (upgrade): #4, ray-cast FWHM**, used as the primary with the median of #2/#3 as a sanity clamp: if `|r_fwhm − r_geo| > 0.5·r_geo`, keep `r_geo`. FWHM is the right answer physically — it does not depend on your threshold choice, only on the local contrast — and it gets you subvoxel accuracy on 1.5–3 mm vessels where the geodesic estimates quantize badly. It costs ~5 ms per candidate. Do this once the pipeline works end to end.

Apply a final clamp `r ∈ [0.5, 0.9·r_ao]`; anything outside is a leak, not a branch.

---

## Part 7 — Direction estimation

### Why `seed − ostium` is not good enough (and when it is)

The chord `(seed − ostium)/‖·‖` has three error sources: (a) the ostium centroid is the noisiest point in the trace — it sits in the junction region where the lumen flares, so its centroid is pulled toward the aortic lumen and biased by the flare's asymmetry; (b) the seed is a single cross-section centroid; (c) for a vessel curving in the first 5 mm, the chord under-estimates the *initial* tangent by roughly half the total turning angle.

However — be honest about magnitudes. For a typical 5 mm segment with 20° of total curvature, the chord differs from the initial tangent by ~10°, and from the mean tangent by ~0°. If the reference direction was itself defined as "unit vector from ostium into daughter" over the first 5 mm (which is what the spec literally says), **the chord is arguably the target quantity**. So: implement the robust fit, but *validate against the chord on dev data* and use whichever has lower angular error. Do not assume sophistication wins.

### Recommended estimator

```
P   = [c(s) in mm for s in 0.5 ... min(6.0, 0.9·s*)]         # ≥ 6 points
w   = 1.0 for all (or w = exp(−s/8) to weight proximal)
# robust total-least-squares via IRLS:
repeat 3 times:
    p̄ = Σ w_i P_i / Σ w_i
    principal axis u = first eigenvector of Σ w_i (P_i − p̄)(P_i − p̄)ᵀ
    residual r_i = ‖(P_i − p̄) − ((P_i − p̄)·u) u‖
    w_i ← w_i · 1/(1 + (r_i/0.6mm)²)                          # Cauchy weights
u  ← u · sign(u · (P_last − ostium))                          # orient outward
```

**Handling curvature.** Compute the fit residual RMS. If `RMS > 0.5 mm`, the segment is curved enough that a straight fit is inappropriate: fit a quadratic `c(s) ≈ a + b s + e s²` componentwise by least squares on arclength `s`, and take the tangent `b + 2 e s₀` evaluated at `s₀ = 2.5 mm` (mid-segment of the first 5 mm), normalized. Using the mid-segment tangent rather than `s = 0` avoids amplifying junction-region noise through extrapolation.

**Consistency check (cheap and worth it).** Require `u · n̄_ostium > 0` — the direction must have a positive outward radial component. If not, the trace ran the wrong way (usually a kissing-vessel geodesic shortcut); flag and reject the candidate.

**Sanity bound.** Angle between `u` and the chord should be < 35°. Larger means the trace is broken.

---

## Part 8 — False-positive rejection

Design principle: **hard rules must be things that are true by anatomy or by task definition; everything statistical goes in the soft score.** A hard rule that is 5 % wrong costs you more recall than the precision it buys, given that discovery is 45 % of the grade and F1-shaped.

### Hard rules (veto, no appeal)

| # | Rule | Threshold | Rationale |
|---|---|---|---|
| H1 | Must be 26-connected to `M_core` through voxels in `LUMEN_OUT` | — | Task definition of "direct daughter" |
| H2 | Geodesic extent ≥ 5.0 mm beyond the wall | 5.0 mm | Task eligibility criterion, stated |
| H3 | Contact patch area ≥ `A_min` | 1.5 mm² | Below this it is noise or a single-voxel bridge |
| H4 | Wall normal at patch not axis-aligned: `|n̄ · t_ao| < cos(35°)` | 35° | Cropped end veto |
| H5 | Patch not adjacent to an image-boundary face of the mask | — | Cropped end veto, complementary to H4 |
| H6 | Estimated radius ≥ `r_min` | 1.0 mm (tune!) | Dataset minimum size |
| H7 | Radius ≤ `0.9·r_ao(local)` | — | Anything this big is the aorta itself or a leak |
| H8 | Threshold persistence ≥ 2 of 4 ladder levels | 2 | Kills partial-volume bridges |
| H9 | Mean HU over the traced segment ≥ `max(150, 0.45·μ_ao)` | — | Kills veins in arterial phase |
| H10 | `u · n̄_ostium > 0` | — | Direction sanity; catches reversed traces |

### Soft scores (feed the score, never veto alone)

- Contrast ratio `mean_HU(trace)/μ_ao`
- Contrast *homogeneity*: 1 − (IQR of HU along trace)/μ_ao — artefacts are heterogeneous
- Elongation: λ₁/(λ₂+λ₃) of the component's first 8 mm
- Length: `min(L, 10)/10`
- Radius plausibility: penalize `r < 1.2 mm` and `r > 6 mm`
- Centerline smoothness: quadratic-fit RMS and max turning angle
- Radius consistency: `1 − |r_area − r_edt|/r`
- Patch compactness: `4π·A/P²` of the contact patch
- Persistence count (2–4)
- Radial outwardness: `u · n̄` (0.3–1.0)
- Optional single-scale vesselness at `c(3 mm)`

### The specific impostors

- **Calcified aortic wall.** Excluded by `T_hi` from `LUMEN_OUT`. Residual risk: an ostium *ringed* by calcium narrows or splits the patch → merge rule (§1.7) recovers it; heavy ostial calcification occluding the opening is a genuine miss and I would not try to fix it.
- **Contrast-enhanced veins.** H9 (HU floor) handles arterial phase. For late-phase cases: soft penalty for `|axis · t_ao| > cos(30°)` combined with radius > 5 mm and radius *growing* with s. Hard veto if `r(10 mm) > 2.2 · r(2 mm)` — real proximal arteries taper or hold; a leak into a vein balloons.
- **Tiny branches below minimum size.** H6. This is the parameter that dominates your F1 — measure the reference radius distribution before choosing it (Part 10).
- **Noise.** H2 + H3 + smoothness score.
- **Adjacent-but-unconnected structures.** Free, by H1.
- **Cropped ends.** H4 + H5 + the `A_patch > 0.4 · π r_ao²` heuristic.
- **Artifacts (streak/motion).** Soft: homogeneity + smoothness. Streaks produce high-HU chains that are thin, non-tubular, and geometrically erratic; they usually fail H8 too.
- **Downstream branches.** Free, by construction (no contact patch).
- **One vessel → multiple detections.** Cross-level dedup + merge rule + final 2.5 mm/25° global dedup. Report a `duplicate_rate` metric in validation so you know.

---

## Part 9 — Candidate scoring

### Concrete formulation

Normalize each feature to [0,1] with a piecewise-linear ramp, then take a weighted sum. Linear is the right choice here: with ~25 cases you cannot fit anything more expressive without overfitting, and a linear score is trivially inspectable when it misbehaves.

```
def ramp(x, lo, hi):  return clip((x - lo) / (hi - lo), 0, 1)

f1  persistence     = ramp(persist,            2,    4)        # w 0.20
f2  contrast        = ramp(meanHU/μ_ao,       0.45, 0.85)      # w 0.20
f3  length          = ramp(L_mm,               5,   10)        # w 0.15
f4  tubularity      = ramp(λ1/(λ2+λ3),         2,    8)        # w 0.12
f5  radius_plaus    = ramp(r_mm,             0.9,  1.8) *
                      (1 - ramp(r_mm,        5.0,  8.0))       # w 0.10
f6  outwardness     = ramp(u·n̄,              0.20, 0.70)       # w 0.08
f7  smoothness      = 1 - ramp(rms_mm,       0.30, 1.20)       # w 0.07
f8  homogeneity     = 1 - ramp(iqrHU/μ_ao,   0.15, 0.50)       # w 0.05
f9  patch_compact   = ramp(4πA/P²,           0.30, 0.75)       # w 0.03

score = Σ wᵢ fᵢ            # weights sum to 1.0
accept if all hard rules pass AND score ≥ τ
```

### Initial thresholds without training

Start at **τ = 0.50** and `r_min = 1.0 mm`. Then, on dev, sweep τ ∈ [0.30, 0.75] in steps of 0.025 and `r_min` ∈ {0.8, 1.0, 1.25, 1.5, 1.75} and pick the (τ, r_min) maximizing mean per-case F1 — *with leave-one-case-out or at minimum a 15/10 split*, because tuning two parameters on 25 cases and reporting the training F1 will flatter you by 5–10 points.

Weights: the top three (persistence, contrast, length) carry 55 % because they are the features that actually separate branches from leaks. The rest are tie-breakers. Do not spend an hour hand-tuning weights — if you have reference labels, fit logistic regression on these nine features (LOOCV) and use the learned coefficients; that is 20 lines and strictly better than intuition. If the fitted model does not beat the hand weights in LOOCV, keep the hand weights and note it.

**Guard against over-suppression:** always log rejected candidates with their features to `prediction_debug.json`. On hidden cases you cannot debug interactively, so a per-case count of "hard-rule rejections by rule" is your only diagnostic. If one rule fires on 40 % of candidates in a hidden-like case, that rule is miscalibrated.

---

## Part 10 — Dataset exploration plan (do this **before** writing the final algorithm)

Budget ~45–60 minutes. Every measurement below directly sets a parameter. Write one script, `explore.py`, that emits a CSV per case plus a montage PNG.

### 10.1 Measurements, and the parameter each one sets

| # | Measurement | Sets |
|---|---|---|
| M1 | `μ_ao, σ_ao, p1, p5, p50, p99` of HU inside `M` eroded 2 mm, per case. Histogram across 25 cases. | Threshold ladder `f_k`, absolute floor |
| M2 | For each reference ostium, the HU profile along the reference direction from 0 → 15 mm (sampled every 0.25 mm). Plot all profiles normalized by `μ_ao`. | The **critical** number: the 5th percentile of branch HU / `μ_ao` → your safe `T_0` |
| M3 | Background: median HU in the shell excluding >150 HU; and IVC HU if you can identify it (largest sub-150 HU blob adjacent to the aorta). | Separation margin; tells you whether cases are arterial or venous phase |
| M4 | Reference radius distribution (histogram, min, p5). | `r_min` — the single highest-leverage parameter |
| M5 | Distance from each reference ostium to the `M` surface (`Φ` at the reference point). If systematically > 1 mm, the mask includes branch stubs. | Whether to implement core/appendage split; ostium bias correction |
| M6 | Distance from each reference ostium to the nearest centerline endpoint. | `d_cap` safety margin for H4/H5 |
| M7 | Angle between reference direction and local aortic tangent; and between reference direction and the wall normal at the ostium. | Confirms that H4 vetoes on the *normal*, not the direction; sets the outwardness ramp |
| M8 | Branches per case; pairwise ostium distances (min). | Expected candidate count; `d_merge` upper bound |
| M9 | Recall-vs-threshold curve: run only Stages 1–6 at `f ∈ {0.40 … 0.85}` step 0.05 and record, per reference branch, whether any candidate patch lands within 5 mm. Plot recall(f) and #candidates(f) on the same axes. | `f_k` ladder; shows you the knee |
| M10 | Component volume vs. threshold for each candidate (persistence curves). Overlay true vs. false. | Confirms/refutes the persistence hypothesis; sets `persist ≥ 2` |
| M11 | Voxel spacing distribution across cases, especially slice thickness. | Whether anisotropy handling is critical (it is if any case has ≥2.5 mm slices) |
| M12 | `cProfile` on 3 cases, per-stage wall time and `tracemalloc` peak. | Where to optimize (probably nowhere except I/O) |

### 10.2 Visualizations to produce

1. **Per-case 3-panel montage:** axial / coronal / sagittal maximum-intensity projection over a 20 mm slab centred on the aorta, with the mask contour in one colour, reference ostia as circles, and predicted ostia as crosses. One PNG per case, 25 PNGs in a folder. This is the highest-value artefact you will produce. Build it in hour 2, not hour 6.
2. **Unrolled aortic wall map:** `max HU` over radial distance 2–8 mm, as an image of (arclength along centerline) × (angle around the centerline, 0–360°). Overlay reference ostia. Every branch should be an obvious bright blob; if a reference ostium is dark on this map, that branch is not findable by intensity and you should stop trying.
3. **Threshold sweep strip:** for one representative case, six thumbnails of the candidate set at six thresholds. Makes the leak behaviour visceral.
4. **Feature scatter:** for all candidates across all cases, `contrast` vs `length`, coloured by matched/unmatched. Tells you within 10 seconds whether your score can separate the classes at all.

### 10.3 What you are looking for

- Is the HU distribution across cases unimodal? If yes, a fixed fractional threshold generalizes. If bimodal (two contrast phases), you need the adaptive ladder unconditionally.
- Does M2's 5th percentile sit above 150 HU? If not, lower your floor and accept more false positives.
- Does M9's recall curve saturate before the candidate count explodes? The gap between those two knees is your entire working margin — if there is no gap, connectivity alone won't do it and you need the unrolled-map second pass.
- Are there reference branches that are simply invisible (M2 profile < 120 HU)? Count them; they are your recall ceiling. Do not spend time chasing a ceiling you cannot move.

---

## Part 11 — CPU optimization

### Where the time actually goes

| Stage | Estimated time | Notes |
|---|---|---|
| gzip decompression of `image.nii.gz` | **3–9 s** | Dominant fixed cost. Irreducible for `.nii.gz` (gzip is sequential; you cannot seek to an ROI). |
| Read mask | 0.3–1.0 s | Masks compress ~50:1 |
| Crop + stats | 0.2 s | |
| `distance_transform_edt` outside `M` (ROI) | 0.8–2.0 s | ~12 M voxels; scipy's EDT is O(N) and single-threaded |
| `distance_transform_edt` inside `M` | 0.3 s | Small domain |
| Φ smoothing + gradient | 0.4 s | `gaussian_filter` on ROI, float32 |
| Threshold + CC labelling × 4 levels | 4 × 0.35 s ≈ 1.4 s | Restrict to shell → ~2–4 M voxels |
| Contact patches × 4 | 4 × 0.15 s ≈ 0.6 s | |
| Geodesic tracing, ~30 candidates | 0.6–2.5 s | Cells are 1–10 k voxels each |
| Radius / direction / scoring | 0.2 s | |
| JSON + debug output | <0.05 s | |
| **Total** | **≈ 10–20 s** | Comfortably inside 60 s |

### Optimizations, ranked by value

1. **Crop to the ROI immediately after reading.** Non-negotiable; it is worth more than everything else combined. A 512×512×700 volume is 180 M voxels; the ROI is typically 8–15 M. 15× on every subsequent operation.
2. **Restrict to the shell before thresholding and labelling.** Another 3–5×. Use `np.where(shell)` index arrays if memory matters, though boolean masks over the ROI are only ~12 MB.
3. **float32 and int16 everywhere.** Never let a float64 full-volume array exist. `Dout.astype(np.float32)` right after the EDT. Peak RSS target: < 2 GB.
4. **Cache the EDT and Φ across ladder levels.** They do not depend on the threshold. Trivially done and saves 4–6 s if you got it wrong.
5. **Do not downsample.** Tempting, but you are localizing 1–3 mm structures with a 2–3 mm accuracy target; downsampling by 2 destroys exactly the signal you need. The runtime budget does not require it. Skip.
6. **Skip multiscale vesselness.** If you use vesselness at all, use one scale, evaluated at a few hundred points along traces, not over the volume. That turns a 20 s operation into a 50 ms one.
7. **Parallelize only if profiling says so.** The natural axis is per-candidate tracing (`concurrent.futures.ThreadPoolExecutor`; `MCP_Geometric` releases the GIL during `find_costs`) or per-ladder-level (processes, but that duplicates the ROI arrays — ~200 MB each, fine for 4 workers in 8 GB). **Only do this if you exceed 40 s.** Parallelism is a reproducibility risk (nondeterministic reduction order) for a 10 % scoring category you are already passing.
8. **Set `OMP_NUM_THREADS=4`** at the top of `run.py` before importing numpy, and pin it explicitly rather than letting it default — this also makes runtime reproducible.
9. **Prune early.** Drop components with < 20 voxels before any per-candidate work; drop patches below `A_min` before tracing. Typically cuts the candidate list from ~150 to ~30.
10. **Avoid `skimage.measure.regionprops` on large label images.** It is slow and allocates. Use `scipy.ndimage.labeled_comprehension` / `sum_labels` / `center_of_mass`, which are C loops.

---

## Part 12 — Implementation architecture

```text
project/
    run.py                 # CLI, orchestration, timing, error handling
    src/
        __init__.py
        io_utils.py        # SimpleITK read/write, grid validation, coordinate helpers
        preprocessing.py   # ROI crop, aortic HU statistics, threshold ladder
        aorta.py           # EDT, signed distance, normals, core/appendage split, centerline
        candidates.py      # shell, lumen mask, connectivity, contact patches, dedup/merge
        tracing.py         # geodesic Voronoi, MCP tracing, level sets, bifurcation detection
        geometry.py        # radius estimators, direction fit, ostium snapping, PCA
        scoring.py         # hard rules, feature extraction, linear score, threshold
        output.py          # JSON schema, deterministic ordering, debug dump
        visualization.py   # MIP montages, unrolled wall map
    tests/
        test_coords.py     # the Part 5 self-tests — write these FIRST
        test_geometry.py   # synthetic cylinder: known radius/direction round-trip
        test_tracing.py    # synthetic Y-junction: bifurcation detected at the right s
    eval/
        evaluate.py        # Part 14 metrics
        explore.py         # Part 10 measurements
    config.py              # every parameter, in one place, with units in the name
    README.md
    requirements.txt       # SimpleITK, numpy, scipy, scikit-image, matplotlib (dev only)
```

**Module contents.**

- `io_utils.py` — `read_pair(image_path, mask_path)` returning `(sitk_img, I_zyx, M_zyx, spacing_zyx)`; `np_to_itk_index`, `itk_phys(img, idx_zyx)`, `phys_batch(img, idx_array)` (vectorized: build `D·diag(S)` once and do a matmul — 1000× faster than looping `TransformContinuousIndexToPhysicalPoint`, and assert equality against ITK on 10 random points).
- `preprocessing.py` — `roi_bbox(M, margin_mm, spacing)`, `aortic_stats(I, M, spacing)`, `threshold_ladder(stats)`.
- `aorta.py` — `distance_fields(M, spacing)`, `surface_normals(phi, spacing, sigma_mm)`, `core_and_appendages(M, Din)`, `centerline(M, spacing)` with both the per-slice and geodesic implementations behind one signature.
- `candidates.py` — `lumen_mask(I, shell, T_lo, T_hi)`, `connected_daughters(lumen, M_core)`, `contact_patches(comp_labels, M_core)`, `merge_patches(...)`, `persistence(patch_sets_per_level)`.
- `tracing.py` — `geodesic_cells(comp, patches, spacing)`, `trace(cell, patch, spacing, step_mm, delta_mm)` → `TraceResult(centerline_mm, band_volumes, s_bifurcation, length_mm)`.
- `geometry.py` — `radius_at(trace, s)`, `direction_fit(points_mm)`, `snap_to_surface(pt, phi, normals)`.
- `scoring.py` — `Features` dataclass, `hard_rules(cand) -> list[str]` (returns the *names* of violated rules, for debugging), `score(features) -> float`.
- `output.py` — `write_json(candidates, path)` with deterministic sorting; `write_debug(candidates, path)`.

**Dependencies.** SimpleITK, NumPy, SciPy, scikit-image. That is enough. **Do not add VTK or PyVista** — you never build a mesh, and adding a 200 MB dependency with a fragile install for zero algorithmic gain is a reproducibility liability. OpenCV is unnecessary (everything is 3D). matplotlib only in `visualization.py`, imported lazily so `run.py` does not depend on it.

---

## Part 13 — Pseudocode

```python
# ============ run.py ============
def main(image_path, mask_path, output_path):
    t0 = time.time()
    img, I, M, sp = read_pair(image_path, mask_path)        # I,M in (z,y,x); sp in mm (z,y,x)
    validate_grids(img, mask_img)
    if M.sum() == 0: write_json([], output_path); return

    # ---- Stage 2: ROI + statistics ----
    bb   = roi_bbox(M, margin_mm=30.0, spacing=sp)
    I, M, off = crop(I, bb), crop(M, bb), bb.origin_zyx
    M    = keep_largest_component(M)
    Mero = binary_erosion(M, ball_mm(2.0, sp))
    mu   = np.median(I[Mero]); sd = 1.4826*median_abs_deviation(I[Mero])

    # ---- Stage 3: distance fields, normals, core, centerline ----
    Din  = edt(M,  sampling=sp).astype(np.float32)
    Dout = edt(~M, sampling=sp).astype(np.float32)
    Phi  = (Dout - Din)
    nrm  = normalize(np.stack(np.gradient(gaussian_filter(Phi, sigma_vox(1.0, sp)), *sp)))
    r_ao = per_slice_max(Din)                                # mm, smoothed over 10 mm
    Mcore = binary_propagation(Din > 0.60*r_ao_broadcast, mask=M)
    if (M.sum() - Mcore.sum()) > 0.15*M.sum(): Mcore = M     # guard
    cl, tang, ends = centerline(Mcore, sp)                   # per-slice centroids + spline

    # ---- Stage 4/5: ladder, connectivity, patches ----
    T_hi = max(700.0, mu + 3.5*sd + 150.0)
    levels = [max(150.0, f*mu) for f in (0.50, 0.60, 0.70, 0.80)]
    shell  = (Dout > 0) & (Dout <= 20.0)
    patch_sets = []
    for T in levels:
        lumen = shell & (I >= T) & (I <= T_hi)
        big   = label26(lumen | M)
        keep  = mode_label(big[M])                           # component containing the aorta
        branches = (big == keep) & ~M
        comp, ncomp = label26(branches)
        remove_small(comp, min_voxels=20)
        contact = comp_nonzero & has_neighbor26(Mcore)
        patches = label26(contact)                           # each = candidate ostium
        patch_sets.append(summarize(patches, comp, Phi, nrm))
    base = patch_sets[0]
    for p in base:
        p.persist = 1 + count(level k>0 has a patch within 2.0 mm of p.centroid
                              whose component extends >= 5 mm)

    # ---- Stage 6-10: per candidate ----
    cands = []
    for comp_id, group in group_by_component(base):
        cells = geodesic_voronoi(comp == comp_id, [p.voxels for p in group], sp)
        for p, cell in zip(group, cells):
            cell = cell & ~( (Dout < 0.5) & (geo_from_patch > 1.0) )   # anti-kissing
            g    = mcp_geodesic(cell, seeds=p.voxels, sampling=sp)
            tr   = trace_levelsets(g, cell, step=0.5, delta=0.4, max_s=10.0)
            #   tr.centerline_idx[s], tr.band_volume[s], tr.n_components[s]
            s_bif = first_persistent_split(tr, min_child_r=max(0.8, 0.45*tr.r0),
                                                persist_mm=1.5)
            L     = min(10.0, (s_bif - 0.5) if s_bif else 10.0, tr.max_s)
            if L < 5.0 and not s_bif: continue               # H2
            pts_mm = [phys(img, idx + off) for idx in tr.centerline_idx[s<=L]]

            s_seed = min(5.0, 0.85*s_bif) if s_bif else 5.0
            seed   = interp(pts_mm, s_seed)
            radius = median([r_area(tr, s_seed), r_edtmax(tr, s_seed),
                             r_area(tr, s_seed-1.0), r_area(tr, s_seed+1.0)])
            radius = clip(radius, 0.5, 0.9*local_r_ao)
            direction, rms = robust_line_fit([q for q,s in pts_mm if s <= min(6.0, L)])
            if rms > 0.5: direction = quadratic_tangent(pts_mm, s0=2.5)
            ost_idx = weighted_centroid(p.surface_voxels)
            ostium  = snap_to_zero_level(phys(img, ost_idx + off), Phi, nrm)

            f = extract_features(p, tr, I, mu, direction, ostium, radius, rms, nrm, tang)
            violations = hard_rules(f, r_min=R_MIN, cap_angle=35.0, persist_min=2)
            cands.append(Candidate(ostium, seed, radius, direction, f,
                                   score(f), violations))

    # ---- Stage 11/12: dedup, filter, emit ----
    cands = merge_duplicates(cands, d_centroid=3.0, d_trace3=2.0, ang=30.0)
    kept  = [c for c in cands if not c.violations and c.score >= TAU]
    kept  = global_dedup(kept, d=2.5, ang=25.0)
    kept.sort(key=lambda c: (-round(c.score, 6), c.ostium[2], c.ostium[1], c.ostium[0]))
    write_json([{ "instance_id": f"branch_{i+1:03d}",
                  "parent_instance_id": "aorta",
                  "ostium_xyz_mm":  list(map(float, c.ostium)),
                  "seed_xyz_mm":    list(map(float, c.seed)),
                  "radius_mm":      float(c.radius),
                  "direction_xyz":  list(map(float, c.direction)) }
                for i, c in enumerate(kept)], output_path)
    write_debug(cands, output_path.replace(".json", "_debug.json"))
    log(f"{len(kept)} branches, {time.time()-t0:.1f}s")


# ============ tracing.trace_levelsets ============
def trace_levelsets(g, cell, step, delta, max_s):
    out = TraceResult()
    for s in arange(step, min(max_s, g[cell].max()) + 1e-6, step):
        band = cell & (g >= s - delta) & (g <= s + delta)
        lab, n = label26(band)
        sizes  = component_volumes_mm3(lab, spacing)
        radii  = sqrt(sizes / (pi * 2*delta))          # equivalent-area radius
        keep   = [i for i in range(n) if radii[i] >= 0.4]
        out.n_components[s] = len(keep)
        out.child_radii[s]  = sorted(radii[keep], reverse=True)
        main = argmax(sizes)
        out.centerline_idx[s] = center_of_mass(lab == main)   # fractional (z,y,x)
        out.band_volume[s]    = sizes[main]
    return out

def first_persistent_split(tr, min_child_r, persist_mm):
    for s in sorted(tr.n_components):
        if tr.n_components[s] >= 2 and tr.child_radii[s][1] >= min_child_r:
            if all(tr.n_components.get(s + k*0.5, 0) >= 2
                   for k in range(1, int(persist_mm/0.5) + 1)):
                return s
    return None


# ============ geometry.robust_line_fit ============
def robust_line_fit(P):                    # P: list of 3-vectors in mm
    w = ones(len(P))
    for _ in range(3):
        pbar = (w[:,None]*P).sum(0)/w.sum()
        C    = ((w[:,None]*(P-pbar)).T @ (P-pbar)) / w.sum()
        u    = eigh(C)[1][:, -1]
        r    = norm((P-pbar) - ((P-pbar) @ u)[:,None]*u, axis=1)
        w    = 1.0/(1.0 + (r/0.6)**2)
    u *= sign(u @ (P[-1] - P[0]))
    return u/norm(u), sqrt((w*r**2).sum()/w.sum())
```

---

## Part 14 — Validation

### One-to-one matching

Use the **Hungarian algorithm**, not greedy. Greedy matching systematically inflates precision when two predictions cluster near one reference.

```python
from scipy.optimize import linear_sum_assignment
D = cdist(pred_ostia, ref_ostia)                  # (P, R) Euclidean mm
D_pad = where(D <= GATE_MM, D, 1e6)               # gate
ri, ci = linear_sum_assignment(D_pad)
matches = [(i, j) for i, j in zip(ri, ci) if D[i, j] <= GATE_MM]
TP = len(matches); FP = P - TP; FN = R - TP
```
Report at **two gates**: a strict 5 mm and a lenient 10 mm. Also report a radius-adaptive gate `max(5, 2·r_ref)` since a 1 mm branch and an 8 mm branch deserve different tolerances. State clearly which gate your headline number uses.

### Metrics

| Metric | Definition |
|---|---|
| Precision / Recall / F1 | per case, then report **mean per-case F1** *and* micro-averaged (pooled) F1. Per-case mean is the one that usually matches leaderboard scoring; report both so you are not surprised. |
| Ostium error | over matches: mean, **median**, p90, max of `‖pred − ref‖`. Median is the honest headline; max tells you about catastrophic failures. |
| Seed-on-vessel | fraction of matched seeds whose distance to the reference centerline < `r_ref`; if no reference centerline, use `‖seed − ref_seed‖ < r_ref` OR `HU(seed) ≥ 0.6·μ_ao` as a proxy |
| Direction error | `degrees(arccos(clip(u_pred·u_ref, -1, 1)))`; mean and median. Flag any > 60° separately — those are traces that ran backwards, a different bug class from imprecision. |
| Radius error | signed `r_pred − r_ref` (to expose systematic bias) *and* `|Δr|/r_ref` |
| Duplicate rate | (# predictions within `GATE` of an already-matched reference) / (# predictions) |
| Runtime | wall-clock per case; report mean and max, and the I/O share separately |
| Peak RAM | `resource.getrusage(RUSAGE_SELF).ru_maxrss` in the child process, or `tracemalloc` + a `psutil` sampler thread |

### Reporting

Emit `results.csv` (one row per case) and `matches.csv` (one row per matched/unmatched pair with all features). The second file is what you actually debug from: sort by ostium error descending and look at the top 10.

**Guard against self-deception:** whenever you tune `τ` or `r_min`, report leave-one-case-out F1, not in-sample F1. With 25 cases and 2 free parameters, in-sample F1 will overstate hidden-set performance by ~5–10 points. If your final number is a tuned in-sample number, you will be surprised on the leaderboard.

---

## Part 15 — Visual verification

Static PNGs, generated automatically, no UI. Build this early; it will find more bugs per minute than any other activity.

### The per-case figure (2×3 panels, one PNG)

1. **Axial MIP** over a 20 mm slab centred on the aorta, windowed to [0, 600] HU. Overlay: aorta mask contour (cyan, 1 px), predicted ostia (red `+`, radius-scaled circle), reference ostia if available (green `o`), branch ID text next to each.
2. **Coronal MIP** over the full aortic extent, same overlays. This is the most informative single panel — nearly all abdominal branches are visible.
3. **Sagittal MIP** — best view for celiac/SMA/IMA, which leave anteriorly.
4. **Direction arrows panel:** coronal MIP with a 10 mm arrow from each ostium along `direction_xyz`, projected into the coronal plane, coloured by score (viridis).
5. **Unrolled wall map:** (arclength × angle) of max HU in the 2–8 mm radial band, with predicted ostia as crosses and references as circles. Missed references show up here as bright blobs with no cross — instantly diagnostic.
6. **Candidate table rendered as text:** ID, score, radius, length, persistence, mean HU ratio, and *rejected* candidates in grey with the violated rule name. Seeing "rejected: H4_cap" next to a real branch tells you the bug in one glance.

### What makes it useful rather than decorative

- **Show the rejects, not just the accepts.** A visualization that only shows output cannot explain a recall failure.
- **Annotate with the numbers that drove the decision** (score, persistence, the violated rule). A picture without the feature values sends you back to the debugger.
- **Use MIP slabs, not single slices.** A single axial slice will miss most ostia and you will conclude your detector is broken when it is not.
- **Fixed windowing across cases** so you can flip through 25 PNGs and see contrast-phase differences immediately.
- **Deterministic filenames** `case_XX.png` so a `montage` call gives you a 25-case contact sheet.

Implementation: pure matplotlib, ~120 lines, ~1.5 s per case. Run it on **all 25** dev cases, not three — the marginal cost is zero once it works, and the outliers are where the information is.

---

## Part 16 — The ten most likely failure modes

**1. Contrast-phase variation breaks the threshold.**
*Why:* a portal-venous or delayed-phase case has `μ_ao ≈ 180 HU` and an IVC at 160 HU; the arterial/venous separation vanishes.
*Detect:* `μ_ao < 220` or bimodality in the shell histogram; a spike in candidate count and in mean component volume.
*Fix:* the adaptive ladder already helps; add a hard radius-growth veto and an axis-parallelism penalty. Optionally detect the phase from `μ_ao` and switch to a stricter parameter set.
*Worth it?* **Yes** — the ladder is core. The phase-switch is worth it only if dev cases show two phases.

**2. Vein leak through a shared wall (IVC, left renal vein crossing anterior to the aorta).**
*Why:* 1–2 voxels of partial volume bridge two bright lumens.
*Detect:* component volume in the shell > ~3000 mm³; radius growing past 5 mm; persistence = 1.
*Fix:* persistence rule (H8) + radius-growth veto.
*Worth it?* **Yes.** This is the #1 precision killer and the fix is 20 lines.

**3. The mask includes proximal branch stubs.**
*Why:* annotation practice varies.
*Detect:* measurement M5 — reference ostia sitting >1 mm inside the mask surface; or `|M_append|/|M|` consistently 3–10 %.
*Fix:* the core/appendage decomposition (§1.1).
*Worth it?* **Yes if M5 says so, no otherwise.** Check before building it — it is ~40 lines and pure waste if the masks are clean.

**4. Cropped end reported as a giant branch.**
*Why:* the aorta continues beyond the mask as bright contrast, producing a huge connected component with a large flat contact patch.
*Detect:* trivially — a "branch" with radius ≈ `r_ao` at a mask extremity.
*Fix:* H4 (wall-normal test) + H5 + H7 (radius cap).
*Worth it?* **Yes, mandatory.** Without it you add 2 guaranteed false positives per case.

**5. `r_min` set wrong.**
*Why:* the dataset's minimum-size rule is unknown and dominates both precision and recall.
*Detect:* asymmetry between FP and FN counts; sweep the parameter and look at the F1 curve's shape.
*Fix:* measurement M4, then sweep with LOOCV.
*Worth it?* **Yes — this is the single highest-leverage hour of your entire project.** More valuable than any algorithmic improvement.

**6. Bifurcation within 5 mm of the ostium.**
*Why:* short common trunks (celiac especially) divide early.
*Detect:* count of candidates where `s_bif < 5 mm`; seed radius implausibly small.
*Fix:* `s_seed = min(5, 0.85·s_bif)` (already in the design).
*Worth it?* **Yes, it is three lines.**

**7. Thick slices (≥2.5 mm) destroy small branches and direction accuracy.**
*Why:* a 1.5 mm branch running in-plane occupies one voxel of thickness; the geodesic centerline staircases.
*Detect:* M11; direction error correlating with slice thickness.
*Fix:* isotropic resampling of the ROI to 0.7 mm before tracing (adds ~2 s and ~400 MB, and uses `sitk.ResampleImageFilter` with linear interpolation). Only for anisotropic cases.
*Worth it?* **Only if M11 shows anisotropic cases.** Otherwise it is 2 s and a resampling-bug surface for nothing.

**8. Tortuous aorta breaks the per-slice centerline, corrupting the cap veto.**
*Why:* at a sharp bend, an axial slice cuts the aorta obliquely or twice.
*Detect:* centerline tangent turning > 40°/cm; centerline points falling outside the mask.
*Fix:* geodesic-diameter centerline.
*Worth it?* **Yes — it is ~30 lines using the MCP machinery you already have**, and it removes a whole class of silent failure.

**9. One branch producing two ostium instances (split by calcium or noise).**
*Why:* a plaque bisects the contact patch.
*Detect:* the duplicate-rate metric; pairs of predictions <3 mm apart with similar directions.
*Fix:* the merge rule of §1.7 (proximity **and** trace convergence).
*Worth it?* **Yes**, but keep the convergence condition — proximity alone will merge true adjacent ostia.

**10. Coordinate-system bug (axis order, missing direction matrix, ROI offset).**
*Why:* five places to get it wrong, and the failure is silent-but-total.
*Detect:* the Part 5 assertions; ostium errors clustered around 100–400 mm or mirrored; predicted points outside the image bounds.
*Fix:* write `tests/test_coords.py` first, use two helper functions everywhere, do all geometry in mm.
*Worth it?* **Yes. Do it in the first 20 minutes.** This is the failure mode most likely to cost you the entire competition, and it is entirely preventable.

*Runner-up (11): bowel/oral contrast and cancellous bone entering the lumen mask.* Mostly handled for free by connectivity, and by `T_hi` for cortical bone. Not worth dedicated code.

---

## Part 17 — Hackathon strategy

Scoring weights should drive everything: **discovery 45 % + ostium 25 % = 70 %**. Instance quality is 15 %, compute 10 %, reproducibility 5 %. Therefore:

- Optimize **detection F1 and ostium position** until they stop improving.
- Treat radius/direction as "get them reasonable, then stop." A 15 % category shared between seed, radius, and direction is worth ~5 % each. Do not spend an hour on FWHM radius refinement while your F1 is 0.6.
- Compute is **pass/fail in practice** — you will be at ~15 s against a 60 s budget. Spend zero additional time on it.
- Reproducibility is 5 % and nearly free: pin versions, fix ordering, set `OMP_NUM_THREADS`, no randomness. 15 minutes, full marks.

### Hour 1 — Baseline that produces valid JSON

1. `io_utils.py` + `tests/test_coords.py`. **Write and pass the coordinate tests before anything else.** (20 min)
2. ROI crop, `μ_ao`, single threshold `T = 0.6·μ_ao` floored at 150. (10 min)
3. `Dout`, shell ≤ 20 mm, lumen mask, `label26(lumen | M)`, keep the aorta component, subtract `M`, label, contact voxels with `M`, `label26` the contact → patches. (15 min)
4. For each patch: ostium = patch centroid (no snapping); seed = centroid of component voxels at `Dout ∈ [4.5, 5.5]` within that component; radius = `Din_lumen(seed)`; direction = normalized `seed − ostium`. (10 min)
5. Filters: patch area ≥ 1.5 mm², component extent ≥ 5 mm (use `Dout` max as a proxy for now), radius ≥ 1.0 mm. Write JSON. (5 min)

**Do NOT** in hour 1: geodesics, ladders, scoring, normals, visualization, ML. Using `Dout` (Euclidean distance from the aorta) instead of geodesic distance is a deliberate, acceptable approximation at this stage — it is wrong for curved branches and right enough for a baseline.

This is Approach A and it will produce something like F1 ≈ 0.5–0.6 with ~3 mm ostium error. **Ship it, commit it, and keep it runnable as a fallback.**

### Hours 2–4 — Strong prototype (this is where the score comes from)

1. **`evaluate.py` with Hungarian matching** — build it *before* the algorithm improvements, or you are tuning blind. (30 min)
2. **`explore.py` / Part 10 measurements M1, M2, M4, M5, M9.** (40 min) — these set your parameters. This feels like a detour and is not.
3. **Visualization montage** (panels 1, 2, 4, 6). (30 min)
4. **Cropped-end veto** (normals + centerline + H4/H5/H7). (25 min) — biggest single precision win.
5. **Geodesic tracing with MCP** replacing the `Dout` proxy: centerline, real length, bifurcation detection, seed placement. (45 min) — biggest single quality win.
6. **Threshold ladder + persistence.** (25 min) — biggest single vein-rejection win.
7. **Hard rules + linear score + τ sweep.** (25 min)

Expected: F1 ≈ 0.72–0.82, ostium median error ≈ 1.5–2.5 mm, runtime ≈ 15 s.

**Do NOT** in this phase: vesselness, isotropic resampling, cylinder fitting, parallelization, ML, or refactoring for elegance.

### Hours 5–6 — Optimized version

Pick from this list **in the order that your `matches.csv` says matters**, not in the order written:

1. `r_min` and `τ` sweep with LOOCV. (30 min) — almost always the top-value item.
2. Robust direction fit + quadratic fallback; validate against the plain chord and keep the winner. (25 min)
3. Radius: add the ray-cast FWHM estimator with a geodesic sanity clamp. (25 min)
4. Ostium snapping to the `Φ = 0` isosurface; measure and correct any systematic bias along `n`. (20 min)
5. Core/appendage split, **if and only if** M5 said the masks include stubs. (35 min)
6. Geodesic-diameter centerline, if any dev case is tortuous. (30 min)
7. Merge/dedup rules, tuned against the duplicate-rate metric. (20 min)
8. Logistic regression on the nine features with LOOCV, replacing the hand weights **only if it wins**. (30 min)

Expected: F1 ≈ 0.80–0.88.

### Final polish (45 min)

- Pin `requirements.txt` to exact versions; `OMP_NUM_THREADS=4`; no RNG anywhere (and if any, seed it).
- Deterministic output ordering; round floats to 4 decimals in JSON.
- Wrap the whole pipeline in try/except so a single bad case emits `{"branches": []}`-shaped valid JSON rather than a traceback and a zero.
- Add a wall-clock watchdog: if elapsed > 45 s, skip the remaining ladder levels and emit what you have.
- README: one-line install, one-line run, a parameter table with units, and a short "known limitations" section (graders notice honesty).
- Regenerate all 25 montages and flip through them once. You will find at least one bug.

### Explicit "do not waste time on" list

Frangi/Sato multiscale vesselness as a detector. Deep learning. VTK/PyVista and mesh generation. Level-set or active-contour segmentation. Full vascular graph construction. RANSAC cylinder fitting. Multiprocessing (you are at 15 s of a 60 s budget). Anatomical labelling or atlas registration. A GUI. Elegant abstractions and a plugin architecture. Chasing branches that measurement M2 shows are below 120 HU — they are not in the image.

---

## Final summary

### 1. Recommended algorithm in 13 steps

1. Read mask, compute ROI bbox, read image, crop; validate that the grids match.
2. Compute aortic HU statistics (`μ_ao`, `σ_ao`) inside the mask eroded by 2 mm; derive a threshold ladder `T_k = max(150, f_k·μ_ao)`, `f = [0.50, 0.60, 0.70, 0.80]`, and a calcium ceiling `T_hi`.
3. Compute `Din`, `Dout`, signed distance `Φ`, smoothed normals `n`; split the mask into `M_core` and thin appendages; extract the aortic centerline and tangents.
4. Build the shell `0 < Dout ≤ 20 mm`.
5. For each ladder level: threshold to a lumen mask, take the 26-connected component of `lumen ∪ M` containing the aorta, subtract `M`, and label the residue — these are the lumen-contiguous daughters.
6. Extract **contact patches** (connected sets of daughter voxels touching `M_core`). Each patch is one candidate instance. Compute cross-level persistence.
7. Partition each multi-patch component into geodesic Voronoi cells, one per patch.
8. Trace each candidate with `MCP_Geometric` from its patch: geodesic level sets every 0.5 mm give cross-sections, centroids give the centerline, a persistent split into ≥2 children gives the bifurcation arclength. Truncate at `min(10 mm, s_bif − 0.5)`.
9. Ostium = patch surface centroid snapped to the `Φ = 0` isosurface. Seed = centerline point at `min(5 mm, 0.85·s_bif)`.
10. Radius = median of equivalent-area and EDT-inradius estimates over three adjacent stations; clamp to `[0.5, 0.9·r_ao]`.
11. Direction = IRLS total-least-squares line fit over 0–6 mm in **physical** coordinates, with a quadratic-tangent fallback when the fit residual exceeds 0.5 mm; oriented outward.
12. Apply hard vetoes H1–H10 (connectivity, ≥5 mm extent, patch area, wall-normal cap test, boundary faces, `r_min`, `r_max`, persistence, HU floor, outwardness), then the soft score with threshold τ; merge duplicates by proximity **and** trace convergence.
13. Sort deterministically, convert all points with `TransformContinuousIndexToPhysicalPoint`, emit JSON plus a debug dump of rejected candidates and their violated rules.

### 2. Top 5 parameters to tune

| Rank | Parameter | Default | Why it dominates |
|---|---|---|---|
| 1 | `r_min` (minimum daughter radius) | 1.0 mm | Directly trades precision against recall on the 45 % category; the dataset's own eligibility rule is unknown |
| 2 | Threshold ladder fractions `f_k` | [0.50, 0.60, 0.70, 0.80] | Controls both small-branch recall and vein leakage; the single most case-variable quantity |
| 3 | Score threshold `τ` | 0.50 | The explicit operating point on the precision–recall curve |
| 4 | Cap-veto angle (`|n̄·t_ao|` limit) | 35° | Too tight → 2 guaranteed FPs/case; too loose → lost branches near the mask ends |
| 5 | Bifurcation child-radius rule | `max(0.8 mm, 0.45·r₀)` | Sets whether trunks are truncated too early (bad seeds/radii) or too late (granddaughter contamination) |

Tune 1–3 jointly by grid search with leave-one-case-out; tune 4–5 by inspecting the montages.

### 3. Top 5 failure modes

1. **Coordinate-system bug** (axis order / missing direction matrix / ROI offset) — total, silent, preventable in 20 minutes.
2. **Vein or bone leak through partial volume**, inflating false positives and corrupting traces.
3. **`r_min` mis-set**, costing 10–20 F1 points regardless of how good the rest is.
4. **Cropped aortic ends** reported as branches — 2 guaranteed FPs per case without the wall-normal veto.
5. **Contrast-phase variation on hidden cases**, collapsing the arterial/venous separation your thresholds assume.

### 4. Minimal viable implementation

~180 lines: read → crop → `μ_ao` → single threshold at `0.6·μ_ao` (floor 150) → 20 mm shell → `label26(lumen | M)` → keep the aorta's component → subtract `M` → label → contact patches with `M` → filter by patch area, `Dout`-extent ≥ 5 mm, and radius ≥ 1 mm → ostium = patch centroid, seed = centroid at `Dout ≈ 5 mm`, radius = EDT at seed, direction = normalized `seed − ostium` → JSON. Runs in ~12 s. Expect F1 ≈ 0.5–0.6, ostium error ≈ 3 mm. Build it in hour 1 and never delete it.

### 5. Best improvement if the baseline works

**The threshold ladder with persistence filtering, combined with geodesic tracing.** These two are the highest-yield additions: persistence is what separates real ostia from partial-volume bridges (the dominant false-positive mechanism), and geodesic tracing simultaneously fixes centerline accuracy, bifurcation handling, seed placement, radius, and direction — five deliverables from one primitive. Together they should move F1 from ~0.55 to ~0.80. After that, the next best marginal move is replacing hand-tuned score weights with LOOCV-validated logistic regression on the nine candidate features — but only after the geometry is right, and only if it actually wins in cross-validation.

### 6. Expected computational cost

| | |
|---|---|
| Total per case | **10–20 s** (target 60 s) |
| Largest component | `.nii.gz` decompression, 3–9 s — irreducible |
| Compute after I/O | 6–11 s, single-threaded |
| Peak RSS | 1.2–2.0 GB (limit 8 GB) |
| Parallelism needed | **None.** Set `OMP_NUM_THREADS=4` and leave it alone. |
| Headroom | ~3× — enough to afford isotropic resampling or an extra ladder level if measurements justify it |
