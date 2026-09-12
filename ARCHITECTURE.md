# Architecture

```text
CTA NIfTI + parent-aorta mask
              |
     validate grid + affine
              |
      crop to 25 mm ROI
              |
 per-case lumen calibration
              |
 bright wall-touching proposals
              |
  >=5 mm path + cap rejection
              |
 physical geometry + graph NMS
          /            \
 strict prediction   diagnostics sidecar
      JSON             + Aorta Map
```

## Detection layer

The baseline in `detector/branchseed.py` is deliberately classical. It can be inspected and timed on CPU. Organizer NIfTI volumes are read through SimpleITK, with physical points produced through the image transform API. Its public contract is a list of physical-space proximal branch measurements, independent of visualization.

The next accuracy slice should replace component PCA with a surface-normal proposal map plus a beam width of 3–5 over the first 10 mm. Keep the same output and diagnostics contracts so the team can improve detection without breaking the demo.

## Product layer

The static browser in `dist/` is dependency-free and offline. It opens on the working surface, not a landing page. The unwrapped map is the primary navigation model; the schematic 3D panel gives context; linked CT evidence, radius plane, confidence components and topology notes explain why each branch survived.

## Safety boundary

The project discovers and measures geometry. It does not identify anatomy, predict invisible vessels, diagnose pathology or recommend care. Every demo surface states that it is a synthetic hackathon prototype.
