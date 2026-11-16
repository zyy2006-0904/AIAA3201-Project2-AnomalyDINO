# Integration Strategy

The integrated project keeps the original repository's complete submission
structure and completed Part 3. The teammate repository supplies stronger
components for the mandatory Part 1 and for localization evaluation.

| Component | Source retained | Reason |
| --- | --- | --- |
| Data split and calibration exclusion | Both; implemented with explicit manifest and tests | Prevents leakage |
| Multi-channel template map | Teammate | Consistently stronger than RGB-only baseline |
| Global closest-template selection | Our project | Literal compliance with the handout |
| Per-pixel template matching | Teammate, ablation only | Useful performance experiment, but changes the baseline definition |
| Gaussian + morphology | Teammate | Reduces isolated pixel responses |
| 99.5th-percentile image score | Teammate | Slightly stronger than top-1% mean in reported ablations |
| Full-frame 448 localization | Our stricter evaluation | Avoids unreported border exclusion |
| 448-pixel AnomalyDINO rescoring | Teammate concept, optimized implementation | More faithful than a 32x32 patch-only metric |
| AnomalyDINO + multi-layer Part 3 | Our project | Teammate repository has no completed Part 3 |
| Report, demo video, output bundle | Our project | Teammate repository lacks these mandatory artifacts |

## Runtime optimizations

- Replaced Python-level multichannel SSIM with a vectorized OpenCV Gaussian SSIM
  implementation.
- Removed optional pixel AP/F1 sorting from the hot path.
- Replaced floating-point pixel sorting with a 16-bit monotonic rank histogram.
- Rescored saved AnomalyDINO maps at 448 pixels without rerunning the backbone.

## Controlled validation

The integrated baseline has two explicit modes:

```bash
# Primary compliant method: one globally closest template
python scripts/run_template_baseline.py --template-mode global

# Teammate-inspired ablation: independently closest template at every pixel
python scripts/run_template_baseline.py \
  --template-mode per_pixel --border-crop 32
```

`scripts/run_part1_ablation.sh` runs the primary method, the per-pixel ablation,
and an RGB-only ablation with the same references.
