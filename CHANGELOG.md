# Integration Changelog

## Multi-channel Part 1 baseline

- Integrated RGB, grayscale, gradient, and SSIM difference channels from the
  teammate implementation.
- Kept `global` closest-template selection as the compliant default.
- Added `per_pixel` multi-template matching as an explicit ablation instead of
  silently changing the task definition.
- Standardized score aggregation to the 99.5th percentile.
- Retained Gaussian smoothing and 3x3 morphological cleanup.
- Added full-frame evaluation by default; 32-pixel border cropping is optional.

## Faster and more consistent evaluation

- Added a vectorized OpenCV Gaussian SSIM implementation.
- Added a 448-pixel in-memory rescorer for saved AnomalyDINO patch maps.
- Added a 16-bit rank-histogram pixel AUROC with sub-0.001 error on controlled
  arrays.
- Removed expensive optional pixel-AP/F1 sorting from the experiment hot path.
- Aggregate output now reads 448-pixel rescored metrics when available.

## Tests and reproducibility

- Added project-level tests for global template selection, metric correctness,
  quantized pixel AUROC agreement, and sample-standard-deviation behavior.
- Preserved exact reference filenames and calibration exclusions.
- Added a reproducible Part 1 ablation script.

## Verified score changes

- Part 1 image AUROC improved from 51.46/52.79/54.52/56.27 to
  56.37/57.59/58.22/60.71 at 1/2/4/8 shots.
- AnomalyDINO image AUROC is unchanged at the protocol level:
  94.61/95.33/96.15/96.62.
- At 448-pixel localization, AnomalyDINO reaches
  98.08/98.21/98.36/98.46, matching the teammate's reported scale.
- Multi-layer fusion improves image AUROC to
  94.99/95.75/96.56/97.04 with a small localization trade-off.
