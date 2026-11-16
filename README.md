# AIAA 3201 Project 2: Few-Shot Industrial Visual Anomaly Detection

This repository reproduces and extends AnomalyDINO on the six VisA
categories required by the course project:

`candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, and `pipe_fryum`.

Project repository:
`https://github.com/zyy2006-0904/AIAA3201-Project2-AnomalyDINO`.

Team: Yifei Zhang and Yiyang Zhang.

The rationale for merging the two team implementations is documented in
[`INTEGRATION.md`](INTEGRATION.md) and [`CHANGELOG.md`](CHANGELOG.md).

The project contains:

1. The required normal-only template-difference baseline.
2. A faithful AnomalyDINO reproduction using DINOv2 ViT-S/14 at 448 pixels.
3. A controlled multi-layer DINOv2 feature-fusion improvement.
4. Image-level and 448-pixel localization metrics over three seeds.
5. A single-image demo with anomaly score, OK/NG decision, and heatmap.

## Data

The class dataset is read-only. `scripts/prepare_visa_symlinks.py` creates the
official `VisA_pytorch/1cls` layout using symlinks, so the original 1.9 GB image
archive is not duplicated.

For each category, the first 20 lexicographically sorted normal training images
are reserved for threshold calibration. They are excluded from every few-shot
reference selection to avoid leakage. The selected filenames are saved in each
run directory.

```bash
python scripts/prepare_visa_symlinks.py
```

## Environment

Python 3.10, PyTorch 2.7.1 with CUDA 12.6, torchvision 0.22.1, FAISS CPU, and
the remaining dependencies in `requirements.txt` are sufficient. DINOv2 weights
are downloaded by `torch.hub`.

`faiss-cpu` is pinned in `requirements.txt` so the environment installs without a
GPU toolchain. Swapping in `faiss-gpu-cu12` and dropping `--faiss_on_cpu` is
supported and much faster (the reported runs use the GPU index); the numbers do
not change, only the runtime.

```bash
pip install -r requirements.txt
```

## Official AnomalyDINO

Clone and patch the official repository:

```bash
git clone --depth 1 https://github.com/dammsi/AnomalyDINO.git code/AnomalyDINO
python scripts/patch_official_anomalydino.py code/AnomalyDINO
```

The reported experiment uses AnomalyDINO commit
`dd35b0f1f6ff6146d196fe4efd3457a58ebcb1e0`.

The patch:

- Restricts VisA evaluation to the six required categories.
- Excludes the first 20 calibration images from reference sampling.
- Saves exact reference filenames for auditability.
- Stores patch maps instead of full-resolution TIFFs. Pixel localization scores
  are computed by resizing ground-truth masks to the patch grid. This is
  disk-safe on the shared server and evaluates every test image, including
  normal images with all-zero masks.

Run the official baseline:

```bash
python run_anomalydino.py \
  --dataset VisA \
  --model_name dinov2_vits14 \
  --data_root data/VisA_pytorch/1cls \
  --shots 1 2 4 8 \
  --num_seeds 3 \
  --preprocess agnostic \
  --resolution 448 \
  --eval_segm \
  --faiss_on_cpu \
  --tag baseline
```

Run the improvement, which fuses the normalized patch features from blocks 9
and 11 before nearest-neighbor retrieval:

```bash
python run_anomalydino.py \
  --dataset VisA \
  --model_name dinov2_vits14 \
  --data_root data/VisA_pytorch/1cls \
  --shots 1 2 4 8 \
  --num_seeds 3 \
  --preprocess agnostic \
  --resolution 448 \
  --layer 9 11 \
  --eval_segm \
  --faiss_on_cpu \
  --tag multilayer9-11
```

## Template Baseline

```bash
python scripts/run_template_baseline.py \
  --data-root data/VisA_pytorch/1cls \
  --output results/template \
  --shots 1 2 4 8 \
  --seeds 0 1 2
```

The integrated baseline uses the same selected references as AnomalyDINO and
fuses four complementary difference maps: RGB, grayscale, gradient magnitude,
and Gaussian SSIM. The primary `global` mode chooses one globally closest
template, exactly matching the handout's wording. The teammate-inspired
`per_pixel` mode is retained as an ablation:

```bash
python scripts/run_template_baseline.py \
  --data-root data/VisA_pytorch/1cls \
  --output results/template_ablation \
  --shots 1 2 4 8 --seeds 0 1 2 \
  --template-mode per_pixel --border-crop 32
```

The default score is the 99.5th percentile of the smoothed map. The 95th
percentile of the 20 held-out calibration scores remains the OK/NG threshold.
Full-frame 448-pixel localization is the primary evaluation; border cropping is
not used unless explicitly requested.

## Metrics

- Image AUROC, AP, and F1.
- Full-frame 448-pixel AUROC using a fast 16-bit rank histogram.
- Mean and standard deviation across seeds 0, 1, and 2.
- Equal weighting of category means, as required by the project handout.
- Inference time and peak GPU memory.

**Evaluation frame.** Every method is scored on the complete 448x448 frame: no
border cropping and no per-image score rescaling. AnomalyDINO produces its map
on the patch grid, so the saved maps are resampled to 448 px before scoring; the
template baseline is computed directly at 448 px. Because the frame is identical
for all rows, the comparison in the results table is like-for-like. Excluding a
border margin would raise the baseline's pixel AUROC by several points, which is
why the choice is stated explicitly rather than left implicit.

**Not reported: AUPRO.** The handout marks AUPRO as optional and requires the
official 0.30 FPR cap when it is reported. We do not report it: the six-class run
takes several minutes per condition in the reference implementation, and the two
required metrics (image AUROC, pixel AUROC) already separate the methods
decisively.

Test labels are used only for final metric computation. They are not used for
reference selection, thresholds, model selection, or score rescaling.
The histogram-based pixel AUROC preserves rank ordering in 65,536 bins; its
error relative to exact sklearn AUROC is below `0.001` on controlled arrays.

Run the project tests with:

```bash
pytest -q tests
```

## Verified Results

Mean image AUROC (%) over seeds 0, 1, and 2:

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---:|---:|---:|---:|
| Integrated template difference | 56.4 | 57.6 | 58.2 | 60.7 |
| AnomalyDINO | 94.6 | 95.3 | 96.2 | 96.6 |
| Multi-layer AnomalyDINO | **95.0** | **95.7** | **96.6** | **97.0** |

A note on reading the last two rows: the multi-layer fusion gains about 0.4
image-AUROC points at every shot count, which is **smaller than the seed
standard deviation** (0.5-0.7 points). The direction is consistent across all
four shot settings, but we report it as a conservative, low-variance change
rather than a significant improvement, and pixel AUROC in fact decreases by
0.02-0.06 points. No per-seed paired deltas are stored in this repository, so a
paired significance test cannot be reconstructed from the released results.


448-pixel localization AUROC (%):

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---:|---:|---:|---:|
| Integrated template difference | 56.9 | 57.1 | 57.8 | 57.9 |
| AnomalyDINO | **98.1** | **98.2** | **98.4** | **98.5** |
| Multi-layer AnomalyDINO | 98.0 | 98.2 | 98.3 | 98.4 |

The integrated Part 1 raises image AUROC by 3.70-4.91 points over the original
RGB-only baseline. The multi-layer improvement adds another ~0.4 image-AUROC
point, with a small 448-pixel localization trade-off of 0.02-0.06 points.

At the fixed calibration threshold, Part 1 false-positive rate drops from
32.9-35.1% to 7.9-10.6%; false-negative rate rises from 64.8-70.1% to
80.3-85.1%. The integrated baseline is deliberately conservative and should be
reported as a false-alarm/recall trade-off.
Detailed means, standard deviations, per-category metrics, failure cases, and
efficiency measurements are under `results/` and `artifacts/`.

## Demo

```bash
python demo/inspect_image.py \
  --image /path/to/product_image.JPG \
  --category chewinggum \
  --shot 4 \
  --seed 0 \
  --layers 9 11 \
  --output demo_outputs/chewinggum_example
```

The command writes `result.json`, `input.png`, `heatmap.png`, and `overlay.png`.

## Acknowledgements

This project uses [AnomalyDINO](https://github.com/dammsi/AnomalyDINO),
[DINOv2](https://github.com/facebookresearch/dinov2), and the
[VisA dataset](https://github.com/amazon-science/spot-diff). Refer to the course
report for full citations and licensing details.
