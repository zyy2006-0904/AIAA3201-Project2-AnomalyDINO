# AIAA 3201 Project 2: Few-Shot Industrial Visual Anomaly Detection

This repository reproduces and extends AnomalyDINO on the six VisA
categories required by the course project:

`candle`, `capsules`, `cashew`, `chewinggum`, `fryum`, and `pipe_fryum`.

Project repository:
`https://github.com/zyy2006-0904/AIAA3201-Project2-AnomalyDINO`.

The project contains:

1. The required normal-only template-difference baseline.
2. A faithful AnomalyDINO reproduction using DINOv2 ViT-S/14 at 448 pixels.
3. A controlled multi-layer DINOv2 feature-fusion improvement.
4. Image-level and patch-grid localization metrics over three seeds.
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

The baseline uses the same selected references as AnomalyDINO. It chooses the
closest template by mean absolute RGB difference, applies Gaussian smoothing,
and uses the 95th percentile of the 20 held-out calibration scores as the
OK/NG threshold. AUROC is computed from unthresholded scores.

## Metrics

- Image AUROC, AP, and F1.
- Patch-grid pixel AUROC, AP, and F1.
- AUPRO with the standard 0.30 false-positive-rate limit for AnomalyDINO.
- Mean and standard deviation across seeds 0, 1, and 2.
- Equal weighting of category means, as required by the project handout.
- Inference time and peak GPU memory.

Test labels are used only for final metric computation. They are not used for
reference selection, thresholds, model selection, or score rescaling.

## Verified Results

Mean image AUROC (%) over seeds 0, 1, and 2:

| Method | 1-shot | 2-shot | 4-shot | 8-shot |
|---|---:|---:|---:|---:|
| Template difference | 51.5 | 52.8 | 54.5 | 56.3 |
| AnomalyDINO | 94.6 | 95.3 | 96.2 | 96.6 |
| Multi-layer AnomalyDINO | **95.0** | **95.7** | **96.6** | **97.0** |

The improvement increases image AUROC by 0.38-0.42 points at every tested
shot count while leaving patch-level localization AUROC essentially unchanged.
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
