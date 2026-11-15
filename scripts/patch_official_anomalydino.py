#!/usr/bin/env python3
"""Apply the small course-protocol changes to a clean AnomalyDINO checkout."""

from __future__ import annotations

import argparse
from pathlib import Path


SIX_CATEGORIES = (
    '["candle", "capsules", "cashew", "chewinggum", "fryum", '
    '"pipe_fryum"]'
)
TWELVE_CATEGORIES = (
    '["candle", "capsules", "cashew", "chewinggum", "fryum", '
    '"macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", '
    '"pipe_fryum"]'
)


OLD_SELECTION = '''    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    if n_ref_samples == -1:
        # full-shot setting
        img_ref_samples = list_images(img_ref_folder)
    else:
        # few-shot setting, pick samples in deterministic fashion according to seed
        img_ref_samples = list_images(img_ref_folder)[seed*n_ref_samples:(seed + 1)*n_ref_samples]
'''

NEW_SELECTION = '''    img_ref_folder = f"{data_root}/{object_name}/train/good/"
    all_ref_images = list_images(img_ref_folder)
    if len(all_ref_images) <= 20:
        raise ValueError(
            f"Not enough training images in {img_ref_folder}; "
            "20 are required for demo calibration."
        )
    calibration_refs = all_ref_images[:20]
    eligible_ref_images = all_ref_images[20:]
    if n_ref_samples == -1:
        # full-shot setting
        img_ref_samples = eligible_ref_images
    else:
        # Few-shot reference selection after excluding the required first 20
        # calibration images.
        img_ref_samples = eligible_ref_images[
            seed * n_ref_samples:(seed + 1) * n_ref_samples
        ]

    reference_dir = f"{plots_dir}/references"
    os.makedirs(reference_dir, exist_ok=True)
    with open(
        f"{reference_dir}/{object_name}_shot={n_ref_samples}_seed={seed}.json",
        "w",
        encoding="utf-8",
    ) as reference_file:
        json.dump(
            {
                "object": object_name,
                "shot": n_ref_samples,
                "seed": seed,
                "calibration_images": calibration_refs,
                "reference_images": img_ref_samples,
            },
            reference_file,
            indent=2,
        )
'''

OLD_TIFF_SAVE = '''                if save_tiffs:
                    anomaly_map = dists2map(d_masked, image_test.shape)
                    tiff.imwrite(f"{plots_dir}/anomaly_maps/seed={seed}/{object_name}/test/{type_anomaly}/{img_test_nr}.tiff", anomaly_map)
'''

NEW_TIFF_SAVE = '''                # Patch-grid maps are retained for all test images. Avoid
                # materializing full-resolution TIFFs, which is unnecessary
                # for patch-level localization evaluation and can exhaust the
                # shared server disk.
'''

OLD_TIFF_DELETE = '''    # optionally delete all tiff files in the anomaly_maps_dir to save disk space
    if delete_tiff_files:
        for pred_name in prediction_filenames:
            os.remove(pred_name + '.tiff')
'''

NEW_TIFF_DELETE = '''    # Patch-grid .npy maps are intentionally retained. They are needed by
    # image-level evaluation immediately after localization scoring.
'''

OLD_HUB_LOAD = (
    "        model = torch.hub.load('facebookresearch/dinov2', self.model_name)\n"
)

NEW_HUB_LOAD = '''        hub_source = os.environ.get(
            "DINOV2_HUB_DIR", "facebookresearch/dinov2"
        )
        model = torch.hub.load(
            hub_source,
            self.model_name,
            source="local" if os.path.isdir(hub_source) else "github",
        )
'''


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text:
        return
    if text.count(old) != 1:
        raise RuntimeError(f"Expected one patch target in {path}, found {text.count(old)}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path)
    args = parser.parse_args()

    utils_path = args.repo / "src/utils.py"
    detection_path = args.repo / "src/detection.py"
    post_eval_path = args.repo / "src/post_eval.py"
    backbones_path = args.repo / "src/backbones.py"
    replace_once(utils_path, TWELVE_CATEGORIES, SIX_CATEGORIES)
    replace_once(
        detection_path,
        "import matplotlib.pyplot as plt\n",
        "import json\nimport matplotlib.pyplot as plt\n",
    )
    replace_once(detection_path, OLD_SELECTION, NEW_SELECTION)
    replace_once(detection_path, OLD_TIFF_SAVE, NEW_TIFF_SAVE)
    replace_once(
        post_eval_path,
        "        prediction = read_tiff(pred_name)\n",
        '        prediction = np.load(pred_name + ".npy")\n',
    )
    replace_once(
        post_eval_path,
        (
            "        if gt_name is not None:\n"
            "            ground_truth.append(np.asarray(Image.open(gt_name)))\n"
            "        else:\n"
            "            ground_truth.append(np.zeros(prediction.shape))\n"
        ),
        (
            "        if gt_name is not None:\n"
            "            mask_image = Image.open(gt_name).convert(\"L\")\n"
            "            mask_image = mask_image.resize(\n"
            "                (prediction.shape[1], prediction.shape[0]),\n"
            "                resample=Image.Resampling.NEAREST,\n"
            "            )\n"
            "            ground_truth.append(\n"
            "                (np.asarray(mask_image) > 0).astype(np.uint8)\n"
            "            )\n"
            "        else:\n"
            "            ground_truth.append(\n"
            "                np.zeros(prediction.shape, dtype=np.uint8)\n"
            "            )\n"
        ),
    )
    replace_once(post_eval_path, OLD_TIFF_DELETE, NEW_TIFF_DELETE)
    replace_once(
        backbones_path,
        "import cv2\n",
        "import cv2\nimport os\n",
    )
    replace_once(backbones_path, OLD_HUB_LOAD, NEW_HUB_LOAD)
    print(
        "Patched AnomalyDINO for six categories, held-out calibration images, "
        "and disk-safe patch-grid localization scoring."
    )


if __name__ == "__main__":
    main()
