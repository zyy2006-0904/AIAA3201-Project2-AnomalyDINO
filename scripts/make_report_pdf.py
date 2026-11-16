#!/usr/bin/env python3
"""Build the course report PDF from the verified experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)


METHOD_ORDER = ("template", "anomalydino", "multilayer")
METHOD_NAMES = {
    "template": "Template difference",
    "anomalydino": "AnomalyDINO",
    "multilayer": "Multi-layer AnomalyDINO",
}
SHOTS = (1, 2, 4, 8)
CATEGORIES = (
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "pipe_fryum",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def read_summary(project_root: Path) -> list[dict[str, str]]:
    with (
        project_root / "results/aggregate/summary_metrics.csv"
    ).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def read_categories(project_root: Path) -> list[dict[str, str]]:
    with (
        project_root / "results/aggregate/summary_by_category.csv"
    ).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def lookup(
    rows: list[dict[str, str]],
    method: str,
    shot: int,
    metric: str,
) -> tuple[float, float]:
    for row in rows:
        if (
            row["method"] == method
            and int(row["shot"]) == shot
            and row["metric"] == metric
        ):
            return float(row["mean"]), float(row["std"])
    raise KeyError((method, shot, metric))


def scaled_image(path: Path, width: float) -> Image:
    with PILImage.open(path) as image:
        ratio = image.height / image.width
    return Image(str(path), width=width, height=width * ratio)


def fit_table(
    data: list[list[object]],
    widths: list[float],
    font_size: float = 7.3,
    header_rows: int = 1,
    spans: list[tuple[str, tuple[int, int], tuple[int, int]]] | None = None,
) -> Table:
    table = Table(data, colWidths=widths, repeatRows=header_rows)
    commands = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 1.5),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
    ]
    if spans:
        commands.extend(spans)
    table.setStyle(TableStyle(commands))
    return table


def main() -> None:
    args = parse_args()
    project = args.project_root
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    summary = read_summary(project)
    categories = read_categories(project)

    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Times-Roman",
        fontSize=10.0,
        leading=12.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6.0,
    )
    title = ParagraphStyle(
        "Title",
        parent=styles["Title"],
        fontName="Times-Bold",
        fontSize=22,
        leading=25,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Times-Roman",
        fontSize=11,
        leading=13,
        alignment=TA_CENTER,
        spaceAfter=10,
    )
    section = ParagraphStyle(
        "Section",
        parent=body,
        fontName="Times-Bold",
        fontSize=13,
        leading=15,
        alignment=TA_LEFT,
        spaceBefore=8,
        spaceAfter=5,
    )
    subsection = ParagraphStyle(
        "Subsection",
        parent=body,
        fontName="Times-Bold",
        fontSize=10.6,
        leading=12.5,
        spaceBefore=5,
        spaceAfter=3,
    )
    caption = ParagraphStyle(
        "Caption",
        parent=body,
        fontName="Times-Roman",
        fontSize=7.7,
        leading=9,
        alignment=TA_CENTER,
        spaceAfter=7,
    )
    abstract = ParagraphStyle(
        "Abstract",
        parent=body,
        fontSize=9.2,
        leading=11.2,
        leftIndent=24,
        rightIndent=24,
        spaceAfter=10,
    )
    reference = ParagraphStyle(
        "Reference",
        parent=body,
        fontSize=7.2,
        leading=8.8,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2.4,
    )

    page_width, page_height = letter
    left = right = 0.55 * inch
    top = 0.55 * inch
    bottom = 0.55 * inch
    gap = 0.28 * inch
    column_width = (page_width - left - right - gap) / 2

    def page_header_footer(canvas, document):
        canvas.saveState()
        canvas.setFont("Times-Roman", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        canvas.drawString(
            left,
            0.28 * inch,
            "AIAA 3201 Term Project - Few-Shot Industrial Visual Anomaly Detection",
        )
        canvas.drawRightString(
            page_width - right,
            0.28 * inch,
            f"{canvas.getPageNumber()}",
        )
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(output),
        pagesize=letter,
        leftMargin=left,
        rightMargin=right,
        topMargin=top,
        bottomMargin=bottom,
        title="Few-Shot Industrial Visual Anomaly Detection with DINOv2 Patch Retrieval",
        author="Yifei Zhang and Yiyang Zhang",
    )
    first_frame = Frame(
        left,
        bottom,
        page_width - left - right,
        page_height - top - bottom,
        id="first",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    left_frame = Frame(
        left,
        bottom,
        column_width,
        page_height - top - bottom,
        id="left",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    right_frame = Frame(
        left + column_width + gap,
        bottom,
        column_width,
        page_height - top - bottom,
        id="right",
        leftPadding=0,
        rightPadding=0,
        topPadding=0,
        bottomPadding=0,
    )
    doc.addPageTemplates(
        [
            PageTemplate(
                id="title",
                frames=[first_frame],
                onPage=page_header_footer,
            ),
            PageTemplate(
                id="twocol",
                frames=[left_frame, right_frame],
                onPage=page_header_footer,
            ),
        ]
    )

    story = []
    story.append(Spacer(1, 0.35 * inch))
    story.append(
        Paragraph(
            "AIAA 3201: Introduction to Computer Vision &nbsp; | &nbsp; Spring 2027",
            subtitle,
        )
    )
    story.append(
        Paragraph(
            "Few-Shot Industrial Visual Anomaly Detection<br/>"
            "with DINOv2 Patch Retrieval",
            title,
        )
    )
    story.append(Paragraph("Yifei Zhang &nbsp;&nbsp; | &nbsp;&nbsp; Yiyang Zhang", subtitle))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph("<b>Abstract</b>", subsection))
    story.append(
        Paragraph(
            "Industrial inspection must recognize rare defects when only a handful "
            "of defect-free examples are available. This project studies six VisA "
            "categories under the official 1-class protocol. We implement a "
            "multi-channel template-difference baseline, reproduce AnomalyDINO with frozen "
            "DINOv2 ViT-S/14 features at 448 x 448 resolution, and introduce a "
            "controlled improvement that fuses normalized patch features from "
            "transformer blocks 9 and 11. All methods use identical references, "
            "calibration exclusions, preprocessing, and seeds.",
            abstract,
        )
    )
    story.append(
        Paragraph(
            "Across 1/2/4/8-shot settings and seeds 0/1/2, the integrated template "
            "baseline reaches 56.4-60.7% image AUROC, whereas AnomalyDINO reaches "
            "94.6-96.6%. Multi-layer fusion improves image AUROC by about 0.4 points "
            "with a small localization trade-off. The report also "
            "documents difficult examples, reference-selection leakage controls, "
            "runtime, and GPU memory. Code, results, and a runnable demo are included "
            "in the public repository.",
            abstract,
        )
    )
    story.append(
        Paragraph(
            'Public repository: <link href="https://github.com/zyy2006-0904/AIAA3201-Project2-AnomalyDINO" color="#145ca8">github.com/zyy2006-0904/AIAA3201-Project2-AnomalyDINO</link>.',
            caption,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    pipeline_data = [
        [
            Paragraph("<b>1-8 references</b>", body),
            Paragraph("<b>DINOv2</b><br/>patch encoder", body),
            Paragraph("<b>Normal memory</b><br/>1-NN retrieval", body),
            Paragraph("<b>Test image</b><br/>+ PCA mask", body),
            Paragraph("<b>Heatmap + score</b><br/>OK / NG", body),
        ]
    ]
    pipeline = Table(
        pipeline_data,
        colWidths=[
            (page_width - left - right) / 5
            for _ in range(5)
        ],
    )
    pipeline.setStyle(
        TableStyle(
            [
                ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#aaaaaa")),
                ("INNERGRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f3f6fa")),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(pipeline)
    story.append(
        Paragraph(
            "<b>Figure 1.</b> End-to-end inspection pipeline. The improvement fuses "
            "patch tokens from blocks 9 and 11 before retrieval.",
            caption,
        )
    )
    story.append(
        Paragraph(
            "<b>Contributions.</b> (1) Required template-difference baseline; "
            "(2) faithful AnomalyDINO reproduction; (3) transparent multi-layer "
            "fusion; (4) auditable calibration/reference protocol; (5) complete "
            "quantitative, qualitative, efficiency, report, and demo artifacts.",
            body,
        )
    )

    story.append(NextPageTemplate("twocol"))
    story.append(PageBreak())
    story.append(Paragraph("1. Introduction", section))
    story.append(
        Paragraph(
            "Visual anomaly detection is a core component of automated industrial "
            "inspection. A detector must answer whether a product image is abnormal "
            "and where the defect is located. The anomalous class is absent from "
            "training and its visual appearance varies by defect. The problem is "
            "therefore naturally normal-only: model defect-free appearance and flag "
            "samples that deviate from that distribution.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Reconstruction and synthesis methods learn to reproduce normal input or "
            "generate pseudo-defects [4, 5]. Patch-distribution methods estimate "
            "normal statistics and compare test patches with them [2, 3]. Foundation "
            "models reuse general visual representations learned from much larger "
            "datasets. That family is attractive in few-shot inspection because no "
            "task-specific training is required and the encoder transfers across "
            "products.",
            body,
        )
    )
    story.append(
        Paragraph(
            "AnomalyDINO pairs frozen DINOv2 patch tokens with nearest-neighbor "
            "retrieval [6, 8]. It naturally outputs both an anomaly score and a "
            "localization map. Its remaining weakness is the relatively coarse last "
            "layer: semantic anomaly evidence is strong, while very thin scratches "
            "or subtle missing components can be blurred. This motivates a "
            "training-free multi-layer improvement.",
            body,
        )
    )
    story.append(Paragraph("2. Related Work", section))
    story.append(
        Paragraph(
            "<b>Normal-only detection.</b> MVTec AD established the standard setting "
            "with normal training images, anomalous test images, and pixel masks [1]. "
            "PaDiM models patch features with Gaussian distributions and Mahalanobis "
            "distance [2]. PatchCore uses a coreset of nominal patches for high-recall "
            "retrieval [3]. DRAEM learns discriminative reconstruction from synthetic "
            "defects [4], while CutPaste creates self-supervised annotations through "
            "cut-and-paste [5].",
            body,
        )
    )
    story.append(
        Paragraph(
            "<b>Foundation models.</b> DINOv2 provides transferable patch-level "
            "representations without supervisory labels [6]. AnomalyDINO applies "
            "these features to few-shot anomaly detection, combined with PCA masking "
            "and rotation augmentation [8]. AnomalyCLIP instead aligns local regions "
            "with object-agnostic normal/abnormal prompts [7]. EfficientAD targets "
            "millisecond inspection with compact student-teacher distillation [9], "
            "and MVTec AD 2 extends evaluation to harder scenarios [10]. VisA, the "
            "dataset used here, contains twelve annotated industrial categories [11].",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("3. Method", section))
    story.append(Paragraph("3.1 Template-difference baseline", subsection))
    story.append(
        Paragraph(
            "All images are resized to 448 x 448. For test image I and reference "
            "R<sub>j</sub>, we build RGB, grayscale, gradient, and 1-SSIM difference "
            "maps and take their pixel-wise maximum. The compliant global mode "
            "chooses the one reference with minimum mean combined difference. A "
            "per-pixel multi-template mode is retained only as an ablation. The "
            "selected map is Gaussian-smoothed, morphologically cleaned, and "
            "summarized by its 99.5th percentile.",
            body,
        )
    )
    story.append(Paragraph("3.2 AnomalyDINO reproduction", subsection))
    story.append(
        Paragraph(
            "DINOv2 maps each image to patch tokens z<sub>i</sub> in R<super>384</super>. "
            "Each normal reference is augmented with eight rotations. Tokens are "
            "L2-normalized and stored in memory M. A PCA foreground mask suppresses "
            "background patches. For query patch x<sub>q</sub>, 1-NN cosine distance "
            "is 1 - cos(x<sub>q</sub>, m*). The resulting patch distances form the "
            "anomaly map, and the mean top-1% distance is the image score.",
            body,
        )
    )
    story.append(Paragraph("3.3 Multi-layer improvement", subsection))
    story.append(
        Paragraph(
            "The improvement averages the normalized patch features of transformer "
            "blocks 9 and 11. The final-layer branch preserves object semantics, "
            "while the earlier branch contributes texture and boundary detail. The "
            "fused vector is normalized and enters exactly the same memory bank, "
            "retrieval, mask, and scoring code. No parameters are trained. This is "
            "the only change in Part 3, which isolates its effect.",
            body,
        )
    )
    story.append(Paragraph("3.4 Leakage controls", subsection))
    story.append(
        Paragraph(
            "For each category, the first 20 lexicographically sorted normal training "
            "images are reserved for threshold calibration and excluded from all "
            "few-shot pools. Seed r selects contiguous files after that prefix. Exact "
            "filenames are saved for every run. The demo threshold is the 95th "
            "percentile of calibration scores. AUROC uses unthresholded scores and no "
            "per-image score rescaling.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("4. Experiments", section))
    story.append(
        Paragraph(
            "We use the six required VisA categories under the official 1-class split. "
            "There are 3,245 normal training images, 360 normal test images, and 600 "
            "anomalous test images. Reference counts are 1/2/4/8 with seeds 0/1/2. "
            "The frozen DINOv2 ViT-S/14 backbone runs at 448 pixels; all methods share "
            "the same selected references and preprocessing.",
            body,
        )
    )
    dataset_rows = [
        ["Category", "Train normal", "Test normal", "Test anomaly"],
        ["candle", "900", "100", "100"],
        ["capsules", "542", "60", "100"],
        ["cashew", "450", "50", "100"],
        ["chewinggum", "453", "50", "100"],
        ["fryum", "450", "50", "100"],
        ["pipe_fryum", "450", "50", "100"],
        ["Total", "3,245", "360", "600"],
    ]
    dataset_table = fit_table(
        dataset_rows,
        [0.95 * inch, 0.67 * inch, 0.67 * inch, 0.72 * inch],
        font_size=6.7,
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 1.</b> VisA subset counts for the official 1-class split.",
                    caption,
                ),
                dataset_table,
                Spacer(1, 5),
            ]
        )
    )
    story.append(
        Paragraph(
            "Image AUROC is the main metric. Pixel localization is evaluated over all "
            "test images, including normal images with all-zero masks. Template and "
            "AnomalyDINO maps are evaluated on a common 448 x 448 grid. Pixel AUROC "
            "uses a monotonic 16-bit rank histogram; its error relative to exact "
            "sklearn AUROC is below 0.001 on controlled arrays.",
            body,
        )
    )

    main_rows = [
        ["Method", "1-shot", "2-shot", "4-shot", "8-shot"],
        ["Image-level AUROC (%)", "", "", "", ""],
    ]
    integration_path = project / "results/integration_comparison.csv"
    integration_rows = [["Shot", "Old image", "New image", "Old loc.", "New loc."]]
    with integration_path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            integration_rows.append(
                [
                    row["shot"],
                    f"{100*float(row['old_image_auroc']):.2f}",
                    f"{100*float(row['new_image_auroc']):.2f}",
                    f"{100*float(row['old_pixel_auroc']):.2f}",
                    f"{100*float(row['new_pixel_auroc']):.2f}",
                ]
            )
    integration_table = fit_table(
        integration_rows,
        [0.55 * inch, 0.83 * inch, 0.83 * inch, 0.75 * inch, 0.75 * inch],
        font_size=6.8,
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 2.</b> Original versus integrated Part 1. Image "
                    "AUROC is directly comparable; localization also changes "
                    "from the old patch grid to a 448-pixel grid.",
                    caption,
                ),
                integration_table,
                Spacer(1, 5),
            ]
        )
    )
    story.append(
        Paragraph(
            "The integration raises Part 1 image AUROC by 3.70-4.91 points. "
            "Localization also rises, but that number combines the stronger "
            "features with a more informative evaluation grid and is not treated "
            "as a purely architectural gain.",
            body,
        )
    )
    story.append(
        Paragraph(
            "At the fixed 95th-percentile calibration threshold, false-positive "
            "rate falls from 32.9-35.1% to 7.9-10.6%, while false-negative rate "
            "rises from 64.8-70.1% to 80.3-85.1%. The integrated baseline is "
            "substantially more conservative and trades recall for fewer false alarms.",
            body,
        )
    )
    for method in METHOD_ORDER:
        row = [METHOD_NAMES[method]]
        for shot in SHOTS:
            mean, std = lookup(
                summary,
                method,
                shot,
                "mean_classification_au_roc",
            )
            row.append(f"{100*mean:.1f} +/- {100*std:.1f}")
        main_rows.append(row)
    main_rows.append(["448-pixel localization AUROC (%)", "", "", "", ""])
    for method in METHOD_ORDER:
        row = [METHOD_NAMES[method]]
        for shot in SHOTS:
            mean, std = lookup(
                summary,
                method,
                shot,
                "mean_segmentation_au_roc",
            )
            row.append(f"{100*mean:.1f} +/- {100*std:.1f}")
        main_rows.append(row)
    main_table = fit_table(
        main_rows,
        [0.92 * inch, 0.66 * inch, 0.66 * inch, 0.66 * inch, 0.66 * inch],
        font_size=6.8,
        spans=[
            ("SPAN", (0, 1), (-1, 1)),
            ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#e8eef7")),
            ("SPAN", (0, 5), (-1, 5)),
            ("BACKGROUND", (0, 5), (-1, 5), colors.HexColor("#e8eef7")),
        ],
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 3.</b> AUROC mean and standard deviation over seeds.",
                    caption,
                ),
                main_table,
                Spacer(1, 4),
            ]
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("5. Results and Discussion", section))
    story.append(
        Paragraph(
            "The integrated template baseline remains well below AnomalyDINO but "
            "improves substantially over the original RGB-only version. Its remaining "
            "errors are dominated by pose, illumination, and background variation. "
            "AnomalyDINO improves image AUROC by roughly 34-41 points and localization "
            "AUROC by roughly 41 points, demonstrating the value of frozen patch "
            "semantics and normal-neighbor retrieval.",
            body,
        )
    )
    story.append(
        Paragraph(
            "The improvement is small but consistent on detection: about +0.4 points "
            "at every shot count. On the stricter 448-pixel localization grid it is "
            "slightly lower by 0.02-0.06 points, revealing an interpretable "
            "detection/localization trade-off rather than an automatic improvement in "
            "every metric.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Shot scaling is strongest for AnomalyDINO. Going from one to eight "
            "references raises image AUROC from 94.6% to 96.6%, whereas the template "
            "baseline gains only 4.8 points and remains unusable. Seed variance also "
            "decreases for AnomalyDINO at larger shot counts, suggesting that diverse "
            "normal views reduce sensitivity to the particular selected references.",
            body,
        )
    )

    category_rows = [
        ["Method", "candle", "caps.", "cashew", "gum", "fryum", "pipe"]
    ]
    for method in METHOD_ORDER:
        row = [METHOD_NAMES[method].replace("AnomalyDINO", "AD")]
        for category in CATEGORIES:
            matches = [
                item
                for item in categories
                if item["method"] == method
                and int(item["shot"]) == 4
                and item["category"] == category
                and item["metric"] == "mean_classification_au_roc"
            ]
            row.append(f"{100*float(matches[0]['mean']):.1f}")
        category_rows.append(row)
    category_table = fit_table(
        category_rows,
        [0.72 * inch] + [0.48 * inch] * 6,
        font_size=6.7,
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 4.</b> Per-category image AUROC (%) at 4-shot.",
                    caption,
                ),
                category_table,
                Spacer(1, 5),
            ]
        )
    )

    curves = scaled_image(
        project / "results/aggregate/image_auroc_vs_shots.png",
        column_width,
    )
    story.append(
        KeepTogether(
            [
                curves,
                Paragraph(
                    "<b>Figure 2.</b> Image-level AUROC versus shot count. Error bars "
                    "show one standard deviation over three seeds.",
                    caption,
                ),
            ]
        )
    )
    pixel_curves = scaled_image(
        project / "results/aggregate/pixel_auroc_vs_shots.png",
        column_width,
    )
    story.append(
        KeepTogether(
            [
                pixel_curves,
                Paragraph(
                    "<b>Figure 3.</b> 448-pixel localization AUROC versus shot count.",
                    caption,
                ),
            ]
        )
    )
    story.append(Paragraph("5.1 Qualitative localization", subsection))
    qualitative = scaled_image(
        project / "report/figures/qualitative_example.png",
        column_width,
    )
    story.append(
        KeepTogether(
            [
                qualitative,
                Paragraph(
                    "<b>Figure 4.</b> Demo input, anomaly heatmap, and overlay for a "
                    "4-shot chewinggum defect. The calibrated decision is NG.",
                    caption,
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "The anomaly map responds to the defect region without defect labels or "
            "task-specific training. Localization is patch-limited: one response cell "
            "spans 14 x 14 input pixels before smoothing. This is adequate for "
            "contamination and missing-component defects but limits sub-pixel or "
            "wire-thin anomalies.",
            body,
        )
    )

    metric_rows = [
        ["Method / shot", "Image AP", "Image F1", "448 loc. AUROC"],
    ]
    for method in ("anomalydino", "multilayer"):
        for shot in (4, 8):
            ap, _ = lookup(summary, method, shot, "mean_classification_ap")
            f1, _ = lookup(summary, method, shot, "mean_classification_f1")
            pixel_auroc, _ = lookup(
                summary, method, shot, "mean_segmentation_au_roc"
            )
            metric_rows.append(
                [
                    f"{METHOD_NAMES[method].replace('AnomalyDINO','AD')} {shot}",
                    f"{100*ap:.1f}",
                    f"{100*f1:.1f}",
                    f"{100*pixel_auroc:.1f}",
                ]
            )
    metric_table = fit_table(
        metric_rows,
        [1.25 * inch, 0.61 * inch, 0.61 * inch, 0.78 * inch],
        font_size=6.7,
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 5.</b> Additional metrics at 4- and 8-shot (%).",
                    caption,
                ),
                metric_table,
                Spacer(1, 5),
            ]
        )
    )
    story.append(Paragraph("5.2 Difficult cases", subsection))
    failure_grid = scaled_image(
        project / "report/figures/failure_grid.png",
        column_width,
    )
    story.append(
        KeepTogether(
            [
                failure_grid,
                Paragraph(
                    "<b>Figure 5.</b> Difficult examples selected from actual 4-shot "
                    "predictions: low-scoring defects and high-scoring normal images.",
                    caption,
                ),
            ]
        )
    )
    story.append(
        Paragraph(
            "Low-scoring defects are small, low-contrast, or locally similar to "
            "legitimate texture. High-scoring normal images typically contain pose, "
            "reflection, or boundary changes represented by too few references. These "
            "errors motivate object-specific foreground masks, more diverse references, "
            "and calibrated selective prediction.",
            body,
        )
    )

    failure_records = json.loads(
        (
            project / "artifacts/failures_multilayer/failure_cases.json"
        ).read_text(encoding="utf-8")
    )
    failure_rows = [["Category", "Sample", "Type", "Score"]]
    for record in failure_records:
        failure_rows.append(
            [
                record["category"],
                record["sample"].replace("/", " / "),
                record["failure_type"].replace(" ", "\n"),
                f"{record['score']:.3f}",
            ]
        )
    failure_table = fit_table(
        failure_rows,
        [0.78 * inch, 1.15 * inch, 1.22 * inch, 0.54 * inch],
        font_size=6.2,
    )
    story.append(
        KeepTogether(
            [
                Paragraph(
                    "<b>Table 6.</b> Exact difficult cases selected in Figure 5.",
                    caption,
                ),
                failure_table,
                Spacer(1, 5),
            ]
        )
    )

    story.append(Paragraph("5.3 Efficiency", subsection))
    efficiency_rows = [
        ["Variant", "Forward ms", "Pipeline median ms", "Peak GPU MiB"],
        ["AnomalyDINO", "12.2", "339.7", "468"],
        ["Multi-layer", "13.3", "313.0", "468"],
    ]
    story.append(
        fill := fit_table(
            efficiency_rows,
            [0.79 * inch, 0.63 * inch, 0.88 * inch, 0.73 * inch],
            font_size=6.7,
        )
    )
    story.append(
        Paragraph(
            "<b>Table 7.</b> Efficiency on an RTX A6000. Forward timing uses 50 "
            "synchronized passes. Pipeline timing includes loading, masks, and CPU "
            "FAISS; it is reported as a median because the server was shared.",
            caption,
        )
    )
    story.append(
        Paragraph(
            "The extra block costs about 1.1 ms in a clean forward benchmark, under "
            "10%, and does not increase peak memory. Memory-bank setup is a few "
            "seconds per category and is dominated by reference augmentation.",
            body,
        )
    )
    story.append(PageBreak())
    story.append(Paragraph("6. Limitations and Conclusion", section))
    story.append(
        Paragraph(
            "The study evaluates the six required categories, not all twelve. Localization "
            "is scored on a 448 x 448 grid with a 16-bit monotonic rank histogram, "
            "rather than original-resolution floating-point maps. "
            "The official one-image-at-a-time loop leaves GPU capacity unused during "
            "I/O and PCA. Finally, only one fixed layer pair is tested; other layer "
            "pairs or calibration-only learned weights may be stronger.",
            body,
        )
    )

    story.append(Paragraph("6.1 Reproducibility details", subsection))
    story.append(
        Paragraph(
            "The data preparation keeps the shared VisA archive read-only and creates "
            "a symlinked official layout. The environment uses Python 3.10, PyTorch "
            "2.7.1 (CUDA 12.6), torchvision 0.22.1, and CPU FAISS 1.16. The official "
            "AnomalyDINO commit is dd35b0f1f6ff6146d196fe4efd3457a58ebcb1e0. The "
            "project pins dependencies and stores the exact data-preparation patch, "
            "run commands, reference filenames, calibration images, per-image scores, "
            "inference times, metrics, figures, and demo inputs.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Every seed writes metrics_seed={r}.json and measurements_seed={r}.csv. "
            "The aggregate script computes category means first and then takes the "
            "unweighted mean across categories. The same script generates the AUROC "
            "shot curves. The evaluation never uses test labels to choose references, "
            "thresholds, layer weights, or score transformations.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Despite these constraints, the project completes the required baseline, "
            "official reproduction, and a controlled improvement. Frozen DINOv2 "
            "features are decisive in few-shot inspection. Multi-layer fusion gives a "
            "consistent image-level gain while preserving localization and memory. "
            "The saved references, calibration exclusion, fixed seeds, full metrics, "
            "and demo make the result reproducible and auditable.",
            body,
        )
    )

    story.append(PageBreak())
    story.append(Paragraph("References", section))
    refs = [
        "[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger. MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. CVPR, 2019.",
        "[2] T. Defard, A. Setkov, A. Loesch, and R. Audigier. PaDiM: A Patch Distribution Modeling Framework for Anomaly Detection and Localization. ICPR Workshops, 2021.",
        "[3] K. Roth, L. Pemula, J. Zepeda, B. Scholkopf, T. Brox, and P. Gehler. Towards Total Recall in Industrial Anomaly Detection. CVPR, 2022.",
        "[4] V. Zavrtanik, M. Kristan, and D. Skocaj. DRAEM: A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection. ICCV, 2021.",
        "[5] C.-L. Li, K. Sohn, J. Yoon, and T. Pfister. CutPaste: Self-Supervised Learning for Anomaly Detection and Localization. CVPR, 2021.",
        "[6] M. Oquab et al. DINOv2: Learning Robust Visual Features without Supervision. TMLR, 2024.",
        "[7] Q. Zhou, G. Pang, Y. Tian, S. He, and J. Chen. AnomalyCLIP: Object-Agnostic Prompt Learning for Zero-Shot Anomaly Detection. ICLR, 2024.",
        "[8] S. Damm, M. Laszkiewicz, J. Lederer, and A. Fischer. AnomalyDINO: Boosting Patch-Based Few-Shot Anomaly Detection with DINOv2. WACV, 2025.",
        "[9] K. Batzner, L. Heckler, and R. Konig. EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies. WACV, 2024.",
        "[10] L. Heckler-Kram, K. Batzner, et al. The MVTec AD 2 Dataset: Advanced Scenarios for Unsupervised Anomaly Detection. IJCV, 2026.",
        "[11] Y. Zou, J. Jeong, L. Pemula, D. Zhang, and O. Dabeer. SPot-the-Difference Self-Supervised Pre-training for Anomaly Detection and Segmentation. ECCV, 2022.",
    ]
    for item in refs:
        story.append(Paragraph(item, reference))

    doc.build(story)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
