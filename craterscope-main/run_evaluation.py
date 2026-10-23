"""
run_evaluation.py — Offline evaluation script.

Runs the detector on the official test split and saves metrics + comparison plots.
Uses the training split for parameter comparison only (no final metrics).

Run:
    python run_evaluation.py
"""

import os
import json
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from detector import PotholeDetector, DEFAULT_PARAMS_A, DEFAULT_PARAMS_B
from evaluation import load_split, evaluate_dataset

# Dataset location: set POTHOLE_DATASET, or place the "archive" folder next to this project.
DATASET_ROOT = os.environ.get(
    "POTHOLE_DATASET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "archive"),
)
ANNO_DIR = os.path.join(DATASET_ROOT, "annotated-images")
SPLITS_JSON = os.path.join(DATASET_ROOT, "splits.json")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_config_comparison():
    """Compare Method A vs Method B on a 50-image subset of the training split."""
    print("\n=== Configuration Comparison (train subset, 50 images) ===")
    train_annos = load_split(SPLITS_JSON, ANNO_DIR, split="train")
    detector = PotholeDetector()

    results = {}
    for label, params in [("A: Edge Density + Darkness", DEFAULT_PARAMS_A),
                           ("B: CLAHE + Morph Gradient", DEFAULT_PARAMS_B)]:
        print(f"\nRunning {label}...")
        m = evaluate_dataset(train_annos, detector, params,
                             iou_threshold=0.5, max_images=50, verbose=False)
        results[label] = m
        print(f"  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")

    # Bar chart comparison
    labels = list(results.keys())
    metrics = ["precision", "recall", "f1"]
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    bars_a = [results[labels[0]][m] for m in metrics]
    bars_b = [results[labels[1]][m] for m in metrics]
    ax.bar(x - width/2, bars_a, width, label=labels[0], color="#2196F3")
    ax.bar(x + width/2, bars_b, width, label=labels[1], color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(["Precision", "Recall", "F1-Score"])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Method A vs Method B — Training Subset (IoU@0.5)\n(Classical methods only — no AI/ML)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    out = os.path.join(RESULTS_DIR, "config_comparison.png")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n[INFO] Comparison chart saved: {out}")
    return results


def run_test_evaluation():
    """Final evaluation on the official test split (Method A, frozen params)."""
    print("\n=== Final Test Evaluation (Method A, official test split) ===")
    test_annos = load_split(SPLITS_JSON, ANNO_DIR, split="test")
    detector = PotholeDetector()

    m = evaluate_dataset(test_annos, detector, DEFAULT_PARAMS_A,
                         iou_threshold=0.5, max_images=None, verbose=False)

    print(f"\n  Images processed : {m['n_processed']}")
    print(f"  Images skipped   : {m['n_skipped']}")
    print(f"  TP  : {m['tp']}")
    print(f"  FP  : {m['fp']}")
    print(f"  FN  : {m['fn']}")
    print(f"  Precision : {m['precision']:.4f}")
    print(f"  Recall    : {m['recall']:.4f}")
    print(f"  F1-Score  : {m['f1']:.4f}")
    print(f"  IoU @     : {m['iou_threshold']}")

    # Save metrics to JSON
    metrics_out = {
        "split": "test",
        "iou_threshold": m["iou_threshold"],
        "n_processed": m["n_processed"],
        "n_skipped": m["n_skipped"],
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
        "precision": round(m["precision"], 4),
        "recall": round(m["recall"], 4),
        "f1": round(m["f1"], 4),
    }
    with open(os.path.join(RESULTS_DIR, "test_metrics.json"), "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\n[INFO] Metrics saved: {os.path.join(RESULTS_DIR, 'test_metrics.json')}")
    return m


def save_qualitative_samples(n: int = 6):
    """Save before/after comparison images for n test-set samples."""
    print(f"\n=== Saving {n} qualitative samples ===")
    test_annos = load_split(SPLITS_JSON, ANNO_DIR, split="test")
    detector = PotholeDetector()

    rng = np.random.default_rng(42)
    indices = rng.choice(len(test_annos), min(n, len(test_annos)), replace=False)

    fig, axes = plt.subplots(n, 4, figsize=(16, n * 3.5))
    if n == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        anno = test_annos[idx]
        img = cv2.imread(anno["img_path"])
        if img is None:
            continue
        result = detector.detect(img, DEFAULT_PARAMS_A)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ann_rgb = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)

        # Draw GT boxes in green on a separate copy
        gt_img = img.copy()
        for (x1, y1, x2, y2) in anno["bboxes"]:
            cv2.rectangle(gt_img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        gt_rgb = cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB)

        # Intermediate: enhanced or edges
        inter = result.intermediates
        mid = inter.get("enhanced", inter.get("edges",
                         inter.get("morphed", np.zeros_like(img[:, :, 0]))))

        axes[row, 0].imshow(img_rgb);         axes[row, 0].set_title("Original")
        axes[row, 1].imshow(gt_rgb);          axes[row, 1].set_title("Ground Truth (green)")
        axes[row, 2].imshow(mid, cmap="gray"); axes[row, 2].set_title("Enhanced / Intermediate")
        axes[row, 3].imshow(ann_rgb);          axes[row, 3].set_title("Detection (red)")
        for ax in axes[row]:
            ax.axis("off")

    plt.suptitle("Qualitative results — test set samples", fontsize=12)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "qualitative_samples.png")
    plt.savefig(out, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Qualitative samples saved: {out}")


def save_intermediate_pipeline():
    """Save a full intermediate pipeline visualisation for one sample image."""
    test_annos = load_split(SPLITS_JSON, ANNO_DIR, split="test")
    if not test_annos:
        return
    anno = test_annos[0]
    img = cv2.imread(anno["img_path"])
    if img is None:
        return

    detector = PotholeDetector()
    result = detector.detect(img, DEFAULT_PARAMS_A)
    inter = result.intermediates

    stages = [
        ("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB), False),
        ("1. Grayscale", inter.get("gray"), True),
        ("2. Gaussian Blur", inter.get("blurred"), True),
        ("3. CLAHE Enhanced", inter.get("enhanced"), True),
        ("4. Adaptive Thresh", inter.get("thresh"), True),
        ("5. Morphology", inter.get("morphed"), True),
        ("6. Detection", cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB), False),
    ]
    stages = [(s, im, g) for s, im, g in stages if im is not None]

    cols = 4
    rows = (len(stages) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    axes = np.array(axes).flatten()

    for i, (title, image, is_gray) in enumerate(stages):
        if is_gray:
            axes[i].imshow(image, cmap="gray")
        else:
            axes[i].imshow(image)
        axes[i].set_title(title, fontsize=9)
        axes[i].axis("off")

    for j in range(len(stages), len(axes)):
        axes[j].axis("off")

    plt.suptitle(f"Pipeline stages — {anno['filename']}", fontsize=11)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "pipeline_stages.png")
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Pipeline stages saved: {out}")


if __name__ == "__main__":
    from preprocess import inspect_dataset, save_sample_grid
    inspect_dataset(verbose=True)
    save_sample_grid(os.path.join(RESULTS_DIR, "sample_grid.png"))
    save_intermediate_pipeline()
    comparison = run_config_comparison()
    test_metrics = run_test_evaluation()
    save_qualitative_samples(n=6)
    print("\n=== Evaluation complete. Results saved to results/ ===")
