"""
preprocess.py — Dataset inspection, statistics, and sample visualisation.

Run this script directly to get a summary of the dataset:
    python preprocess.py
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

# Dataset location: set POTHOLE_DATASET, or place the "archive" folder next to this project.
DATASET_ROOT = os.environ.get(
    "POTHOLE_DATASET",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "archive"),
)
ANNO_DIR = os.path.join(DATASET_ROOT, "annotated-images")
SPLITS_JSON = os.path.join(DATASET_ROOT, "splits.json")


def inspect_dataset(verbose: bool = True) -> dict:
    """
    Walk the annotated-images directory and collect basic statistics.

    Returns a summary dict.
    """
    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    train_set = set(splits["train"])
    test_set = set(splits["test"])

    all_xmls = sorted(Path(ANNO_DIR).glob("*.xml"))
    all_jpgs = sorted(Path(ANNO_DIR).glob("*.jpg"))

    widths, heights, bbox_counts = [], [], []
    missing_images = []
    multi_anno = []

    for xml_path in all_xmls:
        # Check paired image
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except ET.ParseError:
            continue

        fn = root.findtext("filename", "").strip()
        img_path = Path(ANNO_DIR) / fn
        if not img_path.exists():
            missing_images.append(fn)

        size = root.find("size")
        if size is not None:
            widths.append(int(size.findtext("width", 0)))
            heights.append(int(size.findtext("height", 0)))

        objects = root.findall("object")
        n = len(objects)
        bbox_counts.append(n)
        if n > 1:
            multi_anno.append((fn, n))

    summary = {
        "total_xml": len(all_xmls),
        "total_jpg": len(all_jpgs),
        "train_count": len(splits["train"]),
        "test_count": len(splits["test"]),
        "missing_images": missing_images,
        "image_widths":  {"min": min(widths), "max": max(widths),
                          "mean": float(np.mean(widths))} if widths else {},
        "image_heights": {"min": min(heights), "max": max(heights),
                          "mean": float(np.mean(heights))} if heights else {},
        "bboxes_per_image": {
            "min": min(bbox_counts), "max": max(bbox_counts),
            "mean": float(np.mean(bbox_counts)),
            "total": sum(bbox_counts),
        } if bbox_counts else {},
        "multi_annotation_images": len(multi_anno),
    }

    if verbose:
        print("=" * 55)
        print("  POTHOLE DATASET INSPECTION SUMMARY")
        print("=" * 55)
        print(f"  XML files found  : {summary['total_xml']}")
        print(f"  JPG files found  : {summary['total_jpg']}")
        print(f"  Train split      : {summary['train_count']}")
        print(f"  Test split       : {summary['test_count']}")
        if missing_images:
            print(f"  Missing images   : {len(missing_images)}")
            for m in missing_images[:5]:
                print(f"    - {m}")
        else:
            print(f"  Missing images   : 0")
        if widths:
            print(f"  Image widths     : {summary['image_widths']}")
            print(f"  Image heights    : {summary['image_heights']}")
        if bbox_counts:
            b = summary["bboxes_per_image"]
            print(f"  Bboxes per image : min={b['min']} max={b['max']} "
                  f"mean={b['mean']:.2f} total={b['total']}")
        print(f"  Multi-bbox images: {summary['multi_annotation_images']}")
        print("=" * 55)

    return summary


def save_sample_grid(output_path: str = None, n: int = 9):
    """
    Save a grid of n sample images with their ground-truth bounding boxes.
    Uses matplotlib; saves to output_path or results/sample_grid.png.
    """
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(__file__), "results", "sample_grid.png"
        )

    with open(SPLITS_JSON) as f:
        splits = json.load(f)

    rng = np.random.default_rng(42)
    sample_xmls = rng.choice(splits["train"], min(n, len(splits["train"])),
                             replace=False)

    cols = 3
    rows = (len(sample_xmls) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    axes = np.array(axes).flatten()

    for i, xml_name in enumerate(sample_xmls):
        xml_path = os.path.join(ANNO_DIR, xml_name)
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
        except Exception:
            continue
        fn = root.findtext("filename", "").strip()
        img = cv2.imread(os.path.join(ANNO_DIR, fn))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img_rgb)
        axes[i].set_title(fn, fontsize=8)
        axes[i].axis("off")
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            if bb is None:
                continue
            xmin = int(float(bb.findtext("xmin", 0)))
            ymin = int(float(bb.findtext("ymin", 0)))
            xmax = int(float(bb.findtext("xmax", 0)))
            ymax = int(float(bb.findtext("ymax", 0)))
            rect = mpatches.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin,
                linewidth=2, edgecolor="red", facecolor="none"
            )
            axes[i].add_patch(rect)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Sample annotated images (red = GT pothole)", fontsize=11)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Sample grid saved: {output_path}")
    return output_path


if __name__ == "__main__":
    summary = inspect_dataset(verbose=True)
    save_sample_grid()
