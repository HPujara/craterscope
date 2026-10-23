"""
evaluation.py — Annotation parsing and detection evaluation metrics.

Annotation format: Pascal VOC XML (bounding boxes only, no pixel masks).
Evaluation is bounding-box level, not pixel-level segmentation IoU.

Matching rule:
  A predicted box is a True Positive (TP) if it overlaps an unmatched
  ground-truth box with IoU >= iou_threshold (default 0.5).
  Each GT box may be matched at most once (greedy, by descending IoU).
"""

import os
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Annotation parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_annotation(xml_path: str) -> Optional[Dict]:
    """
    Parse a Pascal VOC XML annotation file.

    Returns a dict:
      {
        "filename":  str,             # e.g. "img-1.jpg"
        "width":     int,
        "height":    int,
        "bboxes":    [(xmin, ymin, xmax, ymax), ...]   # one per pothole
      }
    Returns None and prints a warning on parse errors.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as exc:
        print(f"[WARN] Malformed XML {xml_path}: {exc}")
        return None

    filename_el = root.find("filename")
    if filename_el is None or not filename_el.text:
        print(f"[WARN] Missing <filename> in {xml_path}")
        return None

    size_el = root.find("size")
    width = int(size_el.findtext("width", "0")) if size_el is not None else 0
    height = int(size_el.findtext("height", "0")) if size_el is not None else 0

    bboxes = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip().lower()
        if name != "pothole":
            continue
        bb = obj.find("bndbox")
        if bb is None:
            continue
        try:
            xmin = int(float(bb.findtext("xmin", "0")))
            ymin = int(float(bb.findtext("ymin", "0")))
            xmax = int(float(bb.findtext("xmax", "0")))
            ymax = int(float(bb.findtext("ymax", "0")))
        except ValueError:
            print(f"[WARN] Bad bndbox values in {xml_path}")
            continue

        # Clamp to valid range if dimensions are known
        if width > 0 and height > 0:
            xmin = max(0, min(xmin, width - 1))
            ymin = max(0, min(ymin, height - 1))
            xmax = max(0, min(xmax, width))
            ymax = max(0, min(ymax, height))

        if xmax > xmin and ymax > ymin:
            bboxes.append((xmin, ymin, xmax, ymax))

    return {
        "filename": filename_el.text.strip(),
        "width": width,
        "height": height,
        "bboxes": bboxes,
    }


def load_split(splits_json: str, anno_dir: str, split: str = "test") -> List[Dict]:
    """
    Load all annotations for the given split ('train' or 'test').

    Returns list of annotation dicts, skipping missing/invalid files.
    Also prints a summary of processed / skipped counts.
    """
    with open(splits_json) as f:
        splits = json.load(f)

    xml_names = splits.get(split, [])
    annotations = []
    skipped = 0
    invalid = 0

    for xml_name in xml_names:
        xml_path = os.path.join(anno_dir, xml_name)
        if not os.path.exists(xml_path):
            print(f"[WARN] Missing annotation file: {xml_path}")
            skipped += 1
            continue

        anno = parse_annotation(xml_path)
        if anno is None:
            invalid += 1
            continue

        # Verify that the corresponding image file exists
        img_path = os.path.join(anno_dir, anno["filename"])
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found for annotation: {anno['filename']}")
            skipped += 1
            continue

        anno["xml_path"] = xml_path
        anno["img_path"] = img_path
        annotations.append(anno)

    print(f"[INFO] Split '{split}': {len(annotations)} loaded, "
          f"{skipped} missing, {invalid} invalid XML.")
    return annotations


# ──────────────────────────────────────────────────────────────────────────────
# IoU and matching
# ──────────────────────────────────────────────────────────────────────────────

def compute_iou(boxA: Tuple, boxB: Tuple) -> float:
    """
    Compute Intersection-over-Union for two axis-aligned bounding boxes.

    Each box is (xmin, ymin, xmax, ymax).

    IoU = |A ∩ B| / |A ∪ B|
        = intersection_area / (area_A + area_B - intersection_area)
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    area_A = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    area_B = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union_area = area_A + area_B - inter_area

    if union_area == 0:
        return 0.0
    return float(inter_area) / float(union_area)


def match_predictions(
    pred_boxes: List[Tuple],
    gt_boxes: List[Tuple],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Greedy matching of predicted boxes to ground-truth boxes.

    Algorithm:
      For each GT box, find the unmatched predicted box with the highest IoU.
      If that IoU >= threshold, count it as TP; mark both as matched.
      Remaining unmatched predictions are FP; unmatched GTs are FN.

    Returns (tp, fp, fn).
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0
    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    matched_pred = set()
    tp = 0

    for gt in gt_boxes:
        best_iou = 0.0
        best_pred_idx = -1
        for i, pred in enumerate(pred_boxes):
            if i in matched_pred:
                continue
            iou = compute_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = i
        if best_pred_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_pred.add(best_pred_idx)

    fp = len(pred_boxes) - len(matched_pred)
    fn = len(gt_boxes) - tp
    return tp, fp, fn


def compute_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Compute precision, recall, and F1-score from confusion counts.

    Precision = TP / (TP + FP)   — of all detections, what fraction are real?
    Recall    = TP / (TP + FN)   — of all real potholes, what fraction found?
    F1        = 2 * P * R / (P + R)  — harmonic mean
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn}


# ──────────────────────────────────────────────────────────────────────────────
# Full dataset evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_dataset(
    annotations: List[Dict],
    detector,
    params: dict,
    iou_threshold: float = 0.5,
    max_images: int = None,
    verbose: bool = False,
) -> Dict:
    """
    Run the detector on each image and accumulate TP/FP/FN counts.

    Parameters
    ----------
    annotations   : list of dicts from load_split()
    detector      : PotholeDetector instance
    params        : parameter dict to pass to detector.detect()
    iou_threshold : IoU threshold for TP matching
    max_images    : limit evaluation to first N images (for speed)
    verbose       : print per-image results

    Returns dict with:
      - aggregate metrics (precision, recall, f1, tp, fp, fn)
      - per_image list of per-image results
      - n_processed, n_skipped
    """
    import cv2

    total_tp = total_fp = total_fn = 0
    per_image = []
    n_skipped = 0

    subset = annotations if max_images is None else annotations[:max_images]

    for anno in subset:
        img = cv2.imread(anno["img_path"])
        if img is None:
            print(f"[WARN] Could not read image: {anno['img_path']}")
            n_skipped += 1
            continue

        result = detector.detect(img, params)
        tp, fp, fn = match_predictions(result.bboxes, anno["bboxes"], iou_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        per_image.append({
            "filename": anno["filename"],
            "gt_count": len(anno["bboxes"]),
            "pred_count": result.count,
            "tp": tp, "fp": fp, "fn": fn,
            "elapsed_ms": result.elapsed_ms,
        })

        if verbose:
            print(f"  {anno['filename']:20s}  GT={len(anno['bboxes'])}  "
                  f"Pred={result.count}  TP={tp} FP={fp} FN={fn}")

    metrics = compute_metrics(total_tp, total_fp, total_fn)
    metrics["n_processed"] = len(subset) - n_skipped
    metrics["n_skipped"] = n_skipped
    metrics["per_image"] = per_image
    metrics["iou_threshold"] = iou_threshold
    return metrics
