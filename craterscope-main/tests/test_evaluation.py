"""
tests/test_evaluation.py — Unit tests for annotation parsing and evaluation metrics.
Run: python -m pytest tests/ -v
"""

import sys
import os
import xml.etree.ElementTree as ET
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluation import compute_iou, match_predictions, compute_metrics, parse_annotation


# ──────────────────────────────────────────────────────────────────────────────
# IoU tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeIoU:
    def test_identical_boxes(self):
        box = (0, 0, 100, 100)
        assert compute_iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert compute_iou((0, 0, 50, 50), (60, 60, 100, 100)) == pytest.approx(0.0)

    def test_half_overlap(self):
        # boxA = (0,0,100,100) area=10000
        # boxB = (50,0,150,100) area=10000
        # intersection = 50*100=5000, union=15000
        iou = compute_iou((0, 0, 100, 100), (50, 0, 150, 100))
        assert iou == pytest.approx(5000 / 15000, rel=1e-6)

    def test_contained_box(self):
        # inner fully inside outer: IoU = inner_area / outer_area
        iou = compute_iou((25, 25, 75, 75), (0, 0, 100, 100))
        # intersection=2500, union=10000, iou=0.25
        assert iou == pytest.approx(0.25)

    def test_zero_area_box(self):
        assert compute_iou((0, 0, 0, 0), (0, 0, 100, 100)) == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Matching tests
# ──────────────────────────────────────────────────────────────────────────────

class TestMatchPredictions:
    def test_perfect_single_match(self):
        box = (10, 10, 110, 110)
        tp, fp, fn = match_predictions([box], [box], iou_threshold=0.5)
        assert tp == 1 and fp == 0 and fn == 0

    def test_no_predictions(self):
        tp, fp, fn = match_predictions([], [(0, 0, 100, 100)], iou_threshold=0.5)
        assert tp == 0 and fp == 0 and fn == 1

    def test_no_ground_truth(self):
        tp, fp, fn = match_predictions([(0, 0, 100, 100)], [], iou_threshold=0.5)
        assert tp == 0 and fp == 1 and fn == 0

    def test_iou_below_threshold(self):
        # Only 10% overlap — should not match at threshold 0.5
        tp, fp, fn = match_predictions(
            [(0, 0, 100, 100)], [(80, 80, 200, 200)], iou_threshold=0.5
        )
        assert tp == 0 and fp == 1 and fn == 1

    def test_multiple_predictions_one_gt(self):
        gt = [(0, 0, 100, 100)]
        preds = [(0, 0, 100, 100), (200, 200, 300, 300)]
        tp, fp, fn = match_predictions(preds, gt, iou_threshold=0.5)
        assert tp == 1 and fp == 1 and fn == 0

    def test_each_gt_matched_once(self):
        # Two identical predictions vs one GT — only one TP allowed
        gt = [(0, 0, 100, 100)]
        preds = [(0, 0, 100, 100), (0, 0, 100, 100)]
        tp, fp, fn = match_predictions(preds, gt, iou_threshold=0.5)
        assert tp == 1 and fp == 1 and fn == 0


# ──────────────────────────────────────────────────────────────────────────────
# Metrics tests
# ──────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:
    def test_perfect(self):
        m = compute_metrics(10, 0, 0)
        assert m["precision"] == pytest.approx(1.0)
        assert m["recall"] == pytest.approx(1.0)
        assert m["f1"] == pytest.approx(1.0)

    def test_no_detections(self):
        m = compute_metrics(0, 0, 10)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)
        assert m["f1"] == pytest.approx(0.0)

    def test_all_false_positives(self):
        m = compute_metrics(0, 10, 0)
        assert m["precision"] == pytest.approx(0.0)
        assert m["recall"] == pytest.approx(0.0)

    def test_typical_values(self):
        m = compute_metrics(tp=7, fp=3, fn=2)
        assert m["precision"] == pytest.approx(7 / 10)
        assert m["recall"] == pytest.approx(7 / 9)
        expected_f1 = 2 * (7/10) * (7/9) / ((7/10) + (7/9))
        assert m["f1"] == pytest.approx(expected_f1, rel=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# Annotation parsing tests
# ──────────────────────────────────────────────────────────────────────────────

class TestParseAnnotation:
    def _write_xml(self, content: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".xml", delete=False, encoding="utf-8"
        )
        tmp.write(content)
        tmp.flush()
        return tmp.name

    def test_valid_single_bbox(self):
        xml = """<annotation>
            <filename>img-1.jpg</filename>
            <size><width>720</width><height>720</height><depth>3</depth></size>
            <object>
                <name>pothole</name>
                <bndbox>
                    <xmin>100</xmin><ymin>200</ymin>
                    <xmax>300</xmax><ymax>400</ymax>
                </bndbox>
            </object>
        </annotation>"""
        path = self._write_xml(xml)
        anno = parse_annotation(path)
        os.unlink(path)
        assert anno is not None
        assert anno["filename"] == "img-1.jpg"
        assert anno["width"] == 720
        assert len(anno["bboxes"]) == 1
        assert anno["bboxes"][0] == (100, 200, 300, 400)

    def test_multiple_bboxes(self):
        xml = """<annotation>
            <filename>img-2.jpg</filename>
            <size><width>720</width><height>720</height><depth>3</depth></size>
            <object><name>pothole</name>
                <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
            </object>
            <object><name>pothole</name>
                <bndbox><xmin>200</xmin><ymin>200</ymin><xmax>400</xmax><ymax>300</ymax></bndbox>
            </object>
        </annotation>"""
        path = self._write_xml(xml)
        anno = parse_annotation(path)
        os.unlink(path)
        assert len(anno["bboxes"]) == 2

    def test_missing_filename(self):
        xml = """<annotation>
            <size><width>720</width><height>720</height><depth>3</depth></size>
        </annotation>"""
        path = self._write_xml(xml)
        anno = parse_annotation(path)
        os.unlink(path)
        assert anno is None

    def test_malformed_xml(self):
        path = self._write_xml("<annotation><unclosed>")
        anno = parse_annotation(path)
        os.unlink(path)
        assert anno is None

    def test_non_pothole_object_ignored(self):
        xml = """<annotation>
            <filename>img-3.jpg</filename>
            <size><width>720</width><height>720</height><depth>3</depth></size>
            <object><name>car</name>
                <bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>50</ymax></bndbox>
            </object>
        </annotation>"""
        path = self._write_xml(xml)
        anno = parse_annotation(path)
        os.unlink(path)
        assert anno is not None
        assert len(anno["bboxes"]) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Integration: detect + evaluate on a synthetic image
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectorIntegration:
    def test_detect_dark_patch(self):
        """Detector should find a large dark region on a light background."""
        import numpy as np
        from detector import PotholeDetector, DEFAULT_PARAMS_A

        img = np.full((300, 300, 3), 180, dtype=np.uint8)  # light grey
        img[80:200, 80:220] = 40                           # dark patch = pothole

        det = PotholeDetector()
        result = det.detect(img, DEFAULT_PARAMS_A)
        assert result.count >= 1, (
            "Expected at least one detection on a synthetic dark-patch image"
        )

    def test_uniform_image_no_detection(self):
        """Uniform images should produce zero detections."""
        import numpy as np
        from detector import PotholeDetector, DEFAULT_PARAMS_A

        img = np.full((300, 300, 3), 120, dtype=np.uint8)
        det = PotholeDetector()
        result = det.detect(img, DEFAULT_PARAMS_A)
        assert result.count == 0, "Uniform image should have no detections"
