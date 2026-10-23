"""
tests/test_dataset.py — Data-integrity and end-to-end evaluation on the real dataset.

Skipped automatically if the dataset is not present.
"""

import json
import os

import cv2
import numpy as np
import pytest

from tests.conftest import needs_dataset, ANNO_DIR, SPLITS_JSON, PROJECT_ROOT
from detector import PotholeDetector, DEFAULT_PARAMS_A, DEFAULT_PARAMS_B
from evaluation import load_split, evaluate_dataset

pytestmark = needs_dataset


@pytest.fixture(scope="module")
def splits():
    with open(SPLITS_JSON) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_annos():
    return load_split(SPLITS_JSON, ANNO_DIR, split="test")


@pytest.fixture(scope="module")
def train_annos():
    return load_split(SPLITS_JSON, ANNO_DIR, split="train")


class TestDataIntegrity:
    def test_split_sizes(self, splits):
        assert len(splits["train"]) == 532
        assert len(splits["test"]) == 133

    def test_no_train_test_leakage(self, splits):
        assert not set(splits["train"]) & set(splits["test"])

    def test_every_annotation_loads(self, train_annos, test_annos):
        assert len(train_annos) == 532 and len(test_annos) == 133

    def test_boxes_valid_and_inside_image(self, train_annos, test_annos):
        for a in train_annos + test_annos:
            assert a["bboxes"], f"{a['filename']} has no pothole boxes"
            for x1, y1, x2, y2 in a["bboxes"]:
                assert 0 <= x1 < x2 <= a["width"] and 0 <= y1 < y2 <= a["height"]

    def test_xml_size_matches_image(self, test_annos):
        for a in test_annos[:40]:
            img = cv2.imread(a["img_path"])
            assert img is not None
            assert img.shape[:2] == (a["height"], a["width"]), a["filename"]


class TestEndToEnd:
    def test_reproduces_saved_test_metrics(self, test_annos):
        """The numbers in results/test_metrics.json must be reproducible."""
        with open(os.path.join(PROJECT_ROOT, "results", "test_metrics.json")) as f:
            saved = json.load(f)
        m = evaluate_dataset(test_annos, PotholeDetector(), DEFAULT_PARAMS_A, 0.5)
        assert (m["tp"], m["fp"], m["fn"]) == (saved["tp"], saved["fp"], saved["fn"])
        assert m["f1"] == pytest.approx(saved["f1"], abs=1e-4)

    def test_lower_iou_threshold_gives_higher_f1(self, test_annos):
        d = PotholeDetector()
        f1_05 = evaluate_dataset(test_annos[:60], d, DEFAULT_PARAMS_A, 0.5)["f1"]
        f1_03 = evaluate_dataset(test_annos[:60], d, DEFAULT_PARAMS_A, 0.3)["f1"]
        assert f1_03 >= f1_05

    def test_methods_beat_trivial_baseline(self, test_annos):
        """Both methods must beat a 'one box = central half of image' guess."""
        from evaluation import match_predictions, compute_metrics
        tp = fp = fn = 0
        for a in test_annos:
            w, h = a["width"], a["height"]
            t, f, n = match_predictions([(w // 4, h // 4, 3 * w // 4, 3 * h // 4)], a["bboxes"], 0.5)
            tp += t; fp += f; fn += n
        baseline = compute_metrics(tp, fp, fn)["f1"]
        d = PotholeDetector()
        for p in (DEFAULT_PARAMS_A, DEFAULT_PARAMS_B):
            assert evaluate_dataset(test_annos, d, p, 0.5)["f1"] > baseline

    def test_every_test_image_processes_without_error(self, test_annos):
        d = PotholeDetector()
        for a in test_annos:
            r = d.detect(cv2.imread(a["img_path"]), DEFAULT_PARAMS_A)
            assert r.annotated_image is not None
