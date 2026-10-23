"""
tests/test_robustness.py — Behavioural tests on synthetic road scenes.

Each scene has a known ground truth, so these tests check localisation,
invariance (flip, brightness, contrast, compression, noise) and the known
false-positive sources (shadows, lane markings). Known weaknesses are
recorded with strict xfail so the suite flags it if they are ever fixed.
"""

import cv2
import numpy as np
import pytest

from tests.conftest import make_road, add_pothole
from detector import PotholeDetector, DEFAULT_PARAMS_A, DEFAULT_PARAMS_B
from evaluation import compute_iou

GT = (170, 155, 310, 245)        # bbox of the synthetic pothole in road_with_pothole
DET = PotholeDetector()


def best_iou(bboxes, gt=GT):
    return max((compute_iou(b, gt) for b in bboxes), default=0.0)


def centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


# ──────────────────────────────────────────────────────────────────────────────
# Localisation
# ──────────────────────────────────────────────────────────────────────────────

class TestLocalisation:
    @pytest.mark.parametrize("params", [DEFAULT_PARAMS_A, DEFAULT_PARAMS_B], ids=["A", "B"])
    def test_detects_single_pothole(self, params, road_with_pothole):
        img, gt = road_with_pothole
        r = DET.detect(img, params)
        assert r.count == 1
        cx, cy = centre(r.bboxes[0])
        assert gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]   # centred on GT
        assert best_iou(r.bboxes, gt) > 0.3

    def test_detected_box_contains_pothole(self, road_with_pothole):
        img, gt = road_with_pothole
        x1, y1, x2, y2 = DET.detect(img, DEFAULT_PARAMS_A).bboxes[0]
        assert x1 <= gt[0] and y1 <= gt[1] and x2 >= gt[2] and y2 >= gt[3]

    @pytest.mark.xfail(strict=True, reason=(
        "LIMITATION: boxes are inflated ~45% per side by the σ=15 density blur "
        "and 13×13 closing, so even a perfect synthetic pothole only reaches "
        "IoU≈0.39 — below the 0.5 TP threshold. Major cause of low test F1."))
    def test_box_tight_enough_for_iou_05(self, road_with_pothole):
        img, gt = road_with_pothole
        assert best_iou(DET.detect(img, DEFAULT_PARAMS_A).bboxes, gt) >= 0.5

    def test_two_separate_potholes(self):
        img = add_pothole(make_road(), 120, 180, 50, 35, seed=2)
        img = add_pothole(img, 360, 200, 50, 35, seed=3)
        boxes = DET.detect(img, DEFAULT_PARAMS_A).bboxes
        assert len(boxes) == 2
        xs = sorted(centre(b)[0] for b in boxes)
        assert xs[0] < 240 < xs[1]

    def test_clean_road_no_detection_method_a(self, road):
        assert DET.detect(road, DEFAULT_PARAMS_A).count == 0

    @pytest.mark.xfail(strict=True, reason=(
        "LIMITATION: Method B keeps the top 13% of gradient density even on a "
        "pothole-free road, producing ~7 false alarms."))
    def test_clean_road_no_detection_method_b(self, road):
        assert DET.detect(road, DEFAULT_PARAMS_B).count == 0


# ──────────────────────────────────────────────────────────────────────────────
# Invariance to photometric / geometric changes
# ──────────────────────────────────────────────────────────────────────────────

class TestInvariance:
    def test_horizontal_flip_mirrors_box(self):
        img = add_pothole(make_road(), 150, 200, 60, 40)
        w = img.shape[1]
        b = DET.detect(img, DEFAULT_PARAMS_A).bboxes
        bf = DET.detect(cv2.flip(img, 1), DEFAULT_PARAMS_A).bboxes
        assert len(b) == len(bf) >= 1
        for x1, y1, x2, y2 in b:
            mirrored = (w - x2, y1, w - x1, y2)
            assert max(compute_iou(mirrored, f) for f in bf) > 0.9

    @pytest.mark.parametrize("beta", [-50, -20, 20, 40])
    def test_global_brightness_shift(self, beta, road_with_pothole):
        img, gt = road_with_pothole
        shifted = cv2.convertScaleAbs(img, alpha=1.0, beta=beta)
        assert best_iou(DET.detect(shifted, DEFAULT_PARAMS_A).bboxes, gt) > 0.3

    def test_reduced_contrast(self, road_with_pothole):
        img, gt = road_with_pothole
        low = cv2.convertScaleAbs(img, alpha=0.5, beta=60)
        assert best_iou(DET.detect(low, DEFAULT_PARAMS_A).bboxes, gt) > 0.3

    @pytest.mark.parametrize("quality", [90, 50, 25])
    def test_jpeg_compression(self, quality, road_with_pothole):
        img, gt = road_with_pothole
        _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        assert best_iou(DET.detect(cv2.imdecode(enc, 1), DEFAULT_PARAMS_A).bboxes, gt) > 0.3

    def test_mild_sensor_noise(self, road_with_pothole):
        img, gt = road_with_pothole
        rng = np.random.default_rng(5)
        noisy = (img + rng.normal(0, 10, img.shape)).clip(0, 255).astype(np.uint8)
        assert best_iou(DET.detect(noisy, DEFAULT_PARAMS_A).bboxes, gt) > 0.2

    @pytest.mark.xfail(strict=True, reason=(
        "LIMITATION: at σ=25 noise the edge-density cue is swamped by noise "
        "texture; detection drifts off the pothole."))
    def test_heavy_sensor_noise(self, road_with_pothole):
        img, gt = road_with_pothole
        rng = np.random.default_rng(5)
        noisy = (img + rng.normal(0, 25, img.shape)).clip(0, 255).astype(np.uint8)
        assert best_iou(DET.detect(noisy, DEFAULT_PARAMS_A).bboxes, gt) > 0.2

    def test_colour_cast_ignored(self, road_with_pothole):
        """Pipeline works on luminance only; a tint shouldn't change much."""
        img, gt = road_with_pothole
        tinted = img.astype(np.int16)
        tinted[:, :, 0] -= 30          # less blue
        tinted[:, :, 2] += 30          # more red
        tinted = tinted.clip(0, 255).astype(np.uint8)
        assert best_iou(DET.detect(tinted, DEFAULT_PARAMS_A).bboxes, gt) > 0.3


# ──────────────────────────────────────────────────────────────────────────────
# Known false-positive sources
# ──────────────────────────────────────────────────────────────────────────────

def _shadow_scene():
    img = make_road()
    m = np.zeros(img.shape[:2], np.float32)
    cv2.ellipse(m, (240, 200), (90, 60), 0, 0, 360, 1, -1)
    m = cv2.GaussianBlur(m, (0, 0), 8)
    return (img * (1 - 0.5 * m[..., None])).astype(np.uint8)


def _lane_scene():
    img = make_road()
    img[:, 230:250] = 245
    return img


class TestFalsePositiveSources:
    @pytest.mark.xfail(strict=True, reason=(
        "LIMITATION: a smooth, texture-less shadow is flagged as a pothole — "
        "darkness + its soft edge pass every filter."))
    @pytest.mark.parametrize("params", [DEFAULT_PARAMS_A, DEFAULT_PARAMS_B], ids=["A", "B"])
    def test_shadow_not_detected(self, params):
        assert DET.detect(_shadow_scene(), params).count == 0

    @pytest.mark.xfail(strict=True, reason=(
        "LIMITATION: a white lane marking creates strong edges; the closing "
        "step widens it into a blob that passes the aspect-ratio filter."))
    @pytest.mark.parametrize("params", [DEFAULT_PARAMS_A, DEFAULT_PARAMS_B], ids=["A", "B"])
    def test_lane_marking_not_detected(self, params):
        assert DET.detect(_lane_scene(), params).count == 0

    def test_bright_sky_band_rejected(self):
        img = make_road()
        img[:90] = 235                                  # bright sky with a horizon edge
        rng = np.random.default_rng(1)
        img[:90] = (img[:90] + rng.integers(-15, 15, img[:90].shape)).clip(0, 255).astype(np.uint8)
        for x1, y1, x2, y2 in DET.detect(img, DEFAULT_PARAMS_A).bboxes:
            assert y2 > 120, "box located in the sky region"


# ──────────────────────────────────────────────────────────────────────────────
# Parameter sensitivity — knobs move results in the documented direction
# ──────────────────────────────────────────────────────────────────────────────

class TestParameterSensitivity:
    def test_higher_percentile_keeps_fewer_pixels(self, road_with_pothole):
        img, _ = road_with_pothole
        fracs = [(DET.detect(img, dict(DEFAULT_PARAMS_A, percentile=p))
                  .intermediates["thresh"] > 0).mean() for p in (70, 80, 90, 95)]
        assert fracs == sorted(fracs, reverse=True)

    def test_larger_min_area_never_adds_detections(self):
        img = add_pothole(add_pothole(make_road(), 120, 180, 30, 20, seed=2), 360, 200, 60, 40, seed=3)
        counts = [DET.detect(img, dict(DEFAULT_PARAMS_A, min_area=a)).count
                  for a in (500, 1500, 5000, 20000, 60000)]
        assert counts == sorted(counts, reverse=True)
        assert counts[-1] == 0

    def test_strict_brightness_filter_removes_everything(self, road_with_pothole):
        img, _ = road_with_pothole
        assert DET.detect(img, dict(DEFAULT_PARAMS_A, max_brightness=10)).count == 0

    def test_edge_weight_one_ignores_darkness(self, road_with_pothole):
        img, _ = road_with_pothole
        inter = DET.detect(img, dict(DEFAULT_PARAMS_A, edge_weight=1.0, dark_weight=0.0)).intermediates
        m = DEFAULT_PARAMS_A["border_margin"]
        diff = np.abs(inter["combined"].astype(int) - inter["edge_density"].astype(int))
        assert diff[m:-m, m:-m].max() <= 1
