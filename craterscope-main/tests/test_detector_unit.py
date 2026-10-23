"""
tests/test_detector_unit.py — Unit tests for each stage of the classical pipeline.

Covers the helper functions, the region filters, the intermediate feature maps
(darkness, Sobel edge density, CLAHE, morphological gradient) and the
PotholeDetector public API contract.
"""

import cv2
import numpy as np
import pytest

from tests.conftest import make_road, add_pothole
from detector import (
    PotholeDetector,
    DEFAULT_PARAMS_A,
    DEFAULT_PARAMS_B,
    _odd,
    _normalize_to_uint8,
    _solidity,
    _filter_contours,
    contour_to_bbox,
)

BOTH = pytest.mark.parametrize(
    "params", [DEFAULT_PARAMS_A, DEFAULT_PARAMS_B], ids=["methodA", "methodB"]
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    @pytest.mark.parametrize("n,expected", [(1, 1), (2, 3), (4, 5), (5, 5), (0, 1), (-3, 1), (6.0, 7)])
    def test_odd_kernel_sizes(self, n, expected):
        assert _odd(n) == expected

    def test_normalize_range(self):
        arr = np.array([[2.0, 4.0], [6.0, 10.0]], np.float32)
        out = _normalize_to_uint8(arr)
        assert out.dtype == np.uint8
        assert out.min() == 0 and out.max() == 255

    def test_normalize_constant_array(self):
        out = _normalize_to_uint8(np.full((5, 5), 7.0))
        assert out.dtype == np.uint8 and not out.any()

    def _contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cnts[0]

    def test_solidity_of_square_is_one(self):
        m = np.zeros((100, 100), np.uint8)
        m[20:80, 20:80] = 255
        assert _solidity(self._contour(m)) == pytest.approx(1.0, abs=0.02)

    def test_solidity_of_l_shape_below_one(self):
        m = np.zeros((100, 100), np.uint8)
        m[10:90, 10:30] = 255
        m[70:90, 10:90] = 255
        assert 0.3 < _solidity(self._contour(m)) < 0.7

    def test_contour_to_bbox(self):
        m = np.zeros((100, 100), np.uint8)
        m[20:50, 30:70] = 255
        assert contour_to_bbox(self._contour(m)) == (30, 20, 70, 50)


# ──────────────────────────────────────────────────────────────────────────────
# Region filter (_filter_contours) — each rule in isolation
# ──────────────────────────────────────────────────────────────────────────────

class TestRegionFilter:
    H, W = 400, 400

    def _run(self, mask, gray=None, **overrides):
        params = dict(DEFAULT_PARAMS_A, **overrides)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return _filter_contours(cnts, self.H, self.W, params, gray_image=gray)

    def _blank(self):
        return np.zeros((self.H, self.W), np.uint8)

    def test_accepts_normal_blob(self):
        m = self._blank()
        cv2.ellipse(m, (200, 200), (50, 35), 0, 0, 360, 255, -1)
        assert len(self._run(m)) == 1

    def test_rejects_too_small(self):
        m = self._blank()
        m[100:120, 100:120] = 255            # 400 px² < min_area 1500
        assert self._run(m) == []

    def test_rejects_too_large(self):
        m = self._blank()
        m[10:390, 10:390] = 255              # ~90% of image > 35%
        assert self._run(m) == []

    def test_rejects_extreme_aspect_ratio(self):
        m = self._blank()
        m[200:205, 20:380] = 255             # 360×5, aspect 72
        assert self._run(m, min_area=100) == []

    def test_rejects_low_solidity(self):
        # a spiky star: small area relative to its convex hull
        m = self._blank()
        pts = []
        for k in range(16):
            r = 150 if k % 2 == 0 else 25
            a = k * np.pi / 8
            pts.append([200 + r * np.cos(a), 200 + r * np.sin(a)])
        cv2.fillPoly(m, [np.int32(pts)], 255)
        assert self._run(m, min_area=100, min_solidity=0.5) == []
        assert len(self._run(m, min_area=100, min_solidity=0.05)) == 1

    def test_rejects_panoramic_band(self):
        m = self._blank()
        m[150:200, 20:380] = 255             # w=90% img, h=12.5% img, aspect 7.2
        assert self._run(m) == []

    def test_rejects_bright_region(self):
        m = self._blank()
        cv2.circle(m, (200, 200), 50, 255, -1)
        bright = np.full((self.H, self.W), 230, np.uint8)
        dark = np.full((self.H, self.W), 90, np.uint8)
        assert self._run(m, gray=bright) == []
        assert len(self._run(m, gray=dark)) == 1


# ──────────────────────────────────────────────────────────────────────────────
# Feature maps
# ──────────────────────────────────────────────────────────────────────────────

class TestFeatureMaps:
    def test_dark_map_peaks_inside_dark_patch(self):
        img = make_road()
        img[150:230, 200:300] = 40
        inter = PotholeDetector().detect(img, DEFAULT_PARAMS_A).intermediates
        dm = inter["dark_map"].astype(float)
        assert dm[160:220, 210:290].mean() > 5 * dm[20:80, 20:80].mean() + 1

    def test_dark_map_ignores_bright_patch(self):
        img = make_road()
        img[150:230, 200:300] = 230
        dm = PotholeDetector().detect(img, DEFAULT_PARAMS_A).intermediates["dark_map"]
        # excess darkness is one-sided: max(0, background - pixel)
        assert dm[170:210, 220:280].mean() < 30

    def test_edge_density_higher_on_texture(self):
        img = make_road()
        rng = np.random.default_rng(3)
        img[120:240, 160:320] = rng.integers(0, 255, (120, 160, 1), dtype=np.uint8)
        ed = PotholeDetector().detect(img, DEFAULT_PARAMS_A).intermediates["edge_density"]
        assert ed[150:210, 200:280].mean() > 3 * ed[40:100, 40:100].mean()

    @BOTH
    def test_border_is_masked(self, params):
        img = make_road()
        img[:, :30] = 0                      # strong edge right at the border
        inter = PotholeDetector().detect(img, params).intermediates
        m = params["border_margin"]
        morphed = inter["morphed"]
        assert not morphed[:m].any() and not morphed[-m:].any()
        assert not morphed[:, :m].any() and not morphed[:, -m:].any()

    def test_clahe_increases_local_contrast(self):
        img = make_road(noise=2)
        img[100:200, 100:200] = img[100:200, 100:200] // 2 + 60   # low contrast region
        inter = PotholeDetector().detect(img, DEFAULT_PARAMS_B).intermediates
        assert inter["enhanced"].std() > inter["blurred"].std()

    def test_morph_gradient_is_dilate_minus_erode(self):
        img = make_road()
        img[100:200, 150:300] = 30
        inter = PotholeDetector().detect(img, DEFAULT_PARAMS_B).intermediates
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        e = inter["enhanced"]
        ref = cv2.dilate(e, se).astype(int) - cv2.erode(e, se).astype(int)
        m = DEFAULT_PARAMS_B["border_margin"]
        got = inter["morph_gradient"][m:-m, m:-m].astype(int)
        assert np.array_equal(got, ref[m:-m, m:-m])

    def test_threshold_keeps_about_top_percentile(self, road_with_pothole):
        img, _ = road_with_pothole
        inter = PotholeDetector().detect(img, DEFAULT_PARAMS_A).intermediates
        frac = (inter["thresh"] > 0).mean()
        assert 0.05 < frac <= 0.16           # percentile 85 → ≤15% kept

    def test_mask_is_binary(self, road_with_pothole):
        img, _ = road_with_pothole
        for p in (DEFAULT_PARAMS_A, DEFAULT_PARAMS_B):
            inter = PotholeDetector().detect(img, p).intermediates
            for k in ("thresh", "morphed"):
                assert set(np.unique(inter[k])) <= {0, 255}


# ──────────────────────────────────────────────────────────────────────────────
# Public API contract
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectorAPI:
    @BOTH
    def test_result_fields(self, params, road_with_pothole):
        img, _ = road_with_pothole
        r = PotholeDetector().detect(img, params)
        assert r.count == len(r.bboxes) == len(r.contours)
        assert r.annotated_image.shape == img.shape
        assert r.annotated_image.dtype == np.uint8
        assert r.elapsed_ms > 0

    @BOTH
    def test_bboxes_inside_image(self, params, road_with_pothole):
        img, _ = road_with_pothole
        h, w = img.shape[:2]
        for x1, y1, x2, y2 in PotholeDetector().detect(img, params).bboxes:
            assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h

    def test_expected_intermediate_keys(self, road):
        a = PotholeDetector().detect(road, DEFAULT_PARAMS_A).intermediates
        b = PotholeDetector().detect(road, DEFAULT_PARAMS_B).intermediates
        assert set(a) == {"gray", "blurred", "dark_map", "edge_density",
                          "combined", "thresh", "morphed"}
        assert set(b) == {"gray", "blurred", "enhanced", "morph_gradient",
                          "density", "thresh", "morphed"}
        for v in list(a.values()) + list(b.values()):
            assert v.shape == road.shape[:2] and v.dtype == np.uint8

    def test_default_params_is_method_a(self, road_with_pothole):
        img, _ = road_with_pothole
        d = PotholeDetector()
        assert d.detect(img).bboxes == d.detect(img, DEFAULT_PARAMS_A).bboxes

    @BOTH
    def test_deterministic(self, params, road_with_pothole):
        img, _ = road_with_pothole
        d = PotholeDetector()
        assert d.detect(img, params).bboxes == d.detect(img, params).bboxes

    @BOTH
    def test_input_not_mutated(self, params, road_with_pothole):
        img, _ = road_with_pothole
        before = img.copy()
        PotholeDetector().detect(img, params)
        assert np.array_equal(img, before)

    def test_params_dict_not_mutated(self, road):
        p = dict(DEFAULT_PARAMS_A)
        PotholeDetector().detect(road, p)
        assert p == DEFAULT_PARAMS_A

    def test_annotation_draws_red(self, road_with_pothole):
        img, _ = road_with_pothole
        r = PotholeDetector().detect(img, DEFAULT_PARAMS_A)
        assert r.count >= 1
        x1, y1, x2, y2 = r.bboxes[0]
        edge = r.annotated_image[y1, x1:x2]
        assert (edge[:, 2] > 200).all() and (edge[:, 0] < 60).all()

    @pytest.mark.parametrize("shape", [(20, 20, 3), (60, 60, 3), (1, 500, 3)])
    def test_tiny_images_do_not_crash(self, shape):
        r = PotholeDetector().detect(np.zeros(shape, np.uint8))
        assert r.count == 0

    @pytest.mark.parametrize("value", [0, 255])
    @BOTH
    def test_solid_black_and_white(self, value, params):
        img = np.full((300, 300, 3), value, np.uint8)
        assert PotholeDetector().detect(img, params).count == 0

    def test_non_square_image(self):
        img = add_pothole(make_road(300, 560), 280, 150, 70, 45)
        assert PotholeDetector().detect(img, DEFAULT_PARAMS_A).count >= 1

    @pytest.mark.xfail(strict=True, reason=(
        "LIMITATION: percentile threshold always keeps the top 15% of pixels. "
        "At 1280x720 the pothole is ~3% of the frame, so road texture fills "
        "the rest; closing merges it into one 73%-of-frame blob that the "
        "max_area filter rejects, losing the pothole."))
    def test_large_hd_frame_detects_pothole(self):
        img = add_pothole(make_road(720, 1280), 640, 400, 120, 70)
        assert PotholeDetector().detect(img, DEFAULT_PARAMS_A).count >= 1

    @pytest.mark.xfail(strict=True, reason=(
        "BUG: border_margin=0 → arr[-0:, :] = 0 zeroes the WHOLE map, so "
        "nothing can ever be detected. UI slider min is 10 so UI is safe."))
    def test_border_margin_zero_still_detects(self, road_with_pothole):
        img, _ = road_with_pothole
        r = PotholeDetector().detect(img, dict(DEFAULT_PARAMS_A, border_margin=0))
        assert r.count >= 1
