"""
Shared pytest fixtures for the pothole-detection test suite.

Dataset-dependent tests are skipped automatically when the dataset is not
available (e.g. on a fresh clone). Point POTHOLE_DATASET at the "archive"
folder to enable them.
"""

import os
import sys

import cv2
import numpy as np
import pytest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

DATASET_ROOT = os.environ.get(
    "POTHOLE_DATASET", os.path.join(PROJECT_ROOT, "..", "archive")
)
ANNO_DIR = os.path.join(DATASET_ROOT, "annotated-images")
SPLITS_JSON = os.path.join(DATASET_ROOT, "splits.json")
HAS_DATASET = os.path.exists(SPLITS_JSON) and os.path.isdir(ANNO_DIR)

needs_dataset = pytest.mark.skipif(
    not HAS_DATASET, reason="Pothole dataset not found (set POTHOLE_DATASET)"
)


def make_road(h=360, w=480, seed=0, base=135, noise=6):
    """Smooth grey 'asphalt' with mild Gaussian noise."""
    rng = np.random.default_rng(seed)
    img = rng.normal(base, noise, (h, w)).clip(0, 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def add_pothole(img, cx, cy, rx, ry, seed=1, depth=70):
    """Paint a dark, rough elliptical 'pothole' (dark + high texture)."""
    rng = np.random.default_rng(seed)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
    texture = rng.integers(0, 90, img.shape[:2]).astype(np.int16)
    out = img.astype(np.int16)
    for c in range(3):
        ch = out[:, :, c]
        ch[mask > 0] = (ch[mask > 0] - depth + texture[mask > 0] - 45)
    return out.clip(0, 255).astype(np.uint8)


@pytest.fixture
def road():
    return make_road()


@pytest.fixture
def road_with_pothole():
    img = make_road()
    return add_pothole(img, 240, 200, 70, 45), (170, 155, 310, 245)
