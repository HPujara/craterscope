"""
detector.py — Classical image-processing pothole detection pipeline.

Dataset analysis findings (80 training-image sample):
  - 85% of potholes have higher Sobel edge density than surrounding road.
    Mean edge difference: +28.1 Sobel units.
  - Only 46% of potholes are darker than road (intensity diff < -5).
  - 37% have higher local standard deviation.

Conclusion: EDGE DENSITY is the primary detection cue for this dataset.
Multi-scale local darkness is a secondary cue.

Two configurations are implemented:
  A  — Multi-scale darkness + edge density (primary, recommended)
  B  — CLAHE + Canny edge map (alternative)
"""

import cv2
import numpy as np
import time


# ──────────────────────────────────────────────────────────────────────────────
# Default parameter sets
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_PARAMS_A = {
    "method": "density_combined",
    "blur_ksize": 5,

    # Multi-scale excess-darkness (σ values for background estimation)
    "dark_sigmas": [15, 30, 50],
    "dark_weight": 0.3,       # Weight for darkness feature (secondary cue)

    # Edge density (Sobel magnitude smoothed by a large Gaussian)
    "edge_sigma": 15,         # Sigma for density Gaussian on Sobel magnitude
    "edge_weight": 0.7,       # Weight for edge density (primary cue — 85% reliable)

    # Thresholding (tuned on training split, parameter sweep result)
    "percentile": 85,         # Keep top 15% pixels by combined score

    # Morphological operations
    "morph_close_ksize": 13,  # Closing kernel (fills gaps between edges)
    "morph_close_iter": 2,
    "morph_open_ksize": 7,    # Opening kernel (removes thin noise)
    "morph_open_iter": 1,

    # Region filters
    "min_area": 1500,
    "max_area_ratio": 0.35,
    "min_aspect": 0.15,
    "max_aspect": 8.0,
    "min_solidity": 0.15,
    "border_margin": 25,      # Zero image border to prevent boundary artifacts
    "max_brightness": 175,    # Reject sky/bright regions (mean gray > threshold)
}

DEFAULT_PARAMS_B = {
    "method": "morph_gradient",
    "blur_ksize": 5,

    # CLAHE (applied before gradient for contrast enhancement)
    "clahe_clip": 2.5,
    "clahe_tile": 8,

    # Morphological gradient SE size
    "grad_ksize": 7,

    # Density smoothing sigma
    "density_sigma": 12,

    # Thresholding (tuned on training split)
    "percentile": 87,

    # Morphological operations
    "morph_close_ksize": 13,
    "morph_close_iter": 2,
    "morph_open_ksize": 7,
    "morph_open_iter": 1,

    # Region filters
    "min_area": 1500,
    "max_area_ratio": 0.35,
    "min_aspect": 0.15,
    "max_aspect": 8.0,
    "min_solidity": 0.15,
    "border_margin": 25,
    "max_brightness": 175,
}


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def _odd(n: int) -> int:
    n = max(1, int(n))
    return n if n % 2 == 1 else n + 1


def contour_to_bbox(cnt):
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, x + w, y + h)


def _solidity(cnt) -> float:
    area = cv2.contourArea(cnt)
    if area == 0:
        return 0.0
    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    return area / hull_area if hull_area > 0 else 0.0


def _filter_contours(contours, img_h, img_w, params, gray_image=None):
    """
    Discard contours unlikely to be potholes:
      - area: too small (noise) or too large (entire road/sky captured)
      - aspect ratio: extremely elongated shapes (road markings)
      - solidity: near-zero = hairline fragments, not blob regions
      - brightness: sky and bright surfaces (mean gray > max_brightness)
      - panoramic band: very wide and very thin = horizon line, not pothole
    """
    img_area = img_h * img_w
    min_area = params.get("min_area", 1500)
    max_area = params.get("max_area_ratio", 0.35) * img_area
    min_asp = params.get("min_aspect", 0.15)
    max_asp = params.get("max_aspect", 8.0)
    min_sol = params.get("min_solidity", 0.15)
    max_brightness = params.get("max_brightness", 175)

    passed = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        asp = w / h if h > 0 else 0
        if asp < min_asp or asp > max_asp:
            continue
        sol = _solidity(cnt)
        if sol < min_sol:
            continue
        # Reject panoramic horizontal bands (horizon/road-line artifacts)
        if w > 0.75 * img_w and h < 0.20 * img_h:
            continue
        # Reject bright regions (sky, reflections) using original gray
        if gray_image is not None:
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            cv2.drawContours(mask, [cnt], -1, 255, -1)
            mean_val = cv2.mean(gray_image, mask=mask)[0]
            if mean_val > max_brightness:
                continue
        passed.append((cnt, contour_to_bbox(cnt)))
    return passed


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    """Normalise float array to 0–255 uint8."""
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - mn) / (mx - mn) * 255).astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Method A: Multi-scale Darkness + Edge Density
# ──────────────────────────────────────────────────────────────────────────────

def _pipeline_density_combined(bgr_image, params):
    """
    Method A — Primary pipeline.

    Rationale (from statistical analysis of 80 training images):
      - 85% of potholes show higher Sobel edge density than surrounding asphalt.
        This is because potholes have irregular cracks, exposed aggregate, and
        rough texture — more edges per unit area than smooth road surface.
        Mean Sobel edge-density difference = +28.1 units.
      - Only 46% of potholes are darker than the surrounding road. Darkness
        is therefore an unreliable standalone feature, used here as a supplement.

    Pipeline:
    1. Grayscale + Gaussian blur — reduce colour to luminance; suppress noise.
    2. Multi-scale excess darkness —
         dark(σ) = max(0, GaussianBlur(gray, σ) − gray)
       At each scale σ, regions darker than their neighbourhood by ≥1 level are
       captured. Max across scales handles potholes of different sizes.
       Mathematically: background_model(σ) ≈ local mean at radius ≈ 3σ.
    3. Sobel magnitude — |∇I| = √(Gx²+Gy²). Measures first-order gradient
       magnitude (edge strength) at each pixel.
    4. Edge density — GaussianBlur(|∇I|, edge_sigma) creates a smoothed
       'density map' where high values indicate many edges nearby.
       Border pixels are zeroed to prevent image-boundary artifacts.
    5. Normalise + weighted combination — darkness and edge density are each
       normalised to [0,255] then blended: score = 0.3*dark + 0.7*edges.
       Edge weight dominates because it is more statistically reliable.
    6. Percentile threshold — pixels above the (100−P)th percentile of the
       combined score form the candidate binary mask.
    7. Morphological closing — dilation followed by erosion using a rectangular
       SE. Fills small gaps within candidate blobs, merging nearby fragments.
    8. Morphological opening — erosion then dilation. Removes thin noise
       protrusions while preserving compact blobs.
    9. Contour extraction + geometric filtering.
    """
    p = params
    h, w = bgr_image.shape[:2]

    # Step 1
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    ksize = _odd(p.get("blur_ksize", 5))
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Step 2 — multi-scale excess darkness
    dark_maps = []
    for sigma in p.get("dark_sigmas", [15, 30, 50]):
        bg = cv2.GaussianBlur(blurred, (0, 0), float(sigma)).astype(np.int16)
        dark = np.clip(bg - blurred.astype(np.int16), 0, 255).astype(np.uint8)
        dark_maps.append(dark)
    dark_combined = np.maximum.reduce(dark_maps).astype(np.float32)

    # Step 3 — Sobel magnitude
    sx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sx**2 + sy**2).astype(np.float32)

    # Zero image border to avoid boundary artifacts
    m = int(p.get("border_margin", 15))
    mag[:m, :] = 0; mag[-m:, :] = 0; mag[:, :m] = 0; mag[:, -m:] = 0

    # Step 4 — edge density
    edge_sigma = float(p.get("edge_sigma", 15))
    density = cv2.GaussianBlur(mag, (0, 0), edge_sigma)

    # Step 5 — normalise + combine
    dark_n = _normalize_to_uint8(dark_combined)
    edge_n = _normalize_to_uint8(density)
    dw = float(p.get("dark_weight", 0.3))
    ew = float(p.get("edge_weight", 0.7))
    combined = cv2.addWeighted(dark_n, dw, edge_n, ew, 0)

    # Zero image border in combined map too (prevents morphological extension to edges)
    combined[:m, :] = 0; combined[-m:, :] = 0
    combined[:, :m] = 0; combined[:, -m:] = 0

    # Step 6 — threshold
    pct = float(p.get("percentile", 85))
    tv = float(np.percentile(combined, pct))
    _, thresh = cv2.threshold(combined, tv, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)

    # Step 7 — morphological closing
    ck = _odd(p.get("morph_close_ksize", 13))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel,
                              iterations=int(p.get("morph_close_iter", 2)))

    # Step 8 — morphological opening
    ok = _odd(p.get("morph_open_ksize", 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel,
                              iterations=int(p.get("morph_open_iter", 1)))

    # Re-apply border mask after morphology to remove extension artifacts
    opened[:m, :] = 0; opened[-m:, :] = 0
    opened[:, :m] = 0; opened[:, -m:] = 0

    # Step 9 — contours + filter
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = _filter_contours(contours, h, w, p, gray_image=gray)

    intermediates = {
        "gray": gray,
        "blurred": blurred,
        "dark_map": dark_n,
        "edge_density": edge_n,
        "combined": combined,
        "thresh": thresh,
        "morphed": opened,
    }
    return detections, intermediates


# ──────────────────────────────────────────────────────────────────────────────
# Method B: CLAHE + Morphological Gradient Density
# ──────────────────────────────────────────────────────────────────────────────

def _pipeline_morph_gradient(bgr_image, params):
    """
    Method B — CLAHE + Morphological Gradient Density.

    The morphological gradient is defined as:
        gradient(f) = dilation(f, SE) − erosion(f, SE)

    It highlights boundaries between differently-valued regions.
    Unlike the Sobel operator (first derivative of a Gaussian-smoothed image),
    the morphological gradient:
      - Captures the maximum local range of intensity within the SE
      - Is more robust to salt-and-pepper noise
      - With a larger SE, captures coarser-scale boundaries

    Steps:
    1. Grayscale + Gaussian blur.
    2. CLAHE — enhances local contrast so that subtle pothole-road boundaries
       become more pronounced for the gradient operation.
    3. Morphological gradient with an elliptical SE (size = grad_ksize).
       gradient = dilate(enhanced, SE) − erode(enhanced, SE)
    4. Border masking — zero the margin pixels to prevent image-edge artifacts.
    5. Density smoothing — GaussianBlur(gradient, density_sigma) creates a
       smooth map of regions with many nearby boundaries (textured areas).
    6. Percentile threshold — keep top (100−P)% most boundary-dense regions.
    7. Morphological closing + opening.
    8. Contour filtering with geometric + brightness constraints.
    """
    p = params
    h, w = bgr_image.shape[:2]

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    ksize = _odd(p.get("blur_ksize", 5))
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # CLAHE — amplifies local contrast before gradient computation
    clahe = cv2.createCLAHE(
        clipLimit=float(p.get("clahe_clip", 2.5)),
        tileGridSize=(int(p.get("clahe_tile", 8)),
                      int(p.get("clahe_tile", 8)))
    )
    enhanced = clahe.apply(blurred)

    # Morphological gradient: captures intensity range within the SE window
    gk = _odd(p.get("grad_ksize", 7))
    grad_se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk))
    morph_grad = cv2.morphologyEx(enhanced, cv2.MORPH_GRADIENT, grad_se).astype(np.float32)

    # Border masking
    m = int(p.get("border_margin", 25))
    morph_grad[:m, :] = 0; morph_grad[-m:, :] = 0
    morph_grad[:, :m] = 0; morph_grad[:, -m:] = 0

    # Density smoothing
    ds = float(p.get("density_sigma", 12))
    density = cv2.GaussianBlur(morph_grad, (0, 0), ds)
    density[:m, :] = 0; density[-m:, :] = 0
    density[:, :m] = 0; density[:, -m:] = 0

    # Threshold
    pct = float(p.get("percentile", 87))
    tv = float(np.percentile(density, pct))
    _, thresh = cv2.threshold(density, tv, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)

    # Morphological closing
    ck = _odd(p.get("morph_close_ksize", 13))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ck, ck))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, close_kernel,
                              iterations=int(p.get("morph_close_iter", 2)))

    # Opening
    ok = _odd(p.get("morph_open_ksize", 7))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ok, ok))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, open_kernel,
                              iterations=int(p.get("morph_open_iter", 1)))

    # Re-apply border mask after morphology
    opened[:m, :] = 0; opened[-m:, :] = 0
    opened[:, :m] = 0; opened[:, -m:] = 0

    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = _filter_contours(contours, h, w, p, gray_image=gray)

    intermediates = {
        "gray": gray,
        "blurred": blurred,
        "enhanced": enhanced,
        "morph_gradient": morph_grad.astype(np.uint8),
        "density": density.astype(np.uint8),
        "thresh": thresh,
        "morphed": opened,
    }
    return detections, intermediates


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

class PotholeDetector:
    """
    Wraps the two classical detection configurations.

    Usage
    -----
    detector = PotholeDetector()
    result = detector.detect(bgr_image, params=DEFAULT_PARAMS_A)

    result.bboxes          → list of (xmin, ymin, xmax, ymax)
    result.annotated_image → BGR image with red boxes
    result.intermediates   → dict of intermediate single-channel images
    result.elapsed_ms      → float, wall-clock time in milliseconds
    result.count           → int, number of detected regions
    """

    class Result:
        def __init__(self, bboxes, contours, annotated_image, intermediates, elapsed_ms):
            self.bboxes = bboxes
            self.contours = contours
            self.annotated_image = annotated_image
            self.intermediates = intermediates
            self.elapsed_ms = elapsed_ms
            self.count = len(bboxes)

    def detect(self, bgr_image: np.ndarray, params: dict = None) -> "PotholeDetector.Result":
        """
        Run detection on a BGR image (as returned by cv2.imread).

        Parameters
        ----------
        bgr_image : np.ndarray  H×W×3 BGR image
        params    : dict        parameter set; defaults to DEFAULT_PARAMS_A

        Returns
        -------
        Result with bboxes, annotated image, and intermediates.
        """
        if params is None:
            params = DEFAULT_PARAMS_A.copy()

        t0 = time.perf_counter()
        method = params.get("method", "density_combined")

        if method == "morph_gradient":
            detections, intermediates = _pipeline_morph_gradient(bgr_image, params)
        else:
            detections, intermediates = _pipeline_density_combined(bgr_image, params)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        contours = [d[0] for d in detections]
        bboxes = [d[1] for d in detections]

        # Draw: red bounding boxes + semi-transparent red fill
        annotated = bgr_image.copy()
        overlay = annotated.copy()
        cv2.drawContours(overlay, contours, -1, (0, 0, 255), -1)
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)

        for (xmin, ymin, xmax, ymax) in bboxes:
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            cv2.putText(annotated, "pothole",
                        (xmin, max(ymin - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1,
                        cv2.LINE_AA)

        return PotholeDetector.Result(bboxes, contours, annotated, intermediates, elapsed_ms)
