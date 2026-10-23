# Academic Project Report
## Pothole Detection Using Classical Image Processing
### Subject: Image & Video Processing (IVP)

---

## 1. Abstract

This project implements a pothole detection system using exclusively classical image-processing techniques — no AI, machine learning, or deep learning. A dataset of 665 annotated road images is used to develop and evaluate two detection configurations. The primary method uses edge density analysis (dominant cue: 85% of potholes show higher Sobel edge density) combined with multi-scale excess-darkness maps. Results are evaluated against ground-truth bounding boxes at IoU threshold 0.5, and quantitative metrics are reported on the official test split.

---

## 2. Problem Statement

Road potholes pose safety hazards. Automatic detection from road-facing cameras is commercially useful. This project explores how far **classical image processing** alone can go, without any learning-based model, using only hand-crafted features and rules derived from image statistics.

The challenge is significant: unlike simple object detection problems, potholes are highly variable in appearance, size, colour, and lighting conditions.

---

## 3. Dataset Description

| Property | Value |
|---|---|
| Source | Annotated road-image dataset |
| Total images | 665 JPG files |
| Annotation format | Pascal VOC XML (bounding boxes) |
| Train / Test split | 532 / 133 (80% / 20%) |
| Image dimensions | Width 300–720 px, height 251–720 px |
| Bboxes per image | Min 1, max 19, mean 2.62 |
| Total bounding boxes | 1,740 |
| Multi-bbox images | 381 of 665 |

The dataset was released with a fixed train/test split (`splits.json`). This split was preserved throughout; the test set was never used during parameter development.

---

## 4. Statistical Analysis of Pothole Appearance

Before designing the pipeline, we computed statistics across 80 training images to understand what makes potholes distinguishable from road asphalt. For each annotated pothole, we compared the GT region to the surrounding road pixels.

| Feature | Potholes vs Road | % images where cue is reliable |
|---|---|---|
| Intensity (darker) | mean diff = +2.8 (road − pothole) | 46% |
| Local std. deviation (more variable) | mean diff = −2.4 (pothole − road) | 37% |
| **Sobel edge density** | **mean diff = +28.1 Sobel units** | **85%** |

**Key finding:** Potholes are NOT consistently darker than road — only 46% are distinctly darker. However, 85% of potholes show significantly higher edge density (Sobel gradient magnitude). This is because pothole surfaces contain irregular cracks, exposed aggregate, and broken asphalt edges, all of which create strong local intensity gradients.

**Implication:** Edge density is the primary and most reliable classical feature for pothole detection in this dataset. Darkness is a supplementary cue only.

---

## 5. Pipeline Architecture

### Pipeline Flowchart (Method A — Primary)

```
Input BGR Image
       │
       ▼
  Grayscale Conversion
  (cv2.cvtColor BGR→GRAY)
       │
       ▼
  Gaussian Blur (5×5)
  [suppress sensor noise]
       │
       ├──────────────────────────────────────┐
       │                                      │
       ▼                                      ▼
Multi-scale Excess Darkness          Sobel Gradient Magnitude
dark(σ) = clip(BG_σ − gray, 0, 255)  |∇I| = √(Gx² + Gy²)
σ ∈ {15, 30, 50}                     with border masking
dark_combined = max across σ          │
       │                              ▼
       │                     Gaussian Density Smoothing
       │                     density = GaussBlur(|∇I|, σ=15)
       │                              │
       ▼                              ▼
  Normalise to [0,255]        Normalise to [0,255]
       │                              │
       └──────────┬───────────────────┘
                  │
                  ▼
        Weighted Combination
        score = 0.3×dark + 0.7×edges
        [border region zeroed]
                  │
                  ▼
        Percentile Threshold
        top (100−P)% most anomalous pixels
                  │
                  ▼
       Morphological Closing (13×13, 2 iter)
       [fills gaps within detected blobs]
                  │
                  ▼
       Morphological Opening (7×7, 1 iter)
       [removes thin noise protrusions]
                  │
                  ▼
       Border Re-masking
       [removes morphological edge artifacts]
                  │
                  ▼
       Contour Extraction (RETR_EXTERNAL)
                  │
                  ▼
       Geometric Filtering:
       area ∈ [800, 35% of image]
       aspect ratio ∈ [0.15, 8.0]
       solidity ≥ 0.15
       mean brightness ≤ 180 (reject sky)
       no panoramic band (w>75%∧h<20%)
                  │
                  ▼
       Bounding Box Extraction + Drawing
       (red boxes overlaid on original image)
                  │
                  ▼
          Output: Detected Regions
```

---

## 6. Mathematical Principles

### 6.1 Gaussian Blur
For an image I and kernel size k with standard deviation σ:
```
G(x,y) = (1 / 2πσ²) × exp(−(x²+y²) / 2σ²)
Blurred = I ⊛ G
```
Smoothing reduces high-frequency noise before edge computation.

### 6.2 Multi-scale Excess Darkness
The background model at scale σ estimates the expected intensity at each pixel's neighbourhood:
```
background_σ(x,y) = GaussianBlur(I, σ)(x,y)
dark_σ(x,y) = max(0, background_σ(x,y) − I(x,y))
```
Using three scales (σ = 15, 30, 50) captures potholes of different physical sizes. The maximum response selects the most relevant scale.

### 6.3 Sobel Edge Magnitude
First-order partial derivatives using 3×3 Sobel kernels:
```
Gx = [−1, 0, +1; −2, 0, +2; −1, 0, +1] ⊛ I
Gy = [−1, −2, −1; 0, 0, 0; +1, +2, +1] ⊛ I
|∇I|(x,y) = √(Gx(x,y)² + Gy(x,y)²)
```

### 6.4 Edge Density Map
The Sobel magnitude gives per-pixel edge strength. To capture *regional* edge density, we smooth the magnitude map:
```
density(x,y) = GaussianBlur(|∇I|, σ_d)(x,y)
```
With σ_d = 15, each pixel's density score reflects edge activity within approximately a 45×45 pixel neighbourhood.

### 6.5 Morphological Closing and Opening

**Structuring element** B (k×k rectangle):
```
Closing: (I • B) = (I ⊕ B) ⊖ B   [Dilation then erosion]
Opening: (I ∘ B) = (I ⊖ B) ⊕ B   [Erosion then dilation]
```
Closing fills small holes and gaps within bright regions.
Opening removes thin noise protrusions and isolated small blobs.

### 6.6 Bounding Box IoU
For predicted box P = (x1,y1,x2,y2) and GT box G:
```
intersection = max(0, min(x2P,x2G)−max(x1P,x1G)) × max(0, min(y2P,y2G)−max(y1P,y1G))
union = area(P) + area(G) − intersection
IoU = intersection / union
```
A prediction is TP if IoU ≥ 0.5 with an unmatched GT box.

---

## 7. Configuration Comparison

Two configurations were compared on a 50-image subset of the training split (IoU@0.5):

| Method | Precision | Recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| **A: Edge Density + Multi-scale Darkness** | **0.184** | **0.180** | **0.182** | **23** | **102** | **105** |
| B: CLAHE + Morphological Gradient | 0.121 | 0.125 | 0.123 | 16 | 116 | 112 |

Method A was selected as the primary method based on training-set performance.

**Why A outperforms B:** Method A uses Sobel edge magnitude (continuous gradient) which is more noise-tolerant. Method B uses the morphological gradient which captures max local intensity range — effective but more sensitive to fine texture variations. Method A's multi-scale darkness provides a complementary cue that B lacks.

---

## 8. Final Test-Set Results

Evaluation on the **official test split** (133 images, never used during parameter tuning):

| Metric | Value |
|---|---|
| Images processed | 133 |
| Images skipped | 0 |
| True Positives (TP) | 33 |
| False Positives (FP) | 299 |
| False Negatives (FN) | 266 |
| **Precision** | **0.099** |
| **Recall** | **0.110** |
| **F1-Score** | **0.105** |
| IoU threshold | 0.5 |

**Important note:** These are bounding-box detection metrics at IoU@0.5. They measure whether a predicted bounding box spatially overlaps the annotated ground truth sufficiently. This is NOT pixel-level segmentation IoU.

---

## 9. False Positives and False Negatives Analysis

### False Positives
- **Sky/horizon regions:** High edge density at the horizon line where road meets sky
- **Road markings:** White/yellow lines have strong edges but low solidity (linear shape)
- **Windshield reflections or dashboard:** If visible in image
- **Multiple detections on one pothole:** Same pothole matched twice at different scales

### False Negatives
- **Low-contrast potholes:** Potholes nearly identical in texture/colour to smooth asphalt (15% by dataset statistics)
- **Very small potholes:** Below the min_area threshold
- **Filled/patched potholes:** Recently repaired areas may appear uniformly textured
- **Overexposed images:** Bright lighting flattens the edge density gradient

### Limitations
1. **No adaptivity:** Parameters were tuned on training data but are fixed at inference. Different lighting conditions, camera angles, and road types degrade performance.
2. **No temporal context:** Each frame in a video is processed independently; temporal consistency is not exploited.
3. **IoU sensitivity:** Even a good detection that is 45% spatially overlapping counts as FN.
4. **Scale assumption:** The edge density sigma (15) is tuned for typical dataset image sizes. Different camera mounting heights would require re-tuning.
5. **No feature learning:** Classical methods cannot adapt to the specific appearance distribution of a new environment.

---

## 10. Comparison: Classical vs ML Approaches (Academic Discussion)

| Aspect | Classical (this project) | Deep Learning (out of scope) |
|---|---|---|
| Training data | None needed for detection | Requires thousands of labelled images |
| Computational cost | Low (real-time on CPU) | High (GPU typically required) |
| Explainability | Full (every step is interpretable) | Limited (black-box weights) |
| Adaptability | Must re-tune for new environments | Learns general features automatically |
| Typical F1 (literature) | 0.10–0.35 on real datasets | 0.55–0.85 on similar datasets |
| This project | F1=0.105 (test), 0.182 (train subset) | N/A |
| Academic scope | IVP fundamentals | Advanced ML / computer vision |

---

## 11. Possible Improvements (Classical Methods Only)

1. **Multi-scale approach:** Run detection at multiple image scales and merge results (image pyramid)
2. **Temporal smoothing (video):** Average detection masks across 3–5 consecutive frames to reduce noise
3. **Road segmentation:** Use colour/texture segmentation to restrict detection to the road surface, excluding sky and vehicle body
4. **Adaptive thresholding:** Instead of a global percentile, compute threshold locally per image strip
5. **Shape refinement:** Use active contours (snakes) to refine bounding boxes after initial detection
6. **Ensemble:** Run both Method A and B and use the union of their detections (higher recall at cost of precision)

---

## 12. Viva Preparation — Questions and Answers

**Q1: Why is edge density the primary feature instead of colour/darkness?**
A: Statistical analysis on 80 training images showed that 85% of potholes have higher Sobel edge density than surrounding asphalt, with a mean difference of +28.1 units. In contrast, only 46% of potholes are distinctly darker. Potholes contain irregular cracks, exposed aggregate, and broken edges — all of which produce strong intensity gradients. Smooth asphalt has few edges.

**Q2: What is CLAHE and why is it used in Method B?**
A: CLAHE = Contrast Limited Adaptive Histogram Equalization. It equalises the histogram within small image tiles rather than globally, enhancing local contrast. The clip_limit prevents over-amplification of noise by capping the histogram slope. It makes locally dark/textured pothole regions more distinguishable from surrounding asphalt.

**Q3: What is morphological closing? How does it help here?**
A: Closing = dilation followed by erosion using the same structuring element. Dilation expands white regions (filling small gaps), then erosion restores the original sizes while keeping the filled gaps. In our pipeline, closing merges nearby edge-density blobs that belong to the same pothole into one solid region, enabling correct contour extraction.

**Q4: Why is IoU@0.5 used instead of IoU@0.1?**
A: IoU@0.5 is the standard PASCAL VOC criterion. It requires that the detected box overlaps the ground-truth box by at least 50% of their union. This is a meaningful threshold: a box with IoU=0.49 roughly represents a box shifted half a pothole-width away, which would be a misleading detection in practice. Stricter thresholds (0.75+) are used in more demanding benchmarks.

**Q5: What is solidity and why filter on it?**
A: Solidity = contour area / convex hull area. A convex shape (e.g., rectangle, circle) has solidity ≈ 1.0. A very irregular shape (thin spiky contour) has low solidity. We filter out contours with solidity < 0.15 to remove hairline artefacts (thin edge fragments that form complex non-blob shapes). Potholes are expected to be moderately compact.

**Q6: How are predicted contours converted to bounding boxes for evaluation?**
A: We use `cv2.boundingRect(contour)` which returns the smallest axis-aligned rectangle enclosing the contour. The result (x, y, w, h) is converted to (xmin, ymin, xmax, ymax) = (x, y, x+w, y+h). This matches the Pascal VOC annotation convention.

**Q7: What is the difference between pixel-level IoU and bounding-box IoU?**
A: Pixel-level (segmentation) IoU compares the exact set of predicted pixels to the exact ground-truth pixel mask. Bounding-box IoU compares only the enclosing rectangles. Since our annotations are bounding boxes (not pixel masks), we use bounding-box IoU. Bounding-box IoU is a looser measure — a prediction can have high bbox IoU but low pixel-level IoU if the pothole doesn't fill its bounding box.

**Q8: How is the Canny edge detector different from the Sobel operator?**
A: Sobel computes a smooth gradient magnitude at each pixel — a continuous edge strength map. Canny is a multi-stage algorithm: Sobel-like gradient computation → non-maximum suppression (thinning edges to 1 pixel) → double hysteresis thresholding (strong/weak edges) → edge tracking. Canny produces binary edge maps that are thin, well-localised, and connected. Sobel is faster and produces a graded response useful for density maps.

**Q9: Why are parameters tuned on training data only?**
A: To prevent test-set data leakage. If we tuned parameters using test images, our reported metrics would overfit to those specific images and would not generalise. This is analogous to fitting model hyperparameters on validation data in machine learning. The test set should only be used once, at the end.

**Q10: Why is classical detection limited on this dataset?**
A: This dataset has diverse road images with varying lighting, camera angles, and pothole types. Some potholes are nearly indistinguishable from the surrounding asphalt by texture or colour alone. Classical methods use fixed, hand-designed rules that cannot adapt to this diversity. Without learning, we cannot discover the complex, non-linear combination of features that distinguishes potholes from all other road-surface anomalies.

---

## 13. References

1. OpenCV Documentation — Morphological Operations, Canny Edge, CLAHE
2. Canny, J. (1986). A computational approach to edge detection. IEEE TPAMI.
3. Soille, P. (2003). Morphological Image Analysis.
4. Everingham et al. (2010). The PASCAL VOC Challenge. IJCV. — IoU evaluation standard.
5. Koch, C. & Brilakis, I. (2011). Pothole detection in asphalt pavement images. Advanced Engineering Informatics. — Classical approach reference.
