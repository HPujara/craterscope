<div align="center">

# 🕳️ CraterScope

### Pothole Detection with Classical Image & Video Processing

**Detect potholes in road images and videos using only hand-crafted OpenCV operations: no AI, ML or deep learning.**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-app-FF4B4B?logo=streamlit&logoColor=white)
![Tests](https://img.shields.io/badge/tests-127%20passed-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)

<img src="docs/screenshots/02_image_detection.png" alt="Image detection in the Streamlit app" width="850">

</div>

---

## Overview

This is an academic **Image & Video Processing (IVP)** project. A road photo or video frame goes through a fully explainable pipeline (smoothing, multi-scale background subtraction, Sobel edge density, percentile thresholding, morphology and contour analysis) and every detected pothole gets a red bounding box. Every step is a classical operation with a formula. Nothing is learned from data.

The project includes:

- an interactive **Streamlit web app** for images and videos, with live parameter sliders and every intermediate stage shown;
- **two detection methods**, compared on the dataset;
- an **evaluation framework** (Pascal VOC parsing, IoU, precision / recall / F1) that runs on the official test split;
- a **test suite** of 140 automated tests plus a browser end-to-end test.

## Screenshots

| Image detection | Intermediate stages (Method A) |
|---|---|
| ![](docs/screenshots/02_image_detection.png) | ![](docs/screenshots/03_intermediate_stages_A.png) |
| **Method B: CLAHE + morphological gradient** | **Video detection** |
| ![](docs/screenshots/04_intermediate_stages_B.png) | ![](docs/screenshots/06_video_detection.png) |
| **Evaluation dashboard** | **Error handling** |
| ![](docs/screenshots/07_evaluation.png) | ![](docs/screenshots/08_error_handling.png) |

<details>
<summary>More: home screen, Method B controls, phone layout</summary>

| Home | Method B sidebar | Phone (390 px) |
|---|---|---|
| ![](docs/screenshots/01_home.png) | ![](docs/screenshots/05_sidebar_method_b.png) | ![](docs/screenshots/09_mobile.png) |

</details>

### Offline result figures

| Pipeline stages | Qualitative test samples (green = ground truth, red = detection) |
|---|---|
| ![](results/pipeline_stages.png) | ![](results/qualitative_samples.png) |

| Method A vs B (training subset) | Dataset samples |
|---|---|
| ![](results/config_comparison.png) | ![](results/sample_grid.png) |

## Detection pipeline

### Method A: Edge density + multi-scale darkness (primary)

| # | Operation | IVP concept | Why |
|---|---|---|---|
| 1 | BGR → grayscale | Colour models | Potholes differ in brightness/texture, not hue |
| 2 | Gaussian blur 5×5 | Spatial smoothing | Suppress sensor noise |
| 3 | `max(0, Gσ * I − I)` for σ = 15, 30, 50 | Background subtraction, scale-space | Regions darker than their local surroundings |
| 4 | Sobel magnitude `√(Gx² + Gy²)` | First-order edge detection | Broken asphalt has many edges |
| 5 | Gaussian σ=15 of gradient magnitude | Texture / edge density | Region-level "roughness" map |
| 6 | `0.3·dark + 0.7·edge` | Image arithmetic, feature fusion | Edges are the stronger cue (85 % vs 46 %) |
| 7 | Keep pixels > 85th percentile | Histogram-based segmentation | Adapts to each image's contrast |
| 8 | Closing 13×13 ×2, opening 7×7 | Binary morphology | Merge fragments, remove specks |
| 9 | Contours + area / aspect / solidity / brightness / band filters | Region description | Reject shapes that aren't potholes |

### Method B: CLAHE + morphological gradient density (comparison)

Grayscale → Gaussian blur → **CLAHE** (clip 2.5, 8×8 tiles) → **morphological gradient** `(I ⊕ SE) − (I ⊖ SE)` with a 7×7 ellipse → border mask → density blur σ=12 → 87th-percentile threshold → closing / opening → the same contour filters. It is about 10–25× faster than Method A, but it raises more false alarms on plain road texture.

### Why edge density?

A statistical study of training images found that **85 %** of potholes have higher Sobel edge density than the road around them, while only **46 %** are darker. Texture is therefore the primary cue and darkness the secondary one.

## Results

Official **test split** (133 images, never used for tuning), Method A, a match counts at IoU ≥ 0.5:

| TP | FP | FN | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|
| 33 | 299 | 266 | 0.099 | 0.110 | 0.105 |

These numbers are low, and the tests show why: the blurred density map makes detected boxes about 45 % too large, so even a perfectly placed synthetic pothole only reaches IoU ≈ 0.39. The percentile threshold also always selects the top 15 % of pixels, even on a road with no pothole. Shadows, patches and lane markings look like potholes to low-level features. The full analysis is in [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) and [`report.md`](report.md).

## Quick start

```bash
git clone https://github.com/nishantparwani/craterscope.git
cd craterscope
pip install -r requirements.txt
streamlit run app.py              # opens http://localhost:8501
```

You can upload any road image or video without the dataset. The dataset is only needed for evaluation and dataset tests.

### Dataset (for evaluation)

Download [*A Fully Annotated Image Dataset for Pothole Detection*](https://www.kaggle.com/search?q=A+Fully+Annotated+Image+Dataset+for+Pothole+Detection) (665 images, Pascal VOC XML, official `splits.json`). Place its `archive/` folder **next to** this repository, or set an environment variable:

```bash
export POTHOLE_DATASET=/path/to/archive          # PowerShell: $env:POTHOLE_DATASET="C:\path\to\archive"
python preprocess.py                             # dataset statistics + sample grid
python run_evaluation.py                         # metrics + all figures in results/
```

## Testing

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v                       # 140 tests (dataset tests auto-skip if absent)
python -m playwright install chromium
python e2e/ui_e2e.py                             # real-browser test, regenerates screenshots
```

| Suite | Covers |
|---|---|
| `test_evaluation.py` | IoU, greedy matching, metrics, XML parsing |
| `test_detector_unit.py` | each pipeline stage, region filters, API contract, degenerate inputs |
| `test_robustness.py` | brightness, contrast, JPEG, noise, colour cast, flip, shadows, lane lines |
| `test_video_and_performance.py` | video round-trip, moving pothole, speed |
| `test_dataset.py` | split integrity, no leakage, reproducible metrics, beats baseline |
| `test_app_ui.py` | Streamlit layout, slider defaults, method switching, evaluation tab |
| `e2e/ui_e2e.py` | image and video upload in a real browser, error handling, phone layout |

Known limitations are recorded as `xfail` tests so they stay visible. Results: **127 passed, 13 known issues**. See the [test report](docs/TEST_REPORT.md).

## Project structure

```
├── app.py                 Streamlit web app (image, video, evaluation tabs)
├── detector.py            Classical pipelines: Method A, Method B, region filters
├── evaluation.py          Pascal VOC parsing, IoU, matching, precision / recall / F1
├── preprocess.py          Dataset statistics and sample grid
├── run_evaluation.py      Full offline evaluation and figure generation
├── report.md              Academic project report
├── FACULTY_GUIDE.md       Algorithm-by-algorithm guide (where / how / why)
├── docs/
│   ├── TEST_REPORT.md     Test results and findings
│   └── screenshots/       App screenshots (generated by e2e/ui_e2e.py)
├── results/               Test metrics (JSON) and result figures
├── tests/                 pytest suite
└── e2e/ui_e2e.py          Playwright browser test
```

## Limitations and future work

- **False positives:** shadows, road patches, stains and lane markings.
- **Loose boxes:** density smoothing inflates boxes. Re-thresholding inside each candidate would tighten them.
- **Always-on threshold:** a percentile threshold has no absolute floor. Adding one would let clean roads produce no detections.
- **No temporal context in video:** each frame is processed on its own. Tracking or temporal smoothing would reduce flicker.
- **Speed:** the σ=50 background blur dominates run time. Blurring a ¼-scale copy is about 27× faster with ≤1 grey-level difference.

## Team

Group project by a team of four:

| Member | Responsibility |
|---|---|
| Member 1 | Dataset analysis, annotation parsing, evaluation metrics |
| Member 2 | Method A: feature extraction (darkness, Sobel edge density, fusion, thresholding) |
| Member 3 | Method B, morphology, contour filtering, parameter tuning |
| Member 4 | Streamlit app, video pipeline, testing, documentation |

## License

The code is released under the [MIT License](LICENSE). The dataset is **not** included and keeps its own license. Images in `results/` and `docs/screenshots/` are derived from it for illustration.

## Acknowledgements

- Dataset: *A Fully Annotated Image Dataset for Pothole Detection* (Kaggle).
- [OpenCV](https://opencv.org/), [NumPy](https://numpy.org/), [Streamlit](https://streamlit.io/), [Matplotlib](https://matplotlib.org/).
- R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, the reference for the operations used here.
