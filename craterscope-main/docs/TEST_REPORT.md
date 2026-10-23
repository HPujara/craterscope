# Test Report

**Date:** 18 Oct 2026 · **Environment:** Windows 11, Python 3.13.1, OpenCV 4.11.0, NumPy 2.3.3, Streamlit 1.65.0, Playwright 1.60 (Edge)

## Summary

| Suite | File | Tests | Result |
|---|---|---|---|
| Evaluation maths & XML parsing (original) | `tests/test_evaluation.py` | 22 | ✅ 22 passed |
| Pipeline stage unit tests | `tests/test_detector_unit.py` | 50 | ✅ 48 passed · ⚠️ 2 known issues |
| Robustness / behaviour (synthetic scenes) | `tests/test_robustness.py` | 28 | ✅ 21 passed · ⚠️ 7 known issues |
| Video I/O + performance | `tests/test_video_and_performance.py` | 9 | ✅ 7 passed · ⚠️ 2 known issues |
| Real dataset integrity + end-to-end | `tests/test_dataset.py` | 9 | ✅ 9 passed |
| Streamlit UI (headless AppTest) | `tests/test_app_ui.py` | 22 | ✅ 20 passed · ⚠️ 2 known issues |
| Browser end-to-end (real upload, Edge) | `e2e/ui_e2e.py` | 17 checks | ✅ 16 passed · ⚠️ 1 known issue |
| **Total** | | **140 + 17** | **127 passed, 13 expected-fail · E2E 16 passed, 1 known issue** |

"Known issue" tests are marked `xfail(strict=True)`: they encode the *correct* behaviour, fail today,
and will alert you (XPASS → failure) if a future change fixes them.

```bash
python -m pytest tests/ -v          # ~3.5 min with the dataset, ~40 s without
python e2e/ui_e2e.py                # browser test + regenerates docs/screenshots/
```

## What was tested

**Algorithm (unit):** `_odd`, normalisation, solidity, contour→bbox; every region-filter rule in isolation
(area, aspect, solidity, panoramic band, brightness); dark map is one-sided (dark only); Sobel edge density
is higher on texture; border masking; CLAHE raises local contrast; morphological gradient equals
dilate − erode exactly; percentile threshold keeps ≈15 %; masks are binary; API contract (shapes, dtypes,
bboxes inside image, determinism, input not mutated, default = Method A); tiny / black / white / non-square images.

**Robustness:** localisation on synthetic potholes, two separate potholes, clean road, horizontal-flip
equivariance, brightness ±50, reduced contrast, JPEG q=25–90, Gaussian noise σ=10/25, colour cast,
shadows, lane markings, sky band, and monotonic effect of each parameter.

**Video:** encode → `VideoCapture` → detect → `VideoWriter` round-trip keeps frame count, size and fps;
a moving pothole is followed down the frame; static scenes give stable detections; corrupt files are rejected.

**Dataset:** 532/133 split sizes, no train/test leakage, all 665 XMLs parse, every box valid and inside its
image, XML size matches the JPEG, `results/test_metrics.json` is **exactly reproducible** (TP 33, FP 299, FN 266),
and both methods beat a trivial "centre box" baseline.

**UI/UX:** title, disclaimer, 3 tabs, prompts, sidebar sections; slider defaults match `DEFAULT_PARAMS_*`;
darkness weight = 1 − edge weight; switching to Method B swaps controls; extreme slider values don't crash;
evaluation metrics match the JSON; missing-metrics warning; real browser upload of image and video,
7 intermediate stages, download button, corrupt-file error message, stale results cleared, no JS errors, no horizontal overflow at 390 px, sidebar behaviour on phones.

## Findings

| # | Severity | Finding | Evidence |
|---|---|---|---|
| 1 | High (accuracy) | Boxes are inflated by the σ=15 density blur + 13×13 closing; a perfectly placed synthetic pothole only reaches IoU ≈ 0.39 (< 0.5 TP threshold). Main reason test F1 is 0.105. | `test_box_tight_enough_for_iou_05` |
| 2 | High (accuracy) | Percentile threshold always keeps the top 15 % of pixels, even on a clean road. Method B gives ~7 false alarms on a pothole-free road; on 1280×720 frames the real pothole is merged into a 73 %-of-frame blob and lost. | `test_clean_road_no_detection_method_b`, `test_large_hd_frame_detects_pothole` |
| 3 | Medium | Smooth shadows and white lane markings are detected as potholes (both methods). | `TestFalsePositiveSources` |
| 4 | Medium (perf) | Method A ≈ 340 ms per 720×720 image, ≈ 3 fps at 480p. 211 ms of that is one σ=50 `GaussianBlur` on uint8. Downsample ×¼ → blur → upsample is 7.8 ms with ≤ 1 grey-level difference. | `TestPerformance` |
| 5 | Low (bug) | `border_margin=0` zeroes the whole feature map (`arr[-0:] = 0`). Not reachable from the UI (slider min 10). | `test_border_margin_zero_still_detects` |
| 6 | Low (UI) | Percentile slider defaults to 85 for Method B; tuned value is 87. | `test_method_b_percentile_default` |
| 7 | Low (UI text) | Header says "adaptive thresholding"; the code uses percentile thresholding. README's Method B table describes Canny, but code uses morphological gradient. `pipeline_stages.png` labels Method A's threshold as "Adaptive Thresh". | `test_header_describes_actual_methods` |
| 8 | Low (maintenance) | `use_container_width` is deprecated in Streamlit (warned on every image), and `requirements.txt` has no upper bound. | AppTest log |
| 9 | UX | Video tab gives no preview of the processed video, only a download; detections across frames aren't smoothed (24-frame clip of one pothole → 69 boxes). Stage numbering skips "5" for Method A. No built-in sample image to try. | E2E screenshots |

Things that worked well: robust to brightness, contrast, colour cast and JPEG compression; exactly
flip-equivariant; deterministic; never crashes on degenerate input; friendly error on corrupt uploads;
layout usable at phone width; saved metrics reproducible bit-for-bit.

### Additional UX finding

| # | Severity | Finding | Evidence |
|---|---|---|---|
| 10 | UX (mobile) | On a 390 px phone the sidebar opens over the page (300 px wide) and hides the upload area, because `st.set_page_config(initial_sidebar_state="expanded")`. Using `"auto"` collapses it on small screens. Also, after a corrupt upload the previous result stays visible next to the error for 3–6 s until the rerun finishes. | `e2e/ui_e2e.py`, `docs/screenshots/09_mobile.png` |
