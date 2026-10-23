"""
app.py — Streamlit web application for classical pothole detection.
No AI / machine learning / deep learning is used anywhere in this project.
All detection is performed using classical image-processing techniques only.

Run: streamlit run app.py
"""

import os
import io
import json
import time
import tempfile

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from detector import (
    PotholeDetector,
    DEFAULT_PARAMS_A,
    DEFAULT_PARAMS_B,
)

# ──────────────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pothole Detection — Classical IVP",
    page_icon="🕳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
METRICS_PATH = os.path.join(RESULTS_DIR, "test_metrics.json")

detector = PotholeDetector()


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar — parameter controls
# ──────────────────────────────────────────────────────────────────────────────

def build_sidebar():
    st.sidebar.title("Detection Parameters")
    st.sidebar.markdown(
        "_Adjust these to tune the classical image-processing pipeline._"
    )

    method = st.sidebar.selectbox(
        "Detection method",
        [
            "A: Edge Density + Multi-scale Darkness (recommended)",
            "B: CLAHE + Morphological Gradient Density",
        ],
    )
    use_method_a = method.startswith("A")

    st.sidebar.subheader("Pre-processing")
    blur = st.sidebar.slider("Blur kernel size (odd)", 3, 15, 5, step=2)

    if use_method_a:
        st.sidebar.subheader("Feature Weights")
        edge_w = st.sidebar.slider("Edge density weight", 0.0, 1.0, 0.7, 0.1)
        dark_w = round(1.0 - edge_w, 1)
        st.sidebar.caption(f"Darkness weight: {dark_w}")
        edge_sigma = st.sidebar.slider("Edge density sigma", 8, 30, 15, 1)
    else:
        st.sidebar.subheader("CLAHE")
        clahe_clip = st.sidebar.slider("CLAHE clip limit", 0.5, 6.0, 2.5, 0.5)
        st.sidebar.subheader("Gradient")
        grad_ksize = st.sidebar.slider("Morphological gradient size (odd)", 3, 15, 7, 2)
        density_sigma = st.sidebar.slider("Density sigma", 8, 25, 12, 1)

    st.sidebar.subheader("Thresholding")
    percentile = st.sidebar.slider("Percentile threshold", 70, 95, 85, 1)

    st.sidebar.subheader("Morphology")
    morph_close = st.sidebar.slider("Closing kernel size (odd)", 7, 21, 13, 2)
    morph_iter = st.sidebar.slider("Closing iterations", 1, 4, 2)

    st.sidebar.subheader("Region Filtering")
    min_area = st.sidebar.slider("Min contour area (px²)", 500, 5000, 1500, 100)
    min_solidity = st.sidebar.slider("Min solidity", 0.1, 0.6, 0.15, 0.05)
    max_brightness = st.sidebar.slider("Max mean brightness (sky filter)", 130, 220, 175)
    border_margin = st.sidebar.slider("Border mask (px)", 10, 40, 25)

    if use_method_a:
        params = {
            "method": "density_combined",
            "blur_ksize": blur,
            "dark_sigmas": [15, 30, 50],
            "dark_weight": dark_w,
            "edge_sigma": edge_sigma,
            "edge_weight": edge_w,
            "percentile": percentile,
            "morph_close_ksize": morph_close,
            "morph_close_iter": morph_iter,
            "morph_open_ksize": max(3, morph_close // 2),
            "morph_open_iter": 1,
            "min_area": min_area,
            "max_area_ratio": 0.35,
            "min_aspect": 0.15,
            "max_aspect": 8.0,
            "min_solidity": min_solidity,
            "border_margin": border_margin,
            "max_brightness": max_brightness,
        }
    else:
        params = {
            "method": "morph_gradient",
            "blur_ksize": blur,
            "clahe_clip": clahe_clip,
            "clahe_tile": 8,
            "grad_ksize": grad_ksize,
            "density_sigma": density_sigma,
            "percentile": percentile,
            "morph_close_ksize": morph_close,
            "morph_close_iter": morph_iter,
            "morph_open_ksize": max(3, morph_close // 2),
            "morph_open_iter": 1,
            "min_area": min_area,
            "max_area_ratio": 0.35,
            "min_aspect": 0.15,
            "max_aspect": 8.0,
            "min_solidity": min_solidity,
            "border_margin": border_margin,
            "max_brightness": max_brightness,
        }

    return params


# ──────────────────────────────────────────────────────────────────────────────
# Intermediate visualisation helper
# ──────────────────────────────────────────────────────────────────────────────

def show_intermediates(intermediates: dict):
    st.subheader("Intermediate Processing Stages")
    stage_map = {
        "gray":            "1. Grayscale",
        "blurred":         "2. Gaussian Blur",
        "dark_map":        "3a. Dark Map",
        "edge_density":    "3b. Edge Density",
        "combined":        "4. Combined Score",
        "enhanced":        "3. CLAHE Enhanced",
        "morph_gradient":  "4. Morph Gradient",
        "density":         "5. Density Map",
        "thresh":          "6. Threshold Mask",
        "morphed":         "7. After Morphology",
    }
    cols = st.columns(4)
    col_idx = 0
    for key, title in stage_map.items():
        img = intermediates.get(key)
        if img is None:
            continue
        with cols[col_idx % 4]:
            st.image(img, caption=title, use_container_width=True, clamp=True)
        col_idx += 1


# ──────────────────────────────────────────────────────────────────────────────
# Page: Image mode
# ──────────────────────────────────────────────────────────────────────────────

def page_image(params):
    st.header("Image Detection")
    st.markdown(
        "Upload a road image. The pipeline will detect likely pothole regions "
        "using **classical image processing only** (no AI/ML)."
    )

    uploaded = st.file_uploader(
        "Upload image (JPG / PNG)", type=["jpg", "jpeg", "png"]
    )
    if uploaded is None:
        st.info("Upload an image to begin.")
        return

    # Decode with OpenCV
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if bgr is None:
        st.error("Could not decode image. Please upload a valid JPG or PNG.")
        return

    with st.spinner("Running classical detection pipeline…"):
        result = detector.detect(bgr, params)

    rgb_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_annot = cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(rgb_orig, use_container_width=True)
    with col2:
        st.subheader(f"Detected ({result.count} region(s))")
        st.image(rgb_annot, use_container_width=True)

    # Metrics panel
    st.subheader("Detection Summary")
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Regions detected", result.count)
    mcol2.metric("Processing time", f"{result.elapsed_ms:.1f} ms")
    mcol3.metric("Method", params["method"].replace("_", " ").title())

    if result.bboxes:
        st.markdown("**Detected bounding boxes (xmin, ymin, xmax, ymax):**")
        for i, (x1, y1, x2, y2) in enumerate(result.bboxes):
            w, h = x2 - x1, y2 - y1
            area = w * h
            st.markdown(
                f"- Region {i+1}: ({x1}, {y1}, {x2}, {y2}) — "
                f"size {w}×{h}px, area {area} px²"
            )

    show_intermediates(result.intermediates)


# ──────────────────────────────────────────────────────────────────────────────
# Page: Video mode
# ──────────────────────────────────────────────────────────────────────────────

def page_video(params):
    st.header("Video Detection")
    st.markdown(
        "Upload a road video. Each frame is processed with the same classical "
        "pipeline. Detected potholes are highlighted with **red boxes**."
    )
    st.warning(
        "Processing is done frame-by-frame in Python. For a 10-second clip at "
        "30 fps this may take 30–90 seconds depending on resolution."
    )

    uploaded = st.file_uploader("Upload video (MP4 / AVI)", type=["mp4", "avi", "mov"])
    if uploaded is None:
        st.info("Upload a video to begin.")
        return

    # Write to temp file so OpenCV can open it
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_in:
        tmp_in.write(uploaded.read())
        tmp_in_path = tmp_in.name

    cap = cv2.VideoCapture(tmp_in_path)
    if not cap.isOpened():
        st.error("Could not open video file. Try a different format.")
        os.unlink(tmp_in_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.info(f"Video: {width}×{height} px, {total_frames} frames, {fps:.1f} fps")

    tmp_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp_out_path = tmp_out.name
    tmp_out.close()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out_path, fourcc, fps, (width, height))

    progress = st.progress(0)
    frame_idx = 0
    total_detections = 0
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.detect(frame, params)
        writer.write(result.annotated_image)
        total_detections += result.count
        frame_idx += 1
        if total_frames > 0:
            progress.progress(min(frame_idx / total_frames, 1.0))

    cap.release()
    writer.release()
    os.unlink(tmp_in_path)

    elapsed = time.time() - t_start
    st.success(
        f"Processed {frame_idx} frames in {elapsed:.1f}s "
        f"({frame_idx/elapsed:.1f} fps). "
        f"Total detections across all frames: {total_detections}."
    )

    with open(tmp_out_path, "rb") as vf:
        video_bytes = vf.read()
    os.unlink(tmp_out_path)

    st.download_button(
        "Download processed video",
        data=video_bytes,
        file_name="pothole_detected.mp4",
        mime="video/mp4",
    )


# ──────────────────────────────────────────────────────────────────────────────
# Page: Evaluation results
# ──────────────────────────────────────────────────────────────────────────────

def page_evaluation():
    st.header("Dataset Evaluation Results")
    st.markdown(
        """
        **Note:** All detection is heuristic (classical image processing).
        These are not AI predictions. Metrics reflect how well the pipeline's
        geometric rules align with human-annotated ground-truth bounding boxes.

        **Matching rule:** A predicted bounding box counts as a **True Positive (TP)**
        only when it overlaps an unmatched ground-truth box with
        **IoU ≥ 0.5** (bounding-box IoU, not pixel-level segmentation IoU).
        """
    )

    st.subheader("Confusion Count Definitions")
    col1, col2, col3 = st.columns(3)
    col1.info("**TP** — Predicted box matches a GT box at IoU ≥ 0.5")
    col2.warning("**FP** — Predicted box has no matching GT box (false alarm)")
    col3.error("**FN** — GT box has no matching predicted box (missed pothole)")

    st.subheader("Quantitative Metrics (Official Test Split)")

    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            m = json.load(f)

        mcols = st.columns(5)
        mcols[0].metric("Precision", f"{m['precision']:.3f}")
        mcols[1].metric("Recall", f"{m['recall']:.3f}")
        mcols[2].metric("F1-Score", f"{m['f1']:.3f}")
        mcols[3].metric("IoU Threshold", f"{m['iou_threshold']}")
        mcols[4].metric("Images evaluated", m['n_processed'])

        st.markdown(
            f"TP: **{m['tp']}** | FP: **{m['fp']}** | FN: **{m['fn']}** | "
            f"Skipped: **{m['n_skipped']}**"
        )

        # Comparison chart if available
        comp_path = os.path.join(RESULTS_DIR, "config_comparison.png")
        if os.path.exists(comp_path):
            st.subheader("Method A vs Method B Comparison (Training Subset)")
            st.image(comp_path, use_container_width=True)

        # Qualitative samples
        qual_path = os.path.join(RESULTS_DIR, "qualitative_samples.png")
        if os.path.exists(qual_path):
            st.subheader("Qualitative Samples")
            st.image(qual_path, use_container_width=True)

        # Pipeline stages
        pipe_path = os.path.join(RESULTS_DIR, "pipeline_stages.png")
        if os.path.exists(pipe_path):
            st.subheader("Pipeline Intermediate Stages")
            st.image(pipe_path, use_container_width=True)

    else:
        st.warning(
            "No evaluation results found. "
            "Run `python run_evaluation.py` first to generate metrics."
        )

    st.subheader("Limitations")
    st.markdown(
        """
        - **False positives:** Shadows, road patches, painted markings, and stains
          can appear as dark irregular regions and trigger false detections.
        - **False negatives:** Shallow or brightly-lit potholes may not differ
          enough in intensity from the surrounding asphalt to be detected.
        - **Fixed thresholds:** Classical heuristics are not adaptive across
          widely varying lighting, camera angles, or road surface types.
        - **No temporal context:** In video mode, each frame is processed
          independently with no tracking or temporal smoothing.
        """
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🕳️ Pothole Detection — Classical Image & Video Processing")
    st.markdown(
        "> **Academic IVP project.** All detection uses classical OpenCV techniques "
        "(CLAHE, adaptive thresholding, morphological operations, contour analysis). "
        "**No AI, machine learning, or deep learning is used.**"
    )

    params = build_sidebar()

    tab1, tab2, tab3 = st.tabs(["Image Detection", "Video Detection", "Evaluation"])
    with tab1:
        page_image(params)
    with tab2:
        page_video(params)
    with tab3:
        page_evaluation()


if __name__ == "__main__":
    main()
