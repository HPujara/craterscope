"""
tests/test_app_ui.py — Headless UI tests for the Streamlit app (streamlit.testing.AppTest).

File-upload flows are covered by the browser end-to-end script
(e2e/ui_e2e.py), because AppTest cannot drive st.file_uploader.
"""

import os
import warnings

import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import PROJECT_ROOT
from detector import DEFAULT_PARAMS_A, DEFAULT_PARAMS_B

APP = os.path.join(PROJECT_ROOT, "app.py")
METRICS = os.path.join(PROJECT_ROOT, "results", "test_metrics.json")


@pytest.fixture
def at():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        app = AppTest.from_file(APP, default_timeout=60).run()
    assert not app.exception, app.exception
    return app


def slider(at, label):
    return next(s for s in at.sidebar.slider if s.label == label)


class TestLayout:
    def test_title_and_disclaimer(self, at):
        assert "Pothole Detection" in at.title[0].value
        assert any("No AI" in m.value for m in at.markdown)

    def test_three_tabs(self, at):
        assert [t.label for t in at.tabs] == ["Image Detection", "Video Detection", "Evaluation"]

    def test_upload_prompts_shown(self, at):
        infos = [i.value for i in at.info]
        assert "Upload an image to begin." in infos
        assert "Upload a video to begin." in infos

    def test_video_speed_warning_shown(self, at):
        assert any("frame-by-frame" in w.value for w in at.warning)

    def test_sidebar_sections(self, at):
        subs = [s.value for s in at.sidebar.subheader]
        for s in ("Pre-processing", "Thresholding", "Morphology", "Region Filtering"):
            assert s in subs


class TestSidebarControls:
    def test_method_a_defaults_match_detector(self, at):
        p = DEFAULT_PARAMS_A
        assert slider(at, "Blur kernel size (odd)").value == p["blur_ksize"]
        assert slider(at, "Edge density weight").value == p["edge_weight"]
        assert slider(at, "Edge density sigma").value == p["edge_sigma"]
        assert slider(at, "Percentile threshold").value == p["percentile"]
        assert slider(at, "Closing kernel size (odd)").value == p["morph_close_ksize"]
        assert slider(at, "Min contour area (px²)").value == p["min_area"]
        assert slider(at, "Max mean brightness (sky filter)").value == p["max_brightness"]
        assert slider(at, "Border mask (px)").value == p["border_margin"]

    def test_darkness_weight_is_complement(self, at):
        slider(at, "Edge density weight").set_value(0.4).run()
        assert at.sidebar.caption[0].value == "Darkness weight: 0.6"

    def test_switch_to_method_b_changes_controls(self, at):
        at.sidebar.selectbox[0].select_index(1).run()
        labels = [s.label for s in at.sidebar.slider]
        assert "CLAHE clip limit" in labels and "Density sigma" in labels
        assert "Edge density weight" not in labels
        assert not at.exception

    def test_method_b_defaults_match_detector(self, at):
        at.sidebar.selectbox[0].select_index(1).run()
        p = DEFAULT_PARAMS_B
        assert slider(at, "CLAHE clip limit").value == p["clahe_clip"]
        assert slider(at, "Morphological gradient size (odd)").value == p["grad_ksize"]
        assert slider(at, "Density sigma").value == p["density_sigma"]

    @pytest.mark.xfail(strict=True, reason=(
        "UI BUG: the percentile slider always defaults to 85, but the tuned "
        "Method B default is 87 — users get untuned Method B results."))
    def test_method_b_percentile_default(self, at):
        at.sidebar.selectbox[0].select_index(1).run()
        assert slider(at, "Percentile threshold").value == DEFAULT_PARAMS_B["percentile"]

    def test_odd_kernel_sliders_step_by_two(self, at):
        for label in ("Blur kernel size (odd)", "Closing kernel size (odd)"):
            s = slider(at, label)
            assert s.step == 2 and s.min % 2 == 1

    @pytest.mark.parametrize("label,value", [
        ("Percentile threshold", 70), ("Percentile threshold", 95),
        ("Closing iterations", 4), ("Min contour area (px²)", 5000),
        ("Border mask (px)", 10), ("Blur kernel size (odd)", 15),
    ])
    def test_extreme_slider_values_do_not_crash(self, at, label, value):
        slider(at, label).set_value(value).run()
        assert not at.exception


class TestEvaluationTab:
    def test_metrics_match_json(self, at):
        import json
        with open(METRICS) as f:
            m = json.load(f)
        shown = {x.label: x.value for x in at.metric}
        assert shown["Precision"] == f"{m['precision']:.3f}"
        assert shown["Recall"] == f"{m['recall']:.3f}"
        assert shown["F1-Score"] == f"{m['f1']:.3f}"
        assert shown["Images evaluated"] == str(m["n_processed"])

    def test_result_figures_rendered(self, at):
        subs = [s.value for s in at.subheader]
        for s in ("Method A vs Method B Comparison (Training Subset)",
                  "Qualitative Samples", "Pipeline Intermediate Stages", "Limitations"):
            assert s in subs

    def test_tp_fp_fn_definitions(self, at):
        text = " ".join(e.value for e in list(at.info) + list(at.warning) + list(at.error))
        assert "**TP**" in text and "**FP**" in text and "**FN**" in text

    def test_missing_metrics_shows_warning(self, at, tmp_path, monkeypatch):
        """If results/test_metrics.json is absent, the user is told how to create it."""
        import shutil
        backup = tmp_path / "m.json"
        shutil.move(METRICS, backup)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                app = AppTest.from_file(APP, default_timeout=60).run()
            assert any("run_evaluation.py" in w.value for w in app.warning)
        finally:
            shutil.move(backup, METRICS)


class TestCopyAccuracy:
    @pytest.mark.xfail(strict=True, reason=(
        "UI TEXT: header claims 'adaptive thresholding' but the pipeline uses "
        "percentile thresholding — misleading in a viva/demo."))
    def test_header_describes_actual_methods(self, at):
        header = " ".join(m.value for m in at.markdown)
        assert "adaptive thresholding" not in header
