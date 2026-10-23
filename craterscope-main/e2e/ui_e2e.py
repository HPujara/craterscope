"""
e2e/ui_e2e.py — Browser end-to-end test of the Streamlit app + README screenshots.

Starts `streamlit run app.py`, drives it with Playwright (Chromium or Edge),
uploads a real dataset image and a short clip, asserts on what the user sees,
and saves screenshots to docs/screenshots/.

    pip install playwright && python -m playwright install chromium
    python e2e/ui_e2e.py [path/to/road_image.jpg]

Exit code 0 = every check passed.
"""

import os
import re
import socket
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from playwright.sync_api import sync_playwright, expect

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHOTS = os.path.join(ROOT, "docs", "screenshots")
DATASET = os.environ.get("POTHOLE_DATASET", os.path.join(ROOT, "..", "archive"))
DEFAULT_IMAGE = os.path.join(DATASET, "annotated-images", "img-611.jpg")

results = []
known_issues = []


def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")


def known(name, present, detail=""):
    """Record a known UX issue without failing the run."""
    if present:
        known_issues.append(name)
        print(f"  [KNOWN] {name} {detail}")
    else:
        check(f"Fixed: {name}", True)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_clip(image_path, out_path, n=24, fps=8):
    """Slow zoom into a still road photo — a stand-in for dash-cam footage."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(n):
        s = 1 + 0.25 * i / n
        M = cv2.getRotationMatrix2D((w / 2, h * 0.6), 0, s)
        vw.write(cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT))
    vw.release()
    return n


def launch_browser(p):
    for kw in ({}, {"channel": "msedge"}, {"channel": "chrome"}):
        try:
            return p.chromium.launch(**kw)
        except Exception:
            continue
    raise RuntimeError("No Chromium-based browser available for Playwright")


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    if not os.path.exists(image):
        sys.exit(f"Test image not found: {image}")
    os.makedirs(SHOTS, exist_ok=True)

    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", os.path.join(ROOT, "app.py"),
         "--server.headless=true", f"--server.port={port}",
         "--browser.gatherUsageStats=false", "--theme.base=light"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    clip = os.path.join(tempfile.mkdtemp(), "road_clip.mp4")
    n_frames = make_clip(image, clip)

    try:
        with sync_playwright() as p:
            browser = launch_browser(p)
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))

            # ── Load ──────────────────────────────────────────────────────
            url = f"http://127.0.0.1:{port}"
            for _ in range(60):
                try:
                    page.goto(url, timeout=3000)
                    break
                except Exception:
                    time.sleep(1)
            t0 = time.time()
            expect(page.get_by_text("Upload an image to begin.")).to_be_visible(timeout=60000)
            check("App loads and shows upload prompt", True, f"({time.time()-t0:.1f}s)")
            check("Three tabs present", page.get_by_role("tab").count() == 3)
            sliders = page.locator('[data-testid="stSidebar"] [data-testid="stSlider"]')
            expect(sliders).to_have_count(10, timeout=30000)
            check("Sidebar has 10 sliders (Method A)", sliders.count() == 10)
            page.screenshot(path=os.path.join(SHOTS, "01_home.png"))

            # ── Image detection (Method A) ────────────────────────────────
            page.locator('input[type="file"]').first.set_input_files(image)
            heading = page.get_by_text("Detected (", exact=False)
            expect(heading).to_be_visible(timeout=60000)
            check("Image result rendered", True, f"→ '{heading.inner_text()}'")
            check("Processing-time metric shown",
                  page.get_by_text("Processing time").is_visible())
            check("Bounding-box list shown",
                  page.get_by_text("Region 1:", exact=False).count() > 0)
            page.get_by_text("Original", exact=True).first.scroll_into_view_if_needed()
            page.screenshot(path=os.path.join(SHOTS, "02_image_detection.png"))

            stages = page.get_by_text("Intermediate Processing Stages")
            stages.scroll_into_view_if_needed()
            time.sleep(1)
            image_tab = page.locator('[data-testid="stTabPanel"]').first
            n_stage_imgs = image_tab.locator('[data-testid="stImage"]').count() - 2
            check("7 intermediate stage images (Method A)", n_stage_imgs == 7, f"(got {n_stage_imgs})")
            page.screenshot(path=os.path.join(SHOTS, "03_intermediate_stages_A.png"))

            # ── Switch to Method B ────────────────────────────────────────
            page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').click()
            page.get_by_role("option", name="B: CLAHE", exact=False).click()
            expect(page.get_by_text("CLAHE clip limit")).to_be_visible(timeout=30000)
            check("Method B controls appear after switching", True)
            expect(page.get_by_text("3. CLAHE Enhanced")).to_be_visible(timeout=60000)
            check("Method B stages (CLAHE, Morph Gradient) rendered",
                  page.get_by_text("4. Morph Gradient").is_visible())
            page.get_by_text("Intermediate Processing Stages").scroll_into_view_if_needed()
            time.sleep(1)
            page.screenshot(path=os.path.join(SHOTS, "04_intermediate_stages_B.png"))
            page.evaluate("window.scrollTo(0,0)")
            page.locator('[data-testid="stSidebar"]').screenshot(
                path=os.path.join(SHOTS, "05_sidebar_method_b.png"))

            # ── Video tab ─────────────────────────────────────────────────
            page.get_by_role("tab", name="Video Detection").click()
            page.locator('input[type="file"]').nth(1).set_input_files(clip)
            done = page.get_by_text(re.compile(r"Processed \d+ frames"))
            expect(done).to_be_visible(timeout=180000)
            msg = done.inner_text()
            check("Video processed", f"Processed {n_frames} frames" in msg, f"→ '{msg[:70]}…'")
            dl = page.get_by_role("button", name="Download processed video")
            expect(dl).to_be_visible(timeout=30000)
            check("Download button offered", dl.is_visible())
            page.screenshot(path=os.path.join(SHOTS, "06_video_detection.png"))

            # ── Evaluation tab ────────────────────────────────────────────
            page.get_by_role("tab", name="Evaluation").click()
            expect(page.get_by_text("Quantitative Metrics (Official Test Split)")).to_be_visible()
            check("Evaluation metrics displayed",
                  page.get_by_text("F1-Score").first.is_visible())
            page.screenshot(path=os.path.join(SHOTS, "07_evaluation.png"))

            # ── Error handling: corrupt upload ────────────────────────────
            page.get_by_role("tab", name="Image Detection").click()
            bad = os.path.join(tempfile.mkdtemp(), "broken.jpg")
            with open(bad, "wb") as f:
                f.write(b"this is not a jpeg")
            page.locator('input[type="file"]').first.set_input_files(bad)
            err = page.get_by_text("Could not decode image", exact=False)
            try:
                expect(err).to_be_visible(timeout=30000)
                ok = True
            except AssertionError:
                ok = False
            check("Corrupt image shows friendly error", ok)
            # stale elements from the previous run are removed when the rerun ends
            expect(page.locator('[data-testid="stStatusWidget"]')).to_have_count(0, timeout=60000)
            check("Stale results cleared after corrupt upload",
                  page.get_by_text("Detection Summary").count() == 0)
            page.screenshot(path=os.path.join(SHOTS, "08_error_handling.png"))

            # ── Responsive: phone width ───────────────────────────────────
            m = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
            m.goto(url)
            expect(m.get_by_text("Upload an image to begin.")).to_be_visible(timeout=60000)
            m.wait_for_timeout(3000)
            overflow = m.evaluate("document.documentElement.scrollWidth > window.innerWidth + 1")
            check("No horizontal overflow at 390px width", not overflow)
            sidebar_w = m.locator('[data-testid="stSidebar"]').bounding_box()["width"]
            known("Sidebar opens over the content on phones", sidebar_w > 0.5 * 390,
                  f"(sidebar {sidebar_w:.0f}px of 390px; initial_sidebar_state='expanded')")
            m.screenshot(path=os.path.join(SHOTS, "09_mobile.png"))

            check("No JavaScript errors", not errors, f"{errors[:2]}")
            browser.close()
    finally:
        server.terminate()
        server.wait(timeout=10)

    passed = sum(ok for _, ok, _ in results)
    print(f"\nE2E: {passed}/{len(results)} checks passed, {len(known_issues)} known issue(s). "
          f"Screenshots → {SHOTS}")
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
