#!/usr/bin/env python3
"""Verify the continuous, local-file SPEN experiment viewer in real Chrome.

All displayed frame identities and image paths are checked against the independent
frame_index.jsonl. The check does not alter reconstruction arrays or PNG assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlencode, urlparse

from playwright.sync_api import sync_playwright


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--browser", help="Path to Chrome/Chromium executable")
    args = parser.parse_args()
    run = args.run.resolve()
    index = run / "index.html"
    summary = json.loads((run / "summary.json").read_text())
    frames = [json.loads(line) for line in (run / "frame_index.jsonl").read_text().splitlines()]
    cases = {case["id"]: case for case in summary["cases"]}
    frame_map = {frame["frame_id"]: frame for frame in frames}
    experiment_frames = defaultdict(set)
    experiment_cases = defaultdict(set)
    case_frames = defaultdict(list)
    for frame in frames:
        experiment_frames[frame["experiment_name"]].add(frame["frame_id"])
        case_frames[frame["case_id"]].append(frame)
    for case in cases.values():
        experiment_cases[case["experiment_name"]].add(case["id"])
    checks, browser_errors, remote_requests, failed_requests, cancelled_requests = [], [], [], [], []
    screenshot_dir = run / "viewer_checks"
    screenshot_dir.mkdir(exist_ok=True)
    screenshots = []
    report = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "url": index.as_uri(), "index_html_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "viewer_mode": "continuous_experiment_all_frames",
        "frame_manifest_records": len(frames), "case_records": len(cases),
        "checks": checks, "browser_errors": browser_errors, "remote_requests": remote_requests,
        "failed_requests": failed_requests, "navigation_cancelled_requests": cancelled_requests,
        "screenshots": screenshots,
    }
    default_case = "20231207_150817_lxj_spen_231207_1_1_1_scan024"
    volume_case = "20220721_095453_lxj_SPEN_diffusion_test_0721_water_1_1_scan019"
    echo_case = "20211203_152911_lxj_SPEN_water_20211203_1_1_scan027"
    partial_case = "20220721_095453_lxj_SPEN_diffusion_test_0721_water_1_1_scan030"
    preview_case = "20230911_092258_lxj_spen_test_230911_1_1_1_scan048"
    missing_case = "20220314_174800_lxj_SPEN_20220315_1_3_scan018"
    default_experiment = cases[default_case]["experiment_name"]
    compare_keys = ["sorted_samples", "rofft_original", "inva_corrected", "tikhonov_coils"]

    with sync_playwright() as playwright:
        executable = args.browser or shutil.which("google-chrome") or shutil.which("chromium")
        launch = {"headless": True, "args": ["--no-sandbox"]}
        if executable:
            launch["executable_path"] = executable
        browser = playwright.chromium.launch(**launch)
        report["browser"] = {"version": browser.version, "executable": executable}
        page = browser.new_page(viewport={"width": 1440, "height": 1050}, device_scale_factor=1)
        page.set_default_timeout(10000)
        page.on("pageerror", lambda error: browser_errors.append(str(error)))
        page.on("console", lambda message: browser_errors.append(message.text) if message.type == "error" else None)
        page.on("request", lambda request: remote_requests.append(request.url) if urlparse(request.url).scheme in ("http", "https") else None)

        def request_failure(request):
            item = {"url": request.url, "failure": request.failure}
            (cancelled_requests if request.failure == "net::ERR_ABORTED" else failed_requests).append(item)

        page.on("requestfailed", request_failure)

        def screenshot(name):
            path = screenshot_dir / name
            page.screenshot(path=str(path), full_page=False)
            screenshots.append(path.relative_to(run).as_posix())

        def check(name, action):
            try:
                detail = action()
                checks.append({"name": name, "passed": True, "detail": detail})
                print(f"PASS {name}", flush=True)
            except Exception as error:
                checks.append({"name": name, "passed": False, "error": str(error), "traceback": traceback.format_exc()})
                print(f"FAIL {name}: {error}", flush=True)
                screenshot(f"continuous_failure_{len(checks):02d}.png")

        def all_row_ids():
            return page.locator("#experiment-gallery .frame-row").evaluate_all("rows => rows.map(row => row.dataset.frameId)")

        def select_experiment(experiment):
            page.locator("#dataset-select").select_option(experiment)
            expected = experiment_frames[experiment]
            actual = all_row_ids()
            assert len(actual) == len(set(actual)), "Duplicated frame rows"
            assert set(actual) == expected, {"missing": sorted(expected - set(actual)), "extra": sorted(set(actual) - expected)}
            actual_cases = page.locator("#experiment-gallery .scan-section").evaluate_all("sections => sections.map(section => section.dataset.caseId)")
            assert set(actual_cases) == experiment_cases[experiment]
            return actual

        def select_case_experiment(case_id):
            return select_experiment(cases[case_id]["experiment_name"])

        def scan_section(case_id):
            return page.locator(f'#experiment-gallery .scan-section[data-case-id="{case_id}"]')

        def row(frame_id):
            return page.locator(f'#experiment-gallery .frame-row[data-frame-id="{frame_id}"]')

        def find_frame(case_id, s=0, v=0, e=0):
            return next(frame for frame in case_frames[case_id] if (frame["slice_index"], frame["volume_index"], frame["echo_index"]) == (s, v, e))

        def assert_row_paths(frame_id, expected_keys, load=False):
            target = row(frame_id)
            expected = frame_map[frame_id]
            assert target.count() == 1, frame_id
            displayed_keys = target.locator(".image-card[data-panel-key]").evaluate_all("cards => cards.map(card => card.dataset.panelKey)")
            assert displayed_keys == expected_keys, displayed_keys
            if load:
                target.scroll_into_view_if_needed()
                page.wait_for_function("id => Array.from(document.querySelectorAll('.frame-row[data-frame-id=\"' + id + '\"] img')).every(image => image.complete && image.naturalWidth > 0)", arg=frame_id)
            images = target.locator(".image-card img").evaluate_all("images => images.map(image => ({key: image.closest('.image-card').dataset.panelKey, src: image.src, width: image.naturalWidth}))")
            actual_keys = []
            for image in images:
                assert image["key"] in expected["panels"], f"Invented panel {image['key']} in {frame_id}"
                expected_path = (run / expected["panels"][image["key"]]["png_path"]).resolve()
                actual_path = Path(unquote(urlparse(image["src"]).path)).resolve()
                assert actual_path == expected_path, (actual_path, expected_path)
                if load:
                    assert image["width"] > 0
                actual_keys.append(image["key"])
            assert actual_keys == [key for key in expected_keys if key in expected["panels"]], actual_keys
            return actual_keys

        def initial():
            page.goto(report["url"], wait_until="load")
            page.locator("#experiment-gallery .frame-row").first.wait_for()
            assert page.locator("#dataset-select").input_value() == default_experiment
            ids = select_experiment(default_experiment)
            assert len(ids) == 88
            assert_row_paths(ids[0], compare_keys, load=True)
            page.evaluate("scrollTo(0, 0)")
            screenshot("desktop_default.png")
            return {"experiment": default_experiment, "all_frame_rows_visible_in_same_document": len(ids)}

        check("file_url_opens_whole_experiment", initial)

        def complete_coverage():
            options = set(page.locator("#dataset-select option").evaluate_all("options => options.map(option => option.value)"))
            assert options - {"", "all"} == set(experiment_cases)
            details = []
            for experiment in sorted(experiment_cases):
                ids = select_experiment(experiment)
                details.append({"experiment": experiment, "frames": len(ids), "scans": len(experiment_cases[experiment])})
            assert sum(item["frames"] for item in details) == len(frames)
            return {"experiments": len(details), "frames": len(frames), "scans": len(cases), "coverage": details}

        check("all_45_experiments_exact_frame_and_scan_coverage", complete_coverage)

        def slices_and_jump():
            ids_before = select_case_experiment(default_case)
            expected = [frame["frame_id"] for frame in sorted(case_frames[default_case], key=lambda frame: frame["slice_index"])]
            actual = scan_section(default_case).locator(".frame-row").evaluate_all("rows => rows.map(row => row.dataset.frameId)")
            assert actual == expected and len(actual) == 11
            for frame_id in actual:
                assert_row_paths(frame_id, compare_keys)
            page.locator("#scan-select").select_option(default_case)
            assert all_row_ids() == ids_before, "Jumping to a scan filtered away other frames"
            frame3 = find_frame(default_case, 3)
            frame4 = find_frame(default_case, 4)
            assert_row_paths(frame3["frame_id"], compare_keys, load=True)
            row(frame3["frame_id"]).evaluate("element => element.scrollIntoView({block: 'start', behavior: 'instant'})")
            assert row(frame4["frame_id"]).count() == 1
            screenshot("desktop_continuous_slices.png")
            return {"slice_indices": list(range(11)), "experiment_frame_count_after_scan_jump": len(ids_before)}

        check("all_11_slices_continuous_and_scan_jump_does_not_filter", slices_and_jump)

        def volumes_and_echoes():
            select_case_experiment(volume_case)
            expected_volumes = [find_frame(volume_case, v=index)["frame_id"] for index in range(35)]
            assert set(scan_section(volume_case).locator(".frame-row").evaluate_all("rows => rows.map(row => row.dataset.frameId)")) == set(expected_volumes)
            for frame_id in expected_volumes:
                assert_row_paths(frame_id, compare_keys)
            assert_row_paths(expected_volumes[-1], compare_keys, load=True)
            select_case_experiment(echo_case)
            expected_echoes = [find_frame(echo_case, e=index)["frame_id"] for index in range(2)]
            for frame_id in expected_echoes:
                assert_row_paths(frame_id, compare_keys, load=True)
            return {"all_volume_indices_in_one_document": list(range(35)), "all_echo_indices_in_one_document": [0, 1]}

        check("35_volumes_and_both_echoes_without_frame_controls", volumes_and_echoes)

        def last_frame():
            ids = select_experiment(default_experiment)
            final_id = ids[-1]
            keys = assert_row_paths(final_id, compare_keys, load=True)
            box = row(final_id).bounding_box()
            assert box and box["y"] < 1050 and box["y"] + box["height"] > 0
            screenshot("desktop_experiment_last_frame.png")
            return {"last_frame_id": final_id, "loaded_panel_keys": keys, "row_viewport_bounds": box}

        check("experiment_last_frame_scrolls_and_loads_real_png", last_frame)

        def partial_and_preview():
            select_case_experiment(partial_case)
            section_text = scan_section(partial_case).inner_text()
            assert "96 × 95" in section_text, "Actual acquired matrix 96 × 95 not displayed"
            embedded_case = page.locator("#viewer-data").evaluate("(element, id) => JSON.parse(element.textContent).cases.find(c => c.id === id)", partial_case)
            assert embedded_case["matrix"] == [96, 95] and embedded_case["declared_matrix"] == [96, 96]
            partial = find_frame(partial_case)["frame_id"]
            assert assert_row_paths(partial, compare_keys) == ["sorted_samples", "rofft_original"]
            page.locator("#view-select").select_option("uncorrected_pair")
            assert assert_row_paths(partial, ["inva_uncorrected", "tikhonov_uncorrected"], load=True) == ["inva_uncorrected", "tikhonov_uncorrected"]
            page.locator("#view-select").select_option("compare4")
            select_case_experiment(preview_case)
            assert assert_row_paths(find_frame(preview_case)["frame_id"], compare_keys) == ["sorted_samples", "rofft_original"]
            select_case_experiment(missing_case)
            missing = scan_section(missing_case)
            assert missing.locator(".frame-row, img").count() == 0
            assert missing.inner_text().strip(), "Unavailable scan has no explanation"
            return {"partial_actual_matrix": [96, 95], "partial_declared_matrix": [96, 96], "missing_scan_has_no_stale_images": True}

        check("partial_preview_and_unavailable_are_explicit", partial_and_preview)

        def whole_list_view():
            ids = select_experiment(default_experiment)
            for view, keys in [("inva_corrected", ["inva_corrected"]), ("recon_pair", ["inva_corrected", "tikhonov_coils"]), ("scanner_preview", ["scanner_preview"])]:
                page.locator("#view-select").select_option(view)
                assert all_row_ids() == ids
                for frame_id in ids:
                    assert_row_paths(frame_id, keys)
            page.locator("#view-select").select_option("compare4")
            return {"view_changes_apply_to_all_rows": len(ids), "views_checked": ["inva_corrected", "recon_pair", "scanner_preview"]}

        check("view_selection_changes_entire_experiment", whole_list_view)

        def zoom():
            select_experiment(default_experiment)
            frame = find_frame(default_case, 7)
            assert_row_paths(frame["frame_id"], compare_keys, load=True)
            row(frame["frame_id"]).locator('.image-card[data-panel-key="inva_corrected"] img').click()
            dialog = page.locator("#zoom-dialog")
            assert dialog.is_visible()
            context = page.locator("#zoom-context").inner_text()
            assert "Slice 7" in context and "Volume 0" in context and "Echo 0" in context and "24" in context, context
            expected = (run / frame["panels"]["inva_corrected"]["png_path"]).resolve()
            actual = Path(unquote(urlparse(page.locator("#zoom-image").evaluate("image => image.src")).path)).resolve()
            assert actual == expected
            screenshot("desktop_single_enlarged.png")
            page.keyboard.press("Escape")
            assert not dialog.is_visible()
            return {"clicked_frame": frame["frame_id"], "dialog_context": context}

        check("zoom_uses_clicked_frame_context", zoom)

        def history():
            old_hash = urlencode({"case": default_case, "s": 7, "v": 0, "e": 0, "view": "inva_corrected"})
            page.goto(report["url"] + "#" + old_hash, wait_until="load")
            page.wait_for_function("name => document.querySelector('#dataset-select').value === name", arg=default_experiment)
            page.wait_for_function("document.querySelector('#view-select').value === 'inva_corrected'")
            assert len(all_row_ids()) == len(experiment_frames[default_experiment])
            frame_id = find_frame(default_case, 7)["frame_id"]
            assert_row_paths(frame_id, ["inva_corrected"], load=True)
            page.reload(wait_until="load")
            assert page.locator("#dataset-select").input_value() == default_experiment
            assert_row_paths(frame_id, ["inva_corrected"])
            select_case_experiment(echo_case)
            page.go_back(wait_until="load")
            page.wait_for_function("name => document.querySelector('#dataset-select').value === name", arg=default_experiment)
            assert set(all_row_ids()) == experiment_frames[default_experiment]
            assert_row_paths(frame_id, ["inva_corrected"])
            page.locator("#view-select").select_option("compare4")
            return {"legacy_case_hash_supported": True, "reload_and_back_restore_experiment": True}

        check("legacy_hash_reload_and_browser_back", history)

        def responsive():
            select_experiment(default_experiment)
            result = {}
            frame_id = find_frame(default_case, 5)["frame_id"]
            for width, height, name in [(1440, 1050, "desktop"), (390, 844, "mobile")]:
                page.set_viewport_size({"width": width, "height": height})
                assert_row_paths(frame_id, compare_keys, load=True)
                measured = page.evaluate("({viewport: innerWidth, document: document.documentElement.scrollWidth, body: document.body.scrollWidth})")
                assert measured["document"] <= width + 1 and measured["body"] <= width + 1, measured
                screenshot(f"{name}_compare.png")
                result[name] = measured
            return result

        check("desktop_and_mobile_continuous_layout_no_overflow", responsive)

        def clean_browser():
            assert not browser_errors, browser_errors
            assert not remote_requests, remote_requests
            assert not failed_requests, failed_requests
            return {"console_and_page_errors": 0, "network_http_requests": 0, "failed_asset_requests": 0, "cancelled_navigation_requests": len(cancelled_requests)}

        check("offline_local_assets_and_clean_browser", clean_browser)
        browser.close()

    report["passed"] = all(item["passed"] for item in checks)
    report["passed_checks"] = sum(item["passed"] for item in checks)
    report["total_checks"] = len(checks)
    report["index_unchanged_during_check"] = hashlib.sha256(index.read_bytes()).hexdigest() == report["index_html_sha256"]
    report["passed"] = report["passed"] and report["index_unchanged_during_check"]
    output = run / "viewer_browser_checks.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"{report['passed_checks']}/{report['total_checks']} checks passed; report: {output}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
