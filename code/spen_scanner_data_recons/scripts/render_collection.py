#!/usr/bin/env python3
"""Render every SPEN frame with an offline viewer and paginated archive.

Usage: python scripts/render_collection.py --run runs/<collection>

Accepts embedded ``summary.json: cases[].frames`` (the pilot schema), or case
summaries at ``cases/<id>/summary.json``.  Array paths may be run-relative,
case-relative, or absolute.  ``cases/<id>/case.json`` is also supported.
Missing panels are explicitly labelled and are
never counted as completed two-method reconstruction.  All generated files
stay inside the run directory; NPZ files are only read.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
import math
import os
from pathlib import Path
import re
from urllib.parse import quote, unquote, urlsplit

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PANELS = (
    ("sorted_samples", "原始 ADC · log RSS"),
    ("rofft_original", "RO FFT · RSS"),
    ("inva_corrected", "Phase Map + InvA"),
    ("tikhonov_coils", "Tikhonov"),
)
METHOD_KEYS = ("inva_corrected", "tikhonov_coils")
DISPLAY_KEYS = tuple(key for key, _ in PANELS) + ("scanner_preview", "inva_uncorrected", "tikhonov_uncorrected")
CAPTION = (
    "每一帧单独展示，不用代表层代替其他层。ADC 为已排序及反向读出校正的采样信号，"
    "采用 log(1 + RSS)、p1–p99.5 窗口；RO FFT 和重建图采用 RSS、0–p99.5 窗口。"
    "各面板独立设窗，强度不能直接比较。RO FFT 与重建预览翻转前两轴以沿用旧预览方向，"
    "这不定义解剖方向；ADC 与扫描仪图保留原方向。PNG 不作平滑，NPZ 数值和坐标不变。"
)
STYLE = """
:root{color-scheme:light;--ink:#172d39;--muted:#586d78;--line:#dce4e8;--accent:#12677b}
*{box-sizing:border-box}body{margin:0;background:#f2f5f7;color:var(--ink);font:15px/1.6 system-ui,'Noto Sans CJK SC',sans-serif}
main{max-width:1480px;margin:auto;padding:24px 20px 60px}h1{font-size:25px;overflow-wrap:anywhere}h2{font-size:20px}h3{font-size:16px}
header,section,article{background:white;border:1px solid var(--line);border-radius:7px;margin:18px 0;padding:18px}
a{color:var(--accent)}.muted{color:var(--muted);font-size:13px}.notice{border-left:4px solid #b08728;background:#fff8e9;padding:12px}
.good{color:#146848}.partial{color:#966820}.bad{color:#9c3434}img{display:block;max-width:100%;height:auto;image-rendering:pixelated}
.frame-image{width:min(100%,1060px);background:#0c1116}.scanner-image{max-width:260px;max-height:260px}.scroll{overflow:auto}
table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid var(--line)}
th{background:#f6f8fa;white-space:nowrap}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.6 monospace}code{font:12px monospace}
.pages{display:flex;flex-wrap:wrap;gap:8px;margin:14px 0}.pages a,.pages strong{border:1px solid var(--line);padding:5px 10px;border-radius:4px}
.pages strong{background:#dceff2}details{margin:12px 0}summary{cursor:pointer;color:var(--accent)}input{font:inherit;padding:8px;max-width:100%;width:450px}
@media(max-width:700px){main{padding:12px 6px}header,section,article{padding:12px 8px}h1{font-size:21px}td,th{padding:5px}}
"""


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def pretty(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def quality_warnings(case: dict, frame: dict | None = None) -> list[str]:
    values = []
    for item in [case] + (case.get('frames', []) if frame is None else [frame]):
        warnings = item.get("quality_warnings", []) or []
        if not isinstance(warnings, list):
            warnings = [warnings]
        for warning in warnings:
            text = str(warning)
            if text not in values:
                values.append(text)
    return values


def warnings_html(values: list[str]) -> str:
    if not values:
        return ""
    return '<div class="notice"><strong>重建质量提示</strong><ul>' + "".join(
        f'<li>{esc(value)}</li>' for value in values) + '</ul></div>'


def slug(value: str) -> str:
    value = str(value)
    readable = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")[:90] or "item"
    return readable + "_" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]


def natural(value) -> tuple:
    return tuple((0, int(piece)) if piece.isdigit() else (1, piece.lower())
                 for piece in re.split(r"(\d+)", str(value)))


def local_link(page: Path, target: Path) -> str:
    return quote(os.path.relpath(target, page.parent), safe="/")


def doc(title: str, content: str) -> str:
    return (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{esc(title)}</title><style>{STYLE}</style></head>'
            f'<body><main>{content}</main></body></html>')


def font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


FONT = font(13)
SMALL_FONT = font(11)


def trim_text(draw: ImageDraw.ImageDraw, value: str, width: int, chosen_font=FONT) -> str:
    value = str(value)
    while value and draw.textlength(value, font=chosen_font) > width:
        value = value[:-2] + "…" if len(value) > 2 else ""
    return value


def load_cases(run: Path) -> tuple[dict, list[dict], list[dict]]:
    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    source_entries = summary.get("cases", [])
    if isinstance(source_entries, dict):
        source_entries = [{"id": key, **value} for key, value in source_entries.items()]
    if not source_entries:
        found = {path.parent: path for path in sorted((run / "cases").glob("*/summary.json"))}
        found.update({path.parent: path for path in sorted((run / "cases").glob("*/case.json"))})
        source_entries = [{"id": path.parent.name, "summary_path": str(path)} for path in found.values()]
    cases, errors = [], []
    seen = set()
    for entry in source_entries:
        if isinstance(entry, str):
            entry = {"id": Path(entry).parent.name, "summary_path": entry}
        case = dict(entry)
        case_id = str(case.get("id", len(cases)))
        if case_id in seen:
            errors.append({"case_id": case_id, "error": "Duplicate case ID in input summary; not silently merged."})
            continue
        seen.add(case_id)
        case.setdefault("id", case_id)
        candidates = []
        for key in ("summary_path", "case_summary_path", "case_path"):
            if case.get(key):
                candidate = Path(case[key])
                if not candidate.is_absolute():
                    candidate = run / candidate
                candidates.append(candidate / "summary.json" if candidate.suffix != ".json" else candidate)
        candidates.extend((run / "cases" / case_id / "case.json", run / "cases" / case_id / "summary.json"))
        case_dir = run
        for candidate in candidates:
            if candidate.is_file():
                try:
                    nested = json.loads(candidate.read_text())
                    if "cases" in nested:
                        nested_cases = nested["cases"]
                        nested = next((item for item in nested_cases if str(item.get("id")) == case_id),
                                      nested_cases[0] if len(nested_cases) == 1 else {})
                    if "case" in nested and isinstance(nested["case"], dict):
                        nested = {**nested, **nested["case"]}
                    case.update(nested)
                    case_dir = candidate.parent
                    case["_summary_path"] = str(candidate)
                except Exception as exc:
                    errors.append({"case_id": case_id, "summary_path": str(candidate), "error": str(exc)})
                break
        case["_case_dir"] = str(case_dir)
        case.setdefault("frames", [])
        case.setdefault("experiment_name", "experiment_unspecified")
        cases.append(case)
    cases.sort(key=lambda c: (natural(c["experiment_name"]), natural(c.get("scan_id", c["id"]))))
    return summary, cases, errors


def arrays_path(run: Path, case: dict, frame: dict) -> Path | None:
    value = frame.get("arrays_path") or frame.get("npz_path")
    if not value:
        return None
    value = Path(value)
    if value.is_absolute():
        return value
    candidates = (run / value, Path(case["_case_dir"]) / value,
                  run / "cases" / str(case["id"]) / value)
    return next((path for path in candidates if path.is_file()), candidates[0])


def panel_image(value: np.ndarray, key: str) -> tuple[Image.Image, dict, dict]:
    value = np.asarray(value)
    if value.ndim not in (2, 3):
        raise ValueError(f"{key}: expected a 2D image or [PE, RO, coil], got {value.shape}")
    finite = np.isfinite(value)
    checks = {"shape": list(value.shape), "dtype": str(value.dtype), "complex": bool(np.iscomplexobj(value)),
              "elements": int(value.size), "nonfinite_elements": int(value.size - np.count_nonzero(finite)),
              "nonzero_elements": int(np.count_nonzero(value[finite]))}
    magnitude = np.abs(value.astype(np.complex128))
    if value.ndim == 3:
        magnitude = np.sqrt(np.sum(magnitude ** 2, axis=2))
    logarithmic = key == "sorted_samples"
    if logarithmic:
        magnitude = np.log1p(magnitude)
    finite_image = np.isfinite(magnitude)
    values = magnitude[finite_image]
    low = float(np.percentile(values, 1)) if logarithmic and values.size else 0.0
    high = float(np.percentile(values, 99.5)) if values.size else 0.0
    fallback = high <= low
    if fallback:
        low, high = 0.0, max(float(values.max()) if values.size else 0.0, 1.0)
    scaled = np.clip((np.where(finite_image, magnitude, 0.0) - low) / (high - low), 0, 1)
    pixels = np.rint(scaled * 255).astype(np.uint8)
    flipped = key not in ("sorted_samples", "scanner_preview")
    if flipped:
        pixels = np.flip(pixels, axis=(0, 1))
    window = {"vmin": low, "vmax": high, "transform": "log1p(RSS)" if logarithmic else "RSS",
              "requested_window": "p1-p99.5" if logarithmic else "0-p99.5",
              "degenerate_window_fallback": fallback, "flip_axes_0_1": flipped,
              "native_png_shape": list(pixels.shape)}
    return Image.fromarray(pixels), window, checks


def paste_nearest(canvas: Image.Image, source: Image.Image, box: tuple[int, int, int, int]) -> None:
    x, y, width, height = box
    scale = min(width / source.width, height / source.height)
    size = max(1, int(source.width * scale)), max(1, int(source.height * scale))
    resized = source.resize(size, Image.Resampling.NEAREST).convert("RGB")
    canvas.paste(resized, (x + (width - size[0]) // 2, y + (height - size[1]) // 2))


def frame_label(case: dict, frame: dict) -> str:
    if frame.get("frame_type") == "scanner_preview" or frame.get("frame_kind") == "scanner_preview_only":
        return f"scan {case.get('scan_id', '?')} | scanner frame {frame.get('scanner_frame_index', '?')}"
    return (f"scan {case.get('scan_id', '?')} | s={frame.get('slice_index', '?')} "
            f"v={frame.get('volume_index', '?')} e={frame.get('echo_index', 0)}")


def render_frame(run: Path, case: dict, frame: dict, number: int, tile_size: int) -> dict:
    frame_id = str(frame.get("id", f"frame_{number:06d}"))
    key = f"{number:06d}_" + slug(frame_id)
    output_dir = run / "gallery" / "frames" / slug(case["id"])
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {"case_id": case["id"], "experiment_name": case["experiment_name"],
              "scan_id": case.get("scan_id"), "frame_id": frame_id,
              "frame_number_in_case": number, "slice_index": frame.get("slice_index"),
              "volume_index": frame.get("volume_index"), "echo_index": frame.get("echo_index", 0),
              "scanner_frame_index": frame.get("scanner_frame_index"),
              "frame_kind": frame.get("frame_type", frame.get("frame_kind", "acquired_frame")),
              "input_status": frame.get("status", case.get("status", "unspecified")),
              "quality_warnings": quality_warnings(case, frame),
              "trajectory_quality": frame.get("trajectory_quality", case.get("trajectory_quality", {})),
              "arrays_path": None, "panels": {}, "array_checks": {}, "hidden_panels": {}, "errors": []}
    phase_map_status = frame.get("metadata", {}).get("phase_map_status", frame.get("phase_map_status"))
    phase_map_confirmed = frame.get("status") == "completed" or phase_map_status in ("completed", "success", "successful")
    record["phase_map_completion_confirmed"] = phase_map_confirmed
    source = arrays_path(run, case, frame)
    if source is not None:
        record["arrays_path"] = os.path.relpath(source, run)
    panels = {}
    if source is not None and source.is_file():
        try:
            with np.load(source, allow_pickle=False) as archive:
                for source_key in DISPLAY_KEYS:
                    if source_key not in archive:
                        continue
                    panel_key = source_key
                    if panel_key == "inva_corrected" and not phase_map_confirmed:
                        record["hidden_panels"][panel_key] = "Phase Map completion is not confirmed by frame status or phase_map_status."
                        continue
                    if panel_key == "tikhonov_coils" and not phase_map_confirmed:
                        record["hidden_panels"][panel_key] = "No confirmed Phase Map; show only the explicitly uncorrected supplemental preview."
                        if "tikhonov_uncorrected" in archive:
                            continue
                        panel_key = "tikhonov_uncorrected"
                    try:
                        image, window, checks = panel_image(archive[source_key], panel_key)
                        target = output_dir / f"{key}_{panel_key}.png"
                        image.save(target, compress_level=3)
                        panels[panel_key] = image
                        record["panels"][panel_key] = {"png_path": str(target.relative_to(run)), "display_window": window,
                                                       "source_key": source_key}
                        record["array_checks"][panel_key] = checks
                    except Exception as exc:
                        record["errors"].append({"panel": panel_key, "error": f"{type(exc).__name__}: {exc}"})
        except Exception as exc:
            record["errors"].append({"error": f"{type(exc).__name__}: {exc}"})
    elif source is not None:
        record["errors"].append({"error": f"Array file does not exist: {source}"})
    has_methods = all(name in panels for name in METHOD_KEYS)
    methods_finite = has_methods and all(record["array_checks"][name]["nonfinite_elements"] == 0 for name in METHOD_KEYS)
    record["two_method_arrays_present"] = has_methods
    record["two_method_arrays_finite"] = bool(methods_finite)
    record["available_panel_count"] = len(panels)
    record["status"] = ("two_methods_available" if methods_finite else
                        "nonfinite_reconstruction" if has_methods else
                        "partial_preview" if panels else "no_preview")
    record["status_note"] = frame.get("error") or frame.get("reason") or case.get("error") or case.get("reason") or ""
    canvas = Image.new("RGB", (4 * tile_size, tile_size + 78), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text((7, 3), trim_text(draw, frame_label(case, frame), canvas.width - 14), font=FONT, fill="#edf4f7")
    status_text = {"two_methods_available": "双方法数组可用", "nonfinite_reconstruction": "重建含非有限值",
                   "partial_preview": "预览可用；双方法未完成", "no_preview": "无可用预览"}[record["status"]]
    if record["quality_warnings"]:
        status_text += " · 存在质量提示，见详情"
    for column, (panel_key, title) in enumerate(PANELS):
        x = column * tile_size
        if panel_key == "sorted_samples" and panel_key not in panels and "scanner_preview" in panels:
            panel_key, title = "scanner_preview", "扫描仪预览 · 原方向"
        draw.text((x + 5, 26), trim_text(draw, title, tile_size - 10, SMALL_FONT), font=SMALL_FONT, fill="#d7e8ef")
        if panel_key in panels:
            paste_nearest(canvas, panels[panel_key], (x + 4, 47, tile_size - 8, tile_size - 8))
        else:
            draw.rectangle((x + 4, 47, x + tile_size - 4, tile_size + 39), fill="#202b33")
            draw.text((x + 12, 56 + tile_size // 3), "未生成 / 不适用", font=SMALL_FONT, fill="#d7b674")
            if record["status_note"]:
                note = trim_text(draw, str(record["status_note"]), tile_size - 24, SMALL_FONT)
                draw.text((x + 12, 75 + tile_size // 3), note, font=SMALL_FONT, fill="#d7b674")
    draw.text((7, tile_size + 51), status_text, font=SMALL_FONT,
              fill="#b9e3c8" if methods_finite and not record["quality_warnings"] else "#e9bf75")
    figure = output_dir / f"{key}.png"
    canvas.save(figure, compress_level=3)
    record["figure"] = str(figure.relative_to(run))
    record["frame_metadata"] = {key: value for key, value in frame.items()
                                if key not in ("arrays_path", "npz_path", "id")}
    return record


def page_navigation(page: Path, experiment_dir: Path, page_count: int, current: int | None = None) -> str:
    items = [f'<a href="{local_link(page, experiment_dir / "index.html")}">实验总览</a>']
    for index in range(1, page_count + 1):
        if index == current:
            items.append(f"<strong>{index}</strong>")
        else:
            items.append(f'<a href="{local_link(page, experiment_dir / f"page_{index:04d}.html")}">{index}</a>')
    return '<nav class="pages">' + "".join(items) + "</nav>"


def frame_html(run: Path, page: Path, record: dict) -> str:
    figure = local_link(page, run / record["figure"])
    label = (f"scan {record['scan_id']} · slice {record['slice_index']} · "
             f"volume {record['volume_index']} · echo {record['echo_index']}")
    if record.get("scanner_frame_index") is not None:
        label += f" · scanner frame {record['scanner_frame_index']}"
    links = []
    for key, title in (*PANELS, ("inva_uncorrected", "InvA 未做 Phase Map 校正"),
                       ("tikhonov_uncorrected", "Tikhonov 未做 Phase Map 校正")):
        if key in record["panels"]:
            links.append(f'<a href="{local_link(page, run / record["panels"][key]["png_path"])}">{esc(title)} 原像素 PNG</a>')
    source = run / record["arrays_path"] if record.get("arrays_path") else None
    if source is not None and source.is_file():
        links.append(f'<a href="{local_link(page, source)}">未归一化 NPZ</a>')
    scanner_html = ""
    if "scanner_preview" in record["panels"]:
        scanner = local_link(page, run / record["panels"]["scanner_preview"]["png_path"])
        scanner_html = (f'<details><summary>扫描仪原图预览（方向及帧对应需单独核实）</summary>'
                        f'<a href="{scanner}"><img class="scanner-image" loading="lazy" src="{scanner}" alt="扫描仪原图"></a>'
                        '<p class="muted">原方向；不是高分辨率真值。</p></details>')
    if "inva_uncorrected" in record["panels"]:
        preview = local_link(page, run / record["panels"]["inva_uncorrected"]["png_path"])
        scanner_html += (f'<details><summary>InvA 未做 Phase Map 校正 · 补充预览</summary>'
                         f'<a href="{preview}"><img class="scanner-image" loading="lazy" src="{preview}" alt="未做 Phase Map 校正的 InvA"></a>'
                         '<p class="muted">此图不属于已完成的 Phase Map + InvA。</p></details>')
    if "tikhonov_uncorrected" in record["panels"]:
        supplemental = record["panels"]["tikhonov_uncorrected"]
        preview = local_link(page, run / supplemental["png_path"])
        scanner_html += (f'<details><summary>未做 Phase Map 校正的 Tikhonov · 补充预览</summary>'
                        f'<a href="{preview}"><img class="scanner-image" loading="lazy" src="{preview}" alt="未做 Phase Map 校正的 Tikhonov"></a>'
                        f'<p class="muted">来源 NPZ 键：<code>{esc(supplemental["source_key"])}</code>；'
                        '使用未做 Phase Map 校正的信号，不计入主四栏的双方法完成结果。</p></details>')
    status = {"two_methods_available": "两种方法数组可用且数值有限；不等于重建准确性已验证。",
              "nonfinite_reconstruction": "重建数组含 NaN / Inf，请查看数值记录。",
              "partial_preview": "部分采样或扫描仪预览可用，未完成两种方法重建。",
              "no_preview": "本帧无可用预览，保留在全帧索引中。"}[record["status"]]
    status_class = "good" if record["two_method_arrays_finite"] and not record["quality_warnings"] else "partial"
    reason = (f'<p class="notice">{esc(record["status_note"])}</p>'
              if not record["two_method_arrays_finite"] and record["status_note"] else "")
    return (f'<article id="{esc(slug(record["case_id"] + ":" + str(record["frame_number_in_case"])))}">'
            f'<h3>{esc(label)}</h3><p class="{status_class}">{esc(status)}</p>'
            f'<p class="muted">{esc(record["frame_id"])} · 状态：{esc(record["input_status"])}</p>'
            f'{reason}'
            f'{warnings_html(record["quality_warnings"])}'
            f'<a href="{figure}"><img class="frame-image" loading="lazy" src="{figure}" alt="每帧采样及重建对照"></a>'
            f'<p>{" · ".join(links)}</p>{scanner_html}'
            f'<details><summary>采集索引、显示窗口与数值检查</summary><pre>{esc(pretty(record))}</pre></details></article>')


def case_metadata(case: dict) -> dict:
    return {key: value for key, value in case.items() if key != "frames" and not key.startswith("_")}


def make_contact_sheet(run: Path, records: list[dict], target: Path, title: str, tile_size: int) -> None:
    tile_w, tile_h = 4 * tile_size, tile_size + 78
    columns = min(2, len(records))
    rows = math.ceil(len(records) / columns)
    canvas = Image.new("RGB", (columns * tile_w, 44 + rows * tile_h), "#101820")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), trim_text(draw, title, canvas.width - 16), font=FONT, fill="#edf4f7")
    for index, record in enumerate(records):
        with Image.open(run / record["figure"]) as image:
            canvas.paste(image.convert("RGB"), ((index % columns) * tile_w, 44 + (index // columns) * tile_h))
    canvas.save(target, compress_level=3)


def render_experiment(run: Path, experiment: str, cases: list[dict], records: list[dict], page_size: int,
                      tile_size: int) -> dict:
    output_dir = run / "gallery" / "experiments" / slug(experiment)
    output_dir.mkdir(parents=True, exist_ok=True)
    index = output_dir / "index.html"
    count = len(records)
    page_count = math.ceil(count / page_size)
    pages = []
    first_links = {}
    for page_number in range(1, page_count + 1):
        page = output_dir / f"page_{page_number:04d}.html"
        subset = records[(page_number - 1) * page_size:page_number * page_size]
        contact = output_dir / f"contact_{page_number:04d}.png"
        make_contact_sheet(run, subset, contact, f"{experiment} | page {page_number}/{page_count}", tile_size)
        for record in subset:
            anchor = slug(record["case_id"] + ":" + str(record["frame_number_in_case"]))
            record["html_path"] = str(page.relative_to(run))
            record["html_anchor"] = anchor
            first_links.setdefault(record["case_id"], (page, anchor))
        navigation = page_navigation(page, output_dir, page_count, page_number)
        content = (f'<header><p><a href="{local_link(page, run / "index.html")}">全部实验</a></p>'
                   f'<h1>{esc(experiment)}</h1><p>第 {page_number} / {page_count} 页 · '
                   f'帧 {(page_number - 1) * page_size + 1}–{min(page_number * page_size, count)} / {count}</p>'
                   f'<p class="muted">{esc(CAPTION)} slice / volume / echo 沿用数组的 0 起始索引。</p>'
                   f'{navigation}<p><a href="{local_link(page, contact)}">本页全部帧联系图 PNG</a></p></header>'
                   + "".join(frame_html(run, page, record) for record in subset) + navigation)
        page.write_text(doc(f"{experiment} · 第 {page_number} 页", content), encoding="utf-8")
        pages.append(str(page.relative_to(run)))
    rows, metadata = [], []
    per_case = defaultdict(list)
    for record in records:
        per_case[record["case_id"]].append(record)
    for case in cases:
        subset = per_case[case["id"]]
        successful = sum(record["two_method_arrays_finite"] for record in subset)
        link = "无帧输出"
        if case["id"] in first_links:
            page, anchor = first_links[case["id"]]
            link = f'<a href="{local_link(index, page)}#{esc(anchor)}">从本扫描第一帧查看</a>'
        counts = Counter(record["status"] for record in subset)
        warning_text = warnings_html(quality_warnings(case))
        rows.append(f'<tr><td>{esc(case.get("scan_id", case["id"]))}</td>'
                    f'<td>{esc(case.get("status", "见逐帧记录"))}<br>{esc(case.get("reason", ""))}</td>'
                    f'<td>{esc(case.get("expected_frames", "未记录"))}</td><td>{len(subset)}</td><td>{successful}</td>'
                    f'<td>{esc(pretty(dict(counts)))}</td><td>{warning_text or "—"}</td><td>{link}</td></tr>')
        meta = case_metadata(case)
        metadata.append(f'<details><summary>scan {esc(case.get("scan_id", case["id"]))} · 完整来源和状态</summary>'
                        f'<pre>{esc(pretty(meta))}</pre></details>')
    total_success = sum(record["two_method_arrays_finite"] for record in records)
    warning_cases = sum(bool(quality_warnings(case)) for case in cases)
    warning_frames = sum(bool(record["quality_warnings"]) for record in records)
    body = (f'<header><a href="{local_link(index, run / "index.html")}">全部实验</a>'
            f'<h1>{esc(experiment)}</h1><p>{len(cases)} 个扫描记录 · {count} 帧输出记录 · '
            f'{total_success} 帧有两种方法的有限数组 · {page_count} 页。</p>'
            f'<p class="partial">重建质量提示：{warning_cases} 个扫描、{warning_frames} 帧。轨迹与相位拟合限制见逐帧说明；数值有限不代表图像可信。</p>'
            '<p>所有已输出帧依扫描 / echo / volume / slice 的输入顺序分页展示；'
            '未能处理的扫描保留在下表，不计作成功重建。扫描仪预览若单独输出，按 scanner frame 索引展示。</p>'
            f'<p class="muted">{esc(CAPTION)}</p>{page_navigation(index, output_dir, page_count)}</header>'
            '<section><h2>扫描覆盖</h2><div class="scroll"><table><thead><tr><th>scan</th><th>采集处理状态</th>'
            '<th>头文件声明帧</th><th>输出帧记录</th><th>双方法有限帧</th><th>展示状态计数</th><th>重建质量提示</th><th>入口</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div></section><section><h2>逐扫描来源与限制</h2>'
            + "".join(metadata) + '</section>')
    index.write_text(doc(experiment, body), encoding="utf-8")
    return {"experiment_name": experiment, "case_count": len(cases), "frame_count": count,
            "two_method_finite_frame_count": total_success, "page_count": page_count,
            "quality_warning_case_count": warning_cases, "quality_warning_frame_count": warning_frames,
            "index_path": str(index.relative_to(run)), "page_paths": pages}


class LinkCollector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []
        self.ids = set()

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids.add(attrs["id"])
        for key in ("href", "src"):
            if key in attrs:
                self.links.append(attrs[key])


def check_links(run: Path, pages: list[Path]) -> dict:
    parsed = {}
    for page in pages:
        parser = LinkCollector()
        parser.feed(page.read_text(encoding="utf-8"))
        parsed[page.resolve()] = parser
    missing, missing_anchors = [], []
    checked = 0
    for page, parser in parsed.items():
        for url in parser.links:
            parts = urlsplit(url)
            if parts.scheme or parts.netloc:
                continue
            target = (page.parent / unquote(parts.path)).resolve() if parts.path else page
            checked += 1
            if not target.exists():
                missing.append({"page": str(page.relative_to(run)), "url": url, "target": str(target)})
            elif parts.fragment and target in parsed and unquote(parts.fragment) not in parsed[target].ids:
                missing_anchors.append({"page": str(page.relative_to(run)), "url": url})
    return {"html_page_count": len(pages), "checked_local_links": checked,
            "missing_target_count": len(missing), "missing_anchor_count": len(missing_anchors),
            "missing_targets": missing, "missing_anchors": missing_anchors,
            "passed": not missing and not missing_anchors}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--page-size", type=int, default=16, help="All frames are paginated; no representative-frame sampling")
    parser.add_argument("--tile-size", type=int, default=160, help="Preview pixels per panel; single-panel PNGs retain native pixel count")
    args = parser.parse_args()
    if not 1 <= args.page_size <= 64 or not 96 <= args.tile_size <= 512:
        parser.error("page-size must be 1..64 and tile-size 96..512")
    run = args.run.resolve()
    summary, cases, errors = load_cases(run)
    (run / "gallery").mkdir(parents=True, exist_ok=True)
    records, grouped_cases = [], defaultdict(list)
    for case_number, case in enumerate(cases, 1):
        grouped_cases[case["experiment_name"]].append(case)
        for frame_number, frame in enumerate(case.get("frames", [])):
            try:
                record = render_frame(run, case, frame, frame_number, args.tile_size)
                records.append(record)
                for error in record["errors"]:
                    errors.append({"case_id": case["id"], "frame_id": record["frame_id"], **error})
            except Exception as exc:
                errors.append({"case_id": case["id"], "frame_id": frame.get("id"),
                               "error": f"{type(exc).__name__}: {exc}"})
        print(f"Gallery {case_number}/{len(cases)} cases; {len(records)} frame records; {case['id']}", flush=True)
    grouped_records = defaultdict(list)
    for record in records:
        grouped_records[record["experiment_name"]].append(record)
    experiments = [render_experiment(run, name, items, grouped_records[name], args.page_size, args.tile_size)
                   for name, items in grouped_cases.items()]
    input_frame_count = sum(len(case.get("frames", [])) for case in cases)
    counts = Counter(record["status"] for record in records)
    successful = sum(record["two_method_arrays_finite"] for record in records)
    with (run / "frame_index.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    columns = ["experiment_name", "scan_id", "case_id", "frame_id", "slice_index", "volume_index", "echo_index",
               "scanner_frame_index", "frame_kind", "status", "input_status", "two_method_arrays_finite",
               "quality_warnings", "arrays_path", "figure", "html_path", "html_anchor"]
    with (run / "frame_index.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    index = run / "index.html"
    rows = []
    for experiment in experiments:
        name = experiment["experiment_name"]
        rows.append(f'<tr class="experiment" data-name="{esc(name.lower())}">'
                    f'<td><a href="{local_link(index, run / experiment["index_path"])}">{esc(name)}</a></td>'
                    f'<td>{experiment["case_count"]}</td><td>{experiment["frame_count"]}</td>'
                    f'<td>{experiment["two_method_finite_frame_count"]}</td><td>{experiment["page_count"]}</td>'
                    f'<td class="partial">{experiment["quality_warning_case_count"]} 扫描 / {experiment["quality_warning_frame_count"]} 帧</td></tr>')
    summary_link = '<a href="summary.json">运行 summary.json</a> · ' if (run / "summary.json").is_file() else ""
    failures = summary.get("failures", [])
    zero_cases = [case_metadata(case) for case in cases if not case.get("frames")]
    body = (f'<header><h1>SPEN 全数据 · 全帧传统重建</h1>'
            f'<p>{len(experiments)} 组实验 · {len(cases)} 个扫描记录 · '
            f'{len(records)} 帧展示记录 · {successful} 帧包含两种方法的有限数组。</p>'
            '<p>每组实验独立分页，包含本次运行输出的全部 slice / volume / echo。'
            '每帧提供 ADC、RO FFT、Phase Map + InvA 与 Tikhonov 面板及原像素 PNG；'
            '不支持的采集明确显示缺失方法与处理原因。</p>'
            f'<p class="notice">{esc(CAPTION)} 扫描仪图仅作原图预览，帧映射及配准未经核实时不能作为真值。'
            '数值有限仅为数值检查，不能替代图像质量验证。扫描数、slice 数与包含 volume / echo 的帧数分别计数。</p>'
            f'<p>{summary_link}<a href="frame_index.csv">全部帧 CSV</a> · '
            '<a href="frame_index.jsonl">逐帧来源、数组和显示记录 JSONL</a> · '
            '<a href="gallery_checks.json">覆盖及链接检查</a></p>'
            f'<p class="partial">重建质量提示：{sum(experiment["quality_warning_case_count"] for experiment in experiments)} 个扫描、'
            f'{sum(bool(record["quality_warnings"]) for record in records)} 帧。轨迹与相位拟合限制会单独标注；“双方法有限帧”只表示数组已计算且数值有限。</p>'
            f'<p class="muted">输入帧记录 {input_frame_count}；展示帧记录 {len(records)}；'
            f'状态分布 {esc(dict(counts))}。每页最多 {args.page_size} 帧，未抽样。</p></header>'
            '<section><h2>按实验查看全部帧</h2><p><input id="search" placeholder="搜索实验名 / 日期"></p>'
            '<div class="scroll"><table><thead><tr><th>实验</th><th>扫描记录</th><th>全部帧记录</th>'
            '<th>双方法有限帧</th><th>页数</th><th>重建质量提示</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div></section>'
            '<section><h2>无帧输出的扫描与运行限制</h2>'
            f'<p>当前 {len(zero_cases)} 个扫描没有帧输出，未被算作完成重建。逐实验页面也保留这些扫描。</p>'
            f'<details><summary>查看无帧扫描、失败与渲染错误</summary><pre>{esc(pretty({"cases_without_frames": zero_cases, "run_failures": failures, "gallery_errors": errors}))}</pre></details>'
            '<details><summary>运行总体参数与范围</summary><pre>'
            + esc(pretty({key: value for key, value in summary.items() if key not in ("cases", "failures")}))
            + '</pre></details></section>'
            '<script>document.getElementById("search").addEventListener("input",function(){'
            'const q=this.value.toLowerCase();document.querySelectorAll(".experiment").forEach('
            'row=>row.hidden=!row.dataset.name.includes(q));});</script>')
    index.write_text(doc("SPEN 全数据 · 全帧传统重建", body), encoding="utf-8")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "run_dir": str(run),
              "experiment_count": len(experiments), "case_count": len(cases),
              "input_frame_count": input_frame_count, "indexed_frame_count": len(records),
              "rendered_frame_count": len(records),
              "scanner_preview_frames": sum("scanner_preview" in record["panels"] for record in records),
              "scanner_preview_only_frames": sum(record["frame_kind"] in ("scanner_preview", "scanner_preview_only") for record in records),
              "failed_zero_frame_cases": len(zero_cases),
              "quality_warning_case_count": sum(bool(quality_warnings(case)) for case in cases),
              "quality_warning_frame_count": sum(bool(record["quality_warnings"]) for record in records),
              "every_input_frame_indexed": input_frame_count == len(records),
              "two_method_finite_frame_count": successful, "frame_status_counts": dict(counts),
              "case_without_frames_count": len(zero_cases), "gallery_error_count": len(errors),
              "page_size": args.page_size, "representative_frame_sampling": False,
              "display_scaling_and_orientation": CAPTION, "stored_arrays_modified": False,
              "experiments": experiments, "errors": errors}
    checks_path = run / "gallery_checks.json"
    checks_path.write_text(pretty(report) + "\n", encoding="utf-8")
    from build_viewer import build_viewer
    viewer = build_viewer(run)
    report['interactive_viewer'] = {'index':'index.html', 'archive':'archive.html',
                                    'manifest':'viewer_manifest.json', 'stats':viewer['stats']}
    pages = [index, run/'archive.html'] + [run / experiment["index_path"] for experiment in experiments]
    pages += [run / path for experiment in experiments for path in experiment["page_paths"]]
    report["link_checks"] = check_links(run, pages)
    checks_path.write_text(pretty(report) + "\n", encoding="utf-8")
    print(f"Gallery complete: {index}; {len(records)}/{input_frame_count} frames; "
          f"{len(experiments)} experiments; {len(errors)} rendering errors; "
          f"links passed={report['link_checks']['passed']}", flush=True)
    return int(bool(errors) or input_frame_count != len(records) or not report["link_checks"]["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
