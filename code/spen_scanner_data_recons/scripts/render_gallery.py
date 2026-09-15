#!/usr/bin/env python3
"""Render the raw-scan reconstruction gallery from summary.json and frame NPZs.

Usage: python scripts/render_gallery.py --run runs/<run_name>

Display scaling never modifies the stored reconstruction arrays.  The two
reconstruction methods may use different operator scaling, so each panel uses
its own 99.5th percentile; ADC log-RSS uses a p1--p99.5 window. No
image-to-image accuracy metric is inferred.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from urllib.parse import quote

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


PANEL_KEYS = (
    ("sorted_samples", "原始 ADC · log(1 + RSS)", "读出采样点", "SPEN 编码点"),
    ("rofft_original", "RO FFT · RSS", "RO 像素", "SPEN 编码点"),
    ("inva_corrected", "Phase Map + InvA · RSS", "RO 像素", "PE 像素"),
    ("tikhonov_coils", "Tikhonov · RSS", "RO 像素", "PE 像素"),
)


def configure_fonts() -> None:
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ):
        if Path(path).is_file():
            font_manager.fontManager.addfont(path)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=path).get_name()
            break
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9, "savefig.facecolor": "white"})


def rss(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 2:
        return np.abs(value).astype(np.float64)
    if value.ndim != 3:
        raise ValueError(f"Expected [PE, RO, coil] or a 2D image, got {value.shape}")
    return np.sqrt(np.sum(np.abs(value.astype(np.complex128)) ** 2, axis=2))


def display_image(value: np.ndarray, logarithmic: bool = False) -> tuple[np.ndarray, dict]:
    magnitude = rss(value)
    # NaN/Inf values are counted separately in gallery_checks.json and the HTML.
    finite = np.isfinite(magnitude)
    image = np.where(finite, magnitude, 0.0)
    if logarithmic:
        image = np.log1p(image)
    positive = image[finite]
    vmin = float(np.percentile(positive, 1)) if logarithmic and positive.size else 0.0
    vmax = float(np.percentile(positive, 99.5)) if positive.size else 0.0
    fallback = vmax <= vmin
    if fallback:
        vmin = 0.0
        vmax = float(np.max(positive)) if positive.size else 1.0
        if vmax <= 0:
            vmax = 1.0
    window = {"vmin": vmin, "vmax": vmax, "transform": "log1p(RSS)" if logarithmic else "RSS",
              "requested_window": "p1-p99.5" if logarithmic else "0-p99.5",
              "degenerate_window_fallback": bool(fallback)}
    return image, window


def array_check(value: np.ndarray) -> dict:
    value = np.asarray(value)
    finite = np.isfinite(value)
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "complex": bool(np.iscomplexobj(value)),
        "elements": int(value.size),
        "nonfinite_elements": int(value.size - np.count_nonzero(finite)),
        "nonzero_elements": int(np.count_nonzero(value[finite])),
    }


def panel(ax, value: np.ndarray, title: str, xlabel: str, ylabel: str, *, logarithmic=False, flip_image=True) -> dict:
    image, window = display_image(value, logarithmic)
    if flip_image:
        image = np.flip(image, axis=(0, 1))
    ax.imshow(image, cmap="gray", vmin=window["vmin"], vmax=window["vmax"], origin="upper", interpolation="nearest", aspect="equal")
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7, length=2)
    window_label = f"p1–p99.5: {window['vmin']:.3g}–{window['vmax']:.3g}" if logarithmic else f"p99.5 = {window['vmax']:.3g}"
    if window["degenerate_window_fallback"]:
        window_label = f"window fallback: {window['vmin']:.3g}–{window['vmax']:.3g}"
    ax.text(0.98, 0.02, window_label, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 2})
    return window


def load_arrays(run_dir: Path, frame: dict) -> dict[str, np.ndarray]:
    arrays_path = Path(frame["arrays_path"])
    if not arrays_path.is_absolute():
        arrays_path = run_dir / arrays_path
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, *_ in PANEL_KEYS:
        if key not in arrays:
            raise KeyError(f"{arrays_path}: missing required array {key}")
    return arrays


def frame_title(case: dict, frame: dict) -> str:
    return (f"{case.get('label', case['id'])} | 扫描 {case.get('scan_id', '?')} | "
            f"slice index {frame.get('slice_index', '?')} · volume index {frame.get('volume_index', '?')}")


def render_frame(case: dict, frame: dict, arrays: dict, target: Path) -> dict:
    fig, axes = plt.subplots(1, 4, figsize=(14.4, 4.4))
    scales = {}
    for ax, (key, title, xlabel, ylabel) in zip(axes, PANEL_KEYS):
        scales[key] = panel(ax, arrays[key], title, xlabel, ylabel, logarithmic=key == "sorted_samples", flip_image=key != "sorted_samples")
    fig.suptitle(frame_title(case, frame), fontsize=12, y=0.985)
    fig.text(0.5, 0.04, "ADC log：p1–p99.5 窗口，保留测量轴；其余：0–p99.5，翻转 PE / RO。各列独立显示，强度不可直接定量比较；无平滑。", ha="center", fontsize=9)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.2, top=0.85, wspace=0.27)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return scales


def render_detail(case: dict, frame: dict, arrays: dict, target: Path) -> bool:
    scanner_index = int(frame.get("slice_index", 0)) + int(case.get("parameters", {}).get("slices", 1)) * int(frame.get("volume_index", 0))
    candidates = [
        ("rofft_corrected", "相位校正后的 RO FFT · RSS"),
        ("inva_uncorrected", "InvA 未做相位校正 · RSS"),
        ("traditional_adaptive", "InvA · adaptive 合并幅度"),
        ("scanner_preview", f"扫描仪 frame {scanner_index} · 原方向 / 未验证配准"),
    ]
    candidates = [(key, title) for key, title in candidates if key in arrays]
    if not candidates:
        return False
    fig, axes = plt.subplots(1, len(candidates), figsize=(3.7 * len(candidates), 4.4), squeeze=False)
    for ax, (key, title) in zip(axes[0], candidates):
        panel(ax, arrays[key], title, "列索引", "行索引", flip_image=key != "scanner_preview")
    fig.suptitle(frame_title(case, frame) + " · 补充检查", fontsize=11, y=0.985)
    note = "各列独立显示归一化；未校正 InvA 用于观察相位校正效果。"
    if "scanner_preview" in arrays:
        note += "扫描仪预览不是高分辨率真值。"
    fig.text(0.5, 0.04, note, ha="center", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.85, wspace=0.28)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return True


def render_tikhonov_sweep(case: dict, frame: dict, arrays: dict, target: Path) -> bool:
    if "tikhonov_sweep_coils" not in arrays:
        return False
    sweep = arrays["tikhonov_sweep_coils"]
    lambdas = np.asarray(arrays["tikhonov_lambdas"]).reshape(-1)
    if sweep.ndim != 4 or sweep.shape[0] != lambdas.size:
        raise ValueError(f"Tikhonov sweep/lambdas shapes disagree: {sweep.shape}, {lambdas.shape}")
    fig, axes = plt.subplots(1, lambdas.size, figsize=(3.8 * lambdas.size, 4.4), squeeze=False)
    for ax, value, strength in zip(axes[0], sweep, lambdas):
        panel(ax, value, f"Tikhonov · λ relative = {float(strength):g}", "RO 像素", "PE 像素")
    fig.suptitle(frame_title(case, frame) + " · 正则强度探索", fontsize=11, y=0.985)
    selected_lambda = frame.get("tikhonov", {}).get("lambda_relative", "未记录")
    fig.text(0.5, 0.04, f"各图独立按 p99.5 显示；主图配置 λ relative = {selected_lambda}，未按图像效果挑选最优参数。", ha="center", fontsize=8)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.85, wspace=0.28)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return True


def representative_frame(case: dict, valid: list[dict]) -> dict | None:
    if not valid:
        return None
    # Select the central acquired slice from the first available volume.
    first_volume = min(frame.get("volume_index", 0) for frame in valid)
    candidates = [frame for frame in valid if frame.get("volume_index", 0) == first_volume]
    candidates.sort(key=lambda frame: frame.get("slice_index", 0))
    return candidates[len(candidates) // 2]


def render_overview(run_dir: Path, representatives: list[tuple[dict, dict]], output: Path) -> None:
    if not representatives:
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.axis("off")
        ax.text(0.5, 0.5, "尚无可展示的成功重建；请查看 index.html 中的失败记录。", ha="center")
        fig.savefig(output, dpi=150)
        plt.close(fig)
        return
    rows = len(representatives)
    fig, axes = plt.subplots(rows, 4, figsize=(14.4, rows * 3.55 + 0.8), squeeze=False)
    for row, (case, frame) in enumerate(representatives):
        arrays = load_arrays(run_dir, frame)
        for ax, (key, title, xlabel, ylabel) in zip(axes[row], PANEL_KEYS):
            panel(ax, arrays[key], title, xlabel, ylabel, logarithmic=key == "sorted_samples", flip_image=key != "sorted_samples")
    figure_height = rows * 3.55 + 0.8
    fig.suptitle("SPEN 原始采样与传统重建 · 每个扫描的代表层", fontsize=14, y=1 - 0.10 / figure_height)
    fig.text(0.5, 0.13 / figure_height, "ADC 已排序及反向读出校正：log-RSS 按 p1–p99.5 显示；其余图按 0–p99.5。强度不可直接定量比较。首个 volume 的中间 slice。",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=0.045, right=0.99, bottom=0.48 / figure_height + 0.025,
                        top=1 - 1.0 / figure_height, wspace=0.27, hspace=0.7)
    fig.canvas.draw()
    for row, (case, frame) in enumerate(representatives):
        row_top = max(ax.get_position().y1 for ax in axes[row])
        fig.text(0.045, row_top + 0.36 / figure_height, frame_title(case, frame), fontsize=10, ha="left", va="bottom")
    # Retain every scan while bounding raster dimensions for very large runs.
    dpi = min(150, max(60, int(18000 / (rows * 3.55 + 0.8))))
    fig.savefig(output, dpi=dpi)
    plt.close(fig)


def esc(value) -> str:
    return html.escape(str(value))


def pretty(value) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def local_url(relative: str) -> str:
    return quote(relative, safe="/")


def metadata_rows(case: dict) -> str:
    rows = [
        ("实验", case.get("experiment_name", "")),
        ("扫描 ID", case.get("scan_id", "")),
        ("原始扫描路径", case.get("raw_scan_path", "")),
        ("原始数组形状", case.get("raw_shape", "")),
        ("对应历史 MAT", case.get("reference_mat_path") or "未提供"),
        ("采集参数（矩阵 / FOV / 分段等）", case.get("parameters", {})),
    ]
    for key in ("validation", "checks", "warnings", "reference_comparison", "scanner_preview"):
        if case.get(key):
            rows.append((key, case[key]))
    return "".join(f"<tr><th>{esc(label)}</th><td><pre>{esc(pretty(value))}</pre></td></tr>" for label, value in rows)


def render_html(summary: dict, case_entries: list[dict], errors: list[dict], run_dir: Path) -> str:
    total_frames = sum(len(entry["frames"]) for entry in case_entries)
    successful_cases = sum(bool(entry["frames"]) for entry in case_entries)
    options = '<option value="all">所有扫描</option>' + "".join(
        f'<option value="{index}">{esc(entry["case"].get("label", entry["case"]["id"]))} / scan {esc(entry["case"].get("scan_id", ""))}</option>'
        for index, entry in enumerate(case_entries)
    )
    sections = []
    for index, entry in enumerate(case_entries):
        case = entry["case"]
        frames_html = []
        for item in entry["frames"]:
            frame = item["frame"]
            figure_url = local_url(item["figure"])
            arrays_url = local_url(frame["arrays_path"]) if not Path(frame["arrays_path"]).is_absolute() else Path(frame["arrays_path"]).as_uri()
            checks = item["checks"]
            all_finite = all(check["nonfinite_elements"] == 0 for check in checks.values())
            check_label = "数组检查：全部数值有限" if all_finite else "数组检查：含 NaN / Inf，请先查看检查记录"
            tables = "".join(
                f'<tr><td><code>{esc(key)}</code></td><td>{esc(" × ".join(map(str, value["shape"])))}</td>'
                f'<td>{esc(value["dtype"])}</td><td>{value["nonfinite_elements"]}</td><td>{value["nonzero_elements"]}</td></tr>'
                for key, value in checks.items()
            )
            supplement = ""
            if item.get("detail_figure"):
                detail_url = local_url(item["detail_figure"])
                supplement = f'<details><summary>相位校正及补充预览</summary><a href="{detail_url}"><img loading="lazy" src="{detail_url}" alt="相位校正及补充检查"></a></details>'
            if item.get("sweep_figure"):
                sweep_url = local_url(item["sweep_figure"])
                lambda_list = " / ".join(f"{float(value):g}" for value in item["sweep_lambdas"])
                selected_lambda = frame.get("tikhonov", {}).get("lambda_relative", "未记录")
                supplement += f'<details><summary>Tikhonov 正则强度探索 · {esc(lambda_list)}</summary><a href="{sweep_url}"><img loading="lazy" src="{sweep_url}" alt="Tikhonov 正则强度探索"></a><p class="caption">主图配置 λ relative = {esc(selected_lambda)}；本组仅作参数敏感性观察，各列独立显示归一化。</p></details>'
            tikhonov = ""
            if frame.get("tikhonov"):
                tikhonov = f'<p>Tikhonov 求解记录（残差衡量拟合采样的程度，不代表图像准确度）：</p><pre>{esc(pretty(frame["tikhonov"]))}</pre>'
            additional = {key: frame[key] for key in ("validation", "checks", "warnings", "reference_comparison", "mat_regression") if frame.get(key)}
            frames_html.append(f'''<article class="frame">
<h3>{esc(frame_title(case, frame))}</h3>
<a href="{figure_url}" title="打开完整 PNG"><img loading="lazy" src="{figure_url}" alt="ADC、RO FFT、Phase Map + InvA 与 Tikhonov 对比"></a>
<p class="caption">ADC 已做采样排序和反向读出校正，保留测量轴方向；其 log-RSS 显示窗口为 p1–p99.5，增强采样纹理可见性。其他图按 0–p99.5 显示，翻转 PE 与 RO，与旧预览方向保持一致。四列独立显示归一化；两种重建的绝对强度不能直接比较。像素显示采用 nearest，无平滑。</p>
<p><a href="{arrays_url}">下载未归一化数组 NPZ</a> · <a href="{figure_url}">打开 PNG</a></p>
{supplement}
<details><summary>{esc(check_label)}</summary>
<div class="scroll"><table><thead><tr><th>数组</th><th>形状</th><th>类型</th><th>非有限元素</th><th>有限非零元素</th></tr></thead><tbody>{tables}</tbody></table></div>
{tikhonov}<pre>{esc(pretty(additional)) if additional else ""}</pre>
</details></article>''')
        if not frames_html:
            frames_html.append('<p class="notice">本扫描没有可展示的帧；请查看下方失败记录。</p>')
        sections.append(f'''<section class="case" data-case="{index}"><h2>{esc(case.get("label", case["id"]))} · scan {esc(case.get("scan_id", ""))}</h2>
<details><summary>来源与采集参数 · {len(entry["frames"])} 帧</summary><table>{metadata_rows(case)}</table></details>
{"".join(frames_html)}</section>''')
    failures = summary.get("failures", [])
    failure_html = ""
    if failures or errors:
        failure_html = f'<section><h2>失败与跳过记录</h2><p>以下记录未被计入可展示帧。</p><pre>{esc(pretty({"reconstruction_failures": failures, "gallery_errors": errors}))}</pre></section>'
    return f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>SPEN 原始数据重建</title>
<style>
:root{{color-scheme:light;--ink:#182632;--muted:#566877;--line:#dce3e8;--accent:#12677b}}
*{{box-sizing:border-box}} body{{margin:0;background:#f3f5f7;color:var(--ink);font:15px/1.65 system-ui,"Noto Sans CJK SC",sans-serif}}
main{{max-width:1440px;margin:0 auto;padding:28px 24px 64px}} h1{{font-size:28px;margin:0 0 12px}} h2{{font-size:21px;margin:0 0 16px}} h3{{font-size:16px;margin:0 0 12px}}
p{{margin:10px 0}} a{{color:var(--accent)}} header,section{{background:white;border:1px solid var(--line);border-radius:8px;padding:24px;margin-bottom:24px}}
.lead{{font-size:17px}} .caption,.meta{{color:var(--muted);font-size:13px}} .notice{{background:#fff7e6;border-left:3px solid #a67a26;padding:12px 16px}}
.toolbar{{position:sticky;top:0;z-index:2;background:#f3f5f7ef;backdrop-filter:blur(8px);padding:14px 0;margin-bottom:14px;border-bottom:1px solid var(--line)}}
select{{padding:8px 12px;max-width:100%;font:inherit;border:1px solid #abbac4;border-radius:5px;background:white;color:var(--ink)}}
img{{width:100%;height:auto;display:block;background:#fff}} .frame{{border-top:1px solid var(--line);margin-top:22px;padding-top:22px}} details{{margin:12px 0}} summary{{cursor:pointer;color:var(--accent)}}
table{{border-collapse:collapse;width:100%;margin:14px 0;font-size:13px}} th,td{{border-bottom:1px solid var(--line);padding:9px 12px;text-align:left;vertical-align:top}} th{{font-weight:600;white-space:nowrap;background:#f7f9fa}}
pre{{margin:0;white-space:pre-wrap;overflow-wrap:anywhere;font:12px/1.65 ui-monospace,monospace}} code{{font:12px ui-monospace,monospace}} .scroll{{overflow:auto}} [hidden]{{display:none!important}}
@media(max-width:700px){{main{{padding:16px 8px}} header,section{{padding:16px 10px}}h1{{font-size:23px}}th,td{{padding:6px}}}}
</style></head><body><main>
<header><h1>SPEN 原始采样与传统重建</h1>
<p class="lead">Phase Map + InvA 与 Tikhonov：从采样信号到图像的对照。</p>
<p>本次可展示 <strong>{successful_cases}</strong> 个扫描、<strong>{total_frames}</strong> 帧。范围以本次配置和 <a href="summary.json">summary.json</a> 为准。</p>
<p class="notice">这里的“原始”指从扫描仪 ADC 数据整理出的采样信号。它已经过排序及反向读出校正，并非可直接观看的解剖图像。ADC 的 log-RSS 使用 p1–p99.5 显示窗口；RO FFT 仍保留 SPEN 编码。其余图像逐面板独立使用 0–p99.5 窗口。分位数重合时安全退回 0–最大值窗口（全零或无有限值则使用 0–1）；未归一化数组保存在 NPZ 中。</p>
<p>概览每个扫描选取首个 volume 的中间 slice；全部已重建帧见下方。层和 volume 保留 runner 中的索引值。扫描仪预览若存在，仅供观察，其配准与处理流程尚未核实，不能作为高分辨率真值。</p>
<p>显示方向：ADC 保留测量坐标，重建图及 RO FFT 同时翻转 PE / RO 以沿用历史预览约定；这不定义解剖方向。扫描仪预览保留 scanner frame 原方向。NPZ 保留 native axes，不修改数据。主图 Tikhonov 配置 λ relative = {esc(summary.get("lambda_relative", "见逐帧求解记录"))}，可用的正则强度探索附在各帧详情中。</p>
<p class="meta">生成时间：{esc(summary.get("created_at", "未记录"))}<br>数据：{esc(summary.get("data_root", ""))}<br>运行目录：{esc(run_dir)}</p>
<p><a href="overview.png">打开概览 PNG</a> · <a href="gallery_checks.json">数组检查记录</a></p>
<a href="overview.png"><img src="overview.png" alt="每个扫描的代表层概览"></a></header>
<div class="toolbar"><label for="case-filter">查看扫描：</label> <select id="case-filter">{options}</select></div>
{"".join(sections)}{failure_html}
<p class="meta">图像使用灰度和 nearest 像素显示；ADC 使用 log(1 + RSS) 的 p1–p99.5 窗口，其他主图使用 RSS 的 0–p99.5 窗口。有限值检查不等于重建准确性验证。</p>
</main><script>
document.getElementById('case-filter').addEventListener('change', function(){{
  const selected = this.value;
  document.querySelectorAll('.case').forEach(section => {{section.hidden = selected !== 'all' && section.dataset.case !== selected;}});
}});
</script></body></html>'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Run directory containing summary.json")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    with (run_dir / "summary.json").open(encoding="utf-8") as handle:
        summary = json.load(handle)
    configure_fonts()
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    case_entries = []
    representatives = []
    errors = []
    check_records = []
    used_names: set[str] = set()
    for case in summary.get("cases", []):
        entries = []
        for frame in case.get("frames", []):
            try:
                frame_id = str(frame["id"])
                if not frame_id or Path(frame_id).name != frame_id or frame_id in (".", ".."):
                    raise ValueError(f"Frame id must be a file-safe basename: {frame_id!r}")
                if frame_id in used_names:
                    raise ValueError(f"Frame id is not globally unique: {frame_id!r}")
                used_names.add(frame_id)
                arrays = load_arrays(run_dir, frame)
                checks = {key: array_check(value) for key, value in arrays.items()}
                figure_relative = f"figures/{frame_id}.png"
                scales = render_frame(case, frame, arrays, run_dir / figure_relative)
                detail_relative = f"figures/{frame_id}_detail.png"
                has_detail = render_detail(case, frame, arrays, run_dir / detail_relative)
                sweep_relative = f"figures/{frame_id}_tikhonov_sweep.png"
                has_sweep = render_tikhonov_sweep(case, frame, arrays, run_dir / sweep_relative)
                entry = {"frame": frame, "figure": figure_relative, "checks": checks, "display_windows": scales}
                if has_detail:
                    entry["detail_figure"] = detail_relative
                if has_sweep:
                    entry["sweep_figure"] = sweep_relative
                    entry["sweep_lambdas"] = arrays["tikhonov_lambdas"].reshape(-1).tolist()
                entries.append(entry)
                check_records.append({"case_id": case["id"], "frame_id": frame_id, "arrays_path": frame["arrays_path"],
                                      "arrays": checks, "display_windows": scales,
                                      "display_p99_5": {key: value["vmax"] for key, value in scales.items()}, "figure": figure_relative})
                print(f"Rendered {case['id']} / {frame_id}", flush=True)
            except Exception as exc:
                plt.close("all")
                error = {"case_id": case.get("id"), "frame_id": frame.get("id"), "arrays_path": frame.get("arrays_path"),
                         "error_type": type(exc).__name__, "error": str(exc)}
                errors.append(error)
                print(f"Skipped frame: {json.dumps(error, ensure_ascii=False)}", flush=True)
        case_entries.append({"case": case, "frames": entries})
        representative = representative_frame(case, [item["frame"] for item in entries])
        if representative is not None:
            representatives.append((case, representative))
    render_overview(run_dir, representatives, run_dir / "overview.png")
    report = {"summary_path": str(run_dir / "summary.json"), "rendered_case_count": len(representatives),
              "rendered_frame_count": len(check_records), "gallery_error_count": len(errors),
              "display_scaling": "ADC log1p(RSS) uses p1-p99.5; other panels use RSS with 0-p99.5. Each panel is independent. Degenerate windows fall back to 0-max (0-1 for zero/empty finite data).",
              "display_orientation": {"sorted_samples": "native measurement axes; no display flip",
                                      "scanner_preview": "original scanner frame orientation; no display flip; registration unverified",
                                      "reconstruction_and_rofft_panels": "flip first two axes (PE and RO) for legacy preview orientation; not anatomical orientation",
                                      "stored_npz_arrays": "native axes, unchanged"},
              "stored_arrays_modified": False, "frames": check_records, "errors": errors}
    (run_dir / "gallery_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (run_dir / "index.html").write_text(render_html(summary, case_entries, errors, run_dir), encoding="utf-8")
    print(f"Gallery: {run_dir / 'index.html'} ({len(check_records)} frames, {len(errors)} rendering errors)", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
