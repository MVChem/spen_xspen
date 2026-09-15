"""Build traceable old/new 192-pixel comparisons from completed export runs."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw

from build_gallery import put_text, read_preview
from image_processing import fit_plane
from nifti_sources import load_nifti

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]


def read_records(run):
    summary = json.loads((run / "summary.json").read_text())
    records = [json.loads(line) for line in (run / "manifest.jsonl").read_text().splitlines()]
    if len(records) != summary["image_count"]:
        raise ValueError(f"Manifest count does not match completed summary: {run}")
    return records


def key(record):
    return record["source_id"], record["slice_index"]


def middle_per_source(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["source_id"]].append(record)
    return [sorted(group, key=lambda row: row["slice_index"])[len(group) // 2]
            for _, group in sorted(groups.items())]


def choose_examples(old_records, records):
    old_by_key = {key(row): row for row in old_records}
    new_by_key = {key(row): row for row in records}
    chosen, used = [], set()

    def add(record):
        if key(record) not in used:
            chosen.append(record)
            used.add(key(record))

    for number in (534, 561, 818):
        old = next((row for row in old_records if row["num"] == number), None)
        if old and key(old) in new_by_key:
            add(new_by_key[key(old)])
    for dataset, fractions in [("ds005186", [0.5]), ("ds005236", [0.5, 1.0])]:
        representatives = middle_per_source([
            row for row in records if row["dataset"] == dataset and key(row) in old_by_key
            and row["source_id"] not in {item["source_id"] for item in chosen}
        ])
        for fraction in fractions:
            if representatives:
                add(representatives[round((len(representatives) - 1) * fraction)])

    laboratory = middle_per_source([
        row for row in records if row["dataset"] == "lab_mouse" and key(row) in old_by_key
    ])
    seen_shapes, seen_spacings, seen_modes = set(), set(), set()
    for _ in range(min(7, len(laboratory))):
        def novelty(row):
            transform = row["transform"]
            shape = tuple(transform["native_plane_shape"])
            spacing = tuple(round(value, 5) for value in transform["native_spacing_yx"])
            mode = bool(transform["direct_native_crop"])
            return (100 * (shape not in seen_shapes) + 10 * (spacing not in seen_spacings)
                    + 2 * (mode not in seen_modes), -np.prod(shape))

        record = max(laboratory, key=novelty)
        laboratory.remove(record)
        transform = record["transform"]
        seen_shapes.add(tuple(transform["native_plane_shape"]))
        seen_spacings.add(tuple(round(value, 5) for value in transform["native_spacing_yx"]))
        seen_modes.add(bool(transform["direct_native_crop"]))
        add(record)
    rats = middle_per_source([row for row in records if row["dataset"] == "ds002870"])
    for index in np.linspace(0, len(rats) - 1, min(3, len(rats))).round().astype(int):
        add(rats[index])
    if not chosen:
        raise ValueError("No matching old/new or newly added ds002870 examples")
    return chosen, old_by_key


def captions(record):
    transform = record["transform"]
    shape = transform["native_plane_shape"]
    spacing = transform["native_spacing_yx"]
    matrix = f"原生面内 {shape[0]}×{shape[1]} · 原始层号 {record['slice_index']}"
    sampling = f"面内采样 {spacing[0] * 1000:.2f}×{spacing[1] * 1000:.2f} μm"
    if transform["direct_native_crop"]:
        operation = "直接裁取 192×192；无插值"
    else:
        spans = np.array(transform["sampling_yx"]) * 192
        operation = f"约 {spans[0]:.0f}×{spans[1]:.0f} 原生区域缩至 192×192"
    return matrix, sampling, operation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    old_run, run = args.old_run.resolve(), args.run.resolve()
    old_records, records = read_records(old_run), read_records(run)
    chosen, old_by_key = choose_examples(old_records, records)
    sources = {row["source_id"]: row for row in json.loads((run / "selected_sources.json").read_text())}
    gallery = run / "gallery"
    gallery.mkdir(exist_ok=True)
    baselines = gallery / "comparison_baselines"
    baselines.mkdir(exist_ok=True)
    pairs = []
    for record in chosen:
        previous = old_by_key.get(key(record))
        if previous:
            left_run, left_record = old_run, previous
            left_label = f"旧版 #{previous['num']}"
            baseline_kind = "previous_export_same_source_same_slice"
        else:
            source = sources[record["source_id"]]
            volume, metadata = load_nifti(source)
            upper = record["transform"]["normalization_window"][1]
            pixels, transform = fit_plane(volume[record["slice_index"]], metadata["spacing_yx"], upper,
                                          size=192, bit_depth=record["transform"]["bit_depth"], crop_mode="full")
            name = f"new_source_{record['num']}_full_fov.png"
            Image.fromarray(pixels).save(baselines / name)
            left_run = run
            left_record = {"image": f"gallery/comparison_baselines/{name}", "filename": name,
                           "transform": transform}
            left_label = "新增来源 · 全视野参照"
            baseline_kind = "new_source_no_old_export_full_fov_baseline"
        left_image = read_preview(left_run, left_record)
        right_image = read_preview(run, record)
        if left_image.size != (192, 192) or right_image.size != (192, 192):
            raise ValueError("Comparison expects exact 192x192 exported previews")
        pairs.append(dict(record=record, left_record=left_record, left_run=left_run,
                          left_image=left_image, right_image=right_image,
                          left_label=left_label, baseline_kind=baseline_kind))

    columns, cell_width, cell_height, margin = 2, 500, 324, 26
    canvas = Image.new("RGB", (columns * cell_width + 2 * margin,
                               ((len(pairs) + 1) // 2) * cell_height + 160), "#101723")
    draw = ImageDraw.Draw(canvas)
    put_text(draw, (margin, 20), "192×192 鼠脑预处理对比", canvas.width - margin * 2, 29, "white")
    put_text(draw, (margin, 64), "同一来源、同一原始层号；全部图像按原导出亮度窗口显示", canvas.width - margin * 2, 18)
    put_text(draw, (margin, 94), "左：旧版或新增来源的全视野参照　　右：新版（原生≥192，不放大、不补黑边）",
             canvas.width - margin * 2, 16)
    cards, traces = [], []
    for index, pair in enumerate(pairs):
        record = pair["record"]
        transform = record["transform"]
        x = margin + index % columns * cell_width
        y = 144 + index // columns * cell_height
        title = f"{index + 1:02d} · {record['dataset']} · {record['subject']}"
        put_text(draw, (x, y), title, cell_width - 24, 17, "white")
        put_text(draw, (x + 10, y + 28), pair["left_label"], 225, 16)
        put_text(draw, (x + 258, y + 28), f"新版 #{record['num']}", 210, 16, "#8ce6b5")
        canvas.paste(pair["left_image"], (x + 10, y + 54))
        canvas.paste(pair["right_image"], (x + 258, y + 54))
        matrix, sampling, operation = captions(record)
        put_text(draw, (x + 10, y + 250), matrix, 470, 15)
        put_text(draw, (x + 10, y + 272), sampling, 470, 14)
        put_text(draw, (x + 10, y + 294), operation, 470, 15, "#8ce6b5")
        left_path = pair["left_run"] / pair["left_record"]["image"]
        right_path = run / record["image"]
        left_link = Path(os.path.relpath(left_path, gallery)).as_posix()
        right_link = Path(os.path.relpath(right_path, gallery)).as_posix()
        e = html.escape
        details = f"{matrix}；{sampling}；{operation}"
        note = ("此来源本次新增，没有旧版 PNG；左图使用同一源层、同一亮度窗口，将原生全视野缩至 192 作参照。"
                if pair["baseline_kind"].startswith("new_source") else
                "左右 PNG 来自两次正式导出，并以来源 ID 和原始层号匹配。")
        cards.append(f'''<article><h2>{e(title)}</h2><div class="pair">
<figure><figcaption>{e(pair['left_label'])}</figcaption><a href="{e(left_link)}"><img width="192" height="192" src="{e(left_link)}" alt="{e(pair['left_label'])}"></a></figure>
<figure><figcaption>新版 #{record['num']}</figcaption><a href="{e(right_link)}"><img width="192" height="192" src="{e(right_link)}" alt="新版"></a></figure></div>
<p>{e(details)}</p><p class="small">{e(record['source_id'])} · {e(record['sequence'])}<br>{e(note)}</p></article>''')
        traces.append(dict(source_id=record["source_id"], slice_index=record["slice_index"],
                           old_num=pair["left_record"].get("num"), new_num=record["num"],
                           left_image=str(left_path), new_image=str(right_path),
                           baseline_kind=pair["baseline_kind"], native_plane_shape=transform["native_plane_shape"],
                           direct_native_crop=transform["direct_native_crop"], transform=transform))
    jpg_path = gallery / "before_after.jpg"
    canvas.save(jpg_path, quality=95, subsampling=0)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>鼠脑预处理前后对比</title><style>
:root{color-scheme:dark;font:16px system-ui,sans-serif;background:#101723;color:#d6dde7}body{max-width:1080px;margin:auto;padding:28px}
h1{font-size:28px;color:white}h2{font-size:17px;color:white;margin:0 0 14px}p{line-height:1.7}a{color:#9bc9ff}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(430px,1fr));gap:20px}
article{padding:18px;background:#192536;border-radius:10px;overflow:auto}.pair{display:flex;gap:28px}figure{margin:0}figcaption{font-size:15px;margin-bottom:9px}img{display:block;width:192px;height:192px}.small{font-size:13px;color:#aab8c9;overflow-wrap:anywhere}
@media(max-width:500px){body{padding:14px}.grid{display:block}article{margin:20px 0}.pair{gap:10px}}
</style><h1>192×192 鼠脑预处理前后对比</h1>
<p>同一来源、同一原始层号，保留原导出亮度窗口。每幅图按 192×192 显示，点击可打开 PNG 原件。新版只使用原生面内两轴均不小于 192 的数据；前景检测仅定位裁剪区域，不是脑分割。</p>
<p><a href="before_after.jpg">下载整张对比图</a> · <a href="index.html">全部新版图像</a> · <a href="before_after.json">对比来源清单</a></p><div class="grid">''' + "\n".join(cards) + "</div></html>"
    (gallery / "before_after.html").write_text(page)
    (gallery / "before_after.json").write_text(json.dumps(traces, ensure_ascii=False, indent=2) + "\n")
    temporary = WORKSPACE / "tmp/preprocess_crop_260914"
    temporary.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(jpg_path, temporary / "before_after.jpg")
    print(json.dumps(dict(examples=len(pairs), jpg=str(jpg_path), html=str(gallery / "before_after.html"),
                          convenient_preview=str(temporary / "before_after.jpg")), ensure_ascii=False))


if __name__ == "__main__":
    main()
