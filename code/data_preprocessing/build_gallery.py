"""Build an offline, filterable gallery and contact sheets for an exported run."""
from __future__ import annotations

import argparse
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


FONT_PATHS = (
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
)


def font(size):
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def read_preview(run, record):
    """Apply the fixed PNG bit-depth range, preserving the export window."""
    with Image.open(run / record['image']) as image:
        pixels = np.asarray(image)
        bit_depth = record.get('transform', {}).get('bit_depth')
        if bit_depth is None:
            bit_depth = 16 if image.mode.startswith('I') else 8
        if bit_depth not in (8, 16):
            raise ValueError(f"Unsupported bit depth in {record['filename']}: {bit_depth}")
        pixels = np.rint(np.clip(pixels.astype(np.float64), 0, 2 ** bit_depth - 1)
                         * (255 / (2 ** bit_depth - 1))).astype(np.uint8)
    return Image.fromarray(pixels).convert('RGB')


def fit_text(draw, text, width, face):
    text = str(text)
    if draw.textlength(text, font=face) <= width:
        return text
    while text and draw.textlength(text + '…', font=face) > width:
        text = text[:-1]
    return text + '…'


def put_text(draw, xy, text, width, size=17, fill='#d6dde7'):
    face = font(size)
    draw.text(xy, fit_text(draw, text, width, face), font=face, fill=fill)


def paste_preview(canvas, run, record, xy, size=192):
    image = read_preview(run, record)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    x, y = xy
    canvas.paste(image, (x + (size - image.width) // 2, y + (size - image.height) // 2))


def group_sources(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record['dataset'], record['source_id'])].append(record)
    return [(key, sorted(group, key=lambda record: (record['slice_index'], record['num'])))
            for key, group in sorted(groups.items())]


def overview_samples(sources, count):
    datasets = defaultdict(list)
    for (dataset, _), group in sources:
        datasets[dataset].append(group)
    selected = []
    for dataset_groups in datasets.values():
        # Spread choices across the source list, then prefer contrast diversity.
        indices = np.linspace(0, len(dataset_groups) - 1,
                              min(count, len(dataset_groups))).round().astype(int)
        choices = [dataset_groups[index] for index in indices]
        represented = {group[0]['sequence'] for group in choices}
        for group in dataset_groups:
            sequence = group[0]['sequence']
            if sequence in represented:
                continue
            for index in range(len(choices) - 1, -1, -1):
                previous = choices[index][0]['sequence']
                if sum(other[0]['sequence'] == previous for other in choices) > 1:
                    choices[index] = group
                    represented.add(sequence)
                    break
        selected.extend(group[len(group) // 2] for group in choices)
    return selected


def make_overview(run, gallery, sources, count):
    selected = overview_samples(sources, count)
    columns, cell_width, cell_height, margin = 4, 270, 290, 24
    rows = max(1, (len(selected) + columns - 1) // columns)
    canvas = Image.new('RGB', (columns * cell_width + 2 * margin,
                               rows * cell_height + 90), '#101723')
    draw = ImageDraw.Draw(canvas)
    put_text(draw, (margin, 18), f'鼠脑图像概览 · {len(selected)} 个来源', canvas.width - 48, 26, '#ffffff')
    for index, record in enumerate(selected):
        x = margin + (index % columns) * cell_width
        y = 75 + (index // columns) * cell_height
        paste_preview(canvas, run, record, (x + (cell_width - 192) // 2, y))
        put_text(draw, (x, y + 198), record['dataset'], cell_width - 12, 17, '#ffffff')
        put_text(draw, (x, y + 222), f"{record['subject']} · {record['sequence']}", cell_width - 12, 15)
        put_text(draw, (x, y + 246), f"#{record['num']} · slice {record['slice_index']}", cell_width - 12, 15)
    path = gallery / 'overview.jpg'
    canvas.save(path, quality=92, subsampling=0)
    return path


def make_source_pages(run, gallery, sources, per_page):
    output = gallery / 'source_pages'
    output.mkdir(exist_ok=True)
    pages = []
    columns, cell_width, cell_height, margin = 2, 608, 286, 24
    for offset in range(0, len(sources), per_page):
        batch = sources[offset:offset + per_page]
        rows = (len(batch) + columns - 1) // columns
        canvas = Image.new('RGB', (columns * cell_width + margin * 2,
                                   rows * cell_height + 90), '#101723')
        draw = ImageDraw.Draw(canvas)
        page_number = len(pages) + 1
        put_text(draw, (margin, 16), f'来源质检 {page_number} · 首 / 中 / 末张', canvas.width - 48, 26, '#ffffff')
        for index, ((dataset, source), group) in enumerate(batch):
            x = margin + (index % columns) * cell_width
            y = 72 + (index // columns) * cell_height
            put_text(draw, (x, y), f'{dataset} · {source}', cell_width - 20, 16, '#ffffff')
            put_text(draw, (x, y + 23), f"{group[0]['sequence']} · {len(group)} 张", cell_width - 20, 15)
            chosen = [group[0], group[len(group) // 2], group[-1]]
            for column, record in enumerate(chosen):
                image_x = x + column * 198
                paste_preview(canvas, run, record, (image_x, y + 51))
                put_text(draw, (image_x, y + 247), f"#{record['num']} · slice {record['slice_index']}", 190, 14)
        path = output / f'{page_number:03d}.jpg'
        canvas.save(path, quality=90, subsampling=0)
        pages.append(dict(file=f'source_pages/{path.name}', number=page_number,
                          first=offset + 1, last=offset + len(batch)))
    return pages


HTML = r'''<!doctype html>
<html lang="zh-CN">
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>鼠脑图像 · 192 × 192</title>
<style>
:root{color-scheme:dark;font:15px system-ui,sans-serif;background:#101723;color:#d6dde7}
*{box-sizing:border-box}body{max-width:1500px;margin:auto;padding:28px 24px 48px}
h1{font-size:28px;color:#fff;margin:0 0 10px}p{line-height:1.7;margin:8px 0 18px;color:#aab8c9}
a{color:#9bc9ff;text-decoration:none}a:hover{text-decoration:underline}
nav{display:flex;gap:18px;flex-wrap:wrap;margin:18px 0}details{margin:18px 0}summary{cursor:pointer}
.pages{display:flex;gap:10px;flex-wrap:wrap;margin:14px 0}.pages a{padding:5px 9px;background:#1a2638;border-radius:5px}
.filters{display:flex;align-items:end;gap:12px;flex-wrap:wrap;padding:18px;background:#192536;border-radius:10px}
label{display:flex;flex-direction:column;gap:7px;color:#c3cfdd}
select,input,button{font:inherit;color:#e8eef7;background:#101723;border:1px solid #40516a;border-radius:6px;padding:9px 12px}
input{min-width:260px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}
.pagination{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:22px 0}
#count{margin-right:auto}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(210px,1fr));gap:14px}
.card{background:#192536;border-radius:9px;padding:12px;min-width:0}.card>a{display:block;background:#000;text-align:center;border-radius:4px;overflow:hidden}
.card img{display:block;width:192px;height:192px;max-width:100%;object-fit:contain;margin:auto}
.card strong{display:block;color:#fff;font-size:14px;line-height:1.5;margin-top:10px;overflow-wrap:anywhere}
.card p{font-size:13px;margin:6px 0 0;overflow-wrap:anywhere;line-height:1.5}
.empty{padding:40px;text-align:center;color:#aab8c9}footer{margin-top:28px;color:#8797ac;font-size:13px}
</style>
<h1>鼠脑图像 · __SIZE__</h1>
<p>__IMAGE_COUNT__ 张图像 · __SOURCE_COUNT__ 个来源 · __DATASET_COUNT__ 个数据集</p>
<nav><a href="overview.jpg" target="_blank" rel="noopener">查看图像概览</a><a href="../manifest.jsonl">图像清单</a></nav>
<details><summary>逐来源质检 · __PAGE_COUNT__ 页</summary><div class="pages">__SOURCE_LINKS__</div></details>
<div class="filters">
<label>数据集<select id="dataset"><option value="">全部数据集</option></select></label>
<label>对比度 / 序列<select id="sequence"><option value="">全部序列</option></select></label>
<label>搜索<input id="search" type="search" placeholder="编号、文件名、动物或来源"></label>
<label>每页<select id="page-size"><option>24</option><option selected>48</option><option>96</option></select></label>
<button id="reset">重置</button>
</div>
<div class="pagination"><span id="count" aria-live="polite"></span><button id="previous">上一页</button><span id="page"></span><button id="next">下一页</button></div>
<div class="grid" id="grid"></div>
<div class="pagination"><button id="previous-bottom">上一页</button><span id="page-bottom"></span><button id="next-bottom">下一页</button></div>
<footer>点击图像可打开 PNG 原图。</footer>
<script>
const records=__RECORDS__;
const byId=id=>document.getElementById(id);
const dataset=byId('dataset'),sequence=byId('sequence'),search=byId('search'),pageSize=byId('page-size');
let filtered=records,page=0;
function options(select,values){values.forEach(value=>{const option=document.createElement('option');option.value=value;option.textContent=value;select.append(option);});}
options(dataset,[...new Set(records.map(r=>r.dataset))].sort());
options(sequence,[...new Set(records.map(r=>r.sequence))].sort());
function render(){
 const size=Number(pageSize.value),pages=Math.max(1,Math.ceil(filtered.length/size));page=Math.min(page,pages-1);
 byId('count').textContent=`${filtered.length.toLocaleString()} / ${records.length.toLocaleString()} 张图像`;
 ['page','page-bottom'].forEach(id=>byId(id).textContent=`${page+1} / ${pages}`);
 ['previous','previous-bottom'].forEach(id=>byId(id).disabled=page===0);
 ['next','next-bottom'].forEach(id=>byId(id).disabled=page>=pages-1);
 const grid=byId('grid');grid.replaceChildren();
 if(!filtered.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='没有匹配的图像';grid.append(empty);return;}
 filtered.slice(page*size,(page+1)*size).forEach(record=>{
  const card=document.createElement('article');card.className='card';
  const link=document.createElement('a');link.href='../'+record.image.split('/').map(encodeURIComponent).join('/');link.target='_blank';link.rel='noopener';
  const image=document.createElement('img');image.src=link.href;image.loading='lazy';image.decoding='async';image.width=192;image.height=192;image.alt=record.filename;link.append(image);
  const name=document.createElement('strong');name.textContent=record.filename;
  const metadata=document.createElement('p');metadata.textContent=`${record.dataset} · ${record.subject} · ${record.sequence} · slice ${record.slice_index}`;
  card.append(link,name,metadata);grid.append(card);
 });
}
function filter(){const query=search.value.trim().toLocaleLowerCase();filtered=records.filter(r=>(!dataset.value||r.dataset===dataset.value)&&(!sequence.value||r.sequence===sequence.value)&&(!query||`${r.num} ${r.filename} ${r.subject} ${r.source_id}`.toLocaleLowerCase().includes(query)));page=0;render();}
dataset.addEventListener('change',filter);sequence.addEventListener('change',filter);search.addEventListener('input',filter);pageSize.addEventListener('change',()=>{page=0;render();});
byId('reset').addEventListener('click',()=>{dataset.value='';sequence.value='';search.value='';filter();});
['previous','previous-bottom'].forEach(id=>byId(id).addEventListener('click',()=>{if(page>0){page--;render();if(id.endsWith('bottom'))byId('count').scrollIntoView();}}));
['next','next-bottom'].forEach(id=>byId(id).addEventListener('click',()=>{if((page+1)*Number(pageSize.value)<filtered.length){page++;render();if(id.endsWith('bottom'))byId('count').scrollIntoView();}}));
render();
</script>
</html>
'''


def make_html(gallery, records, sources, pages, summary):
    # Embed the small display manifest so opening index.html via file:// also works.
    fields = ('num', 'filename', 'image', 'dataset', 'subject', 'sequence', 'source_id', 'slice_index')
    display_records = [{key: record[key] for key in fields} for record in records]
    serialized = json.dumps(display_records, ensure_ascii=False, separators=(',', ':'))
    serialized = serialized.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e')
    replacements = {
        '__SIZE__': ' × '.join(str(value) for value in summary.get('size', [192, 192])),
        '__IMAGE_COUNT__': f'{len(records):,}',
        '__SOURCE_COUNT__': str(len(sources)),
        '__DATASET_COUNT__': str(len({record['dataset'] for record in records})),
        '__PAGE_COUNT__': str(len(pages)),
        '__SOURCE_LINKS__': ''.join(f'<a href="{html.escape(page["file"])}" target="_blank" rel="noopener">'
                                  f'{page["number"]} · 来源 {page["first"]}–{page["last"]}</a>' for page in pages),
    }
    content = HTML
    for token, value in replacements.items():
        content = content.replace(token, value)
    content = content.replace('__RECORDS__', serialized)
    (gallery / 'index.html').write_text(content, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Export run with manifest.jsonl and summary.json')
    parser.add_argument('--sources-per-page', type=int, default=12)
    parser.add_argument('--overview-per-dataset', type=int, default=4)
    args = parser.parse_args()
    if args.sources_per_page < 1 or args.overview_per_dataset < 1:
        parser.error('Contact sheet counts must be positive')
    run = args.run.resolve()
    with (run / 'manifest.jsonl').open(encoding='utf-8') as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    if not records:
        parser.error('Manifest contains no images')
    records.sort(key=lambda record: record['num'])
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    sources = group_sources(records)
    gallery = run / 'gallery'
    gallery.mkdir(exist_ok=True)
    make_overview(run, gallery, sources, args.overview_per_dataset)
    pages = make_source_pages(run, gallery, sources, args.sources_per_page)
    make_html(gallery, records, sources, pages, summary)
    print(json.dumps(dict(gallery=str(gallery / 'index.html'), images=len(records),
                          sources=len(sources), source_pages=len(pages)), ensure_ascii=False))


if __name__ == '__main__':
    main()
