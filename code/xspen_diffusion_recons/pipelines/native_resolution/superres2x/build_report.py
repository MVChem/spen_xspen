"""CPU-only six-column report for saved 2x xSPEN reconstructions.

Reads completed case NPZ/JSON. It never runs inference, resizes reconstruction
arrays, normalizes individual methods, or modifies source reports/checkpoints.
Formal output requires all 96 frozen cases plus a complete evaluation summary.
"""
from __future__ import annotations

import argparse
import base64
from collections import defaultdict
import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from PIL import Image

HERE = Path(__file__).resolve().parent
ORDER = ['MID112', 'MID114', 'MID27', 'MID51', 'MID613', 'MID615', 'MID106', 'MID108']
VIEWS = {'MID112': '轴位', 'MID114': '矢状位', 'MID27': '轴位', 'MID51': '矢状位',
         'MID613': '轴位 · 面内旋转90°', 'MID615': '冠状位 · 训练视图以外',
         'MID106': '轴位', 'MID108': '矢状位'}
METHODS = ['degraded', 'native_tikhonov', 'native_diffusion', 'tikhonov_bicubic2x',
           'native_diffusion_bicubic2x', 'diffusion2x']
TITLES = ['原始 RO + RSS\nPE 编码幅度图', 'Tikhonov\n原生矩阵', 'EDM + DiffPIR\n原生矩阵',
          'Tikhonov\n双三次插值 2×', '原生 EDM 结果\n双三次插值 2×', 'EDM + DiffPIR 2×\n较细网格重建']
REPS = {m: [0, 9, 17] if m in ['MID27', 'MID51'] else [0, 8, 15] for m in ORDER}
SLICES = {m: [16, 26, 37, 48] if m in ['MID112', 'MID114'] else
          [11, 18, 26, 34] if m == 'MID27' else [17, 23, 29, 35] for m in ORDER}
EXPECTED = {(m, r, s) for m in ORDER for r in REPS[m] for s in SLICES[m]}
REPRESENTATIVES = {m: (m, 0, SLICES[m][1]) for m in ORDER}
FONT = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if FONT.exists():
    plt.rcParams['font.family'] = FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'savefig.facecolor': 'white'})


def atomic_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text, encoding='utf-8')
    temp.replace(path)


def write_json(path, value):
    atomic_text(Path(path), json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def local_link(path: Path, directory: Path):
    return quote(os.path.relpath(path.resolve(), directory.resolve()), safe='/')


def identity(record):
    return record['scan'], int(record['repeat']), int(record['slice_index'])


def load_cases(evaluation: Path, partial=False):
    summary_path = evaluation / 'summary.json'
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    if not partial and summary.get('status') != 'complete':
        raise ValueError('Formal gallery requires evaluation/summary.json status=complete; use --partial for preview.')
    records, images = {}, {}
    for metadata_path in sorted(evaluation.glob('MID*/MID*_rep*_slice*.json')):
        record = json.loads(metadata_path.read_text())
        if record.get('status') != 'complete':
            if partial:
                continue
            raise ValueError(f'Incomplete case: {metadata_path}')
        if not partial and (record.get('smoke') is not False or record.get('parameters', {}).get('steps') != 60):
            raise ValueError(f'Formal case is not a 60-step non-smoke result: {metadata_path}')
        key = identity(record)
        if key not in EXPECTED or key in records:
            raise ValueError(f'Unexpected or duplicate frozen case: {key}')
        native = tuple(record['native_shape']); output = tuple(record['output_shape'])
        if len(native) != 2 or min(native) < 1 or output != tuple(2*v for v in native):
            raise ValueError(f'Not exactly 2x each axis: {metadata_path}')
        expected_native = (60, 64) if key[0] in ['MID112', 'MID114'] else (46, 48)
        if native != expected_native:
            raise ValueError(f'Unexpected native matrix: {key}, {native}')
        fov = np.asarray(record['fov_mm'], dtype=float)
        if fov.shape != (2,) or not np.isfinite(fov).all() or min(fov) <= 0:
            raise ValueError(f'Invalid physical FOV: {metadata_path}')
        if not np.isfinite(record['magnitude_scale']) or record['magnitude_scale'] <= 0:
            raise ValueError(f'Invalid shared receiver scale: {metadata_path}')
        npz_path = Path(record.get('npz', metadata_path.with_suffix('.npz')))
        if not npz_path.is_absolute():
            candidate = metadata_path.parent / npz_path
            npz_path = candidate if candidate.exists() else evaluation / npz_path
        if not npz_path.is_file():
            raise FileNotFoundError(npz_path)
        values = {}
        with np.load(npz_path, allow_pickle=False) as stored:
            for i, method in enumerate(METHODS):
                a = stored[method]
                shape = native if i < 3 else output
                if a.shape != (1, 1, *shape) or a.dtype.kind != 'f' or not np.isfinite(a).all():
                    raise ValueError(f'Invalid saved {method}: {npz_path}, {a.shape}, {a.dtype}')
                # Shared scale was fixed by inference. No per-method gain or clipping.
                values[method] = (a[0, 0].astype(np.float64) + 1.) / 2.
        record = dict(record, case=npz_path.stem, npz=str(npz_path.resolve()), metadata_json=str(metadata_path.resolve()))
        record['display_clipped_fractions'] = {m: {'below_zero': float(np.mean(a < 0)),
                                                   'above_one': float(np.mean(a > 1))} for m, a in values.items()}
        record['display_native_pixel_mm_pe_ro'] = (fov / np.asarray(native)).tolist()
        record['display_output_pixel_mm_pe_ro'] = (fov / np.asarray(output)).tolist()
        records[key], images[key] = record, values
    if not records:
        raise ValueError('No completed case arrays available.')
    if not partial:
        if set(records) != EXPECTED:
            raise ValueError(f'Expected all 96 frozen cases; missing {sorted(EXPECTED-set(records))}')
        if 'records' not in summary or not isinstance(summary['records'], list) or len(summary['records']) != 96:
            raise ValueError('Complete summary must list all 96 records.')
        if summary.get('smoke') is True:
            raise ValueError('Smoke outputs cannot be used for a formal report.')
    return records, images, summary


def draw_rows(keys, records, images, title, out: Path, partial=False):
    """Render arrays in physical coordinates; only raw extent gets half native PE shift."""
    rows = len(keys)
    fig, axes = plt.subplots(rows, 6, figsize=(17.8, 2.60*rows + 1.60), squeeze=False)
    fig.suptitle(title + (' · 未完成预览 / 非效果结论' if partial else ''), fontsize=17, fontweight='bold', y=1-.20/(2.60*rows+1.60))
    for row, key in enumerate(keys):
        rec = records[key]
        height, width = rec['fov_mm']
        native_dy = height / rec['native_shape'][0]
        for col, method in enumerate(METHODS):
            ax = axes[row, col]
            shift = native_dy/2 if method == 'degraded' else 0
            ax.imshow(images[key][method], cmap='gray', vmin=0, vmax=1, interpolation='nearest',
                      extent=(0, width, height-shift, -shift), aspect='equal')
            ax.set_xlim(0, width); ax.set_ylim(height, 0)
            ax.set_facecolor('black'); ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(TITLES[col], fontsize=11.1, pad=10,
                             color='#126b6e' if col == 5 else '#25323f')
            if col == 0:
                native = '×'.join(map(str, rec['native_shape']))
                output = '×'.join(map(str, rec['output_shape']))
                label = f"{rec['scan']} · {VIEWS[rec['scan']]}\n{native} → {output}\n层 {rec['slice_index']} · 采集 {rec['repeat']}"
                ax.text(-.065, .5, label, transform=ax.transAxes, ha='right', va='center', fontsize=9.2)
            if col == 5:
                xend, ybar = width-8, height-8
                ax.plot([xend-30, xend], [ybar, ybar], color='white', linewidth=2)
                ax.text(xend-15, ybar-3, '30 mm', ha='center', color='white', fontsize=8)
    footer = ('各行共用原始观测强度尺度与灰度窗 [0, 1] · 同一物理 FOV · 原生层厚保持不变\n'
              '2× 表示长宽各加倍的输出网格；双三次插值是对照，较细网格不等于已证实的有效分辨率提升 · 无真实 GT')
    fig.text(.555, .025, footer, ha='center', fontsize=10, color='#3e4d5a')
    # Physical axes remain identical in each row despite native/2x pixel counts.
    fig.subplots_adjust(left=.138, right=.995, top=1-1.20/(2.60*rows+1.60),
                        bottom=.093 if rows <= 2 else .068 if rows <= 4 else .042,
                        wspace=.055, hspace=.105)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.stem + '.tmp.png')
    fig.savefig(temporary, dpi=145)
    plt.close(fig)
    temporary.replace(out)
    with Image.open(out) as check:
        check.verify()


def draw_roi(keys, records, images, title, out, partial=False):
    """Fixed central 50% FOV, identical physical crop across all three columns."""
    cols = ['native_diffusion', 'native_diffusion_bicubic2x', 'diffusion2x']
    titles = ['EDM + DiffPIR 原生', '原生 EDM 双三次 2×', 'EDM + DiffPIR 2×']
    rows = len(keys)
    fig, axes = plt.subplots(rows, 3, figsize=(11.8, 2.70*rows+1.40), squeeze=False)
    fig.suptitle(title + (' · 未完成预览 / 非效果结论' if partial else ''), fontsize=15, y=1-.22/(2.70*rows+1.40), fontweight='bold')
    for row, key in enumerate(keys):
        r=records[key]; height,width=r['fov_mm']
        for col, method in enumerate(cols):
            ax=axes[row,col]
            ax.imshow(images[key][method],cmap='gray',vmin=0,vmax=1,interpolation='nearest',
                      extent=(0,width,height,0),aspect='equal')
            ax.set_xlim(.25*width,.75*width);ax.set_ylim(.75*height,.25*height)
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if row==0:ax.set_title(titles[col],fontsize=11,pad=10,color='#126b6e' if col==2 else '#25323f')
            if col==0:
                ax.text(-.08,.5,f"{r['scan']}\n层 {r['slice_index']} · 采集 {r['repeat']}\n中央 50% FOV",transform=ax.transAxes,ha='right',va='center',fontsize=10)
            if col==2:
                end=.75*width-5; yy=.75*height-5
                ax.plot([end-20,end],[yy,yy],color='white',lw=2)
                ax.text(end-10,yy-2,'20 mm',color='white',fontsize=8,ha='center')
    fig.text(.57,.024,'每个方向保留中央 50% FOV · ROI 预先固定、未按输出挑选\n同一灰度窗 [0,1] · nearest 显示原有像素 · 较细网格不等于已验证的有效分辨率',ha='center',fontsize=10,color='#3e4d5a')
    fig.subplots_adjust(left=.19,right=.995,top=1-1.05/(2.70*rows+1.40),bottom=.105 if rows==1 else .048,wspace=.08,hspace=.11)
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.with_name(out.stem+'.tmp.png');fig.savefig(temporary,dpi=150);plt.close(fig);temporary.replace(out)
    with Image.open(out) as check:check.verify()


HTML = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light"><title>xSPEN 人脑 · 长宽各 2× 重建</title>
<style>
:root{font:15px/1.6 system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#213342;background:#f3f6f8}*{box-sizing:border-box}body{margin:0}header,main,footer{max-width:1820px;margin:auto;padding:20px 24px}header{padding-bottom:4px}h1{font-size:clamp(25px,3vw,35px);line-height:1.3;margin:10px 0}h2{font-size:18px;margin:0}p{margin:8px 0}.badge{display:inline-block;background:#e1f0ef;color:#155c60;border-radius:30px;padding:3px 12px;margin-right:8px;font-size:13px}.muted{color:#61717e}.controls,.card{background:white;border:1px solid #d7e0e6;border-radius:12px}.controls{padding:14px 16px;margin-bottom:15px}.row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}.row+.row{margin-top:12px;padding-top:12px;border-top:1px solid #e7edf1}.spacer{flex:1}button,select,a{font:inherit}button,select{color:inherit;background:white;border:1px solid #bdcbd5;border-radius:7px;padding:8px 12px;min-height:42px}button{cursor:pointer}button:hover{background:#edf4f5}button[aria-pressed="true"]{background:#176b70;color:white;border-color:#176b70}button:disabled{opacity:.4;cursor:default}button:focus-visible,select:focus-visible,a:focus-visible,img:focus-visible{outline:3px solid #50a3ad;outline-offset:3px}label{display:flex;align-items:center;gap:8px;font-size:14px}.card{overflow:hidden}.heading{padding:13px 16px;border-bottom:1px solid #dce4e9}.heading p{font-size:13px;margin:2px 0}.image-scroll{overflow-x:auto}.image-scroll img{display:block;width:100%;min-width:1200px;height:auto;cursor:zoom-in}.note{padding:10px 16px;border-top:1px solid #e4ebef;font-size:13px;color:#5d6e7a;margin:0}a{color:#176b70;text-decoration:none}a:hover{text-decoration:underline}.downloads{margin-top:15px;background:white;border:1px solid #d7e0e6;border-radius:10px;padding:12px 16px}.downloads h2{font-size:16px}.download-list{display:flex;gap:9px 20px;flex-wrap:wrap;margin:8px 0;font-size:14px}.tiny{font-size:13px}footer{font-size:13px;color:#61717e;padding-top:0}dialog{border:0;border-radius:10px;padding:0;width:calc(100vw - 28px);height:calc(100dvh - 28px);max-width:none;max-height:none;background:white}dialog::backdrop{background:#112233dd}.zoom-shell{display:flex;flex-direction:column;height:100%;min-height:0}.zoom-header{padding:10px 14px;border-bottom:1px solid #d8e2e7}.zoom-body{flex:1;overflow:auto;min-height:0;background:#e5edf2}.zoom-body img{display:block;width:auto;max-width:none;height:auto}.zoom-body.fit img{width:100%;height:auto;max-width:100%}[hidden]{display:none!important}@media(max-width:700px){header,main,footer{padding-left:12px;padding-right:12px}.spacer{display:none}.heading{padding:12px}.heading h2{font-size:16px}.zoom-title{flex-basis:100%}}
</style></head><body><header><span class="badge" id="status"></span><span class="badge" id="counts"></span>
<h1>真实 xSPEN 人脑 · 长宽各 2× 重建</h1>
<p>同时比较原生重建、双三次插值和较细网格上的 EDM EMA + DiffPIR。60×64 → 120×128；46×48 → 92×96。物理 FOV 和层厚不变。</p>
<p class="muted">同一观测共用强度尺度与灰度窗 [0,1]，不分别调亮。2× 是输出网格大小，尚未证实有效空间分辨率提高 2 倍；真实数据无干净参考图。</p>
</header><main><section class="controls" aria-label="图册筛选">
<div class="row"><button data-mode="overview" aria-pressed="true">8 扫描总览</button><button data-mode="pages" aria-pressed="false">逐扫描 / 采集页</button><button data-mode="detail" aria-pressed="false">中央50%细节</button><span class="spacer"></span><span class="tiny muted">6 列对照 · 点击图像放大</span></div>
<div class="row"><label>扫描<select id="scan-select" aria-label="选择扫描"></select></label><label id="occurrence-label" hidden>采集序号<select id="occurrence-select" aria-label="选择采集序号"></select></label><button id="previous" hidden>← 上一页</button><button id="next" hidden>下一页 →</button><span class="tiny muted" id="page-count" aria-live="polite"></span></div>
</section><section class="card"><div class="heading row"><div><h2 id="image-title"></h2><p class="muted" id="image-meta"></p></div><span class="spacer"></span><button id="open-zoom">放大查看</button><a id="save-image" download>保存当前图</a></div><div class="image-scroll"><img id="main-image" tabindex="0" role="button" aria-label="放大对照图" alt="xSPEN 原生与2倍网格重建对照"></div><p class="note">原始 RO+RSS 按原生 PE 半像素中心定位；各重建列使用同一物理范围。双三次对照的插值已保存在数组中，图册不额外平滑重建像素。</p></section>
<section class="downloads"><h2>当前图对应的数组</h2><div class="download-list" id="downloads"></div><p class="tiny muted">NPZ 保存各方法的数值与元数据；数据链接指向本地相对路径，没有内嵌大型数组。分享可下载版本时，请同时保留 evaluation 目录。</p><p class="tiny"><a href="report_manifest.json" download>下载图册清单 JSON</a> · <a href="超分2x结果.md">文字说明</a></p></section>
</main><footer><p>全部图片已嵌入本 HTML，可离线看图。NPZ 下载需要原 evaluation 目录。扫描和 occurrence 不是独立受试者计数，occurrence 未独立核实为 b0 或扩散方向。</p><p>MID615 为冠状位，超出先验的轴位/矢状训练视图；MID613 保留采集的90°面内旋转。固定案例沿用原生重建的选择，没有按2×效果筛选。较细网格的新增信息受先验与简化编码模型约束。</p></footer>
<dialog id="zoom"><div class="zoom-shell"><div class="zoom-header row"><strong id="zoom-title" class="zoom-title"></strong><span class="spacer"></span><button id="zoom-fit" aria-pressed="false">适合窗口</button><button id="zoom-size" aria-pressed="true">原图大小</button><a id="save-zoom" download>保存图</a><button id="close-zoom" autofocus>关闭 ×</button></div><div class="zoom-body" id="zoom-body"><img id="zoom-image" alt="放大的重建图"></div></div></dialog>
<script id="gallery-data" type="application/json">__DATA__</script><script>
'use strict';
const data=JSON.parse(document.getElementById('gallery-data').textContent), el=id=>document.getElementById(id);
const state={mode:'overview',scan:'all',page:data.pages[0].id};
function option(value,label){const o=document.createElement('option');o.value=String(value);o.textContent=label;return o;}
function selectedPages(){return data.pages.filter(p=>state.scan==='all'||p.scan===state.scan);}
function selection(){const ps=selectedPages();if(!ps.some(p=>p.id===state.page))state.page=ps[0].id;el('occurrence-select').replaceChildren(...ps.map(p=>option(p.id,`${p.scan} · occurrence ${p.occurrence}`)));el('occurrence-select').value=state.page;}
function visible(){if(state.mode==='pages')return data.pages.find(p=>p.id===state.page);if(state.mode==='detail')return state.scan==='all'?data.detail:data.scans.find(s=>s.scan===state.scan).detail;if(state.scan==='all')return data.overview;return data.scans.find(s=>s.scan===state.scan).overview;}
function render(){
 document.querySelectorAll('[data-mode]').forEach(b=>b.setAttribute('aria-pressed',String(state.mode===b.dataset.mode)));
 const isPage=state.mode==='pages';['occurrence-label','previous','next'].forEach(id=>el(id).hidden=!isPage);
 const item=visible(), source=data.images[item.image];el('main-image').src=source;el('main-image').alt=item.title;el('image-title').textContent=item.title;el('image-meta').textContent=item.description;el('save-image').href=source;el('save-image').download=item.filename;
 const ps=selectedPages(), index=ps.findIndex(p=>p.id===state.page);el('previous').disabled=index<=0;el('next').disabled=index>=ps.length-1;el('page-count').textContent=isPage?`第 ${index+1} / ${ps.length} 页 · 每页最多4层`:`当前总览 ${item.cases.length} 个固定观测`;
 el('downloads').replaceChildren(...item.cases.map(c=>{const a=document.createElement('a');a.href=c.npz;a.download=c.filename;a.textContent=`${c.scan} · 层${c.slice} · 采集${c.occurrence} · NPZ`;return a;}));
}
function move(delta){const ps=selectedPages(), next=ps.findIndex(p=>p.id===state.page)+delta;if(next<0||next>=ps.length)return;state.page=ps[next].id;el('occurrence-select').value=state.page;render();}
function zoomFit(fit){el('zoom-body').classList.toggle('fit',fit);el('zoom-fit').setAttribute('aria-pressed',String(fit));el('zoom-size').setAttribute('aria-pressed',String(!fit));}
function openZoom(){el('zoom-image').src=el('main-image').src;el('zoom-title').textContent=el('image-title').textContent;el('save-zoom').href=el('save-image').href;el('save-zoom').download=el('save-image').download;zoomFit(false);el('zoom').showModal();el('zoom-body').scrollTo(0,0);}
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;selection();render();}));
el('scan-select').addEventListener('change',e=>{state.scan=e.target.value;selection();render();});el('occurrence-select').addEventListener('change',e=>{state.page=e.target.value;render();});el('previous').addEventListener('click',()=>move(-1));el('next').addEventListener('click',()=>move(1));
el('open-zoom').addEventListener('click',openZoom);el('main-image').addEventListener('click',openZoom);el('main-image').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openZoom();}});el('close-zoom').addEventListener('click',()=>el('zoom').close());el('zoom').addEventListener('click',e=>{if(e.target===el('zoom'))el('zoom').close();});el('zoom-fit').addEventListener('click',()=>zoomFit(true));el('zoom-size').addEventListener('click',()=>zoomFit(false));
document.addEventListener('keydown',e=>{if(state.mode!=='pages'||el('zoom').open||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft'){e.preventDefault();move(-1);}if(e.key==='ArrowRight'){e.preventDefault();move(1);}});
el('status').textContent=data.partial?'未完成预览 / 含smoke时不可评价效果':'正式完成';el('counts').textContent=`${data.scans.length} 个扫描 · ${data.case_count} 个观测`;
el('scan-select').replaceChildren(option('all','全部扫描'),...data.scans.map(s=>option(s.scan,`${s.scan} · ${s.view}`)));
selection();render();
</script></body></html>'''


def build(evaluation: Path, out: Path, partial=False):
    records, images, source_summary = load_cases(evaluation, partial)
    out.mkdir(parents=True, exist_ok=True)
    pages, scans, image_payload, png_metadata = [], [], {}, {}

    def render(keys, filename, title, description, roi=False):
        path = out / filename
        (draw_roi if roi else draw_rows)(keys, records, images, title, path, partial)
        content = path.read_bytes()
        image_payload[filename] = 'data:image/png;base64,' + base64.b64encode(content).decode('ascii')
        png_metadata[filename] = dict(bytes=len(content), sha256=hashlib.sha256(content).hexdigest())
        downloads = [dict(scan=k[0], occurrence=k[1], slice=k[2], case=records[k]['case'],
                          npz=local_link(Path(records[k]['npz']), out),
                          filename=Path(records[k]['npz']).name) for k in keys]
        return dict(image=filename, filename=path.name, title=title, description=description, cases=downloads)

    selected = [REPRESENTATIVES[m] for m in ORDER if REPRESENTATIVES[m] in records]
    if not selected:
        if not partial:
            raise ValueError('No fixed overview representatives.')
        selected = [next(iter(records))]
    overview = render(selected, 'overview.png', '真实 xSPEN：原生、插值与长宽各2×重建',
                      f'{len(selected)}个扫描各1个预先固定观测；各行保留本扫描的采集朝向。')
    detail = render(selected, 'overview_detail.png', '固定中央50% FOV：原生、插值与2×重建',
                    '每个平面方向取中央50% FOV；固定位置，无输出驱动选区。', roi=True)
    for filename, subset in [('overview_first3.png', selected[:3]), ('overview_other5.png', selected[3:]),
                             ('overview_middle3.png', selected[3:6]), ('overview_last2.png', selected[6:])]:
        if subset:
            # Convenience images are not duplicated inside the HTML payload.
            draw_rows(subset, records, images, '真实 xSPEN：长宽各2×重建对照', out/filename, partial)
    for mid in ORDER:
        keys = sorted(k for k in records if k[0] == mid)
        if not keys:
            continue
        rec = records[keys[0]]
        target = REPRESENTATIVES[mid]
        representative = target if target in records else keys[0]
        scan_overview = render([representative], f'pages/{mid}_overview.png', f'{mid} · 固定代表观测',
                               f"{VIEWS[mid]}；原生 {'×'.join(map(str,rec['native_shape']))} → {'×'.join(map(str,rec['output_shape']))}。")
        scan_detail = render([representative], f'pages/{mid}_detail.png', f'{mid} · 固定中央50% FOV',
                             '原生EDM、双三次2×、较细网格EDM；相同物理ROI与[0,1]灰度窗。', roi=True)
        scan_pages = []
        for occurrence in sorted({k[1] for k in keys}):
            page_keys = [k for k in keys if k[1] == occurrence]
            p = render(page_keys, f'pages/{mid}_rep{occurrence:02d}_page01.png',
                       f'{mid} · 采集 {occurrence} · 长宽各2×重建',
                       f"{VIEWS[mid]} · 本页{len(page_keys)}层 · 同一FOV，原生 {'×'.join(map(str,rec['native_shape']))} → {'×'.join(map(str,rec['output_shape']))}；索引从0开始。")
            p.update(id=f'{mid}_rep{occurrence:02d}', scan=mid, occurrence=occurrence)
            pages.append(p); scan_pages.append(p)
        scans.append(dict(scan=mid, view=VIEWS[mid], case_count=len(keys), native_shape=rec['native_shape'],
                          output_shape=rec['output_shape'], fov_mm=rec['fov_mm'], thickness_mm=rec['thickness_mm'],
                          output_pixel_mm_pe_ro=rec['display_output_pixel_mm_pe_ro'], overview=scan_overview, detail=scan_detail,
                          pages=scan_pages, expected_overview_case=f'{mid}_rep00_slice{SLICES[mid][1]:03d}'))
    gallery = dict(partial=partial, case_count=len(records), scans=scans, pages=pages, overview=overview, detail=detail,
                   images=image_payload)
    payload = json.dumps(gallery, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    payload = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    atomic_text(out/'gallery.html', HTML.replace('__DATA__', payload))
    manifest = dict(status='partial' if partial else 'complete', partial=partial, case_count=len(records),
                    source_evaluation=str(evaluation), source_summary=str(evaluation/'summary.json'),
                    frozen_expected_case_count=96, methods=METHODS, titles=TITLES,
                    display=dict(window=[0,1], per_method_normalization=False, magnitude_conversion='(model_x+1)/2, no clipping',
                                 reconstructed_pixel_resampling=False, interpolation='nearest',
                                 raw_extent_shift='-0.5 native PE pixel only; output grid uses full physical FOV',
                                 fov_and_thickness_unchanged=True, detail_roi_fov_fraction=[.25,.75,.25,.75]),
                    inference_origins=dict((origin, sum(r.get('inference_origin','unrecorded')==origin for r in records.values()))
                                           for origin in sorted({r.get('inference_origin','unrecorded') for r in records.values()})),
                    overview_cases=[records[k]['case'] for k in selected], scans=scans, pngs=png_metadata,
                    gallery_bytes=(out/'gallery.html').stat().st_size,
                    limitations=['Output matrix doubled along each in-plane axis, not independently measured effective resolution.',
                                 'No paired real clean GT; no real PSNR/SSIM.',
                                 'Comparison cases fixed before this reconstruction; occurrence is not verified diffusion direction.',
                                 'Display PNGs embedded for offline viewing; numerical NPZ files are linked, not embedded.'],
                    cases=[{k:v for k,v in records[key].items() if k in ['case','scan','repeat','slice_index','npz','metadata_json',
                            'native_shape','output_shape','fov_mm','thickness_mm','magnitude_scale','native_reference_npz',
                            'sr_checkpoint','sr_checkpoint_sha256','sr_checkpoint_step','inference_origin','measurement_residual',
                            'display_clipped_fractions','display_native_pixel_mm_pe_ro','display_output_pixel_mm_pe_ro']}
                           for key in sorted(records)])
    write_json(out/'report_manifest.json',manifest)
    lines=['# 真实 xSPEN 人脑：长宽各2×重建','',
           ('**未完成预览，不作为效果结论；如输入为smoke则仅执行2步。**' if partial else '**正式对照已完成。**') + f"当前包含{len(scans)}份扫描、{len(records)}个固定观测。",
           '输出矩阵在PE与RO两个方向各增加一倍，FOV和层厚保持不变。较细的输出网格不等于已经证明有效分辨率提高两倍。','',
           '![全部扫描总览](overview.png)','',
           '[离线交互图册](gallery.html) · [中央50%细节](overview_detail.png) · [结果清单](report_manifest.json)','',
           '六列依次为原始RO+RSS、原生Tikhonov、原生EDM+DiffPIR、Tikhonov双三次2×、原生EDM结果双三次2×、较细网格EDM+DiffPIR。插值对照用来区分单纯增加像素与加入较细网格先验；各方法共用原始观测导出的强度尺度和[0,1]灰度窗。图册只用nearest显示已存像素，不再对重建图平滑。','',
           '| 扫描 | 方向 | 原生 → 输出矩阵 | 输出名义平面间距 mm | 层厚 mm | 观测数 |','|---|---|---|---|---:|---:|']
    for scan in scans:
        shape=lambda a:'×'.join(map(str,a))
        spacing='×'.join(f'{v:.4f}' for v in scan['output_pixel_mm_pe_ro'])
        lines.append(f"| {scan['scan']} | {scan['view']} | {shape(scan['native_shape'])} → {shape(scan['output_shape'])} | {spacing} | {scan['thickness_mm']:g} | {scan['case_count']} |")
    lines += ['', '原始RO+RSS的PE中心与重建体素中心相差半个原生像素，已通过物理extent定位；重建列均覆盖相同FOV，没有单独移动或旋转方法输出。MID613的90°面内旋转和MID615的冠状位保留原采集状态，后者超出先验的轴位/矢状训练视图。','',
              '总览沿用预先固定的rep0：MID112/MID114取slice26，MID27取slice18，其余取slice23。全部96个选择来自既有原生对照的四层×首/中/末occurrence，未根据2×输出或指标挑例。扫描、切片和occurrence不等于独立人次，occurrence尚未独立核实为b0或扩散方向。','',
              '真实数据没有配对干净GT；图册不提供真实PSNR/SSIM。测量残差若记录，仅表示对当前固定相位/线圈与简化编码模型的拟合，不能直接当作分辨率或真实性指标。','', '## 全部采集页','']
    for scan in scans:
        links=' · '.join(f"[采集{p['occurrence']}]({p['image']})" for p in scan['pages'])
        lines.append(f"- {scan['scan']}：{links}")
    lines += ['', 'HTML内嵌图片可独立离线查看；NPZ不内嵌，下载按钮指向本地evaluation目录。分享需要数值结果的版本时，保留该目录和相对路径。图册生成仅读取已存结果，未启动推理或GPU。']
    atomic_text(out/'超分2x结果.md','\n'.join(lines)+'\n')
    return {k:manifest[k] for k in ['status','case_count','overview_cases','inference_origins','gallery_bytes']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,default=HERE/'evaluation')
    parser.add_argument('--out',type=Path,default=None)
    parser.add_argument('--partial',action='store_true',help='Preview available completed cases in a separate preview directory.')
    args=parser.parse_args()
    out=args.out or (HERE/'preview' if args.partial else HERE)
    print(json.dumps(build(args.evaluation.resolve(),out.resolve(),args.partial),ensure_ascii=False))


if __name__=='__main__':
    main()
