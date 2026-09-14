"""Build a standalone Chinese gallery from the completed expanded human evaluation."""

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib


HERE = Path(__file__).resolve().parent
VIEWS = {
    'MID51': '矢状位',
    'MID108': '矢状位',
    'MID613': '轴状位 · 面内旋转 90°',
    'MID106': '轴状位',
    'MID615': '冠状位 · 训练视图以外',
}

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>新增 xSPEN 人脑扫描 · 原生重建对比</title>
<style>
:root{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#18293a;background:#f4f6f8;line-height:1.6;font-size:15px;font-synthesis:none}*{box-sizing:border-box}body{margin:0}header,main,footer{max-width:1750px;margin:auto;padding:20px 24px}header{padding-bottom:4px}main{padding-top:12px}h1{font-size:clamp(24px,3vw,34px);line-height:1.3;margin:10px 0}p{margin:8px 0}.badge{display:inline-block;border-radius:99px;background:#e4eef5;color:#205b82;padding:3px 10px;font-size:13px;margin-right:7px}.subtitle{color:#536273;max-width:1160px}button,select,a{font:inherit}button,select{border:1px solid #c7d1dc;background:#fff;color:inherit;border-radius:8px;min-height:41px;padding:7px 12px}button{cursor:pointer}button:hover{background:#eef4f8}button[aria-pressed="true"]{color:#fff;background:#205b82;border-color:#205b82}button:disabled{opacity:.45;cursor:default}button:focus-visible,select:focus-visible,a:focus-visible,img:focus-visible{outline:3px solid #488ab8;outline-offset:3px}.controls,.card{background:#fff;border:1px solid #d9e1e7;border-radius:12px}.controls{padding:14px 16px;margin-bottom:16px}.control-row{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.control-row+.control-row{border-top:1px solid #e7ecf0;margin-top:12px;padding-top:12px}.control-row[hidden]{display:none}.group{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.spacer{flex:1}label{display:flex;align-items:center;gap:7px;color:#536273;font-size:13px}.count{color:#637182;font-size:13px}.card{overflow:hidden}.heading{padding:13px 17px;border-bottom:1px solid #e1e7ec;display:flex;gap:12px;align-items:center;flex-wrap:wrap}h2{font-size:17px;margin:0}.meta{font-size:13px;color:#59697b;margin:3px 0 0}.save{color:#205b82;text-decoration:none;border-bottom:1px solid #b3cbdc}.save:hover{text-decoration:underline}.image-scroll{overflow-x:auto;overscroll-behavior-x:contain}.image-scroll img{display:block;width:100%;min-width:1100px;height:auto;cursor:zoom-in}.note{margin:0;padding:10px 17px;border-top:1px solid #e7ecf0;color:#667585;font-size:13px}footer{padding-top:0;padding-bottom:24px;color:#667585;font-size:13px}footer p{margin:5px 0}.sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
dialog{border:0;border-radius:12px;padding:0;width:calc(100vw - 28px);height:calc(100dvh - 28px);max-width:none;max-height:none;color:#18293a;background:#fff;box-shadow:0 8px 64px #0005}dialog::backdrop{background:#0b152bd9}.zoom-shell{display:flex;flex-direction:column;height:100%;min-height:0}.zoom-header{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:10px 14px;border-bottom:1px solid #d9e0e6}.zoom-title{flex:1;margin:0;font-size:14px}.zoom-body{flex:1;min-height:0;overflow:auto;overscroll-behavior:contain;background:#e9edf1}.zoom-body img{display:block;max-width:none;width:auto;height:auto;background:#fff}.zoom-body.fit img{width:100%;max-width:100%;height:auto}
@media(max-width:700px){header,main,footer{padding-left:12px;padding-right:12px}.controls{padding:12px}.heading{padding:12px}.spacer{display:none}.control-row{gap:9px}button,select{min-height:43px}.note{padding:10px 12px}h2{font-size:16px}.zoom-title{flex-basis:100%}}
</style>
</head>
<body>
<header>
<span class="badge">真实采集</span><span class="badge" id="counts"></span>
<h1>新增 xSPEN 人脑扫描 · 原生重建对比</h1>
<p class="subtitle">采集输入、传统重建与扩散模型共 6 列对照。本次使用现有 IXI 人脑原生 EDM EMA 20k 权重进行 DiffPIR 60 步推理，没有新增训练；原生重建保持采集矩阵 46 × 48。</p>
<p class="subtitle">同一观测的各方法共用灰度窗 [0, 1]，不分别调亮。真实数据没有干净参考图，不展示 PSNR 或 SSIM。</p>
</header>
<main>
<section class="controls" aria-label="选择对比图">
<div class="control-row">
<div class="group"><button type="button" data-view="overview" aria-pressed="true">代表观测总览</button><button type="button" data-view="pages" aria-pressed="false">按扫描浏览</button></div>
<span class="spacer"></span><span class="count">每页 4 层 · 6 列比较</span>
</div>
<div class="control-row" id="selection" hidden>
<label>扫描<select id="scan-select" aria-label="选择扫描"></select></label>
<label>采集序号<select id="occurrence-select" aria-label="选择 occurrence"></select></label>
<div class="group"><button type="button" id="previous">← 上一页</button><button type="button" id="next">下一页 →</button></div>
<span class="count" id="page-count" aria-live="polite"></span>
</div>
</section>
<section class="card" aria-label="重建比较图">
<div class="heading"><div><h2 id="image-title"></h2><p class="meta" id="image-meta"></p></div><span class="spacer"></span><button type="button" id="open-zoom">放大查看</button><a class="save" id="save-image" download>保存当前图</a></div>
<div class="image-scroll"><img id="main-image" tabindex="0" role="button" aria-label="放大当前比较图" alt="新增人脑扫描的重建比较图"></div>
<p class="note">可横向滚动查看各列，点击图像放大。列名与切片索引见图内标注；128 × 128 模型列是迁移对照。</p>
</section>
<p class="sr-only" id="announcement" aria-live="polite"></p>
</main>
<footer>
<p>MID51、MID108 为矢状位；MID613、MID106 为轴状位；MID615 为冠状位，超出当前模型的轴状/矢状训练视图。MID613 保留头文件记录的 90° 面内旋转。各扫描按原测量朝向显示。</p>
<p>采集序号（occurrence，图内标作 rep）和切片索引从 0 开始；不同序号不等于已核定的扩散方向。60 个观测是扫描 × 序号 × 切片组合，不代表 60 名独立受试者。</p>
<p>所有图片已内嵌，可离线打开并分享此 HTML 文件。</p>
</footer>
<dialog id="zoom-dialog" aria-label="放大比较图"><div class="zoom-shell"><div class="zoom-header"><p class="zoom-title" id="zoom-title"></p><button type="button" id="zoom-size" aria-pressed="true">原图大小</button><button type="button" id="zoom-fit" aria-pressed="false">适合窗口</button><a class="save" id="save-zoom" download>保存图片</a><button type="button" id="close-zoom" autofocus>关闭 ×</button></div><div class="zoom-body" id="zoom-body"><img id="zoom-image" alt="放大的重建比较图"></div></div></dialog>
<script id="gallery-data" type="application/json">__DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById('gallery-data').textContent);
const el=id=>document.getElementById(id);
const state={view:'overview',scan:data.scans[0].scan,page:data.scans[0].pages[0].id};
const pages=data.scans.flatMap(s=>s.pages.map(p=>({...p,scan:s.scan,view:s.view,geometry:s.geometry_note})));
function option(value,text){const o=document.createElement('option');o.value=String(value);o.textContent=text;return o;}
function currentScan(){return data.scans.find(s=>s.scan===state.scan);}
function currentPage(){return pages.find(p=>p.id===state.page);}
function populateOccurrences(){const scan=currentScan();if(!scan.pages.some(p=>p.id===state.page))state.page=scan.pages[0].id;el('occurrence-select').replaceChildren(...scan.pages.map(p=>option(p.id,`occurrence ${p.occurrence}`)));el('occurrence-select').value=state.page;}
function render(){
 document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===state.view)));
 el('selection').hidden=state.view!=='pages';
 let image,title,meta,filename;
 if(state.view==='overview'){image=data.overview;title='代表观测总览 · 5 个扫描';meta='每行一个扫描，各方法共用该观测的幅度标度。';filename='overview.png';}
 else{const p=currentPage();image=p.image;title=`${p.scan} · occurrence ${p.occurrence}`;meta=`${p.view} · 本页 4 个切片 · 46 × 48 原生矩阵 · 索引从 0 开始`;filename=p.filename;}
 const source=data.images[image];el('main-image').src=source;el('main-image').alt=title;el('image-title').textContent=title;el('image-meta').textContent=meta;el('save-image').href=source;el('save-image').download=filename;
 const index=pages.findIndex(p=>p.id===state.page);el('previous').disabled=index===0;el('next').disabled=index===pages.length-1;el('page-count').textContent=`第 ${index+1} / ${pages.length} 页`;el('announcement').textContent=title;
}
function changePage(delta){const index=pages.findIndex(p=>p.id===state.page)+delta;if(index<0||index>=pages.length)return;state.scan=pages[index].scan;state.page=pages[index].id;el('scan-select').value=state.scan;populateOccurrences();render();}
function fitZoom(fit){el('zoom-body').classList.toggle('fit',fit);el('zoom-fit').setAttribute('aria-pressed',String(fit));el('zoom-size').setAttribute('aria-pressed',String(!fit));}
function openZoom(){el('zoom-image').src=el('main-image').src;el('zoom-image').alt=el('main-image').alt;el('zoom-title').textContent=el('image-title').textContent+' · 按 Esc 关闭';el('save-zoom').href=el('save-image').href;el('save-zoom').download=el('save-image').download;fitZoom(false);el('zoom-dialog').showModal();el('zoom-body').scrollTo(0,0);}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;render();}));
el('scan-select').addEventListener('change',e=>{state.scan=e.target.value;populateOccurrences();render();});
el('occurrence-select').addEventListener('change',e=>{state.page=e.target.value;render();});
el('previous').addEventListener('click',()=>changePage(-1));el('next').addEventListener('click',()=>changePage(1));
el('open-zoom').addEventListener('click',openZoom);el('main-image').addEventListener('click',openZoom);el('main-image').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();openZoom();}});
el('close-zoom').addEventListener('click',()=>el('zoom-dialog').close());el('zoom-dialog').addEventListener('click',e=>{if(e.target===el('zoom-dialog'))el('zoom-dialog').close();});
el('zoom-size').addEventListener('click',()=>fitZoom(false));el('zoom-fit').addEventListener('click',()=>fitZoom(true));
document.addEventListener('keydown',e=>{if(state.view!=='pages'||el('zoom-dialog').open||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft'){e.preventDefault();changePage(-1);}if(e.key==='ArrowRight'){e.preventDefault();changePage(1);}});
el('counts').textContent=`${data.scans.length} 个扫描 · ${data.case_count} 个观测`;
el('scan-select').replaceChildren(...data.scans.map(s=>option(s.scan,`${s.scan} · ${s.view}`)));
populateOccurrences();render();
</script>
</body>
</html>
'''


def checked_png(path):
    """Reject a missing, truncated or corrupt PNG before embedding any gallery."""
    content = path.read_bytes()
    if not content.startswith(b'\x89PNG\r\n\x1a\n'):
        raise ValueError(f'不是 PNG：{path}')
    offset, kinds = 8, []
    while offset + 12 <= len(content):
        size = struct.unpack('>I', content[offset:offset + 4])[0]
        kind = content[offset + 4:offset + 8]
        end = offset + 8 + size
        if end + 4 > len(content):
            raise ValueError(f'PNG 尚未写完：{path}')
        expected = struct.unpack('>I', content[end:end + 4])[0]
        if zlib.crc32(content[offset + 4:end]) & 0xffffffff != expected:
            raise ValueError(f'PNG CRC 不匹配：{path}')
        kinds.append(kind)
        offset = end + 4
        if kind == b'IEND':
            break
    if not kinds or kinds[0] != b'IHDR' or kinds[-1] != b'IEND' or b'IDAT' not in kinds or offset != len(content):
        raise ValueError(f'PNG 不完整：{path}')
    return content


def read_gallery(summary_path, overview_path):
    source_bytes = summary_path.read_bytes()
    source = json.loads(source_bytes)
    if source.get('status') != 'complete' or source.get('smoke') is not False:
        raise ValueError('需要完成的正式 evaluation/summary.json，不能使用 smoke 输出')
    if source.get('case_count') != 60 or len(source.get('scans', [])) != 5:
        raise ValueError('正式图册应包含 5 个扫描、60 个观测')
    scans, images, image_metadata, all_ids = [], {}, {}, set()

    def embed(path):
        path = path.resolve()
        content = checked_png(path)
        key = str(path.relative_to(HERE)) if path.is_relative_to(HERE) else path.name
        if key in images:
            raise ValueError(f'图片重复：{path}')
        images[key] = 'data:image/png;base64,' + base64.b64encode(content).decode('ascii')
        image_metadata[key] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
        return key

    overview = embed(overview_path)
    for scan in source['scans']:
        name = scan['scan']
        if name not in VIEWS or scan.get('case_count') != 12 or len(scan.get('pages', [])) != 3:
            raise ValueError(f'扫描结构或数量不符合正式选择：{name}')
        if scan.get('checkpoint_step') != 20000 or scan.get('parameters', {}).get('steps') != 60:
            raise ValueError(f'{name} 不是 EMA 20k / DiffPIR 60 步的正式结果')
        if scan.get('parameters', {}).get('native_image_resize') is not False:
            raise ValueError(f'{name} 未确认无原生图像 resize')
        rows = []
        for png in scan['pages']:
            path = Path(png)
            if not path.is_absolute():
                path = summary_path.parent / path
            match = re.fullmatch(re.escape(name) + r'_rep(\d+)_page(\d+)\.png', path.name)
            if not match:
                raise ValueError(f'无法识别采集序号：{path.name}')
            occurrence = int(match.group(1))
            page_id = f'{name}_occ{occurrence}'
            if page_id in all_ids or int(match.group(2)) != 1:
                raise ValueError(f'预期每个 occurrence 只有 1 页：{path.name}')
            all_ids.add(page_id)
            rows.append({'id': page_id, 'occurrence': occurrence, 'image': embed(path), 'filename': path.name})
        rows.sort(key=lambda row: row['occurrence'])
        scans.append({'scan': name, 'view': VIEWS[name], 'case_count': scan['case_count'],
                      'geometry_note': scan['geometry_note'], 'pages': rows})
    if {s['scan'] for s in scans} != set(VIEWS) or len(images) != 16:
        raise ValueError('扫描集合或 16 张内嵌资源不完整')
    return {'scans': scans, 'case_count': source['case_count'], 'overview': overview, 'images': images,
            'image_metadata': image_metadata, 'summary_sha256': hashlib.sha256(source_bytes).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=HERE / 'evaluation/summary.json')
    parser.add_argument('--overview', type=Path, default=HERE / 'overview.png')
    parser.add_argument('--out', type=Path, default=HERE / 'gallery.html')
    args = parser.parse_args()
    data = read_gallery(args.summary.resolve(), args.overview.resolve())
    payload = json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    args.out.write_text(HTML.replace('__DATA__', payload), encoding='utf-8')
    print(json.dumps({'output': str(args.out.resolve()), 'scans': len(data['scans']),
                      'pages': sum(len(s['pages']) for s in data['scans']), 'cases': data['case_count'],
                      'embedded_pngs': len(data['images']), 'bytes': args.out.stat().st_size,
                      'summary_sha256': data['summary_sha256']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
