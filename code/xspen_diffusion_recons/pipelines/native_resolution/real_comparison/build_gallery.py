"""Build a local, single-file Chinese gallery with every comparison PNG embedded."""
import argparse
import base64
import json
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent

HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light">
<title>真实 xSPEN 人脑重建对比</title>
<style>
:root{font-family:system-ui,-apple-system,"Segoe UI","Microsoft YaHei",sans-serif;color:#172434;background:#f4f6f8;font-synthesis:none;font-size:15px;line-height:1.6}
*{box-sizing:border-box}body{margin:0}button,select,a{font:inherit}button,select{min-height:40px;border:1px solid #c8d2dc;border-radius:8px;background:#fff;color:inherit;padding:7px 12px}button{cursor:pointer}button:hover{background:#eef4f8}button:disabled{cursor:default;opacity:.4}button[aria-pressed="true"]{color:#fff;background:#205b82;border-color:#205b82}button:focus-visible,select:focus-visible,a:focus-visible{outline:3px solid #4e93bd;outline-offset:3px}
header,.workspace,footer{max-width:1700px;margin:auto;padding:20px 24px}header{padding-bottom:10px}h1{font-size:clamp(24px,3vw,34px);font-weight:700;line-height:1.25;margin:8px 0 12px}p{margin:8px 0}.subtitle{color:#536273;max-width:1050px}.badge{display:inline-block;color:#205b82;background:#e5eff6;padding:3px 10px;border-radius:99px;font-size:13px;margin-right:8px}.workspace{padding-top:8px}.controls{background:#fff;border:1px solid #d9e0e6;border-radius:12px;padding:14px 16px;margin-bottom:16px}.control-line{display:flex;align-items:center;gap:12px;flex-wrap:wrap}.control-line+.control-line{border-top:1px solid #e7ecf0;margin-top:12px;padding-top:12px}.group{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.group-label{font-size:13px;color:#586678;margin-right:4px}.spacer{flex:1}label{font-size:13px;color:#536273;display:flex;align-items:center;gap:7px}#case-select{min-width:290px;max-width:100%}#case-count{color:#536273;font-size:13px}#selection-line[hidden]{display:none}.download{color:#205b82;text-decoration:none;border-bottom:1px solid #b6cbd9;font-size:14px}.download:hover{text-decoration:underline}
.image-card{background:#fff;border:1px solid #d9e0e6;border-radius:12px;overflow:hidden}.image-heading{padding:14px 18px;border-bottom:1px solid #e1e7ec;display:flex;align-items:center;gap:12px;flex-wrap:wrap}.image-title{font-weight:650;margin:0;font-size:17px}.image-meta{font-size:13px;color:#5b6877;margin:3px 0 0}.image-scroll{overflow-x:auto;overscroll-behavior-x:contain;background:#fff}.image-scroll img{display:block;width:100%;min-width:1100px;height:auto;cursor:zoom-in}.image-scroll.raw img{min-width:850px}.image-note{padding:10px 18px;color:#667585;font-size:13px;margin:0;border-top:1px solid #e7ecf0}footer{font-size:13px;color:#667585;padding-top:0;padding-bottom:24px}footer p{margin:4px 0}
dialog{border:0;border-radius:12px;padding:0;width:calc(100vw - 28px);height:calc(100dvh - 28px);max-width:none;max-height:none;box-shadow:0 8px 64px #0005;background:#fff;color:#172434}dialog::backdrop{background:#0b152bd9}.zoom-shell{display:flex;flex-direction:column;height:100%;min-height:0}.zoom-header{flex:0 0 auto;padding:10px 14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-bottom:1px solid #d9e0e6}.zoom-title{flex:1;font-size:14px;margin:0}.zoom-body{flex:1;min-height:0;overflow:auto;background:#e9edf1;overscroll-behavior:contain}.zoom-body img{display:block;max-width:none;width:auto;height:auto;background:#fff}.zoom-body.fit img{width:100%;max-width:100%;height:auto}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}
@media(max-width:700px){header,.workspace,footer{padding-left:12px;padding-right:12px}.controls{padding:12px}.control-line{gap:9px}.spacer{display:none}#case-select{min-width:0;width:100%}.case-label{width:100%;align-items:flex-start;flex-direction:column}.image-heading{padding:12px}.image-title{font-size:16px}.image-scroll img{min-width:1100px}.image-scroll.raw img{min-width:850px}button,select{min-height:43px}.image-note{padding:10px 12px}}
</style>
</head>
<body>
<header>
<div><span class="badge">真实采集</span><span class="badge" id="case-badge"></span></div>
<h1>真实 xSPEN 人脑重建对比</h1>
<p class="subtitle">查看采集输入、传统重建与扩散模型结果。同一例的图像共用灰度窗；真实数据没有干净参考图，不展示 PSNR 或 SSIM。</p>
</header>
<main class="workspace">
<section class="controls" aria-label="图像选择">
<div class="control-line">
<div class="group" aria-label="浏览方式">
<button type="button" data-view="overview" aria-pressed="true">代表总览</button>
<button type="button" data-view="cases" aria-pressed="false">逐例浏览</button>
<button type="button" data-view="raw" aria-pressed="false">采集输入</button>
</div>
<span class="spacer"></span>
<div class="group" id="mode-group" aria-label="比较列数"><span class="group-label">对比</span><button type="button" data-mode="main" aria-pressed="true">6 列</button><button type="button" data-mode="extended" aria-pressed="false">8 列</button></div>
</div>
<div class="control-line" id="selection-line" hidden>
<label>扫描<select id="scan-select" aria-label="选择扫描"></select></label>
<label>重复<select id="repeat-select" aria-label="选择重复"></select></label>
<label class="case-label">图像<select id="case-select" aria-label="选择图像"></select></label>
<div class="group"><button type="button" id="previous" aria-label="上一例">← 上一例</button><button type="button" id="next" aria-label="下一例">下一例 →</button></div>
<span id="case-count" aria-live="polite"></span>
</div>
</section>
<section class="image-card" aria-label="重建比较图">
<div class="image-heading"><div><h2 class="image-title" id="image-title"></h2><p class="image-meta" id="image-meta"></p></div><span class="spacer"></span><button type="button" id="open-zoom">放大查看</button><a class="download" id="download-image" download>保存当前图</a></div>
<div class="image-scroll" id="image-scroll"><img id="main-image" alt="真实 xSPEN 重建比较图" tabindex="0" role="button" aria-label="放大当前比较图"></div>
<p class="image-note" id="image-note">可横向滚动查看各列，点击图像查看原图大小。</p>
</section>
<p class="sr-only" id="announcement" aria-live="polite"></p>
</main>
<footer>
<p>重建比较共用归一化幅度窗 [0, 1]，未按方法分别调亮。采集输入页的测量域显示以图内说明为准。</p>
<p>重复编号和切片索引从 0 开始。各例是扫描、重复与切片组合，不代表相同数量的独立受试者。</p>
<p>所有图片已包含在此页面内，可离线打开并分享此 HTML 文件。</p>
</footer>
<dialog id="zoom-dialog" aria-label="放大比较图"><div class="zoom-shell"><div class="zoom-header"><p class="zoom-title" id="zoom-title"></p><button type="button" id="zoom-size" aria-pressed="true">原图大小</button><button type="button" id="zoom-fit" aria-pressed="false">适合窗口</button><button type="button" id="close-zoom" autofocus>关闭 ×</button></div><div class="zoom-body" id="zoom-body"><img id="zoom-image" alt="放大的重建比较图"></div></div></dialog>
<script id="gallery-data" type="application/json">__GALLERY_DATA__</script>
<script>
"use strict";
const data=JSON.parse(document.getElementById('gallery-data').textContent);
const el=id=>document.getElementById(id);
const state={view:'overview',mode:'main',scan:'all',repeat:'all',case:data.records[0].case};
let filtered=data.records.slice();
const natural=(a,b)=>String(a).localeCompare(String(b),'zh-CN',{numeric:true});
function option(value,label){const o=document.createElement('option');o.value=String(value);o.textContent=label;return o;}
function caseLabel(r){return `${r.scan} · 重复 ${r.repeat} · 切片 ${r.slice_index}`;}
function populateScans(){const select=el('scan-select');select.replaceChildren(option('all','全部扫描'));[...new Set(data.records.map(r=>r.scan))].sort(natural).forEach(v=>select.append(option(v,v)));select.value=state.scan;}
function populateRepeats(){const rows=data.records.filter(r=>state.scan==='all'||r.scan===state.scan);const values=[...new Set(rows.map(r=>r.repeat))].sort((a,b)=>a-b);if(state.repeat!=='all'&&!values.some(v=>String(v)===state.repeat))state.repeat='all';const select=el('repeat-select');select.replaceChildren(option('all','全部重复'));values.forEach(v=>select.append(option(v,`重复 ${v}`)));select.value=state.repeat;}
function populateCases(){filtered=data.records.filter(r=>(state.scan==='all'||r.scan===state.scan)&&(state.repeat==='all'||String(r.repeat)===state.repeat));if(!filtered.some(r=>r.case===state.case))state.case=filtered[0].case;el('case-select').replaceChildren(...filtered.map(r=>option(r.case,caseLabel(r))));el('case-select').value=state.case;}
function chosen(){return data.records.find(r=>r.case===state.case);}
function render(){
 document.querySelectorAll('[data-view]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.view===state.view)));
 document.querySelectorAll('[data-mode]').forEach(b=>{b.setAttribute('aria-pressed',String(b.dataset.mode===state.mode));b.disabled=state.view==='raw';});
 el('selection-line').hidden=state.view!=='cases';
 const extended=state.mode==='extended';let image,title,meta,filename;
 if(state.view==='overview'){image=extended?data.extended_overview_png:data.overview_png;title=`代表观测总览 · ${extended?'8':'6'} 列`;meta='3 例代表图；按图内标注对照采集输入、传统重建与模型结果。';filename=extended?'overview_extended.png':'overview.png';}
 else if(state.view==='raw'){image=data.raw_input_png;title='真实采集输入';meta='原始测量与图像域输入的区别，见图内标注。';filename='raw_input.png';}
 else{const r=chosen();image=extended?r.extended_png:r.main_png;title=caseLabel(r);meta=`原生矩阵 ${r.native_shape[0]} × ${r.native_shape[1]} · ${extended?'8':'6'} 列比较 · 索引从 0 开始`;filename=r.case+(extended?'_extended':'')+'.png';}
 const source=data.images[image];el('main-image').src=source;el('main-image').alt=title;el('image-title').textContent=title;el('image-meta').textContent=meta;el('download-image').href=source;el('download-image').download=filename;
 el('image-scroll').classList.toggle('raw',state.view==='raw');
 el('image-note').textContent=state.view==='raw'?'按图内标注区分测量域与图像域；点击可放大。':'同一例各方法共用灰度窗 [0, 1]。可横向滚动，点击图像查看原图大小。';
 const index=filtered.findIndex(r=>r.case===state.case);el('previous').disabled=index<=0;el('next').disabled=index>=filtered.length-1;el('case-count').textContent=`筛选后 ${filtered.length} 例 · 当前 ${index+1}/${filtered.length}`;el('announcement').textContent=title;
}
function changeCase(delta){const index=filtered.findIndex(r=>r.case===state.case);const next=index+delta;if(next<0||next>=filtered.length)return;state.case=filtered[next].case;el('case-select').value=state.case;render();}
function setZoomFit(fit){el('zoom-body').classList.toggle('fit',fit);el('zoom-size').setAttribute('aria-pressed',String(!fit));el('zoom-fit').setAttribute('aria-pressed',String(fit));}
function zoom(){el('zoom-image').src=el('main-image').src;el('zoom-image').alt=el('main-image').alt;el('zoom-title').textContent=el('image-title').textContent+' · 可滚动查看，按 Esc 关闭';setZoomFit(false);el('zoom-dialog').showModal();el('zoom-body').scrollTo(0,0);}
document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>{state.view=b.dataset.view;render();}));
document.querySelectorAll('[data-mode]').forEach(b=>b.addEventListener('click',()=>{state.mode=b.dataset.mode;render();}));
el('scan-select').addEventListener('change',e=>{state.scan=e.target.value;state.repeat='all';populateRepeats();populateCases();state.view='cases';render();});
el('repeat-select').addEventListener('change',e=>{state.repeat=e.target.value;populateCases();state.view='cases';render();});
el('case-select').addEventListener('change',e=>{state.case=e.target.value;render();});
el('previous').addEventListener('click',()=>changeCase(-1));el('next').addEventListener('click',()=>changeCase(1));
el('open-zoom').addEventListener('click',zoom);el('main-image').addEventListener('click',zoom);el('main-image').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();zoom();}});
el('close-zoom').addEventListener('click',()=>el('zoom-dialog').close());el('zoom-dialog').addEventListener('click',e=>{if(e.target===el('zoom-dialog'))el('zoom-dialog').close();});
el('zoom-size').addEventListener('click',()=>setZoomFit(false));el('zoom-fit').addEventListener('click',()=>setZoomFit(true));
document.addEventListener('keydown',e=>{if(state.view!=='cases'||el('zoom-dialog').open||e.target.tagName==='SELECT')return;if(e.key==='ArrowLeft'){e.preventDefault();changeCase(-1);}if(e.key==='ArrowRight'){e.preventDefault();changeCase(1);}});
el('case-badge').textContent=`${data.records.length} 例对比`;
populateScans();populateRepeats();populateCases();render();
</script>
</body>
</html>
'''


def read_manifest(path):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest.get('records'), list) or not manifest['records']:
        raise ValueError('manifest.records 必须是非空列表')
    required = ['case','scan','repeat','slice_index','native_shape','main_png','extended_png']
    seen = set()
    rows = []
    for row in manifest['records']:
        if any(key not in row for key in required):
            raise ValueError('每条 record 必须包含 '+', '.join(required))
        if row['case'] in seen:
            raise ValueError(f'重复的 case: {row["case"]}')
        if len(row['native_shape'])!=2 or min(row['native_shape'])<=0:
            raise ValueError('native_shape 必须是两个正整数')
        seen.add(row['case'])
        rows.append({key:row[key] for key in required})
    paths = [manifest[key] for key in ['overview_png','extended_overview_png','raw_input_png']]
    paths += [row[key] for row in rows for key in ['main_png','extended_png']]
    images = {}
    for name in dict.fromkeys(paths):
        asset = (path.parent / name).resolve()
        if not asset.is_relative_to(path.parent.resolve()) or asset.suffix.lower()!='.png':
            raise ValueError(f'图片必须是 manifest 目录内的 PNG: {name}')
        content = asset.read_bytes()
        if not content.startswith(b'\x89PNG\r\n\x1a\n') or not content.endswith(b'\x00\x00\x00\x00IEND\xaeB`\x82'):
            raise ValueError(f'PNG 尚未完整生成或格式不正确: {asset}')
        images[name] = 'data:image/png;base64,'+base64.b64encode(content).decode('ascii')
    return dict(records=rows, images=images,
                **{key:manifest[key] for key in ['overview_png','extended_overview_png','raw_input_png']})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=HERE/'manifest.json')
    parser.add_argument('--out',type=Path,default=HERE/'gallery.html')
    parser.add_argument('--wait-seconds',type=float,default=0)
    args=parser.parse_args()
    deadline=time.monotonic()+max(0,args.wait_seconds)
    announced=False
    while True:
        try:
            data=read_manifest(args.manifest.resolve())
            break
        except (FileNotFoundError,json.JSONDecodeError,ValueError) as error:
            if time.monotonic()>=deadline:
                raise SystemExit(str(error)) from error
            if not announced:
                print('等待 manifest 和全部 PNG 生成完成。',flush=True)
                announced=True
            time.sleep(min(2,max(.1,deadline-time.monotonic())))
    payload=json.dumps(data,ensure_ascii=False,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e').replace('&','\\u0026')
    page=HTML.replace('__GALLERY_DATA__',payload)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.out.with_suffix('.html.tmp')
    temporary.write_text(page,encoding='utf-8')
    temporary.replace(args.out)
    print(json.dumps(dict(output=str(args.out.resolve()),cases=len(data['records']),
                         embedded_pngs=len(data['images']),bytes=args.out.stat().st_size),ensure_ascii=False))


if __name__=='__main__':
    main()
