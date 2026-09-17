"""Build a complete gallery with PNG assets inside its preview directory."""
from __future__ import annotations

import argparse
from collections import Counter
import filecmp
import html
import json
import os

import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from urllib.parse import quote


HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>96 × 96 鼠脑 · 全部数据</title>
<style>
:root{color-scheme:dark;font:15px/1.6 system-ui,sans-serif;color:#e5eaf3;background:#10151e}
*{box-sizing:border-box}body{max-width:1600px;margin:auto;padding:30px 24px 60px}h1{font-size:28px;margin:0 0 6px}p{margin:6px 0 18px;color:#acb8cc}a{color:#9bcaff}button,select,input{font:inherit;color:inherit;background:#121c2b;border:1px solid #41516a;border-radius:7px;padding:8px 12px}button{cursor:pointer}button:hover:not(:disabled){border-color:#93bfff;background:#23364f}button:disabled{opacity:.4;cursor:default}button:focus-visible,a:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #a8cdff;outline-offset:3px}
.stats{display:flex;gap:12px;flex-wrap:wrap;margin:22px 0}.stats button{min-width:150px;text-align:left;padding:13px 18px}.stats strong{display:block;font-size:25px;color:#fff}.stats button.active{background:#20344e;border-color:#8fb8eb}.location{overflow-wrap:anywhere;font-size:13px;color:#9aabc3}.filters{display:flex;gap:14px;align-items:end;flex-wrap:wrap;background:#1b2534;padding:18px;border-radius:10px}label{display:flex;flex-direction:column;gap:6px;font-size:13px;color:#bbc8db}select,input{font-size:14px}#search{width:280px;max-width:100%}.pagination{display:flex;align-items:center;gap:9px;flex-wrap:wrap;margin:22px 0}.count{margin-right:auto}#jump{width:80px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(188px,1fr));gap:14px}.card{background:#1b2534;padding:10px;border-radius:9px;min-width:0}.picture{padding:0;border:0;border-radius:5px;display:block;width:100%;background:#000;overflow:hidden}.picture img{display:block;width:192px;height:192px;object-fit:contain;max-width:100%;margin:auto}.tag{color:#a8ccff;font-size:12px;margin:9px 0 3px}.subject{font-size:13px;color:#fff;overflow-wrap:anywhere}.detail{font-size:12px;color:#9eafc8;overflow-wrap:anywhere}.empty{padding:50px;color:#bec8da}.error{color:#ffbaba}footer{color:#98aac3;font-size:13px;margin-top:30px}.bottom{justify-content:center}
dialog{width:min(900px,96vw);max-height:94vh;overflow:auto;background:#151f2d;color:#e5eaf3;border:1px solid #4e6587;border-radius:12px;padding:20px}dialog::backdrop{background:#000c}.modal-top{display:flex;gap:12px;justify-content:space-between;align-items:center}.modal-image{display:block;width:480px;height:480px;max-width:100%;object-fit:contain;background:#000;margin:18px auto}.filename{overflow-wrap:anywhere;font-size:13px;color:#c3d1e5}.modal-actions{display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:16px}#modal-meta{margin:12px 0}#load-errors:empty{display:none}@media(max-width:600px){body{padding:20px 12px}.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.picture img{width:100%;height:auto;aspect-ratio:1}.stats button{min-width:calc(50% - 8px)}.modal-image{height:auto;aspect-ratio:1}}
</style>
</head>
<body>
<h1>96 × 96 鼠脑 · 全部数据</h1>
<p>共 __TOTAL__ 张图像 · 包含旧数据和本次新增数据 · 点击图片放大查看</p>
<div class="location">图片目录：__ROOT_LABEL__</div>
<div class="stats">
<button data-split="" class="active"><span>全部图片</span><strong>__TOTAL__</strong></button>
<button data-split="train"><span>训练集 train</span><strong>__TRAIN__</strong></button>
<button data-split="val"><span>验证集 val</span><strong>__VAL__</strong></button>
<button data-split="test"><span>测试集 test</span><strong>__TEST__</strong></button>
</div>
<div class="filters" id="filters">
<label>数据划分<select id="split"><option value="">全部划分</option><option value="train">训练集 train</option><option value="val">验证集 val</option><option value="test">测试集 test</option></select></label>
<label>数据范围<select id="batch"><option value="">全部数据</option><option value="old">旧数据</option><option value="lab">新增实验室数据</option><option value="public">新增公开数据</option></select></label>
<label>数据来源<select id="dataset"><option value="">全部来源</option></select></label>
<label>序列 / 视图<select id="sequence"><option value="">全部序列 / 视图</option></select></label>
<label>搜索<input id="search" type="search" placeholder="文件名、动物、扫描编号、切片号" autocomplete="off"></label>
<label>每页<select id="page-size"><option>48</option><option selected>96</option><option>192</option></select></label>
<button id="reset">重置</button>
</div>
<div class="pagination">
<span id="count" class="count" role="status"></span>
<button id="first">首页</button><button id="previous">上一页</button><span id="page"></span><button id="next">下一页</button><button id="last">末页</button>
<label for="jump">跳至页</label><input id="jump" type="number" min="1" value="1" aria-label="跳至页"><button id="go">跳转</button>
</div>
<div id="load-errors" class="error" role="status"></div>
<div id="grid" class="grid"></div>
<div class="pagination bottom"><button id="previous-bottom">上一页</button><span id="page-bottom"></span><button id="next-bottom">下一页</button></div>
<footer>显示全部已导出的 PNG，按页加载。__PREVIEW_NOTE__实验室 RAT 是来源名称，未逐一扫描确认物种。</footer>
<dialog id="viewer" aria-labelledby="modal-title">
<div class="modal-top"><strong id="modal-title">图像查看</strong><button id="close">关闭 Esc</button></div>
<img id="modal-image" class="modal-image" alt="">
<div id="modal-meta"></div><div id="modal-filename" class="filename"></div>
<div class="modal-actions"><button id="modal-previous">← 上一张</button><span id="modal-position"></span><button id="modal-next">下一张 →</button><a id="original">打开原始 PNG</a></div>
</dialog>
<script id="image-records" type="application/json">__RECORDS__</script>
<script>
'use strict';
const rows=JSON.parse(document.getElementById('image-records').textContent);
// Columns: partition, filename, dataset, subject, scan/index, slice, sequence, batch.
const atlas=__ATLAS_JS__;
const root=__ROOT_JS__,byId=id=>document.getElementById(id),fmt=n=>n.toLocaleString('en-US');
const controls={split:byId('split'),batch:byId('batch'),dataset:byId('dataset'),sequence:byId('sequence'),search:byId('search'),size:byId('page-size')};
const batchNames={old:'旧数据',lab:'新增实验室',public:'新增公开'};
const sourceNames={'lab-RAT':'实验室 RAT','figshare-aging-28433102':'公开 Aging'};
let filtered=rows,page=0,activeImage=0,renderToken=0;
const viewer=byId('viewer');
const searchable=rows.map(r=>r.join(' ').toLowerCase());
function sourceName(value){return sourceNames[value]||value;}
function url(r){return root+encodeURIComponent(r[0])+'/'+encodeURIComponent(r[1]);}
const rowIndices=new Map(rows.map((r,i)=>[r,i])),sheetCache=new Map();
function sheetUrl(index){return atlas.root+'sheet_'+String(index).padStart(4,'0')+'.jpg';}
function loadSheet(index){
 if(!sheetCache.has(index)){
  const pending=new Promise((resolve,reject)=>{const im=new Image();im.onload=()=>resolve(im);im.onerror=()=>reject(new Error('Preview sheet could not load'));im.src=sheetUrl(index);});
  sheetCache.set(index,pending);if(sheetCache.size>8)sheetCache.delete(sheetCache.keys().next().value);
 }
 return sheetCache.get(index);
}
function displayImage(element,r){
 element.dataset.imageKey=r[0]+'/'+r[1];const key=element.dataset.imageKey;
 if(!atlas){element.src=url(r);return;}
 const index=rowIndices.get(r),sheet=Math.floor(index/atlas.count),tile=index%atlas.count,rowsPerSheet=atlas.count/atlas.columns;
 // CSS sprites work in an opaque sandbox origin; canvas pixel reads do not.
 element.src='data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7';
 element.style.backgroundImage='url("'+sheetUrl(sheet)+'")';
 element.style.backgroundSize=`${atlas.columns*100}% ${rowsPerSheet*100}%`;
 element.style.backgroundPosition=`${(tile%atlas.columns)/(atlas.columns-1)*100}% ${Math.floor(tile/atlas.columns)/(rowsPerSheet-1)*100}%`;
 element.dataset.previewLoaded='false';
 loadSheet(sheet).then(()=>{if(element.dataset.imageKey===key)element.dataset.previewLoaded='true';}).catch(()=>{byId('load-errors').textContent='预览图未能加载，请重新生成预览。';});
}
function options(select,column,label){const counts=new Map();rows.forEach(r=>counts.set(r[column],(counts.get(r[column])||0)+1));[...counts.keys()].sort().forEach(value=>{const o=document.createElement('option');o.value=value;o.textContent=`${label(value)} (${fmt(counts.get(value))})`;select.append(o);});}
options(controls.dataset,2,sourceName);options(controls.sequence,6,v=>v);
function pageCount(){return Math.max(1,Math.ceil(filtered.length/Number(controls.size.value)));}
function setPage(value,scroll=false){page=Math.max(0,Math.min(pageCount()-1,value));render();if(scroll)byId('filters').scrollIntoView({behavior:'smooth',block:'start'});}
function render(){
 const size=Number(controls.size.value),pages=pageCount();page=Math.min(page,pages-1);const start=page*size,token=++renderToken;
 byId('count').textContent=`${fmt(filtered.length)} / ${fmt(rows.length)} 张 · 当前 ${filtered.length?fmt(start+1):0}–${fmt(Math.min(start+size,filtered.length))}`;
 ['page','page-bottom'].forEach(id=>byId(id).textContent=`第 ${page+1} / ${pages} 页`);
 ['previous','previous-bottom','first'].forEach(id=>byId(id).disabled=page===0);
 ['next','next-bottom','last'].forEach(id=>byId(id).disabled=page===pages-1);
 byId('jump').max=pages;byId('jump').value=page+1;
 document.querySelectorAll('[data-split]').forEach(b=>b.classList.toggle('active',b.dataset.split===controls.split.value));
 const grid=byId('grid'),fragment=document.createDocumentFragment();grid.replaceChildren();byId('load-errors').textContent='';
 if(!filtered.length){const empty=document.createElement('div');empty.className='empty';empty.textContent='没有匹配的图像，请调整筛选或搜索。';grid.append(empty);return;}
 filtered.slice(start,start+size).forEach((r,i)=>{
  const card=document.createElement('article');card.className='card';card.dataset.filename=r[1];
  const button=document.createElement('button');button.className='picture';button.type='button';button.title=r[1];button.setAttribute('aria-label','放大 '+r[1]);button.addEventListener('click',()=>openImage(start+i));
  const image=document.createElement('img');displayImage(image,r);image.alt=r[1];image.width=192;image.height=192;image.loading='lazy';image.decoding='async';image.addEventListener('error',()=>{if(token===renderToken)byId('load-errors').textContent='部分图片未能加载，请确认图片目录可以访问。';});button.append(image);
  const tag=document.createElement('div');tag.className='tag';tag.textContent=`${r[0]} · ${batchNames[r[7]]} · ${sourceName(r[2])}`;
  const subject=document.createElement('div');subject.className='subject';subject.textContent=r[3];
  const detail=document.createElement('div');detail.className='detail';detail.textContent=`${r[5]?'切片 '+r[5]:'旧图 #'+r[4]} · ${r[6]}`;
  card.append(button,tag,subject,detail);fragment.append(card);
 });grid.append(fragment);
}
function filter(){
 const queries=controls.search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
 filtered=rows.filter((r,i)=>(!controls.split.value||r[0]===controls.split.value)&&(!controls.batch.value||r[7]===controls.batch.value)&&(!controls.dataset.value||r[2]===controls.dataset.value)&&(!controls.sequence.value||r[6]===controls.sequence.value)&&queries.every(q=>searchable[i].includes(q)));
 page=0;render();
}
function openImage(index){
 activeImage=Math.max(0,Math.min(filtered.length-1,index));const r=filtered[activeImage];if(!r)return;
 byId('modal-title').textContent=sourceName(r[2])+' · '+r[3];displayImage(byId('modal-image'),r);byId('modal-image').alt=r[1];
 byId('modal-meta').textContent=`${r[0]} · ${batchNames[r[7]]} · ${r[5]?'切片 '+r[5]:'旧图 #'+r[4]} · ${r[6]}`;
 byId('modal-filename').textContent=r[0]+'/'+r[1];byId('modal-position').textContent=`${fmt(activeImage+1)} / ${fmt(filtered.length)}`;
 byId('original').href=url(r);if(atlas){byId('original').textContent='原始 PNG（需本地或 HTTP 打开）';byId('original').title='聊天预览限制跨目录符号链接；原图可从 code/data 目录打开';}byId('modal-previous').disabled=activeImage===0;byId('modal-next').disabled=activeImage===filtered.length-1;
 if(!viewer.open)viewer.showModal();
}
['split','batch','dataset','sequence'].forEach(name=>controls[name].addEventListener('change',filter));
controls.search.addEventListener('input',filter);controls.size.addEventListener('change',()=>setPage(0));
document.querySelectorAll('[data-split]').forEach(b=>b.addEventListener('click',()=>{controls.split.value=b.dataset.split;filter();}));
byId('reset').addEventListener('click',()=>{['split','batch','dataset','sequence','search'].forEach(k=>controls[k].value='');filter();});
['previous','previous-bottom'].forEach(id=>byId(id).addEventListener('click',()=>setPage(page-1,id.endsWith('bottom'))));
['next','next-bottom'].forEach(id=>byId(id).addEventListener('click',()=>setPage(page+1,id.endsWith('bottom'))));
byId('first').addEventListener('click',()=>setPage(0));byId('last').addEventListener('click',()=>setPage(pageCount()-1));
function jump(){const n=Number(byId('jump').value);if(Number.isInteger(n)&&n>=1)setPage(n-1);else byId('jump').value=page+1;}
byId('go').addEventListener('click',jump);byId('jump').addEventListener('keydown',e=>{if(e.key==='Enter')jump();});
byId('close').addEventListener('click',()=>viewer.close());
byId('modal-previous').addEventListener('click',()=>openImage(activeImage-1));byId('modal-next').addEventListener('click',()=>openImage(activeImage+1));
viewer.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'){e.preventDefault();openImage(activeImage-1);}if(e.key==='ArrowRight'){e.preventDefault();openImage(activeImage+1);}});
viewer.addEventListener('click',e=>{if(e.target===viewer){const r=viewer.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)viewer.close();}});
render();
</script>
</body></html>
'''


def collect_images(root):
    """Read only image filenames; no legacy manifest/array dependency."""
    records = []
    for part in ('train', 'val', 'test'):
        directory = root / part
        if not directory.is_dir():
            raise ValueError(f'Missing image folder: {directory}')
        for path in sorted(directory.glob('*.png')):
            fields = path.stem.split('__')
            if len(fields) != 5:
                raise ValueError(f'Unrecognized exported image filename: {path.name}')
            if fields[0] == 'old':
                _, dataset, subject, index, sequence = fields
                source, slice_index, batch = index, '', 'old'
            else:
                dataset, subject, source, slice_index, sequence = fields
                if not slice_index.startswith('sl'):
                    raise ValueError(f'Missing slice index: {path.name}')
                slice_index = slice_index[2:]
                batch = 'lab' if dataset == 'lab-RAT' else 'public'
            records.append([part, path.name, dataset, subject, source, slice_index, sequence, batch])
    if not records:
        raise ValueError('No PNG images found')
    return records


def prepare_assets(root, output, records):
    """Copy exact PNG bytes within the HTML's directory grant (no symlinks).

    The remote chat preview intentionally denies parent-directory traversal and
    symlinks resolving outside the HTML directory. Keep that boundary intact.
    Independent copies also let users delete the temporary preview safely.
    """
    assets = output.parent / 'images' / root.name
    if not assets.resolve().is_relative_to(output.parent) or assets.resolve().is_relative_to(root):
        raise ValueError('Preview assets must stay inside the preview directory, outside the dataset')
    copied = 0
    for number, record in enumerate(records, 1):
        source = root / record[0] / record[1]
        target = assets / record[0] / record[1]
        if target.is_symlink() or not target.resolve().is_relative_to(output.parent):
            raise ValueError(f'Preview asset points outside its directory: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file() or not filecmp.cmp(source, target, shallow=False):
            shutil.copy2(source, target)
            copied += 1
        if number % 5000 == 0:
            print(f'Preview assets: {number:,}/{len(records):,}', flush=True)
    print(f'Packaged {len(records):,} PNGs ({copied:,} copied) inside the preview directory', flush=True)
    return quote(assets.relative_to(output.parent).as_posix(), safe='/') + '/'


def prepare_symlink_assets(root, output, records):
    """One data symlink plus small 8-bit JPEG contact sheets for sandboxed preview.

    Contact sheets are display derivatives; training continues to read original
    uint16 PNGs only. The original PNG link is usable in file/HTTP browsers.
    """
    assets = output.parent / 'images' / root.name
    assets.parent.mkdir(parents=True, exist_ok=True)
    if not assets.parent.resolve().is_relative_to(output.parent):
        raise ValueError('Preview images directory must stay inside the HTML directory')
    if assets.is_symlink():
        if assets.resolve() != root:
            raise ValueError('Existing preview symlink points to a different dataset')
    elif assets.exists():
        raise ValueError('Replace the verified PNG copy with a symlink before using --asset-mode symlink')
    else:
        assets.symlink_to(os.path.relpath(root, assets.parent), target_is_directory=True)
    folder = output.parent / 'preview_tiles'
    folder.mkdir(exist_ok=True)
    if folder.is_symlink():
        raise ValueError('Preview contact sheets must be stored inside the HTML directory')
    count, columns = 128, 16
    for start in range(0, len(records), count):
        canvas = Image.new('L', (columns*96, (count//columns)*96))
        for index, record in enumerate(records[start:start+count]):
            with Image.open(root/record[0]/record[1]) as source:
                # Proper uint16 -> display uint8 conversion; no convert("L") clipping.
                pixels = np.rint(np.asarray(source, dtype=np.float32)/257.).astype(np.uint8)
            canvas.paste(Image.fromarray(pixels), ((index % columns)*96, (index//columns)*96))
        canvas.save(folder/f'sheet_{start//count:04d}.jpg', quality=92, optimize=True)
    print(f'Linked original PNG folder; built {(len(records)+count-1)//count} preview contact sheets', flush=True)
    return (quote(assets.relative_to(output.parent).as_posix(), safe='/')+'/',
            dict(root='preview_tiles/', count=count, columns=columns))


def build_gallery(root, output, asset_mode='copy'):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError('Put the HTML outside the image-only data directory')
    records = collect_images(root)
    counts = Counter(r[0] for r in records)
    output.parent.mkdir(parents=True, exist_ok=True)
    atlas = None
    if asset_mode == 'symlink':
        root_url, atlas = prepare_symlink_assets(root, output, records)
    elif asset_mode == 'copy':
        root_url = prepare_assets(root, output, records)
    else:
        raise ValueError('Unknown asset mode')
    replacements = {
        '__TOTAL__': f'{len(records):,}', '__TRAIN__': f'{counts["train"]:,}',
        '__VAL__': f'{counts["val"]:,}', '__TEST__': f'{counts["test"]:,}',
        '__ROOT_LABEL__': html.escape(str(root)),
        '__ROOT_JS__': json.dumps(root_url),
        '__ATLAS_JS__': json.dumps(atlas),
        '__PREVIEW_NOTE__': ('原图为 96 × 96、16 位灰度，位于 code/data；此页面使用轻量 8 位 JPEG 预览图，训练不使用预览图。' if atlas else '原图为 96 × 96、16 位灰度；页面仅放大显示，不改变图片。'),
    }
    content = HTML
    for key, value in replacements.items():
        content = content.replace(key, value)
    # Embedded display index works offline; no sidecar JSON files are created.
    serialized = json.dumps(records, ensure_ascii=False, separators=(',', ':'))
    content = content.replace('__RECORDS__', serialized.replace('&', '\\u0026').replace('<', '\\u003c').replace('>', '\\u003e'))
    output.write_text(content, encoding='utf-8')
    print(f'Gallery: {output}\nImages: {len(records):,}; partitions: {dict(counts)}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path, help='Image-only train/val/test dataset')
    parser.add_argument('--out', required=True, type=Path, help='HTML path outside the dataset')
    parser.add_argument('--asset-mode', choices=['copy', 'symlink'], default='copy',
                        help='symlink: link originals and use small contact sheets for chat preview')
    args = parser.parse_args()
    build_gallery(args.data, args.out, args.asset_mode)


if __name__ == '__main__':
    main()
