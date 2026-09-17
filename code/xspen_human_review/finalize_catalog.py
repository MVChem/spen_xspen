"""Assemble audited acquisition arrays, existing Bruker results and old figures."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageOps

from build_unified import dump, digest
from catalog_all import HERE, ROOT, export_array, public_entry


def flash_reference(run):
    folder=run/'archive_sources/pvmatlab_flash_testdata'
    source=json.loads((folder/'source.json').read_text())
    record=dict(path=source['path'],source=source['source'],sha256=digest(source['path']),kind='reference')
    details=dict(source,title='FLASH · 工具箱参考示例',subtitle='pvtools TestData · 非 SPEN · 5 层',
        original_shape=source['shape'],fov_unit='mm',sequence=source['pulse_program'],
        display_note='工具箱附带的 FLASH 参考数据，非 SPEN，人脑身份未核实。按原始小端 int16 读取，应用 VisuCoreDataSlope/Offs，保留原切片顺序。')
    a=np.load(folder/'flash_reference.npz')['image']
    entry=export_array(run,record,'FLASH_reference',a,[dict(label='切片',size=5,values=source['slice_positions_mm'])],source['fov_mm'],details,source_kind='reference')
    return entry


def old_figures(run):
    records=json.loads((run/'inventory/legacy_image_exports.json').read_text())['records']
    records=[r for r in records if r.get('include_in_legacy_export_page') or (r['kind']=='eps' and r['classification']=='legacy_scientific_export')]
    external=Path('/home/data2/chk/workspace/2026/08/14/01_工作项目/xspen_recons/outputs')
    records.extend(dict(path=str(p),kind='png',classification='legacy_scientific_export',relative_path='MID253 旧结果/'+p.name,sha256=digest(p)) for p in sorted(external.glob('*.png')))
    folder=run/'legacy_figures';folder.mkdir(exist_ok=True)
    items=[];failures=[];seen={}
    for r in records:
        path=Path(r['path']);key=r['sha256'][:20]
        if key in seen:
            for item in seen[key]:item['source_aliases'].append(str(path))
            continue
        try:
            source=path
            if r['kind']=='eps':
                source=folder/(key+'_eps.png')
                if not source.is_file():
                    subprocess.run(['gs','-dSAFER','-dBATCH','-dNOPAUSE','-dEPSCrop','-sDEVICE=png16m','-r150',f'-sOutputFile={source}',str(path)],check=True,capture_output=True,timeout=60)
            with Image.open(source) as im:
                added=[]
                for i in range(getattr(im,'n_frames',1)):
                    im.seek(i);output=folder/(key+f'_{i}.png');thumb=folder/(key+f'_{i}_thumb.jpg')
                    picture=ImageOps.exif_transpose(im).convert('RGB')
                    picture.save(output);picture.thumbnail((600,420));picture.save(thumb,quality=90)
                    item=dict(id=key+f'_{i}',title=path.stem+(f' · 第 {i+1} 页' if getattr(im,'n_frames',1)>1 else ''),
                        source=str(path),source_aliases=[],source_sha256=r['sha256'],relative_source=r['relative_path'],
                        kind=r['kind'],original_mode=im.mode,shape=[im.height,im.width],
                        image=str(output.relative_to(run)),thumbnail=str(thumb.relative_to(run)))
                    items.append(item);added.append(item)
                seen[key]=added
        except Exception as error:failures.append(dict(source=str(path),reason=str(error)))
    data=dict(items=items,source_files=len(records),failures=failures,
        note='历史已导出的彩色/灰度图片及 EPS 栅格预览；不增加扫描数量。MATLAB FIG 底层数值另在主页面 MAT 分类。')
    dump(run/'inventory/legacy_figure_gallery.json',data)
    (run/'legacy_figures.js').write_text('window.LEGACY_FIGURES='+json.dumps(data,ensure_ascii=False)+';\n')
    page='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>旧导出图 · SPEN / xSPEN</title><link rel="stylesheet" href="viewer.css"><script src="legacy_figures.js"></script><style>main{max-width:1480px}.tools{display:flex;gap:18px;align-items:center;flex-wrap:wrap}.tools input{border:1px solid #ddd;border-radius:8px;padding:9px 12px;width:min(380px,100%)}.card{border:0;padding:0;text-align:left;background:none}.card-image{background:#f7f7f7}.card-sub{white-space:normal;overflow-wrap:anywhere}.modal-image{background:#fafafa}#large-image{max-width:100%;max-height:75vh;object-fit:contain}#image-title,#image-source{overflow-wrap:anywhere}#image-source{font-size:11px;color:#888}</style></head><body><header class="masthead"><a class="brand" href="index.html">SPEN / xSPEN <span>← 返回数据浏览</span></a><span class="offline">本地数据</span></header><main><h1>旧导出图</h1><p class="intro">保留原图颜色、拼图与标注。这里只展示历史图片，不增加扫描数量；FIG 数值数组在主页面的 MAT 分类中。</p><div class="tools"><input id="figure-search" placeholder="搜索图名、来源路径…" aria-label="搜索旧导出图"><span id="figure-count"></span></div><div id="figures" class="cards" style="margin-top:28px"></div><div class="pagination"><button id="prev">上一页</button><span id="page"></span><button id="next">下一页</button></div><footer>静态导出图不能替代浮点数组。<a href="inventory/legacy_figure_gallery.json">来源清单 ↗</a></footer></main><dialog id="modal"><div class="modal-header"><span id="image-title"></span><button id="close">×</button></div><div class="modal-image"><img id="large-image" alt="旧导出原图"></div><p id="image-source"></p><a id="download" download>保存显示图 ↓</a></dialog><script>
const data=window.LEGACY_FIGURES.items,$=id=>document.getElementById(id);let page=0,query='';const size=36;
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function draw(){const list=data.filter(x=>(x.title+' '+x.relative_source).toLowerCase().includes(query));page=Math.max(0,Math.min(page,Math.ceil(list.length/size)-1));$('figure-count').textContent=list.length+' 张';$('page').textContent=(page+1)+' / '+Math.max(1,Math.ceil(list.length/size));$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=list.length;$('figures').innerHTML=list.slice(page*size,(page+1)*size).map(x=>`<button class="card" data-id="${x.id}"><div class="card-image"><img src="${x.thumbnail}" loading="lazy" alt="${esc(x.title)}"></div><div class="card-body"><span class="card-title">${esc(x.title)}</span><div class="card-sub">${esc(x.relative_source)}</div></div></button>`).join('');}
$('figure-search').oninput=()=>{query=$('figure-search').value.toLowerCase();page=0;draw();};$('prev').onclick=()=>{page--;draw();window.scrollTo(0,0)};$('next').onclick=()=>{page++;draw();window.scrollTo(0,0)};$('figures').onclick=event=>{const b=event.target.closest('[data-id]');if(!b)return;const x=data.find(x=>x.id===b.dataset.id);$('large-image').src=x.image;$('image-title').textContent=x.title;$('image-source').textContent=x.source;$('download').href=x.image;$('modal').showModal()};$('close').onclick=()=>$('modal').close();draw();
</script></body></html>'''
    (run/'legacy_figures.html').write_text(page)
    return data


def coverage_page(run,manifest,figures):
    core=json.loads((run/'inventory/source_coverage.json').read_text())
    archives=json.loads((run/'inventory/archive_container_coverage.json').read_text())
    groups=Counter(e['group'] for e in manifest['entries'] if e['family']=='archive')
    rows=[('已核验人脑 xSPEN', '8 份 / 6,812 帧'),('已核验 Hybrid SPEN','16 份 / 1,898 帧'),('原选例','10 个快捷入口，已包含在对应扫描中')]
    rows.extend(({'mat':'MAT / FIG 数值数组','nifti':'NIfTI 影像','dicom':'DICOM 序列','raw':'Siemens 原始 ADC','reference':'其他参考示例'}[k],f'{v:,} 个条目') for k,v in groups.items())
    rows.extend([('Bruker SPEN 旧结果',f"{manifest['summary'].get('bruker_scans',0)} 份 / {manifest['summary'].get('bruker_primary_frames',0)} 帧"),('历史导出图片',f"{len(figures['items'])} 张，保留原颜色和标注")])
    exceptions=[r for r in core['files'] if r['status'] not in ['displayed','duplicate_file']]
    failures=[r for r in core['files'] if r['status'] in ['read_error','partially_displayed']]
    esc=html.escape
    body=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>覆盖清单 · SPEN / xSPEN</title><link rel="stylesheet" href="viewer.css"><style>main{{max-width:1100px}}table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;vertical-align:top;border-bottom:1px solid #eee;padding:12px 8px}}td{{overflow-wrap:anywhere}}details{{margin:25px 0}}li{{margin:12px 0}}a{{text-decoration:underline}}.file{{font-size:11px;color:#888;overflow-wrap:anywhere}}</style></head><body><header class="masthead"><a class="brand" href="index.html">SPEN / xSPEN <span>← 返回数据浏览</span></a></header><main><h1>数据覆盖</h1><p class="intro">这里列出实际检索的范围与未显示原因。数组条目、处理阶段、切片帧和扫描采集是不同的计数。</p><table><tbody>'''
    body+=''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)}</td></tr>' for a,b in rows)
    body+=f'''</tbody></table><p>核心来源：<span class="file">{esc(str(ROOT))}</span>。检索 MAT、FIG、NIfTI、DICOM、Siemens DAT，并读取压缩包内相应文件。按文件 SHA256 合并完全相同的副本，保留每个来源别名。</p><p>当前核心清单共 {len(core['files']):,} 个源文件记录；读取失败或部分读取：{len(failures)} 个。顶层压缩包 {len(archives)} 个；递归检查另见嵌套归档记录。</p><p>Bruker SPEN 来自旧 spen_diffusion_recons 的 543 份既有实测结果，保留原始输入与旧重建。它们的人脑身份未核实，部分扫描的物种字段存在冲突；页面另列分类并标注。FLASH 是工具箱测试数据，明确标为非 SPEN。</p><p>Bruker 非 SPEN 参考序列另有 144 份（934 帧）：81 EPI、44 FLASH、19 RARE；连同工具箱 FLASH 5 帧，一共 145 个参考条目。还发现 5 个空 SPEN 扫描和 2 个空参考扫描，无可用图像；详情见来源审计。</p><p>全部原始源目录只读。新增通用数组只做读取和显示，不自动认定其解剖方向，也不把不同文件拼成未经核验的重建前后配对。</p><details><summary>没有二维影像的文件（{len(exceptions)} 个）</summary><ul>'''
    for r in exceptions:
        reason=r.get('reason') or '仅参数、向量、符号对象或不含二维图像的绘图结构；变量级原因见 JSON。'
        body+=f'<li>{esc(Path(r["source"]).name)} — {esc(reason)}<div class="file">{esc(r["source"])}</div></li>'
    body+='''</ul></details><p>不作为新采集重复导入：已经对应到原 DAT 的逐线圈 FID 文本、旧项目的逐文件镜像、IXI 等训练数据和模拟训练缓存、软件图标。旧目录中的部分参考图和处理系数可查看，但不计入人脑扫描数。</p><p>MAT / FIG 仅将至少两个维度达到 16 的数值数组纳入图像浏览；小矩阵、坐标向量、MATLAB 符号对象等在逐变量清单中说明。FIG 只提取底层数组；绘图布局和色标没有自动复刻。</p><ul>'''
    for file,title in [('source_coverage.json','核心逐文件覆盖'),('archive_container_coverage.json','顶层压缩包'),('nested_archive_audit.json','嵌套归档审计'),('additional_source_audit.json','其他来源与排除范围'),('bruker_import_sources.json','Bruker 来源'),('bruker_verification.json','Bruker 导入校验'),('bruker_reference_entries.json','Bruker 非 SPEN 参考序列及空文件'),('legacy_figure_gallery.json','旧导出图片来源')]:
        if (run/'inventory'/file).exists():body+=f'<li><a href="inventory/{file}">{title} ↗</a></li>'
    body+='</ul></main></body></html>'
    (run/'coverage.html').write_text(body)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);args=parser.parse_args()
    run=args.run.resolve();assert run.is_relative_to(HERE/'runs')
    manifest=json.loads((run/'manifest.json').read_text())
    entries=[e for e in manifest['entries'] if e['family']!='bruker' and e.get('group')!='reference']
    if (run/'archive_sources/pvmatlab_flash_testdata/source.json').is_file():entries.append(flash_reference(run))
    references=run/'inventory/bruker_reference_entries.json'
    if references.is_file():
        refs=json.loads(references.read_text());entries.extend(refs['entries'] if isinstance(refs,dict) else refs)
    bruker=json.loads((run/'inventory/bruker_entries.json').read_text())
    if isinstance(bruker,dict):bruker=bruker['entries']
    entries.extend(bruker)
    figures=old_figures(run)
    summary=manifest['summary'];summary.update(entry_count=len(entries),dataset_count=sum(not e.get('is_selected') for e in entries),
        archive_entries=sum(e['family']=='archive' for e in entries),archive_frames=sum(e['frame_count'] for e in entries if e['family']=='archive'),
        bruker_scans=len(bruker),bruker_primary_frames=sum(e['frame_count'] for e in bruker),legacy_figure_count=len(figures['items']),
        catalog_roots=[str(ROOT),'/home/data2/chk/workspace/2026/08/14/spen_diffusion_recons/results/real_batch','/home/data2/chk/workspace/2026/08/14/spen_recons/data/scanner'])
    manifest=dict(summary=summary,entries=entries)
    dump(run/'manifest.json',manifest);(run/'manifest.js').write_text('window.REVIEW_MANIFEST='+json.dumps(manifest,ensure_ascii=False)+';\n')
    coverage_page(run,manifest,figures)
    for name in ['index.html','viewer.js','viewer.css','archive_pixels.js']:shutil.copy2(HERE/'web'/name,run/name)
    selection=json.loads((HERE/'selection.json').read_text());v=selection['visualization']
    v.update(builder=['build_unified.py','expand_scans.py','catalog_all.py','import_bruker.py','import_bruker_reference.py','finalize_catalog.py'],
        scope='核心 xSPEN_项目的可读数值影像和压缩包；另含旧 Bruker SPEN 结果、FLASH 工具参考、历史导出图。逐项覆盖与排除原因见 coverage.html。',
        dataset_count=summary['dataset_count'],archive_entries=summary['archive_entries'],archive_array_frames=summary['archive_frames'],
        bruker_scans=len(bruker),bruker_primary_frames=summary['bruker_primary_frames'],legacy_figure_count=summary['legacy_figure_count'],
        coverage_html='runs/unified_review_260916/coverage.html',coverage_json='runs/unified_review_260916/inventory/source_coverage.json')
    v['interface']='浅色简洁界面，按数据类型筛选、全文来源搜索、分页、切片/附加轴切换、连续浏览、选例跳转和点击放大；大型数组按需加载。'
    dump(HERE/'selection.json',selection);dump(run/'selection.json',selection)
    provenance=json.loads((run/'provenance.json').read_text());provenance.update(catalog_finalized_utc=datetime.now(timezone.utc).isoformat(),catalog_summary=summary)
    for path in [*HERE.glob('*.py'),HERE/'selection.json',*list((HERE/'web').glob('*'))]:
        if path.is_file():provenance['source_hashes'][str(path)]=digest(path)
    dump(run/'provenance.json',provenance)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
