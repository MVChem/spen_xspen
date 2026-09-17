'use strict';
const manifest = window.REVIEW_MANIFEST;
const entries = manifest.entries;
const $ = id => document.getElementById(id);
const escapeHTML = s => String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const groups = [['all','全部影像'],['verified','已核验扫描'],['crossed','xSPEN 对照'],['hybrid','Hybrid SPEN 对照'],['selected','10 个选例'],['mat','MAT 数组'],['nifti','NIfTI 影像'],['dicom','DICOM 序列'],['raw','原始 ADC'],['bruker','Bruker SPEN'],['reference','其他参考序列']];
const isSelected = e => Boolean(e.is_selected ?? (e.family === 'crossed' && e.frame_count === 1));
let filter = 'all', query = '', active = null, stages = [], coords = [], loadToken = 0;
let filmAxis = 0, filmStage = 0, filmPage = 0, catalogPage = 0, renderToken = 0;
let renderBusy = false;
const CATALOG_PAGE_SIZE = 48, FILM_PAGE_SIZE = 64;
const loaded = new Map();
const match = (e, group) => group === 'selected' ? isSelected(e) : !isSelected(e) && (group === 'all' || (group === 'verified' && ['crossed','hybrid'].includes(e.family)) || e.family === group || e.group === group);
const matchesSearch = e => `${e.id} ${e.title} ${e.subtitle} ${e.family} ${e.status_label} ${familyLabel(e)} ${e.source} ${e.source_details?.variable||''}`.toLowerCase().includes(query.toLowerCase());
const visibleEntries = () => entries.filter(e => match(e, filter) && matchesSearch(e));
const axisText = (a,n) => a.values ? `${a.values[n]} (${n+1}/${a.size})` : `${n+1} / ${a.size}`;
const indexOf = (s, point) => s.axes.reduce((index, axis, i) => index * axis.size + point[i], 0);
const isAvailable = (s, index) => !s.frame_available || s.frame_available[index];
const number = n => Number(n).toLocaleString('zh-CN');
const familyLabel = e => e.family === 'archive' ? ({mat:'MAT',nifti:'NIfTI',dicom:'DICOM',raw:'ADC',reference:'参考'})[e.group] : e.family === 'crossed' ? 'xSPEN' : e.family === 'bruker' ? 'Bruker SPEN' : 'Hybrid SPEN';
const stageLabel = s => ({raw_ro_rss:'输入',raw_ro:'原始输入',legacy_recipe_rss:'旧流程复算',legacy_saved_rss:'旧 MATLAB',tikhonov_rss:'Tikhonov',phasemap_rss:'PhaseMap + InvA',raw_adc_log:'原始 ADC',recon:'已有重建',before:'重建前'})[s.id] || s.label;
const positionText = () => isSelected(active) ? `slice ${active.slice} · occurrence ${active.occurrence}` : stages[0].axes.map((a,i)=>`${a.label} ${axisText(a,coords[i])}`).join(' · ');

function init() {
  const summary=manifest.summary;
  const scans=entries.filter(e=>!isSelected(e));
  $('stats').innerHTML=`<strong>${number(scans.length)}</strong> 个影像条目 · ${summary.distinct_scans} 份 xSPEN / Hybrid 对照 · ${entries.filter(isSelected).length} 个选例`;
  $('scope-note').textContent=summary.archive_entries ? `已检索 xSPEN_项目资料目录中的 MAT / FIG、NIfTI、DICOM、Siemens DAT 及压缩包内对应文件，共 ${number(summary.source_files)} 个源文件记录。新增 ${number(summary.archive_entries)} 个独立数组/序列条目，原有 ${summary.distinct_scans} 份已核验扫描保留对照。另含 ${summary.bruker_scans||0} 份 Bruker SPEN 旧结果及参考序列。重复文件、非影像内容和范围限制均列入覆盖清单。` : '已核验扫描与 10 个选例；资料库扩展见覆盖清单。';
  $('filters').innerHTML=groups.map(([id,label])=>`<button data-filter="${id}">${label}<span class="count">${entries.filter(e=>match(e,id)).length}</span></button>`).join('');
  $('filters').addEventListener('click',event=>{const b=event.target.closest('[data-filter]');if(!b)return;filter=b.dataset.filter;catalogPage=0;location.hash='overview';showOverview();});
  $('search').oninput=()=>{query=$('search').value.trim();catalogPage=0;location.hash='overview';showOverview();};
  $('prev-entry').onclick=()=>stepEntry(-1);$('next-entry').onclick=()=>stepEntry(1);
  for(const id of ['window-mode','window-width','gamma','show-adc'])$(id).addEventListener('input',renderFrames);
  $('reset-display').onclick=()=>{$('window-mode').value='frame';$('window-width').value=1;$('gamma').value=1;renderFrames();};
  $('film-axis').onchange=()=>{filmAxis=Number($('film-axis').value);filmPage=Math.floor(coords[filmAxis]/FILM_PAGE_SIZE);renderFrames();};
  $('film-stage').onchange=()=>{filmStage=Number($('film-stage').value);renderFrames();};
  $('catalog-prev').onclick=()=>{catalogPage--;showOverview();window.scrollTo(0,0);};$('catalog-next').onclick=()=>{catalogPage++;showOverview();window.scrollTo(0,0);};
  $('film-prev').onclick=()=>{filmPage--;renderFrames();};$('film-next').onclick=()=>{filmPage++;renderFrames();};
  $('modal-close').onclick=()=>$('image-modal').close();
  $('image-modal').addEventListener('click',event=>{if(event.target===$('image-modal')){const r=event.target.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)event.target.close();}});
  window.addEventListener('hashchange',route);
  let resizeTimer;window.addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(()=>{if(active)renderFrames();},150);});
  document.addEventListener('keydown',e=>{if(!active||$('image-modal').open||/INPUT|SELECT|BUTTON/.test(e.target.tagName))return;if(e.key==='ArrowRight'||e.key==='ArrowLeft'){e.preventDefault();setCoord(0,coords[0]+(e.key==='ArrowRight'?1:-1));}});
  route();
}
function renderNavigation(){
  document.querySelectorAll('[data-filter]').forEach(b=>b.classList.toggle('active',b.dataset.filter===filter));
  const list=visibleEntries().slice(0,80);
  $('entry-nav').innerHTML=list.map(e=>`<a class="scan-link ${active?.id===e.id?'active':''}" data-entry="${escapeHTML(e.id)}" href="#${encodeURIComponent(e.id)}"><span>${escapeHTML(e.title)}</span>${!isSelected(e)?`<span class="nav-sub">${number(e.frame_count)}</span>`:''}</a>`).join('');
}
function showOverview(){
  loadToken++;renderToken++;renderBusy=false;active=null;stages=[];$('overview').hidden=false;$('detail').hidden=true;
  if($('image-modal').open)$('image-modal').close();
  renderNavigation();const selected=visibleEntries();
  $('collection-title').textContent=groups.find(g=>g[0]===filter)[1];$('result-count').textContent=`${number(selected.length)} 个条目`;
  catalogPage=Math.max(0,Math.min(catalogPage,Math.ceil(selected.length/CATALOG_PAGE_SIZE)-1));
  $('catalog-pages').hidden=selected.length<=CATALOG_PAGE_SIZE;$('catalog-prev').disabled=catalogPage===0;$('catalog-next').disabled=(catalogPage+1)*CATALOG_PAGE_SIZE>=selected.length;
  $('catalog-page-label').textContent=`${catalogPage*CATALOG_PAGE_SIZE+1}–${Math.min((catalogPage+1)*CATALOG_PAGE_SIZE,selected.length)} / ${number(selected.length)}`;
  $('empty-state').hidden=selected.length>0;
  $('cards').innerHTML=selected.slice(catalogPage*CATALOG_PAGE_SIZE,(catalogPage+1)*CATALOG_PAGE_SIZE).map(e=>`<a class="card" href="#${encodeURIComponent(e.id)}" data-card="${escapeHTML(e.id)}"><div class="card-image"><img src="${e.thumbnail}" loading="lazy" alt="${escapeHTML(e.title)} 代表帧"></div><div class="card-body"><div class="card-heading"><span class="card-title">${escapeHTML(e.title)}</span><span class="family-label">${familyLabel(e)}</span></div><div class="card-sub">${escapeHTML(e.subtitle)}</div><div class="card-bottom"><span class="badge ${e.status}">${escapeHTML(e.status_label)}</span><span class="frame-count">${isSelected(e)?'固定选例':number(e.frame_count)+' 帧'}</span></div></div></a>`).join('');
}
function route(){let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{id='overview';}const e=entries.find(e=>e.id===id);if(e)openEntry(e);else showOverview();}
function stepEntry(delta){let list=visibleEntries();if(!list.some(e=>e.id===active?.id))list=entries.filter(e=>isSelected(e)===isSelected(active));const at=list.findIndex(e=>e.id===active?.id);if(list.length)location.hash=encodeURIComponent(list[(at+delta+list.length)%list.length].id);}
async function loadPayload(e){
  if(!loaded.has(e.id))loaded.set(e.id,new Promise((resolve,reject)=>{
    const script=document.createElement('script');script.src=e.payload;
    script.onload=()=>{try{const payload=window.REVIEW_PAYLOADS[e.id];
      for(const s of payload){if(s.chunked)continue;const binary=atob(s.data),bytes=new Uint8Array(binary.length);for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
        if(new Uint8Array(new Uint16Array([1]).buffer)[0]===1)s.pixels=new Uint16Array(bytes.buffer);
        else{const view=new DataView(bytes.buffer);s.pixels=new Uint16Array(bytes.length/2);for(let i=0;i<s.pixels.length;i++)s.pixels[i]=view.getUint16(i*2,true);}
        delete s.data;
      }
      delete window.REVIEW_PAYLOADS[e.id];script.remove();resolve(payload);
    }catch(error){loaded.delete(e.id);script.remove();reject(error);}};
    script.onerror=()=>{loaded.delete(e.id);script.remove();reject(new Error('无法读取图像。请保留完整输出文件夹，再打开 index.html。'));};document.head.append(script);
  }));
  const promise=loaded.get(e.id);loaded.delete(e.id);loaded.set(e.id,promise);
  const payload=await promise;
  for(const id of [...loaded.keys()]){if(loaded.size<=3)break;if(id!==active?.id&&id!==e.id)loaded.delete(id);}
  return payload;
}
async function openEntry(e){
  const token=++loadToken;renderToken++;renderBusy=false;active=e;stages=[];
  if(!match(e,filter))filter=isSelected(e)?'selected':e.family==='archive'?e.group:e.family;
  if(!matchesSearch(e)){query='';$('search').value='';}
  $('overview').hidden=true;$('detail').hidden=false;$('viewer').hidden=true;$('loading').hidden=false;$('loading').textContent='正在读取图像…';
  $('detail-family').textContent=familyLabel(e)+(isSelected(e)?' · 选例':'');$('detail-title').textContent=e.title;$('detail-subtitle').textContent=e.subtitle;
  const g=e.geometry;
  $('facts').innerHTML=[`${e.stages[0].shape[1]} × ${e.stages[0].shape[2]}`,g.fov_unit==='px'?'像素比例':`${g.fov_mm.map(x=>Number(x.toFixed(3))).join(' × ')} ${g.fov_unit||'mm'}`,`${number(e.frame_count)} 帧`,g.r_value!=null?`R = ${g.r_value}`:null].filter(Boolean).map(v=>`<span class="fact">${escapeHTML(v)}</span>`).join('');
  const warnings=e.warnings.filter(w=>/没有同名|实际仅记录|缺失 PE|排除了|重建失败|具体解剖部位|NaN|物种|元数据冲突/.test(w));
  $('notice').innerHTML=warnings.map(w=>`<p>${escapeHTML(w)}</p>`).join('');$('notice').hidden=!warnings.length;
  const full=entries.find(x=>x.scan===e.scan&&!isSelected(x)&&x.family!=='archive');$('full-scan-link').hidden=!(isSelected(e)&&full);if(full)$('full-scan-link').href='#'+encodeURIComponent(full.id);
  const bookmarks=entries.filter(x=>x.scan===e.scan&&isSelected(x));$('bookmarks').hidden=isSelected(e)||!bookmarks.length;
  $('bookmarks').innerHTML='<span>选例</span>'+bookmarks.map(x=>`<a href="#${encodeURIComponent(x.id)}">slice ${x.slice} · occurrence ${x.occurrence}</a>`).join('');
  renderNavigation();
  const rows=[['来源文件',e.source],['Scanner H5',e.scanner_h5||'—'],['旧重建 MAT',e.source_mat||'见原始提取记录或各阶段说明'],['序列',g.sequence],[e.family==='archive'?'显示矩阵（行 × 列）':'头部 PE × RO',g.header_pe_ro?.join(' × ')],['FOV / 层厚',`${g.fov_mm.join(' × ')} ${g.fov_unit||'mm'} / ${g.thickness_mm??'未知'}`],['TE',g.echo_time_ms!=null?g.echo_time_ms+' ms':'见原始记录'],['显示顺序',e.source_details?.display_note||'PE 纵轴，RO 横轴。切片位置从 1 开始；选例的 slice / occurrence 和原始 counter 从 0 开始。各数组层序保留对应来源。'],['原始数组',e.source_details?.original_shape?.join(' × ')||e.source_details?.shape_rep_slice_coil_pe_ro?.join(' × ')||'见元数据']];
  $('metadata').innerHTML=`<table>${rows.map(([k,v])=>`<tr><th>${escapeHTML(k)}</th><td>${escapeHTML(v)}</td></tr>`).join('')}</table><p><a href="${e.arrays}" download>显示数组 NPZ ↓</a>${e.complex_arrays?`<a href="${e.complex_arrays}" download>ADC / 复数数组 ↓</a>`:''}<a href="${e.metadata}" target="_blank">完整元数据 ↗</a></p><h4>显示阶段</h4><ul>${e.stages.map(s=>`<li>${escapeHTML(stageLabel(s))}：${escapeHTML(s.note)}</li>`).join('')}</ul><h4>数据说明</h4><ul>${e.warnings.map(w=>`<li>${escapeHTML(w)}</li>`).join('')}</ul><p>页面使用 16 位显示数据，下载数组保留 float32。每帧独立窗默认为 p99.5；ADC 对数幅度始终独立设窗。旧 reader 与当前 reader 的幅度尺度可能不同。</p>`;
  window.scrollTo({top:0,behavior:'instant'});
  try{const result=await loadPayload(e);if(token!==loadToken)return;stages=result;coords=[...e.default_coords];if(e.id==='MID253')coords[0]=19;
    $('show-adc').checked=false;$('show-adc').parentElement.hidden=!stages.some(s=>s.log_display);
    $('window-mode').value='frame';$('window-width').value=1;$('gamma').value=1;$('display-settings').open=false;
    $('axis-controls').innerHTML=stages[0].axes.map((a,i)=>a.size>1?`<label class="axis-control"><span class="axis-head"><span>${escapeHTML(a.label)}</span><output id="axis-value-${i}"></output></span><input aria-label="${escapeHTML(a.label)}" data-axis="${i}" type="range" min="0" max="${a.size-1}" step="1" value="${coords[i]}"></label>`:'').join('');
    document.querySelectorAll('[data-axis]').forEach(input=>input.oninput=()=>setCoord(Number(input.dataset.axis),Number(input.value)));
    const variableAxes=stages[0].axes.map((a,i)=>[a,i]).filter(([a])=>a.size>1);
    $('film-section').hidden=!variableAxes.length;filmAxis=variableAxes[0]?.[1]??0;filmStage=0;filmPage=Math.floor(coords[filmAxis]/FILM_PAGE_SIZE);
    $('film-axis').innerHTML=variableAxes.map(([a,i])=>`<option value="${i}">${escapeHTML(a.label)} · ${a.size}</option>`).join('');
    $('film-stage').innerHTML=stages.map((s,i)=>`<option value="${i}">${escapeHTML(stageLabel(s))}</option>`).join('');
    $('loading').hidden=true;$('viewer').hidden=false;renderFrames();
  }catch(error){if(token===loadToken)$('loading').textContent=error.message;}
}
function setCoord(axis,n){if(!stages.length)return;coords[axis]=Math.max(0,Math.min(stages[0].axes[axis].size-1,n));if(axis===filmAxis)filmPage=Math.floor(coords[axis]/FILM_PAGE_SIZE);renderFrames();}
function windowFor(s,index,point){
  let limit=ArchivePixels.read(s,index).p995;const mode=$('window-mode').value;
  if(!s.log_display&&mode==='shared')limit=Math.max(...stages.filter(x=>!x.log_display&&isAvailable(x,indexOf(x,point))).map(x=>ArchivePixels.read(x,indexOf(x,point)).p995));
  if(!s.log_display&&mode==='scan')limit=s.global_window;
  return Math.max(limit,1e-30)*Number($('window-width').value);
}
function paint(canvas,s,index,point,boxWidth=620,boxHeight=460){
  const [,h,w]=s.shape;canvas.width=w;canvas.height=h;
  const info=ArchivePixels.read(s,index),gamma=Number($('gamma').value);let limit,lower=0;
  if(s.signed||($('window-mode').value==='shared'&&stages.some(x=>x.signed))){const mode=$('window-mode').value;let hi=mode==='scan'?s.global_window:info.p995,lo=mode==='scan'?(s.global_lower||0):(info.p005||0);if(mode==='shared'){const values=stages.filter(x=>!x.log_display&&isAvailable(x,indexOf(x,point))).map(x=>ArchivePixels.read(x,indexOf(x,point)));hi=Math.max(...values.map(x=>x.p995));lo=Math.min(...values.map(x=>x.p005||0));}const center=(hi+lo)/2,half=(hi-lo)*Number($('window-width').value)/2;lower=center-half;limit=center+half;}else limit=windowFor(s,index,point);
  const ctx=canvas.getContext('2d'),image=ctx.createImageData(w,h),offset=info.offset,span=info.max-info.min;
  for(let i=0;i<w*h;i++){const pos=offset+i,invalid=info.invalid&&((info.invalid[pos>>3]>>(pos&7))&1),value=info.pixels[pos]/65535*span+info.min;const v=invalid?128:Math.round(255*Math.pow(Math.max(0,Math.min(1,(value-lower)/Math.max(limit-lower,1e-30))),1/gamma));image.data[i*4]=v;image.data[i*4+1]=v;image.data[i*4+2]=v;image.data[i*4+3]=255;}
  ctx.putImageData(image,0,0);
  const ratio=s.fov[1]/s.fov[0],parent=canvas.parentElement,available=parent.clientWidth-(parent.classList.contains('canvas-wrap')?20:0),drawW=Math.max(1,Math.min(boxWidth,boxHeight*ratio,available));canvas.style.width=drawW+'px';canvas.style.height=(drawW/ratio)+'px';canvas.style.aspectRatio=String(ratio);
  return limit;
}
function openImage(s){
  if(renderBusy)return;
  const index=indexOf(s,coords);if(!isAvailable(s,index))return;
  $('modal-title').textContent=`${active.scan} · ${stageLabel(s)}`;$('modal-caption').textContent=positionText();
  $('image-modal').showModal();paint($('modal-canvas'),s,index,coords,820,Math.min(innerHeight*.72,850));
}
async function renderFrames(){
  if(!stages.length)return;const token=++renderToken;renderBusy=true;
  try{const requests=stages.map(s=>[s,indexOf(s,coords)]);if(!$('film-section').hidden){const fs=stages[filmStage],count=fs.axes[filmAxis].size;for(let n=filmPage*FILM_PAGE_SIZE;n<Math.min((filmPage+1)*FILM_PAGE_SIZE,count);n++){const point=[...coords];point[filmAxis]=n;requests.push([fs,indexOf(fs,point)]);if($('window-mode').value==='shared')for(const s of stages)requests.push([s,indexOf(s,point)]);}}
  const keep=await ArchivePixels.ensure(requests);if(token!==renderToken)return;
  stages[0].axes.forEach((a,i)=>{const out=$('axis-value-'+i);if(out)out.textContent=axisText(a,coords[i]);const input=document.querySelector(`[data-axis="${i}"]`);if(input)input.value=coords[i];});
  $('width-value').textContent=Number($('window-width').value).toFixed(2)+'×';$('gamma-value').textContent=Number($('gamma').value).toFixed(2);$('view-caption').textContent=positionText();
  const visible=stages.filter(s=>!s.log_display||$('show-adc').checked);$('stage-grid').style.setProperty('--columns',visible.length);
  $('stage-grid').innerHTML=visible.map(s=>`<article class="stage"><h3>${escapeHTML(stageLabel(s))}</h3><div class="canvas-wrap" role="button" tabindex="0" aria-label="放大${escapeHTML(stageLabel(s))}">${isAvailable(s,indexOf(s,coords))?`<canvas aria-label="${escapeHTML(stageLabel(s))}" data-stage="${s.id}"></canvas>`:'<span class="unavailable">此帧重建不可用</span>'}</div><div class="stage-note"><span class="stage-window"></span></div></article>`).join('');
  visible.forEach((s,i)=>{const card=$('stage-grid').children[i],canvas=card.querySelector('canvas');if(canvas){const limit=paint(canvas,s,indexOf(s,coords),coords);card.querySelector('.stage-window').textContent=`${s.shape[1]} × ${s.shape[2]} · 窗 ${limit.toPrecision(3)}`;}const box=card.querySelector('.canvas-wrap');box.onclick=()=>openImage(s);box.onkeydown=event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();openImage(s);}};});
  renderFilm();ArchivePixels.prune(keep);renderBusy=false;
  }catch(error){if(token===renderToken){renderBusy=false;$('loading').hidden=false;$('loading').textContent=error.message;}}
}
function renderFilm(){
  if(!stages.length||$('film-section').hidden)return;
  const s=stages[filmStage],axis=s.axes[filmAxis],start=filmPage*FILM_PAGE_SIZE,length=Math.min(FILM_PAGE_SIZE,axis.size-start);
  $('film-pages').hidden=axis.size<=FILM_PAGE_SIZE;$('film-prev').disabled=filmPage===0;$('film-next').disabled=start+length>=axis.size;$('film-page-label').textContent=`${start+1}–${start+length} / ${number(axis.size)}`;
  $('filmstrip').innerHTML=Array.from({length},(_,offset)=>{const n=start+offset,point=[...coords];point[filmAxis]=n;return `<button class="film-frame ${coords[filmAxis]===n?'active':''}" data-frame="${n}"><span class="film-image">${isAvailable(s,indexOf(s,point))?`<canvas aria-label="${escapeHTML(axis.label)} ${n+1}"></canvas>`:'<span class="unavailable">不可用</span>'}</span><span class="film-label">${escapeHTML(axis.label)} ${escapeHTML(axisText(axis,n))}</span></button>`;}).join('');
  Array.from($('filmstrip').children).forEach((button,offset)=>{const n=start+offset,point=[...coords];point[filmAxis]=n;const canvas=button.querySelector('canvas');if(canvas)paint(canvas,s,indexOf(s,point),point,170,Math.min(116,button.querySelector('.film-image').clientHeight));button.onclick=()=>setCoord(filmAxis,n);});
}
init();
