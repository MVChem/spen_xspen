"""Static contact sheets and a self-contained interactive viewer for selected cases."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
if Path(FONT).exists():
    from matplotlib import font_manager
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams['axes.unicode_minus'] = False

TITLES = ['原始 ADC 幅度（log RSS）', '输入：RO FFT + RSS', '旧 MATLAB Img / 同流程复算',
          '奇偶校正 + Tikhonov', 'PhaseMap + 加窗 InvA']
VIEWS = {'axial': '轴位', 'sagittal': '矢状位', 'coronal': '冠状位'}


def values(data):
    raw = np.sqrt(np.sum(np.abs(data['raw_adc']) ** 2, axis=0))
    scale = float(np.quantile(raw, .995))
    adc = np.log1p(raw / max(scale, 1e-30) * 100)
    keys = ['raw_ro_rss', 'legacy_saved_rss' if 'legacy_saved_rss' in data else 'legacy_recipe_rss',
            'tikhonov_rss', 'phasemap_rss']
    ims = [data[key] for key in keys]
    window = max(float(np.quantile(im, .995)) for im in ims)
    return [adc] + ims, window, scale


def draw_row(axes, record, data, number, title=False, shared=False):
    ims, window, _ = values(data)
    m, k = ims[1].shape
    meta = record['metadata']
    aspect = (meta['fov_mm'][0] / m) / (meta['fov_mm'][1] / k)
    for j, (ax, im) in enumerate(zip(axes, ims)):
        ax.imshow(im, cmap='gray', vmin=0, vmax=float(np.quantile(im, .995)) if j == 0 or not shared else window,
                  interpolation='nearest', aspect='auto' if j == 0 else aspect)
        if title:
            ax.set_title(TITLES[j], fontsize=11, pad=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if j == 0:
            ax.set_ylabel(f"{number:02d}  {record['scan']}\n{VIEWS[record['view']]}  s{record['slice']} / o{record['occurrence']}\n{m}×{k}", fontsize=10)
        if j == 2:
            ax.set_xlabel('旧文件 Img' if record['original_mat'] else
                '同流程复算（旧 MAT 未配对）' if record.get('unmatched_mat_candidate') else '同流程复算（无同名 MAT）', fontsize=9)


def verify(out, records):
    checks = []
    for record in records:
        with np.load(out / 'cases' / record['id'] / 'arrays.npz') as data:
            assert all(np.isfinite(data[key]).all() for key in data.files)
            raw = data['processed_raw']
            expected = record['metadata']['shape_rep_slice_coil_pe_ro'][2:]
            assert list(raw.shape) == expected
            assert data['raw_adc'].shape == (raw.shape[0], raw.shape[1], 2 * raw.shape[2])
            assert np.all(np.abs(data['raw_adc']).sum(axis=(0, 2)) > 0)
            for key, coils in [('raw_ro_rss', 'ro_complex'), ('tikhonov_rss', 'tikhonov_coils'), ('phasemap_rss', 'phasemap_coils')]:
                expected_im = np.sqrt(np.sum(np.abs(data[coils]) ** 2, axis=0))
                np.testing.assert_allclose(data[key], expected_im, rtol=2e-6, atol=2e-7)
            a, x = data['encoding_a'], data['tikhonov_coils']
            rhs = a.conj().T @ data['tikhonov_corrected_ro']
            residual = (a.conj().T @ a + .01 * np.eye(a.shape[1])) @ x - rhs
            stationarity = float(np.linalg.norm(residual) / np.linalg.norm(rhs))
            assert stationarity < 2e-5
            phase_error = float(np.max(np.abs(np.abs(data['phasemap_corrected_ro']) - np.abs(data['ro_complex']))))
            np.testing.assert_allclose(np.abs(data['phasemap_corrected_ro']), np.abs(data['ro_complex']), rtol=5e-7, atol=1e-7)
            np.testing.assert_array_equal(data['phasemap_corrected_ro'][:, ::2], data['ro_complex'][:, ::2])
            checks.append(dict(id=record['id'], all_finite=True, native_shape=expected[-2:],
                fresh_raw_reader_relative_error=record['adc_verification']['fresh_reader_relative_error'],
                tikhonov_normal_equation_relative_residual=stationarity,
                phasemap_magnitude_max_abs_error=phase_error,
                old_mat_correlation=record['original_mat']['recipe_correlation'] if record['original_mat'] else None))
    result = dict(passed=True, cases=len(checks), distinct_scans=len({r['scan'] for r in records}), checks=checks)
    (out / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def render(out):
    records = json.loads((out / 'manifest.json').read_text())
    checks = verify(out, records)
    panels = []
    for start, count, filename in [(0, 10, 'overview.png'), (0, 5, 'overview_01_05.png'), (5, 5, 'overview_06_10.png'), (0, 10, 'overview_shared_window.png')]:
        fig, axes = plt.subplots(count, 5, figsize=(14.5, count * 2.8 + 1.2), squeeze=False)
        for i, record in enumerate(records[start:start + count]):
            with np.load(out / 'cases' / record['id'] / 'arrays.npz') as data:
                draw_row(axes[i], record, data, start + i + 1, title=i == 0, shared='shared' in filename)
        fig.suptitle('xSPEN 人脑：原始数据、旧重建与传统逆编码', fontsize=17, y=.99)
        caption = '每行图像列共用窗，旧 reader 的幅度尺度与当前 reader 不同。' if 'shared' in filename else '各图独立 p99.5 窗，仅比较结构；ADC 为对数显示。'
        fig.text(.5, .012, caption + '原生 PE/RO 坐标，无插值重建；s / o 均从 0 编号。', ha='center', fontsize=10)
        fig.subplots_adjust(left=.105, right=.99, top=.95 if count == 10 else .90, bottom=.04, wspace=.055, hspace=.24)
        fig.savefig(out / filename, dpi=150)
        plt.close(fig)
    for i, record in enumerate(records):
        folder = out / 'cases' / record['id']
        with np.load(folder / 'arrays.npz') as data:
            ims, window, adc_scale = values(data)
            fig, axes = plt.subplots(1, 5, figsize=(16, 4.2))
            draw_row(axes, record, data, i + 1, title=True)
            fig.suptitle(f"{record['id']} — {record['selection_reason']}", fontsize=12)
            fig.tight_layout(rect=(0, .03, 1, .87))
            fig.savefig(folder / 'comparison.png', dpi=150)
            plt.close(fig)
            display = dict(default='per-panel p99.5 for structure; independent display only, not stored-array normalization',
                common_window=[0, window], per_panel_upper=[float(np.quantile(im, .995)) for im in ims], percentile=.995, gamma=1,
                adc_log_formula='log1p(100 * RSS_coils(raw_adc) / p99.5(RSS_coils(raw_adc)))',
                adc_scale=adc_scale, orientation='Unmodified stored PE/RO axes; no anatomical left/right labels',
                interpolation='nearest for display only; no image-array resampling')
            (folder / 'display.json').write_text(json.dumps(display, indent=2) + '\n')
            panels.append(dict(id=record['id'], scan=record['scan'], view=VIEWS[record['view']],
                slice=record['slice'], occurrence=record['occurrence'], reason=record['selection_reason'],
                shape=list(ims[1].shape), fov=record['metadata']['fov_mm'],
                raw_shape=list(data['raw_adc'].shape), r=record['metadata']['r_value'],
                protocol_b=record['metadata']['nominal_protocol_b_value'],
                source=record['raw_source'], mat=record['original_mat'], unmatched=record.get('unmatched_mat_candidate'), window=window,
                images=[dict(shape=list(im.shape), values=im.astype(np.float32).ravel().tolist(),
                    p995=float(np.quantile(im, .995))) for im in ims]))
    write_html(out, panels)
    write_report(out, records, checks)
    print(f'Complete: {out / "index.html"}', flush=True)


def write_html(out, panels):
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>xSPEN 人脑数据查看</title>
<style>
:root{font-family:system-ui,"Noto Sans CJK SC",sans-serif;color:#e5eaf3;background:#10151e}body{max-width:1580px;margin:auto;padding:28px}h1{font-size:27px}p{line-height:1.65;color:#b9c5d6}a{color:#92c6ff}header{max-width:1080px}nav{position:sticky;top:0;z-index:5;background:#192230f5;padding:16px;border-radius:12px;display:flex;gap:20px;align-items:center;flex-wrap:wrap}select,button{background:#273448;border:1px solid #4e617d;color:white;padding:7px;border-radius:6px}article{margin:24px 0;padding:20px;background:#192230;border-radius:12px}h2{font-size:20px;margin:0 0 8px}.tag{font-size:13px;color:#aac5df}.grid{display:grid;grid-template-columns:repeat(5,minmax(100px,1fr));gap:10px}.panel{background:#080a0e;padding:10px;border-radius:8px}.panel p{font-size:13px;margin:0 0 8px;color:#d5deed;height:36px}.canvasbox{width:100%;aspect-ratio:1;display:flex;align-items:center}canvas{width:100%;image-rendering:pixelated;display:block}.note{font-size:13px}.source{overflow-wrap:anywhere}details{margin-top:12px}summary{cursor:pointer;color:#9fc9f4}input{vertical-align:middle}.status{color:#80d9b0}footer{padding:24px 0;font-size:14px}@media(max-width:850px){body{padding:12px}.grid{grid-template-columns:repeat(2,minmax(120px,1fr))}.panel p{height:auto}}
</style><header><h1>xSPEN 人脑 · 10 个代表样本</h1>
<p>8 组扫描，覆盖轴位 / 矢状位 / 冠状位、约 3 mm / 4 mm 采样网格。每例从原始 Siemens .dat 重新提取 ADC，并核对到相同切片与采集时序。这里的 10 例是 10 个观测，不代表 10 位独立受试者。</p>
<p>左起：原始 ADC 幅度 → 仅做读出 FFT 的输入 → 旧 MATLAB Img（缺失时按同一流程复算）→ 奇偶相位校正 + Tikhonov → PhaseMap + 加窗 InvA。旧 Img 包含 RO 高斯窗和线圈合成，没有 PE 逆编码；后两列采用当前项目的 xSPEN sinc 模型。</p>
<p class="status">10 / 10 原始数据配对及数值检查通过。未使用学习先验。</p></header>
<nav><label>方向 <select id="view"><option>全部</option><option>轴位</option><option>矢状位</option><option>冠状位</option></select></label><label>显示 <select id="mode"><option value="own">各图独立 p99.5 窗</option><option value="shared">每例图像列共用窗</option></select></label><label>窗宽 <input id="window" type="range" min="40" max="180" value="100"><span id="level">100%</span></label><button id="reset">恢复默认</button><a href="overview.png">整页对照图</a><a href="report.md">数据说明</a></nav>
<p class="note">ADC 为 log RSS，使用独立灰度窗。其余列默认独立 p99.5 窗，便于比较结构，不能比较幅度。共用窗另供检查：旧 MATLAB reader 的额外幅度尺度尚未统一，旧 Img 通常约为当前流程复算的 2 倍。保留原生 PE/RO 显示方向，不标注未经核验的解剖左右。</p><main id="cases"></main>
<footer>采集时序 occurrence 不是已核实的 b0 / 扩散方向；b 字段只表示名义协议参数。当前 sinc 编码模型尚非独立波形标定，图像锐利程度不等同于真实分辨率。<br><a href="verification.json">数值检查</a> · <a href="manifest.json">逐例来源</a> · <a href="inventory.json">旧目录扫描清单</a></footer>
<script>const data=__DATA__;
const titles=['原始 ADC · log RSS','输入 · RO FFT + RSS','旧 MATLAB / 同流程复算','奇偶校正 + Tikhonov','PhaseMap + 加窗 InvA'];
const root=document.querySelector('#cases');
function el(tag,text,cls){let n=document.createElement(tag);if(text)n.textContent=text;if(cls)n.className=cls;return n}
data.forEach((r,i)=>{let card=el('article');card.dataset.view=r.view;card.append(el('h2',`${String(i+1).padStart(2,'0')} · ${r.scan} · ${r.view} · 切片 ${r.slice} / 时序 ${r.occurrence}`));card.append(el('p',`${r.shape.join(' × ')}；FOV ${r.fov.map(x=>x.toFixed(1)).join(' × ')} mm；R=${r.r}；协议 b 字段=${r.protocol_b}。${r.reason}`,'tag'));let grid=el('div',null,'grid');r.images.forEach((im,j)=>{let p=el('div',null,'panel');p.append(el('p',j===2?(r.mat?'旧 MATLAB Img · 已配对':r.unmatched?'旧流程复算 · 旧 MAT 未配对':'旧流程复算 · 无同名 MAT'):titles[j]));let box=el('div',null,'canvasbox');let canvas=el('canvas');canvas.width=im.shape[1];canvas.height=im.shape[0];if(j===0)canvas.style.height='100%';box.append(canvas);p.append(box);p.append(el('p',j===0?`${r.raw_shape.join(' × ')} · coil/PE/ADC`:`原生 ${r.shape.join(' × ')}`,'note'));grid.append(p)});card.append(grid);let links=el('p',null,'note');[['逐例 PNG',`cases/${r.id}/comparison.png`],['复数 ADC 与重建 NPZ',`cases/${r.id}/arrays.npz`],['来源与参数',`cases/${r.id}/metadata.json`]].forEach(([text,url])=>{let a=el('a',text);a.href=url;links.append(a,document.createTextNode('　'))});card.append(links);let details=el('details');details.append(el('summary','原始文件及旧结果配对'));details.append(el('p',r.source,'source'));details.append(el('p',r.mat?`Img frame=${r.mat.frame_zero_based}；按旧流程复算相关性=${r.mat.recipe_correlation.toFixed(6)}；最佳匹配时序=${r.mat.best_matching_occurrence}`:r.unmatched?'同名 MAT 与预期观测未通过配对，本列为同次原始观测按旧脚本流程新计算；未配对候选保存在 NPZ 中。':'未找到同名 MAT，本列为从同次原始观测按旧脚本流程新计算。'));card.append(details);root.append(card)});
function draw(){const mode=document.querySelector('#mode').value,factor=Number(document.querySelector('#window').value)/100,view=document.querySelector('#view').value;document.querySelector('#level').textContent=Math.round(factor*100)+'%';[...root.children].forEach((card,i)=>{const r=data[i];card.hidden=view!=='全部'&&r.view!==view;card.querySelectorAll('canvas').forEach((canvas,j)=>{const im=r.images[j],limit=(j===0||mode==='own'?im.p995:r.window)*(j===0?1:factor),ctx=canvas.getContext('2d'),pixels=ctx.createImageData(canvas.width,canvas.height);im.values.forEach((value,k)=>{const gray=Math.round(255*Math.min(1,Math.max(0,value/limit)));pixels.data.set([gray,gray,gray,255],k*4)});ctx.putImageData(pixels,0,0)})})}
['view','mode','window'].forEach(id=>document.getElementById(id).addEventListener('input',draw));document.querySelector('#reset').onclick=()=>{document.querySelector('#view').value='全部';document.querySelector('#mode').value='own';document.querySelector('#window').value=100;draw()};draw();</script></html>'''
    # Embed data so the viewer also works when opened directly through file://.
    encoded = json.dumps(panels, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    (out / 'index.html').write_text(template.replace('__DATA__', encoded))


def write_report(out, records, checks):
    lines = ['# xSPEN 人脑 10 例原始数据与传统重建', '',
        '选择了 8 组已确认人脑扫描中的 10 个观测，覆盖三种采集平面、约 3/4 mm 网格、R=46/48/60 和两组同层不同时序。样本先看输入选定，没有按重建效果筛选。', '',
        '[交互查看](index.html) · [总览 PNG](overview.png) · [前 5 例](overview_01_05.png) · [后 5 例](overview_06_10.png)', '',
        '## 来源与选择', '',
        '原始数据位于 `/home/data2/chk/workspace/2026/08/14/xSPEN_项目/08_Eddy_Human_xSPEN_DTI/`。本轮读取原始 .dat、旧 HDF5 和同名 .mat，未修改旧文件。完整路径、原始 SHA256、切片计数及采集时序保存在 manifest.json 和逐例 metadata.json。', '',
        '旧清单含 40 个 Siemens 扫描条目，其中有其他 SPEN 家族与体模。已确认人脑的 8 组全部覆盖；MID78、MID530/533/535/537、MID74/76 为体模，MID80 为尚未确认的人脑协议同目录扫描，均未混入。其余未审查数据不据此判定为非人脑。', '',
        '|例|扫描|平面|切片 / 时序（0 基）|PE×RO|选择理由|', '|---|---|---|---|---|---|']
    for i, r in enumerate(records):
        m, k = r['metadata']['shape_rep_slice_coil_pe_ro'][-2:]
        lines.append(f"|{i+1}|{r['scan']}|{VIEWS[r['view']]}|{r['slice']} / {r['occurrence']}|{m}×{k}|{r['selection_reason']}|")
    lines += ['', '## 每列是什么', '',
        '|列|内容|', '|---|---|',
        '|原始 ADC|从 .dat 按原始 slice/line 和时间出现顺序提取的 32 通道复数采样，保留 RO oversampling 与采样极性。图中显示 log RSS，NPZ 保留全部复数值。|',
        '|RO FFT + RSS|反射方向校正、ramp regrid、去 RO oversampling 后，仅做 RO FFT 和 RSS；尚未逆解 PE 编码。|',
        '|旧 MATLAB Img / 同流程复算|原作者脚本的 RO 高斯窗 exp(-0.001*((1:Nro)-Nro/2)^2)、RO FFT、RSS。5 例旧 Img 配对通过；MID613/MID615 的 2 例同名 Img 未通过配对，MID106 和 MID108 的 3 例缺少同名 MAT，后 5 例明确展示同流程新计算结果。旧 Img 除以 sqrt(Nro) 只统一 FFT 系数，未校正旧 reader 的额外幅度尺度。|',
        '|奇偶校正 + Tikhonov|邻行相位差拟合校正后，在原生 sinc 编码矩阵上逐线圈求 (AᴴA+0.01I)⁻¹Aᴴ，再 RSS。A 已按谱范数归一化。|',
        '|PhaseMap + 加窗 InvA|奇偶子集分别加窗重建，拟合相位图，修正原始偶数 RO 图像行，再用高斯加窗伴随和 RSS。width=0.8；这里 InvA 是历史命名，不是严格矩阵逆。|', '',
        '后两列是当前项目已有传统方法在 xSPEN sinc 模型上的实现；不能称作原采集实验室已经独立标定的 PE 逆重建。两列相位估计也不同，差异同时包括相位与反演选择。没有使用学习先验、超分插值或干净 GT。', '',
        '## 图像观察', '',
        'RO FFT + RSS 输入本身已能看见脑部轮廓。1、4 例的 PhaseMap + 加窗 InvA 仍有明显横向条带；较噪的 2、5、10 例在传统逆编码后仍有明显噪声纹理，不能概括为所有样本都变清晰。6 例保留了输入中的条带问题，适合作为后续相位/算子排查样本。', '',
        'MID613/MID615 的同名旧 Img 尚未通过本轮同帧配对；可见显示方向或处理流程差异，具体来源尚未核清。未配对候选保存在 arrays.npz 的 unmatched_mat_candidate_rss 中，不与配对成功的旧结果混用。', '',
        '## 显示与验证', '',
        '所有图像保留原生 PE/RO 网格和原始显示方向。默认各图独立使用 p99.5 窗，仅便于比较结构，不能比较方法间幅度；ADC 单独用 log 灰度。逐例 display.json 记录显示参数，网页可切换共同窗。已配对的旧 Img 在除以 sqrt(Nro) 后，仍约为当前 reader 同流程结果的 2 倍，且存在小的处理差异；没有按图拟合增益或修改存储数组。overview_shared_window.png 保留共用窗的幅度检查图。', '',
        f"- {checks['cases']} 例均从原始 .dat 重新提取，fresh reader 与旧 HDF5 相同帧核对通过；8 个源 .dat 的完整 SHA256 均重新核对。",
        '- 旧 Img 按原始切片计数映射到相同帧，并与全部候选 occurrence 逐一比较；接受的匹配要求预期 occurrence 相关性最高且大于 0.98。',
        '- 复数数组有限、原生尺寸、ADC 行覆盖、RSS 计算、Tikhonov 正规方程和相位校正幅度保持检查均通过，详见 verification.json。', '',
        'occurrence 仅是同一原始 slice/line 按时间出现的编号，不能直接叫 b0、特定扩散方向或同条件重复。文件中 b=600/1000 是名义协议字段。8 组扫描不等于 8 位独立受试者。', '',
        '当前模型使用头部参数与简化 sinc 核，尚无独立完整波形/B0 标定。网格尺寸不是实测有效分辨率，真实数据没有配对 clean GT，因此不报告 PSNR/SSIM 或按图像锐利度排方法优劣。', '',
        '## 产物', '',
        '- `cases/<ID>/arrays.npz`：原始 ADC、预处理复数输入、RO 复图像、两种传统重建的复数线圈图像及 RSS、相位场、旧 Img/同流程复算、编码矩阵。',
        '- `cases/<ID>/metadata.json`：源文件、原始计数、物理参数、算法参数和旧 Img 对齐信息。',
        '- `inventory.json`、`selection.json`、`manifest.json`、`provenance.json`、`verification.json`：候选清单、选例依据、逐例记录、代码与环境来源、数值检查。', '']
    (out / 'report.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    render(parser.parse_args().run.resolve())
