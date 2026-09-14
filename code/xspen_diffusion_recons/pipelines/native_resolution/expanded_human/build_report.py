"""Summarize completed additional human reconstructions, preserving stored pixels."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
EXP=HERE.parent
PRIOR=EXP/'real_comparison'
sys.path.insert(0,str(PRIOR))
spec=importlib.util.spec_from_file_location('previous_comparison_builder',PRIOR/'build_comparison.py')
draw=importlib.util.module_from_spec(spec)
spec.loader.exec_module(draw)


def main():
    summary=json.loads((HERE/'evaluation/summary.json').read_text())
    assert summary['status']=='complete' and not summary['smoke'] and summary['case_count']==60
    images,records={},{}
    order=['MID51','MID613','MID615','MID106','MID108']
    views={'MID51':'矢状位 · 4.02 mm','MID613':'轴位 · 4 mm','MID615':'冠状位 · 4 mm',
           'MID106':'轴位 · 4 mm','MID108':'矢状位 · 4 mm'}
    draw.VIEWS.update(views)
    draw.TITLES['diffusion_native']='原生 EDM EMA + DiffPIR\n46×48 · 人脑先验'
    mapping={'input_ro_rss':'degraded','complex_tikhonov':'native_tikhonov','magnitude_l2':'tikhonov',
             'phasemap_inva':'phasemap_inva','diffusion128':'baseline128','diffusion_native':'diffusion'}
    names=[]
    for mid in order:
        name=f'{mid}_rep00_slice023'
        path=HERE/'evaluation'/mid/f'{name}.npz'
        record=json.loads(path.with_suffix('.json').read_text())
        assert record['steps']==60 and not record['smoke'] and record['provenance']['checkpoint_step']==20000
        with np.load(path) as arrays:
            images[name]={key:(arrays[source][0,0]+1)/2 for key,source in mapping.items()}
        record['case']=name
        records[name]=record
        names.append(name)
    title='新增真实人脑 xSPEN：原生 EDM EMA + DiffPIR'
    draw.save_comparison(images,records,names,draw.MAIN,HERE/'overview.png',title)
    draw.save_comparison(images,records,names[:3],draw.MAIN,HERE/'overview_first3.png',title)
    draw.save_comparison(images,records,names[3:],draw.MAIN,HERE/'overview_other2.png',title)
    raw_observations=0
    per_scan=[]
    for scan in summary['scans']:
        meta=json.loads((HERE/'scanner'/f"{scan['scan']}.json").read_text())
        nrep,ns,c,m,k=meta['shape_rep_slice_coil_pe_ro']
        raw_observations+=nrep*ns
        per_scan.append(dict(scan=scan['scan'],retained_observations=nrep*ns,selected_cases=scan['case_count'],
                             shape=meta['shape_rep_slice_coil_pe_ro'],fov_mm=meta['fov_mm'],
                             r_value=meta['r_value'],view=views[scan['scan']],pages=scan['pages']))
    record=dict(status='complete',additional_human_scans=5,additional_comparison_cases=60,
                new_retained_raw_observations=raw_observations,total_human_scans_with_comparisons=8,
                total_comparison_cases=96,total_retained_raw_observations=2876+raw_observations,
                native_shape=[46,48],model='IXI human native-grid EDM EMA',checkpoint_step=20000,
                reconstruction='DiffPIR',steps=60,trained_new_weights_this_extension=False,
                no_clean_ground_truth=True,per_scan=per_scan,
                acquisition_note='Slice x occurrence counts are not independent subjects; 6812 available observations does not mean 6812 diffusion reconstructions.',
                coverage_note='5 confirmed human scans from 11 inspected full-FOV candidates; the other 6 are phantoms. Six small-FOV scans and previous phantom/companion exclusions MID78/MID80 remain separate; MID80 was not independently reclassified here.',
                geometry_note='Raw per-scan R is used (R46 for51; R48 for613/615/106/108). FOV differs up to0.52% from prior training. Coronal is outside trained axial/sagittal view set. MID613 has 90-degree in-plane rotation; original acquisition orientation retained.')
    (HERE/'results_summary.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    lines=[
        '# 新增 xSPEN 人脑：原生 EDM EMA + DiffPIR', '',
        '**新增 5 份确认人脑扫描、60 组正式对照已经完成。** 结合上一轮，共有 8 份人脑扫描、96 组方法对照。'
        '本轮在训练完成的 46×48 人脑原生 EMA 上推理，使用 60 步 DiffPIR，没有重新训练模型。', '',
        '![新增扫描总览](overview.png)', '',
        '[离线交互图册](gallery.html) · [前三组大图](overview_first3.png) · [其余两组大图](overview_other2.png) · '
        '[上一轮真实对照](../real_comparison/真实数据对照.md)', '',
        '| 扫描 | 方向 | 原生矩阵 | FOV，mm | 编码 R | 保留的原始观测 | 本次对照 |',
        '|---|---|---|---|---:|---:|---:|',
    ]
    for scan in per_scan:
        lines.append(f"| {scan['scan']} | {scan['view']} | 46×48 | {scan['fov_mm'][0]:.3f}×{scan['fov_mm'][1]:.3f} | {scan['r_value']:g} | {scan['retained_observations']} | {scan['selected_cases']} |")
    lines.extend(['',
        f'新增保留观测共 {raw_observations:,} 组，加上之前 2,876 组，共 {2876+raw_observations:,} 组可读取人脑观测。'
        '这里只对 60 组新增观测完成各方法对照；不将原始可读总量当成已完成重建数或独立人数。', '',
        '每个扫描在重建前固定选层 17、23、29、35，取第 0、中间、最后一次 occurrence，共 12 组；'
        '依据输入脑部覆盖选层，没有根据 Diffusion 输出或指标挑选。occurrence 标签未独立核实为扩散方向或 b0。', '',
        '## 与 09/11 小鼠模型的对应', '',
        '同一 U-Net/注意力结构、EDM 预条件及加权去噪训练目标，推理使用 EMA。'
        '人脑权重来自 IXI T2/PD 的独立人脑训练，再于对应物理网格微调 20,000 步。'
        'EDM 是图像先验；真实 xSPEN 编码和复数多线圈数据一致性由 DiffPIR 接入。'
        '本轮固定 60 步、sigma_noise=0.02、sigma_min=0.02、sigma_max=80、lambda=1、xi=0。', '',
        '原生网络对 46×48 输入仅在内部补边到 48×48，输出裁回 46×48；原生方法没有 resize。'
        '对照图的旧 128² 列明确使用双线性迁移。'
        '[模型与周报逐项核对](../data_inventory/EDM_0911_alignment.md)。', '',
        '## 数据与几何核验', '',
        '本轮检查了 11 份全脑尺度 FOV 候选：5 份确认人脑，MID530/533/535/537/74/76 为体模并单列。'
        '不能因目录包含 Brain 就把所有文件计为人脑。'
        '[对象与原始图 QC](anatomy_qc_review.md)、[每层法向、MDH 位置与旋转检查](anatomy_qc_geometry.json)。', '',
        'MID51 与此前 MID27 的原生几何匹配；其余四份的 FOV 与该先验训练 FOV 最大相差约 0.52%。'
        'MID615 为冠状位，属于原 axial/sagittal 训练视图以外的应用；MID613 的平面内旋转为 90°。'
        '图册保留各自采集朝向，只保证同一行方法之间对齐，没有把所有扫描旋转成同一朝向。'
        '编码算子使用每份 raw 的实际 R46/R48、矩阵和观测；chirp R 不是欠采样加速率。', '',
        '## 显示与评价边界', '',
        '六列为原始 RO+RSS、逐线圈复数 Tikhonov、幅度 L2、PhaseMap+加窗 InvA、旧 128² 先验和原生 EDM EMA。'
        '同一例所有方法共用测量导出的强度尺度、[0,1]灰度窗、物理长宽比及 nearest 显示；不逐方法调亮。'
        'PhaseMap 为独立的 xSPEN sinc 算子适配。实采没有配对干净真值，未计算真实 PSNR/SSIM；'
        '记录的测量残差仅反映当前固定相位/线圈/简化物理模型的拟合。', '',
        '## 全部新增对照', '',
    ])
    for scan in per_scan:
        pages=' · '.join(f'[采集页 {i+1}]({Path(p).relative_to(HERE)})' for i,p in enumerate(scan['pages']))
        lines.append(f"- {scan['scan']}：{pages}")
    lines.extend(['', '[结构化结果](results_summary.json) · [冻结病例选择](selection.json) · '
                  '[推理代码及参数](RECONSTRUCTION.md) · [最终数值检查](final_verification.md) · '
                  '[GPU 空闲检查与运行记录](gpu_launch_status.json)', '',
                  '仅在连续三次空闲检查通过后使用 GPU 3；程序不发送进程终止信号、不重置 GPU。原始 raw 和旧模型/结果保持不变。'])
    (HERE/'新增人脑原生EDM结果.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(record,ensure_ascii=False))


if __name__=='__main__':main()
