"""Summarize the four fixed real inverse diagnostics without choosing a winner."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


METHODS = [('phase_inva','PhaseMap + InvA'), ('baseline_replay','Diffusion\n8 inner steps'),
           ('inva_init','Diffusion\nInvA initialization'), ('inner32','Diffusion\n32 inner steps'),
           ('lambda1','Diffusion\nλ = 1'), ('sigma3','Diffusion\nstart σ = 3'),
           ('data_only_vae480','VAE data fit\nno DiT')]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--rois',type=Path,required=True)
    args=p.parse_args()
    root=args.root.resolve()
    rois=json.loads(args.rois.read_text())['rois']
    cases,metrics,audits,rows=[],{},[],[]
    for index in [0,1,5,8]:
        folder=root/f'case_{index:02d}'
        assert json.loads((folder/'completed.json').read_text())['completed']
        with np.load(folder/'arrays.npz') as archive:
            arrays={key:archive[key] for key in archive.files}
        records=json.loads((folder/'metrics.json').read_text())
        probe=root/'initialization_probe'/f'case_{index:02d}'
        with np.load(probe/'arrays.npz') as archive:
            arrays.update({key:archive[key] for key in archive.files})
        records+=json.loads((probe/'metrics.json').read_text())
        lookup={r['method']:r for r in records}
        baseline=((arrays['baseline_replay']+1)/2).clip(0,1)
        for record in records:
            image=((arrays[record['method']]+1)/2).clip(0,1)
            record['display_rmse_to_current_baseline']=float(np.sqrt(np.mean((image-baseline)**2)))
            for field,key in [('display_rmse_to_original_diffusion','original_diffusion'),('display_rmse_to_inva','phase_inva')]:
                record[field]=float(np.sqrt(np.mean((image-((arrays[key]+1)/2).clip(0,1))**2)))
            rows.append(record)
        metrics[str(index)]=lookup
        cases.append((index,{k:np.rot90(((arrays[k]+1)/2).clip(0,1),2) for k,_ in METHODS}))
        audits.append(json.loads((folder/'operator_checks.json').read_text()))
    fields=['index','key','method','measurement_nrmse','display_rmse_to_current_baseline',
            'display_rmse_to_original_diffusion','display_rmse_to_inva','seconds']
    with (root/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)

    width,height=19.8,8.8
    fig=plt.figure(figsize=(width,height),facecolor='white')
    fig.text(.085,.970,'Real acquisitions · Inverse-solver diagnostics',fontsize=16,weight='bold')
    fig.text(.085,.935,'Fixed ROI and grayscale [0, 1] · Values: full-image measurement NRMSE (lower = closer fit, not higher resolution)',fontsize=10,color='#52606D')
    x0,context_w,roi_w,gap=.085,.075,.104,.010
    method_x=x0+context_w+.035
    for col,(_,label) in enumerate(METHODS):
        fig.text(method_x+col*(roi_w+gap)+roi_w/2,.875,label,ha='center',va='center',fontsize=10)
    fig.text(x0+context_w/2,.875,'ROI location\non InvA',ha='center',va='center',fontsize=10)
    for row,(index,images) in enumerate(cases):
        roi=rois[index];x,y,w,h=(roi[k] for k in ['x','y','width','height'])
        bottom=.685-row*.200
        fig.text(.077,bottom+.069,f"FOV {roi['fov_mm']} mm\n{roi['label']}",ha='right',va='center',fontsize=10)
        context_h=context_w*width/height
        ax=fig.add_axes([x0,bottom-.010,context_w,context_h])
        ax.imshow(images['phase_inva'],cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.set_axis_off()
        ax.add_patch(Rectangle((x-.5,y-.5),w,h,fill=False,ec='#20B9C5',lw=1))
        roi_h=roi_w*width/height*h/w
        for col,(method,_) in enumerate(METHODS):
            ax=fig.add_axes([method_x+col*(roi_w+gap),bottom,roi_w,roi_h])
            crop=images[method][y:y+h,x:x+w]
            ax.imshow(crop,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color('#20B9C5');spine.set_linewidth(.8)
            fig.text(method_x+col*(roi_w+gap)+roi_w/2,bottom-.023,
                     f"{metrics[str(index)][method]['measurement_nrmse']:.4f}",ha='center',va='center',fontsize=10)
    fig.savefig(root/'solver_comparison.png',dpi=300,bbox_inches='tight',pad_inches=.15)
    fig.savefig(root/'solver_comparison.pdf',dpi=300,bbox_inches='tight',pad_inches=.15)
    plt.close(fig)

    max_adjoint=max(max(a['adjoint_relative_errors']) for a in audits)
    max_gradient=max(a['image_gradient_relative_error'] for a in audits)
    best_fd=[min(v['relative_error'] for v in a['decoder_directional_derivatives'] if v['direction']==d)
             for a in audits for d in ['gradient','random']]
    summary=dict(cases=[i for i,_ in cases],metrics=metrics,
                 max_adjoint_relative_error=max_adjoint,max_image_gradient_relative_error=max_gradient,
                 max_best_decoder_fd_error=max(best_fd),operator_audits=audits,
                 no_ground_truth=True,no_parameter_selection=True)
    (root/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n')
    lines=['# 实采反演检查 · 260916','',
           '固定病例：FOV16 #5、#13；FOV24 #3、#15。权重、原始观测来源、线圈/相位和重建种子固定；各对照改变的初始化、求解参数或强度尺度分别注明。没有按结果挑病例或选正式参数。',
           '', '## 数值实现', '',
           f'- 实际实采算子的伴随检查，最大相对点积误差：{max_adjoint:.3g}。',
           f'- 自动微分与解析图像梯度最大相对误差：{max_gradient:.3g}。',
           f'- 实际VAE解码器链式梯度，以3种步长做方向有限差分；每方向取最小误差后，最大为{max(best_fd):.3g}。',
           '- 这些检查支持内部数学实现一致，不等同于前向模型与真实扫描完全一致。', '',
           '## 测量 NRMSE', '',
           '| 病例 | InvA | 原方案重跑 | InvA初值 | 内迭代32 | λ=1 | 起始σ=3 | 仅VAE数据拟合 |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for i,_ in cases:
        r=rois[i];m=metrics[str(i)]
        lines.append(f"| FOV{r['fov_mm']} {r['label']} | "+' | '.join(f"{m[k]['measurement_nrmse']:.4f}" for k in
                     ['phase_inva','baseline_replay','inva_init','inner32','lambda1','sigma3','data_only_vae480'])+' |')
    lines += ['', '![局部对照](solver_comparison.png)', '',
              '## 如何解读', '',
              '- 原方案为60外层步、8内层步、λ=0.1、起始σ=1；内迭代、λ、起始σ三个diffusion变体各只改一个参数。另加的InvA初值对照只更换初始图像。',
              '- “仅VAE数据拟合”不调用DiT、不额外注入latent噪声；从相同初值的VAE编码出发，用480次梯度更新拟合观测，约束仍包括固定VAE的可表示图像。它是数据拟合参考，不能当作严格等预算、只移除一个因素的消融。',
              '- 所有数据项在未截断图像上计算；图像显示统一clip至[0,1]并旋转180度，与原图相同。',
              '- 各方法对当前基线的图像RMSE只衡量变化大小，不是对GT的准确率。',
              '- 相同参数/种子的重跑与历史结果有微小数值差异，原因未单独定位；图中用同次诊断的基线作对照。不能将很小的纹理变化当作稳定收益。',
              '- 此处没有高分辨率GT、模体或MTF测量，测量残差降低不证明空间分辨率提高。',
              '- 当前实现是潜空间近端与EDM式更新的自定义组合，不是已验证的精确后验采样器。参考框架：[DiffPIR](https://arxiv.org/abs/2305.08995)、[潜空间逆问题](https://arxiv.org/abs/2307.00619)。这些文献不验证本项目的具体适配。',
              '', '[逐例指标](metrics.csv) · [结构化诊断](summary.json)', '']
    (root/'反演诊断_260916.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(max_adjoint=max_adjoint,max_image_gradient=max_gradient,max_decoder_fd=max(best_fd),out=str(root)),indent=2))


if __name__=='__main__':main()
