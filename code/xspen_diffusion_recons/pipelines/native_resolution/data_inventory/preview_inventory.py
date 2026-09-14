"""Retained analysis recipe; execute explicitly after supplying its input artifacts."""

def main():
    """CPU inventory preview: legacy MAT images, with explicit missing-preview cards."""
    import json
    from pathlib import Path
    import numpy as np
    from scipy.io import loadmat
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    
    OUT=Path(__file__).resolve().parent
    EXP=OUT.parent
    font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    plt.rcParams['font.family']=font.get_name()
    rows=[r for r in json.loads((EXP/'audit/siemens_header_manifest.json').read_text())
          if r['family']=='crossed_chirp_bipolarDiff']
    fig,axes=plt.subplots(4,6,figsize=(18,13))
    notes=[]
    for ax,r in zip(axes.flat,rows):
        path=Path(r['path']).with_suffix('.mat')
        note={'scan_id':r['scan_id'],'source_raw':r['path'],'mat_exists':path.exists()}
        if path.exists():
            arr=loadmat(path,variable_names=['Img']).get('Img')
            if arr is not None and arr.ndim>=3:
                arr=arr.reshape(arr.shape[0],arr.shape[1],-1,order='F')
                frame=min(int(r['header_slices'])//2,arr.shape[2]-1)
                im=np.abs(arr[:,:,frame])
                assert np.isfinite(im).all() and im.max()>0
                scale=float(np.quantile(im,.995))
                ax.imshow(im,cmap='gray',vmin=0,vmax=scale,interpolation='nearest')
                note.update(mat=str(path),image_shape=list(arr.shape),display_frame_zero_based=frame,
                            display_scale=scale,preview_kind='Original saved MAT Img; historical processing not newly verified for every scan; not new diffusion reconstruction')
            else:
                ax.text(.5,.5,'raw 存在\nMAT 无 Img 字段',ha='center',va='center',transform=ax.transAxes)
        else:
            ax.set_facecolor('#e8edf1')
            ax.text(.5,.5,'raw 存在\n尚无同名 MAT 预览',ha='center',va='center',transform=ax.transAxes,color='#475666')
        selected=bool(r['selected'])
        shape=r['native_header_shape_pe_ro']
        tag='已纳入对照' if selected else '未纳入本次对照'
        ax.set_title(f"{r['scan_id']} · {tag}\n头部 {shape[0]}×{shape[1]} · {r['view']}",fontsize=10,
                     color='#087d80' if selected else '#17232d')
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color('#087d80' if selected else '#c7ced5');spine.set_linewidth(2 if selected else .6)
        notes.append(note)
        print(r['scan_id'],path.exists(),flush=True)
    for ax in list(axes.flat)[len(rows):]:ax.axis('off')
    fig.suptitle('22 份双 chirp xSPEN 原始扫描：数据目录预览',fontsize=19,y=.98)
    fig.text(.5,.018,'图像来自已有 MATLAB Img，逐图调窗，仅用于盘点采集内容；空白卡表示缺少现成预览，不表示缺少 raw。\n'
                      '青色标注：已纳入前一轮 36 组对照的 3 份扫描。其余对象、采集计数与模型适配需逐项核对。',
             ha='center',fontsize=11)
    fig.subplots_adjust(left=.02,right=.99,top=.925,bottom=.09,wspace=.17,hspace=.29)
    fig.savefig(OUT/'all_22_raw_inventory.png',dpi=150)
    plt.close(fig)
    (OUT/'preview_manifest.json').write_text(json.dumps(notes,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    main()
