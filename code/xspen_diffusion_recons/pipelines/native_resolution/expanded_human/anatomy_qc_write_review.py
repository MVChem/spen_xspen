"""Retained analysis recipe; execute explicitly after supplying its input artifacts."""

def main():
    """Serialize independent manual visual judgments after actual contact-sheet inspection."""
    from pathlib import Path
    import json,datetime
    import numpy as np
    B=Path(__file__).resolve().parent
    G=json.loads((B/'anatomy_qc_geometry.json').read_text())
    for g in G['scans']:
     m=json.loads((B/'scanner'/f"{g['scan']}.json").read_text())
     g['export_sort_matches']=m['slice_order']==g['position_sorted_original_slice_order']
     pos={r['original_slice_counter']:r['positions_lps_mm'][0] for r in g['per_slice_mdh']}
     g['export_positions_match']=bool(np.allclose(m['positions_lps_mm'],[pos[s] for s in m['slice_order']],atol=1e-6))
    (B/'anatomy_qc_geometry.json').write_text(json.dumps(G,indent=2)+'\n')
    geom={r['scan']:r for r in G['scans']}
    H={
    'MID51':('sagittal',[8,38],'可见连续皮层、脑实质、近中线脑干/小脑样结构与外侧小截面；不是均匀容器。','0–2及43–47主要为空气；3–7与39–42为少量周边组织。中线与外侧的外观变化连续，MDH几何没有换方向。'),
    'MID613':('axial',[13,42],'可见双侧脑实质、后颅窝、脑沟和脑室样结构，层间从颅底过渡至颅顶。','0–12为颅底/颈部及低脑组织占比，43–47为颅顶端部；脑实质内可见横向条带/信号不均。平面内旋转为90度，与MID106的显示朝向不同。'),
    'MID615':('coronal',[8,42],'可见非均匀脑实质、皮层和连续的解剖结构，并非容器。','0–7与43–47脑组织截面较小；存在条带、外围亮边和部分边缘模糊。冠状方向由所有header normals与MDH quaternion共同确认，不由肉眼或文件名猜测。'),
    'MID106':('axial',[12,41],'可见小脑、双侧大脑、脑室及皮层，解剖连续。','0–11为颅底/低脑组织占比，42–47为颅顶小截面；occurrence间对比及噪声水平不同。'),
    'MID108':('sagittal',[8,38],'可见皮层、侧脑实质与近中线后颅窝结构，左右外侧到中央连续。','0–3及44–47主要为空气；4–7与39–43多为周边小截面。中线处存在明显信号不均/条纹。')}
    P={
    'MID530':('coronal','跨层平滑变径的均匀椭球体，内部仅平滑亮度/条带变化，未见脑沟、脑室或脑实质分区。'),
    'MID533':('axial','均匀圆/椭圆容器，跨层变径且一侧有小容器突出；没有脑解剖。'),
    'MID535':('oblique','均匀容器的斜切截面，多个层出现直边/缺口；没有脑组织结构。'),
    'MID537':('oblique','均匀容器斜切截面，端部小亮点及直边与容器形态一致；没有脑解剖。'),
    'MID74':('axial','中心层有平顶/缺口的均匀圆形容器，内部为平滑阴影；不是人脑。'),
    'MID76':('sagittal','平滑均匀容器，端部有分离小突出，跨层呈规则变径；未见脑实质、脑沟或脑室。')}
    records=[]
    for mid in ['MID51','MID613','MID615','MID106','MID108','MID530','MID533','MID535','MID537','MID74','MID76']:
     m=json.loads((B/'scanner'/f'{mid}.json').read_text());q=json.loads((B/'qc'/f'{mid}.json').read_text());g=geom[mid];human=mid in H
     r=dict(scan=mid,classification='human_brain' if human else 'phantom',confidence='high',include_in_human_comparison=human,review_basis='Actual manual visual inspection of RO FFT + coil RSS, before any PE inverse or learned reconstruction.',input_preview=str(B/'qc'/f'{mid}_raw.png'),preview_slices_viewed=q['preview_slices'],preview_occurrences_viewed=q['preview_repeats'],shape_rep_slice_coil_pe_ro=m['shape_rep_slice_coil_pe_ro'],source=m['source'],source_sha256_from_export=m['source_sha256'],hash_independently_recomputed_in_this_qc=False,normal_lps=g['header_normals_lps'][0],all_header_and_mdh_geometry_consistent=g['all_geometry_constant_except_slice_position'],header_inplane_rotation_rad=g['header_inplane_rotation_rad_unique'][0],geometry_sort_matches_export=g['export_sort_matches'],fov_mm_pe_ro=m['fov_mm'],thickness_mm=m['thickness_mm'],nominal_pixel_mm_pe_ro=[f/n for f,n in zip(m['fov_mm'],m['shape_rep_slice_coil_pe_ro'][-2:])],r_value=m['r_value'],excluded_original_slice_counters=m['excluded_original_slice_counters'])
     if human:
      orient,valid,evidence,issues=H[mid];reps=[0,m['shape_rep_slice_coil_pe_ro'][0]//2,m['shape_rep_slice_coil_pe_ro'][0]-1]
      r.update(orientation_from_header_and_mdh=orient,visual_evidence_zh=evidence,issues_zh=issues,all_slices_viewed_at_occurrence=reps[1],all_slices_contact_sheet=str(B/f'anatomy_qc_{mid}_all_slices.png'),conservative_brain_slice_range_inclusive=valid,range_scope='All 48 slices manually reviewed at middle occurrence only; conservative brain-containing range, not a formal brain mask or all-occurrence quality guarantee.',recommended_slices_zero_based=[17,23,29,35],recommended_occurrences_zero_based=reps,selected_cases_viewed=True,selected_contact_sheet=str(B/f'anatomy_qc_{mid}_selected.png'),selected_case_count=12)
     else:
      r.update(orientation_from_header_and_mdh=P[mid][0],visual_evidence_zh=P[mid][1],issues_zh='对象为体模，保留作独立物理/噪声测试；本轮人脑比较排除。',conservative_brain_slice_range_inclusive=None,recommended_slices_zero_based=[],recommended_occurrences_zero_based=[],selected_case_count=0)
     records.append(r)
    result=dict(schema_version=1,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reviewer='independent data_audit agent',complete=True,count_definition='Local scan files / observations, not independent human subjects.',counts=dict(reviewed_scan_files=11,human_brain_scan_files=5,phantom_scan_files=6,unconfirmed_scan_files=0,recommended_comparison_cases=60,human_slice_occurrence_observations=sum(r['shape_rep_slice_coil_pe_ro'][0]*r['shape_rep_slice_coil_pe_ro'][1] for r in records if r['include_in_human_comparison'])),selection_rule='Accept only clear human brain anatomy in raw RO FFT + RSS. For all accepted scans, fixed geometry-order slices [17,23,29,35] and occurrences [0,nocc//2,nocc-1]; chosen and inspected without learned/PE-inverse images or quantitative method scores.',limitations=['The contact sheets use per-frame 99.5th percentile display scaling; brightness is not quantitatively comparable across occurrences.', 'RO FFT + RSS is a raw-input anatomy preview; no PE inverse or diffusion reconstruction was run by this reviewer.', 'Chronological occurrence is not a verified b0/direction label. The first/middle/last selections must keep occurrence labels.', 'FOV/matrix yields nominal sampling spacing, not measured effective resolution; thickness remains 4 mm.', 'Scan geometry is constant within all 11 files; this validates current position sorting for these scans only. It does not establish scanner calibration, diagnostic image quality or all-raw waveform correctness.', 'MID613 and MID106 are both axial with different in-plane rotation. Retain PE/RO coordinates in the model and record orientation; do not rotate raw arrays to match montage appearance without also transforming operator geometry.', 'This QC has not viewed any DiffPIR, prior model or traditional PE-inverse result and cannot rank their performance.'],geometry_evidence=str(B/'anatomy_qc_geometry.json'),records=records)
    (B/'anatomy_qc_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    md=['# 扩展 xSPEN 扫描的独立解剖 QC','', '11 份新导出已逐图目视核查：**5 份明确人脑，6 份体模，0 份未确认**。本轮人脑比较纳入 MID51、MID613、MID615、MID106、MID108；排除 MID530、MID533、MID535、MID537、MID74、MID76。这里计扫描文件，不计独立受试者。','', '判断仅依据 raw 的 RO FFT + coil RSS；未看深度模型或 PE 反演结果。5 份人脑另逐层查看中间 occurrence 的全部 48 层，并查看统一固定 4 层×3 occurrence。','', '| MID | 对象 | 几何方向 | 建议脑组织层范围（含两端，0起） | 人脑比较 |','|---|---|---|---|---|']
    for r in records:md.append(f"| {r['scan']} | {'人脑' if r['include_in_human_comparison'] else '体模'} | {r['orientation_from_header_and_mdh']} | {r['conservative_brain_slice_range_inclusive'] or '无'} | {'纳入' if r['include_in_human_comparison'] else '排除'} |")
    md+=['','层范围是在中间 occurrence 全 48 层检查后给出的保守解剖范围，不是正式 brain mask，也不保证每个 occurrence 的质量。','', '固定示例为每份 **slice 17、23、29、35**；MID51 使用 occurrence **0、9、17**，另四份使用 **0、8、15**，共 **60 例**。这些选定原始输入均已实际看过。5 份导出合计 3,936 个 slice×occurrence 观测，不能称作 3,936 个独立样本/人次。','', '## 每份的直接视觉证据','']
    for r in records:
     md += [f"**{r['scan']}**：{r['visual_evidence_zh']} {r['issues_zh']}", f"[原始预览](qc/{r['scan']}_raw.png)"+(f"；[全48层](anatomy_qc_{r['scan']}_all_slices.png)；[固定12例](anatomy_qc_{r['scan']}_selected.png)" if r['include_in_human_comparison'] else ''),'']
    md+=['## 层排序与方向核查','', '独立读取 11 份 raw 的全部 `sSliceArray.asSlice` 几何及全部 image MDH 的位置/quaternion；没有读取 `mdb.data`。每份内部所有 header normal 相同、只有一种 MDH quaternion，且每个原始 slice counter 在不同 line/occurrence 的位置完全一致。所有 MDH quaternion 的法向与 header 相符（最大误差低于 5×10⁻⁸），MDH 位置与最近 header 层位置的误差低于 5×10⁻⁶ mm。重新按位置在 normal 上的投影排序，与所有 11 份导出的 `slice_order` 及 `positions_lps_mm` 相符。','', '因此，这 11 份中使用首层 normal 加每个原始 slice counter 的位置排序是充分的，没有同扫描混方向的证据。但该结论不能推广到未经核查的新协议。MID51/MID108 为 sagittal，MID613/MID106 为 axial，MID615 为 coronal；其几何确认来自全部层头与 MDH，不来自文件名。','', '**MID613 的平面内旋转确实不同。** 其 `dInPlaneRot=1.570796327`、MDH quaternion 约 `[1,0,0,0]`；MID106 为 `dInPlaneRot=0`、quaternion 约 `[0.7071,0,0,0.7071]`。这解释二者 axial 图在当前 PE/RO 数组坐标中的不同朝向，不应为了图像外观而单独旋转输入而不同时处理算子坐标。','', '[逐层完整几何证据](anatomy_qc_geometry.json) 与 [可复查只读脚本](anatomy_qc_geometry.py) 已保存。','', '## 比较时保留的限制','', '- occurrence 仅表示 chronological acquisition occurrence，尚未核实为 b0 或某个 diffusion direction。','- 各图按本帧 99.5% 分位显示，不能从显示亮度比较不同 occurrence 的绝对信号强度。','- 原始图可见噪声、条带、模糊与信号不均；这些留作后续方法比较的输入问题，不能据此声称模型已经解决。','- 人脑5份均为46×48、4 mm层厚；MID51的FOV约184.958×193 mm/R46，其余184×192 mm/R48。FOV/矩阵是名义采样间距，并未测量有效空间分辨率。','- 体模分类根据实际容器形态与无脑解剖；它们仍可独立用于物理/噪声测试，不混入人脑统计。','- 本轮未启动GPU、PE反演、DiffPIR或训练；原始输入与root导出未修改。']
    (B/'anatomy_qc_review.md').write_text('\n'.join(md)+'\n')
    print(json.dumps(result['counts'],indent=2))

if __name__ == '__main__':
    main()
