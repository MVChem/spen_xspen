"""Retained anatomy audit; execute explicitly after preparing its inputs."""

def main():
    """Read-only geometry audit of full-FOV expansion candidates; MDH headers only.
    No mdb.data access, reconstruction, GPU or original-file modifications.
    """
    from pathlib import Path
    import json,datetime
    from collections import defaultdict
    import numpy as np
    import twixtools
    from twixtools.geometry import quat_to_rotmat
    B=Path(__file__).resolve().parent
    entries=json.loads((B.parent/'audit/siemens_header_manifest.json').read_text())
    MIDS=['MID51','MID613','MID615','MID106','MID108','MID530','MID533','MID535','MID537','MID74','MID76']
    rows=[]
    for mid in MIDS:
     source=next(r['path'] for r in entries if r['scan_id']==mid and r['family']=='crossed_chirp_bipolarDiff')
     t=twixtools.read_twix(source,parse_geometry=False,parse_pmu=False,verbose=False)[0]
     ss=t['hdr']['MeasYaps']['sSliceArray']['asSlice']
     ns=np.asarray([[s.get('sNormal',{}).get(a,0) for a in ['dSag','dCor','dTra']] for s in ss],float)
     ps=np.asarray([[s.get('sPosition',{}).get(a,0) for a in ['dSag','dCor','dTra']] for s in ss],float)
     rotations=[s.get('dInPlaneRot',0) for s in ss]
     geom=np.asarray([[s['dPhaseFOV'],s['dReadoutFOV'],s['dThickness']] for s in ss])
     collected=defaultdict(lambda: {'positions':set(),'quaternions':set(),'image_mdh_rows':0})
     for m in t['mdb']:
      if not m.is_image_scan():continue
      sl=int(m.mdh.Counter.Sli);x=collected[sl];p=m.mdh.SliceData.SlicePos
      x['positions'].add(tuple(float(v) for v in [p.Sag,p.Cor,p.Tra]))
      x['quaternions'].add(tuple(float(v) for v in m.mdh.SliceData.Quaternion));x['image_mdh_rows']+=1
     records=[]
     for sl,x in sorted(collected.items()):
      p=np.asarray(sorted(x['positions']));q=np.asarray(sorted(x['quaternions']))
      nearest=int(np.argmin(np.linalg.norm(ps-p[0],axis=1)))
      qnormals=np.asarray([quat_to_rotmat(*qq)[:,2] for qq in q])
      records.append(dict(original_slice_counter=sl,image_mdh_rows=x['image_mdh_rows'],positions_lps_mm=p.tolist(),quaternions_scalar_first=q.tolist(),matching_header_slice=nearest,max_position_variation_mm=float(np.max(np.linalg.norm(p-p[0],axis=1))),position_match_error_mm=float(np.linalg.norm(ps[nearest]-p[0])),normal_from_quaternion=qnormals.tolist(),max_quaternion_normal_error=float(np.max(np.linalg.norm(qnormals-ns[nearest],axis=1)))))
     order=sorted(collected,key=lambda s:float(np.dot(ns[0],records[s]['positions_lps_mm'][0])))
     sortedpos=np.asarray([records[s]['positions_lps_mm'][0] for s in order]);proj=sortedpos@ns[0]
     meta_path=B/'scanner'/f'{mid}.json';meta=json.loads(meta_path.read_text()) if meta_path.exists() else None
     normal_unique=np.unique(ns.round(7),axis=0).tolist()
     q_unique=sorted(set(q for x in collected.values() for q in x['quaternions']))
     consistent=len(normal_unique)==1 and len(q_unique)==1 and all(len(x['positions'])==1 for x in collected.values())
     row=dict(scan=mid,source=source,header_slice_count=len(ss),header_normals_lps=normal_unique,header_normal_identical_all_slices=len(normal_unique)==1,header_inplane_rotation_rad_unique=sorted(set(rotations)),header_fov_pe_ro_thickness_unique=np.unique(geom.round(7),axis=0).tolist(),mdh_original_slice_count=len(collected),image_mdh_rows=sum(x['image_mdh_rows'] for x in collected.values()),quaternions_scalar_first_unique=q_unique,per_slice_mdh=records,position_sorted_original_slice_order=order,signed_adjacent_slice_spacing_mm=np.diff(proj).tolist(),all_geometry_constant_except_slice_position=consistent,max_normal_error=max(x['max_quaternion_normal_error'] for x in records),max_mdh_to_header_position_error_mm=max(x['position_match_error_mm'] for x in records),export_sort_matches=meta['slice_order']==order if meta else None,export_positions_match=bool(np.allclose(meta['positions_lps_mm'],sortedpos,atol=1e-6)) if meta else None,sort_conclusion='First normal plus per-counter position sorting is sufficient for this scan: all header normals and all image MDH quaternions are identical, no within-counter position variation.' if consistent else 'Geometry varies: inspect individual slice records; cannot assume one normal for all slices.')
     rows.append(row)
     print(mid, 'normal',normal_unique,'q',len(q_unique),'sort',row['export_sort_matches'],'normalerror',row['max_normal_error'],flush=True)
    (B/'anatomy_qc_geometry.json').write_text(json.dumps(dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All sSliceArray slices and all image MDH position/quaternion headers; no mdb.data reads, no full-file hashing or reconstruction.',quaternion_convention='twixtools.geometry.quat_to_rotmat, scalar first; rotation column2 is slice normal. In-plane PE/RO columns require sign adjustment per prs2sct_mdb.',scans=rows),indent=2)+'\n')

if __name__ == '__main__':
    main()
