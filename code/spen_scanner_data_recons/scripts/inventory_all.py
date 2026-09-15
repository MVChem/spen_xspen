#!/usr/bin/env python3
"""Inventory every imported SPEN/xSPEN scan and every declared image frame.

This is a read-only metadata inventory, not a reconstruction success report.
Frame counts are slice x diffusion/repetition volume x echo; receiver channels
and acquisition segments do not count as independent anatomical slices.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "spenpy"))
from spenpy._legacy.bruker.param import read_pv_param

DEFAULT_DATA = PROJECT.parent / "data/spen_acquired_260915"
LEGACY_MATLAB = Path("/home/data1/musong/workspace/2026/03/17/spen_matlab")
LEGACY_EXPORT = Path("/home/data1/musong/workspace/python/spen_recons/scripts/0523_find_scanner_meta_data.py")
IMAGING = {"spen_imaging", "xspen"}


def safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def values(value):
    if value is None:
        return []
    return np.asarray(value).reshape(-1).tolist()


def first(value, default=None):
    vals = values(value)
    return vals[0] if vals else default


def param(path, name):
    return safe(read_pv_param(str(path), name))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trajectory_info(scan_dir):
    try:
        traj = np.asarray(values(param(scan_dir, "PVM_EpiTrajAdjkx")), dtype=float)
        return {"present": bool(traj.size), "samples": int(traj.size),
                "all_finite": bool(np.isfinite(traj).all()),
                "nonzero_positive": bool(np.any(traj > 0)),
                "maximum": float(traj.max()) if traj.size else None}
    except (ValueError, TypeError) as exc:
        return {"present": False, "samples": 0, "nonzero_positive": False,
                "all_finite": False, "error": str(exc)}


def datalist_map(study):
    """Preserve the established explicit PV5 EPI/SPEN datalist pairing rule."""
    path = study / "datalist.txt"
    if not path.exists():
        return {}, None
    tokens = path.read_text().split()
    if any(not re.fullmatch(r"\d+", token) for token in tokens):
        return {}, {"path": str(path), "problem": "invalid_noninteger_datalist"}
    ids = list(map(int, tokens))
    if len(ids) < 3:
        return {}, {"path": str(path), "problem": "datalist_has_fewer_than_three_ids"}
    tail = ids[2:]
    flags = [param(study / str(scan), "SpenGyGaussStren") is not None for scan in tail[:3]]
    pv5 = len(flags) >= 3 and flags == [True, False, True]
    source = {"kind": "explicit_datalist", "path": str(path), "sha256": digest(path),
              "legacy_rule_source": str(LEGACY_EXPORT),
              "rule": "PV5 alternating EPI/SPEN pairs from [datalist[1], *datalist[2:]]" if pv5 else "PV360 SPEN scan IDs from datalist[2:]",
              "first_three_spen_parameter_presence": flags}
    if pv5:
        pairs = ids[1:]
        mapping = {pairs[i + 1]: {"regrid_flavor": "pv5", "trajectory_scan_id": pairs[i],
                                 "source": source} for i in range(0, len(pairs) - 1, 2)}
    else:
        mapping = {scan: {"regrid_flavor": "pv360", "trajectory_scan_id": -1,
                          "source": source} for scan in tail}
    return mapping, source


def inventory_scan(data_root, study, scan, mat, dlist):
    relative = scan["destination_relative_path"]
    path = data_root / relative
    problems, warnings = [], []
    names = ["Method", "ACQ_sw_version", "PVM_Matrix", "PVM_Fov", "PVM_FovCm",
             "PVM_EncNReceivers", "PVM_SPackArrNSlices", "PVM_ObjOrderList",
             "PVM_DwNDiffExp", "DwNDiffExp", "PVM_NRepetitions", "PVM_NEchoImages",
             "NSegments", "PVM_SliceThick", "PVM_SPackArrSliceDistance", "PVM_DwEffBval",
             "SpenGyGaussStren", "SpatEncDuration", "PVM_Phase1Offset", "PVM_EpiEchoSpacing",
             "PVM_SpatDimEnum", "PVM_EpiRampMode", "ACQ_size", "ACQ_word_size", "BYTORDA"]
    p = {}
    for name in names:
        try:
            p[name] = param(path, name)
        except (ValueError, TypeError) as exc:
            p[name] = None
            problems.append(f"parameter_parse_error:{name}:{exc}")
    matrix, fov = values(p["PVM_Matrix"]), values(p["PVM_Fov"])
    slice_values = values(p["PVM_SPackArrNSlices"])
    slices = int(sum(slice_values)) if slice_values else 1
    coils = int(first(p["PVM_EncNReceivers"], 1))
    diffusion = int(first(p["PVM_DwNDiffExp"], 0))
    diffusion_source = "PVM_DwNDiffExp"
    if diffusion < 1:
        diffusion = int(first(p["DwNDiffExp"], 1))
        diffusion_source = "DwNDiffExp" if p["DwNDiffExp"] is not None else "legacy_reader_default_1"
    diffusion = max(1, diffusion)
    repetitions = int(first(p["PVM_NRepetitions"], 1))
    echoes = int(first(p["PVM_NEchoImages"], 1))
    segments = int(first(p["NSegments"], 1))
    volumes = diffusion * repetitions
    expected_frames = slices * volumes * echoes
    defaulted = [name for name in ("PVM_EncNReceivers", "PVM_SPackArrNSlices", "PVM_NRepetitions", "PVM_NEchoImages", "NSegments") if p[name] is None]
    if defaulted:
        warnings.append("legacy_reader_default_1_for:" + ",".join(defaulted))
    if min(slices, coils, volumes, echoes, segments) < 1:
        problems.append("nonpositive_frame_or_channel_dimension")
    if len(matrix) != 2:
        problems.append("matrix_is_not_2d")
    if len(fov) < 2:
        problems.append("missing_2d_field_of_view")
    if p["PVM_SpatDimEnum"] and "3d" in str(p["PVM_SpatDimEnum"]):
        problems.append("3d_spatial_encoding_requires_separate_layout")
    if scan["raw_status"] != "nonempty":
        problems.append("raw_payload_" + scan["raw_status"])
    for name in ("method", "acqp"):
        if not (path / name).is_file():
            problems.append("missing_" + name)
    if scan["classification"] == "xspen":
        problems.append("xspen_requires_its_own_forward_model")
    else:
        for name in ("SpenGyGaussStren", "SpatEncDuration"):
            if p[name] is None:
                problems.append("missing_reconstruction_parameter:" + name)

    version = str(p["ACQ_sw_version"] or "").lower()
    title = next((line for line in (path / "method").read_text(errors="replace").splitlines() if line.startswith("##TITLE=")), "") if (path / "method").exists() else ""
    version_text = version + " " + title.lower()
    acquisition_version = "pv360" if "360" in version_text else "pv6" if re.search(r"(?:pv |paravision )6\.", version_text) else "pv5" if "pv 5." in version_text else "unknown"
    config_sources = []
    flavor, trajectory_id = None, None
    if mat:
        metadata = mat["metadata"]
        flavor, trajectory_id = metadata.get("recon_flavor"), metadata.get("trajectory_scan_id", -1)
        config_sources.append({"kind": "existing_mat_metadata", "path": mat["mat_path"], "sha256": mat["mat_sha256"],
                               "fields": ["metadata.recon_flavor", "metadata.trajectory_scan_id"]})
    elif int(scan["scan"]) in dlist:
        config = dlist[int(scan["scan"])]
        flavor, trajectory_id = config["regrid_flavor"], config["trajectory_scan_id"]
        config_sources.append(config["source"])
    elif acquisition_version in ("pv360", "pv6", "pv5"):
        flavor = "pv360" if acquisition_version == "pv360" else "pv5"
        trajectory_id = -1
        config_sources.append({"kind": "scanner_version_and_own_method", "path": relative,
                               "ACQ_sw_version": p["ACQ_sw_version"], "method_title": title,
                               "rule": "PV360 uses one_d_regridding_pv360; PV5/PV6 use one_d_regridding_pv6 (API flavor pv5); own method only",
                               "legacy_rule_source": str(LEGACY_MATLAB / ("spen/Function_Process_NewPE_SPEN_OddNumWithMask_bruker_PV6.m" if acquisition_version == "pv6" else "pv360.m" if acquisition_version == "pv360" else "pv5.m"))})
    if flavor not in ("pv360", "pv5"):
        problems.append("regrid_flavor_unresolved")
    if trajectory_id is not None:
        trajectory_id = int(trajectory_id)
    trajectory_path = path if trajectory_id in (None, -1) else path.parent / str(trajectory_id)
    trajectory = trajectory_info(trajectory_path)
    trajectory["raw_scan_path"] = str(trajectory_path.relative_to(data_root))
    trajectory["uses_own_method"] = trajectory_path == path
    if not trajectory["nonzero_positive"] or not trajectory["all_finite"]:
        problems.append("no_verified_nonzero_readout_trajectory")
    if trajectory_id not in (None, -1):
        if not (trajectory_path / "method").exists():
            problems.append("explicit_trajectory_method_missing")
        trajectory["source_method"] = param(trajectory_path, "Method")

    payloads = [{"file": item["file"], "bytes": (path / item["file"]).stat().st_size if (path / item["file"]).is_file() else None} for item in scan["raw_payloads"]]
    actual_status = "nonempty" if any((x["bytes"] or 0) > 0 for x in payloads) else "empty" if any(x["bytes"] is not None for x in payloads) else "missing"
    if actual_status != scan["raw_status"]:
        problems.append("raw_availability_changed_since_import")
    total_bytes = sum(x["bytes"] or 0 for x in payloads)
    parameters = {"method": p["Method"], "matrix_ro_pe": matrix, "fov_mm": fov,
                  "coils": coils, "slices": slices, "volumes": volumes, "echoes": echoes,
                  "n_segments": segments, "diffusion_experiments": diffusion,
                  "repetitions": repetitions, "slice_thickness_mm": p["PVM_SliceThick"],
                  "effective_b_values_s_mm2": p["PVM_DwEffBval"],
                  "acquisition_version": acquisition_version, "regrid_flavor": flavor,
                  "trajectory_scan_id": trajectory_id}
    source_sha = {name: digest(path / name) for name in ("method", "acqp") if (path / name).is_file()}
    if (trajectory_path / "method").is_file():
        source_sha["trajectory_method"] = digest(trajectory_path / "method")
    return {"id": f"{study['experiment_name']}_scan{int(scan['scan']):03d}",
            "experiment_name": study["experiment_name"], "scan_id": int(scan["scan"]),
            "raw_scan_path": relative, "classification": scan["classification"],
            "raw_status": actual_status, "import_raw_status": scan["raw_status"],
            "raw_payloads": payloads, "raw_bytes": total_bytes,
            "parameters": parameters, "source_parameters": p, "source_parameter_sha256": source_sha,
            "expected_frames": expected_frames, "expected_frames_basis": "sum(PVM_SPackArrNSlices) * diffusion_experiments * PVM_NRepetitions * PVM_NEchoImages; no coil or segment factor",
            "frame_axes": ["volume", "echo", "slice"], "diffusion_count_source": diffusion_source,
            "frame_count_kind": "declared_parameters_before_binary_decode",
            "regrid_flavor": flavor, "trajectory_scan_id": trajectory_id,
            "trajectory": trajectory, "reconstruction_config_sources": config_sources,
            "reference_mat_path": mat["mat_path"] if mat else None,
            "reference_mat_shape": mat["signal_shape"] if mat else None,
            "problems": problems, "warnings": warnings,
            "eligible_for_attempt": actual_status == "nonempty" and not problems,
            "scanner_2dseq_available": (path / "pdata/1/2dseq").is_file()}


def build_inventory(data_root):
    data_root = Path(data_root).resolve()
    raw = json.loads((data_root / "raw_manifest.json").read_text())
    mats = json.loads((data_root / "manifest.json").read_text())
    mat_by_scan = {r["raw_scan_path"]: r for r in mats["records"]}
    if len(mat_by_scan) != len(mats["records"]):
        raise ValueError("Duplicate MAT mappings require explicit resolution")
    records, studies, excluded = [], [], []
    auxiliary = Counter()
    for study in raw["experiments"]:
        if study["destination_relative_path"].split("/")[0] != "raw":
            for scan in study["scans"]:
                auxiliary[(scan["classification"], scan["raw_status"])] += 1
                excluded.append({"experiment_name": study["experiment_name"], "scan_id": int(scan["scan"]) if scan["scan"].isdigit() else scan["scan"],
                                 "raw_scan_path": scan["destination_relative_path"], "classification": scan["classification"],
                                 "raw_status": scan["raw_status"], "reason": "spectroscopy_experiment_not_2d_imaging"})
            continue
        path = data_root / study["destination_relative_path"]
        dlist, dlist_source = datalist_map(path)
        group = []
        for scan in study["scans"]:
            if scan["classification"] not in IMAGING:
                auxiliary[(scan["classification"], scan["raw_status"])] += 1
                excluded.append({"experiment_name": study["experiment_name"], "scan_id": int(scan["scan"]) if scan["scan"].isdigit() else scan["scan"],
                                 "raw_scan_path": scan["destination_relative_path"], "classification": scan["classification"],
                                 "raw_status": scan["raw_status"], "reason": "auxiliary_scan_not_spen_imaging"})
                continue
            row = inventory_scan(data_root, study, scan, mat_by_scan.get(scan["destination_relative_path"]), dlist)
            records.append(row)
            group.append(row)
        studies.append({"experiment_name": study["experiment_name"], "raw_experiment_path": study["destination_relative_path"],
                        "scan_records": len(group), "nonempty_scans": sum(r["raw_status"] == "nonempty" for r in group),
                        "expected_nonempty_frames": sum(r["expected_frames"] for r in group if r["raw_status"] == "nonempty"),
                        "datalist_source": dlist_source})
    nonempty = [r for r in records if r["raw_status"] == "nonempty"]
    summary = {"imaging_experiments": len(studies), "imaging_scan_records": len(records),
               "raw_status_counts": dict(Counter(r["raw_status"] for r in records)),
               "nonempty_classification_counts": dict(Counter(r["classification"] for r in nonempty)),
               "expected_nonempty_frames": sum(r["expected_frames"] for r in nonempty),
               "expected_quadratic_spen_nonempty_frames": sum(r["expected_frames"] for r in nonempty if r["classification"] == "spen_imaging"),
               "eligible_scan_attempts": sum(r["eligible_for_attempt"] for r in records),
               "expected_eligible_frames": sum(r["expected_frames"] for r in records if r["eligible_for_attempt"]),
               "nonempty_problem_counts": dict(Counter(p for r in nonempty for p in r["problems"])),
               "nonempty_acquisition_version_counts": dict(Counter(r["parameters"]["acquisition_version"] for r in nonempty)),
               "nonempty_slice_count_histogram": dict(Counter(str(r["parameters"]["slices"]) for r in nonempty)),
               "nonempty_volume_count_histogram": dict(Counter(str(r["parameters"]["volumes"]) for r in nonempty)),
               "nonempty_echo_count_histogram": dict(Counter(str(r["parameters"]["echoes"]) for r in nonempty)),
               "nonempty_segment_count_histogram": dict(Counter(str(r["parameters"]["n_segments"]) for r in nonempty)),
               "existing_mat_reference_count": sum(r["reference_mat_path"] is not None for r in records),
               "auxiliary_and_spectroscopy_counts": [{"classification": k[0], "raw_status": k[1], "scans": v} for k, v in sorted(auxiliary.items())]}
    return {"schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
            "data_root": str(data_root), "scope": "All imported quadratic SPEN and xSPEN scan records under raw/, including empty/missing data; auxiliary and spectroscopy scans are listed separately.",
            "frame_count_note": "Declared frames include all slices, diffusion/repetition volumes and echoes. Binary decoding validates actual frame counts separately; this is not a count of successful reconstructions.",
            "trajectory_policy": "Existing MAT metadata first, then established explicit datalist pairing, then the scan's own method plus recorded ParaVision version. No nearest-scan trajectory guess.",
            "source_manifest_sha256": {name: digest(data_root / name) for name in ("raw_manifest.json", "manifest.json")},
            "summary": summary, "experiments": studies, "records": records, "auxiliary_and_spectroscopy_records": excluded}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = build_inventory(args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.out)
    print(json.dumps({"output": str(args.out), **result["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
