#!/usr/bin/env python3
"""Independently verify the case manifests and their referenced frame arrays.

Always reads inventory.json and case.json afresh; summary.json is not trusted.
Default numerical checks cover every referenced raw frame. Historical MAT
comparisons retain the original strict relative-L2 < 1e-6 criterion. Their
scientific mismatches are reported separately from storage/numerical integrity.
Exit status is nonzero for integrity failure, including a changing collection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import sys
import time

# Keep this read-only audit from monopolizing CPU threads during other jobs.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
from scipy.io import loadmat

MAT_TOLERANCE = 1e-6
INVA_TOLERANCE = 1e-6
NORMAL_TOLERANCE = 1e-9
REFERENCE_ARRAYS = {
    "rofft_original": "spen_original_signal_rofft",
    "rofft_corrected": "spen_phase_corrected_signal_rofft",
    "inva_corrected": "traditional_sr_data",
}
RAW_AXES = ["readout", "spen", "slice", "volume", "coil", "echo"]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    data = path.read_bytes()
    return json.loads(data), hashlib.sha256(data).hexdigest()


def local_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Referenced path leaves its root: {relative}")
    return path


def relative_l2(actual, expected):
    return float(np.linalg.norm(actual - expected) / max(np.linalg.norm(expected), 1e-30))


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def audit(run_dir, *, data_root=None, case_ids=None, numeric_every=1):
    start = time.monotonic()
    run_dir = Path(run_dir).resolve()
    inventory, inventory_sha = read_json(run_dir / "inventory.json")
    data_root = Path(data_root or inventory["data_root"]).resolve()
    mat_manifest, mat_manifest_sha = read_json(data_root / "manifest.json")
    mat_by_scan = {r["raw_scan_path"]: r for r in mat_manifest["records"]}
    records = inventory["records"]
    ids = [r["id"] for r in records]
    errors, warnings, results = [], [], []
    referenced_npz, array_owners = set(), {}
    source_case_sha, changed_files = {}, []
    case_statuses, frame_statuses = Counter(), Counter()
    declared_statuses = Counter(r["raw_status"] for r in records)
    numerical = {"frames_checked": 0, "inva_corrected_checks": 0, "inva_uncorrected_checks": 0,
                 "tikhonov_checks": 0, "max_inva_relative_l2": 0., "max_tikhonov_normal_residual": 0.}
    mat_recorded, mat_recomputed = Counter(), Counter()
    mat_array_recorded, mat_array_recomputed = Counter(), Counter()
    mat_mismatches, mat_not_checked = [], []
    mat_case_paths, mat_reference_frame_counts = set(), {}
    matrix_cache = {}
    expected_decoded_total = decoded_total = referenced_array_files = scanned_array_values = 0
    checked_nonempty = unavailable_count = 0
    numeric_index = 0
    full_collection = not case_ids

    def issue(code, case=None, frame=None, detail=None):
        row = {"code": code}
        if case is not None:
            row["case_id"] = case
        if frame is not None:
            row["frame_id"] = frame
        if detail is not None:
            row["detail"] = detail
        errors.append(row)

    if len(ids) != len(set(ids)):
        issue("duplicate_inventory_ids")
    if len(records) != 782 or declared_statuses != {"nonempty": 751, "missing": 25, "empty": 6}:
        issue("unexpected_collection_scope", detail={"records": len(records), "raw_status_counts": dict(declared_statuses)})
    if len(mat_by_scan) != 501 or len(mat_manifest["records"]) != 501:
        issue("unexpected_mat_reference_scope", detail=len(mat_manifest["records"]))
    if case_ids:
        unknown = sorted(set(case_ids) - set(ids))
        if unknown:
            issue("requested_cases_not_in_inventory", detail=unknown)
        records = [r for r in records if r["id"] in case_ids]

    for case_number, record in enumerate(records, 1):
        cid = record["id"]
        case_path = run_dir / "cases" / cid / "case.json"
        result = {"id": cid, "raw_status": record["raw_status"], "frame_checks": []}
        results.append(result)
        if not case_path.is_file():
            issue("missing_case_manifest", cid)
            continue
        try:
            case, case_sha = read_json(case_path)
        except (OSError, ValueError) as exc:
            issue("unreadable_case_manifest", cid, detail=str(exc))
            continue
        source_case_sha[str(case_path.relative_to(run_dir))] = case_sha
        case_statuses[case.get("status", "missing_status")] += 1
        result["case_status"] = case.get("status")
        result["dimension_correction"] = case.get("dimension_correction")
        if case.get("id") != cid or case.get("raw_scan_path") != record["raw_scan_path"]:
            issue("case_identity_mismatch", cid)
        if case.get("status") in ("running", "pending") or not case.get("finished_at"):
            issue("case_not_finished", cid)
        if case.get("raw_status") != record["raw_status"]:
            issue("case_raw_status_changed", cid)
        frames = case.get("frames", [])
        raw_frames = [f for f in frames if f.get("frame_type") == "raw_reconstruction"]
        if len({f.get("id") for f in frames}) != len(frames):
            issue("duplicate_frame_ids", cid)
        counts, expected_indices, raw_shape = case.get("decoded_counts"), set(), case.get("raw_shape")
        if record["raw_status"] == "nonempty":
            checked_nonempty += 1
            if not isinstance(counts, dict) or not all(positive_int(counts.get(k)) for k in ("slices", "volumes", "coils", "echoes")):
                issue("nonempty_scan_has_no_valid_decoded_counts", cid, detail=case.get("reason"))
            else:
                expected_indices = set(product(range(counts["slices"]), range(counts["volumes"]), range(counts["echoes"])))
                actual_indices = [(f.get("slice_index"), f.get("volume_index"), f.get("echo_index")) for f in raw_frames]
                index_set = set(actual_indices)
                missing, extra = sorted(expected_indices - index_set), sorted(index_set - expected_indices, key=str)
                if len(actual_indices) != len(index_set):
                    issue("duplicate_decoded_frame_indices", cid)
                if missing or extra:
                    issue("incomplete_decoded_cartesian_product", cid, detail={"missing": missing, "extra": extra})
                expected_decoded_total += len(expected_indices)
                decoded_total += len(raw_frames)
                result["expected_decoded_frames"] = len(expected_indices)
                result["raw_frame_count"] = len(raw_frames)
                if case.get("decoded_frames") != len(expected_indices):
                    issue("case_decoded_frame_count_mismatch", cid)
                if not isinstance(raw_shape, list) or len(raw_shape) != 6 or raw_shape[2:] != [counts[k] for k in ("slices", "volumes", "coils", "echoes")]:
                    issue("raw_shape_axes_or_counts_mismatch", cid, detail=raw_shape)
                elif math.prod(raw_shape) * 8 != record.get("raw_bytes"):
                    issue("sorted_complex_sample_count_does_not_match_int32_source_bytes", cid,
                          detail={"sorted_shape": raw_shape, "complex_int32_bytes": math.prod(raw_shape) * 8, "source_bytes": record.get("raw_bytes")})
                if case.get("raw_axes") != RAW_AXES:
                    issue("unexpected_raw_axis_names", cid, detail=case.get("raw_axes"))
                if len(expected_indices) != record.get("expected_frames") and not case.get("dimension_correction"):
                    issue("declared_frame_count_changed_without_documented_correction", cid)
        else:
            unavailable_count += 1
            if case.get("status") != "raw_unavailable" or not case.get("reason") or raw_frames:
                issue("unavailable_raw_scan_not_explicitly_reported", cid)
            result["unavailable_reason"] = case.get("reason")

        mat_record = mat_by_scan.get(record["raw_scan_path"])
        reference = None
        if mat_record:
            mat_case_paths.add(mat_record["mat_path"])
            reference_path = local_path(data_root, mat_record["mat_path"])
            expected_reference_frames = int(np.prod(mat_record["signal_shape"][2:])) // int(record["parameters"]["coils"])
            mat_reference_frame_counts[cid] = expected_reference_frames
            if case.get("reference_mat_path") != mat_record["mat_path"]:
                issue("case_mat_reference_mapping_mismatch", cid)
            try:
                if sha(reference_path) != mat_record["mat_sha256"]:
                    raise ValueError("Reference MAT SHA256 differs from the import manifest")
                reference = loadmat(reference_path, variable_names=list(REFERENCE_ARRAYS.values()))
            except (OSError, ValueError) as exc:
                issue("cannot_read_verified_mat_reference", cid, detail=str(exc))

        actual_statuses = Counter(f.get("status", "missing_status") for f in frames)
        if case.get("frame_status_counts") is not None and case["frame_status_counts"] != dict(actual_statuses):
            issue("frame_status_summary_mismatch", cid)
        if case.get("status") == "completed" and (not raw_frames or any(f.get("status") != "completed" for f in raw_frames)):
            issue("case_completed_status_overstates_frame_completion", cid)

        for frame in frames:
            fid = frame.get("id", "missing_id")
            status = frame.get("status", "missing_status")
            frame_statuses[status] += 1
            diagnostic = {"id": fid, "status": status}
            result["frame_checks"].append(diagnostic)
            raw_frame = frame.get("frame_type") == "raw_reconstruction"
            if status == "failed":
                issue("frame_failed", cid, fid, frame.get("reason"))
            relative = frame.get("arrays_path")
            if not relative:
                issue("frame_has_no_array_file", cid, fid)
                continue
            try:
                array_path = local_path(run_dir, relative)
                if array_path.suffix != ".npz":
                    raise ValueError("Expected frame arrays in an NPZ file")
                relative = str(array_path.relative_to(run_dir))
                if relative in array_owners:
                    issue("array_file_referenced_by_multiple_frames", cid, fid, array_owners[relative])
                array_owners[relative] = {"case_id": cid, "frame_id": fid}
                referenced_npz.add(relative)
                before = array_path.stat()
                with np.load(array_path, allow_pickle=False) as archive:
                    arrays = {key: archive[key] for key in archive.files}
                after = array_path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    changed_files.append(relative)
                referenced_array_files += 1
                if not arrays:
                    raise ValueError("Empty NPZ archive")
                for name, array in arrays.items():
                    if not (np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_)):
                        raise ValueError(f"Non-numeric array {name}: {array.dtype}")
                    if not array.size or not np.isfinite(array).all():
                        raise ValueError(f"Empty or nonfinite array {name}")
                    scanned_array_values += int(array.size)
                    if name in frame.get("array_shapes", {}) and list(array.shape) != frame["array_shapes"][name]:
                        issue("saved_array_shape_differs_from_manifest", cid, fid, name)
                for name in frame.get("array_shapes", {}):
                    if name not in arrays:
                        issue("manifest_array_missing_from_npz", cid, fid, name)
            except Exception as exc:
                issue("array_file_validation_failed", cid, fid, str(exc))
                continue
            diagnostic["arrays_checked"] = len(arrays)
            if raw_frame:
                if "sorted_samples" not in arrays or "rofft_original" not in arrays:
                    issue("raw_frame_missing_original_samples_or_rofft", cid, fid)
                elif isinstance(raw_shape, list) and len(raw_shape) == 6:
                    expected_shape = [raw_shape[1], raw_shape[0], raw_shape[4]]
                    if list(arrays["sorted_samples"].shape) != expected_shape:
                        issue("frame_raw_sample_axes_mismatch", cid, fid,
                              {"shape": list(arrays["sorted_samples"].shape), "expected_pe_ro_coil": expected_shape})
                if status == "completed":
                    required = {"encoding", "inva_weighted_adjoint", "rofft_corrected", "inva_corrected", "tikhonov_coils"}
                    missing = sorted(required - arrays.keys())
                    if missing:
                        issue("completed_frame_missing_reconstruction_arrays", cid, fid, missing)
                elif status in ("partial_reconstruction", "preview_only") and not frame.get("reason"):
                    issue("limited_frame_has_no_explanation", cid, fid)

            perform_numeric = raw_frame and numeric_index % numeric_every == 0
            if raw_frame:
                numeric_index += 1
            if perform_numeric:
                numerical["frames_checked"] += 1
                try:
                    inv = arrays.get("inva_weighted_adjoint")
                    for x_name, y_name in (("inva_corrected", "rofft_corrected"), ("inva_uncorrected", "rofft_original")):
                        if x_name not in arrays:
                            continue
                        if inv is None or y_name not in arrays:
                            raise ValueError(f"Cannot verify {x_name}: missing saved operator or observation")
                        prediction = np.einsum("ij,jrc->irc", inv.astype(np.complex128), arrays[y_name].astype(np.complex128))
                        error = relative_l2(arrays[x_name], prediction)
                        diagnostic[x_name + "_recomputed_relative_l2"] = error
                        numerical[x_name + "_checks"] += 1
                        numerical["max_inva_relative_l2"] = max(numerical["max_inva_relative_l2"], error)
                        if not np.isfinite(error) or error >= INVA_TOLERANCE:
                            issue("inva_operator_application_mismatch", cid, fid, {"array": x_name, "relative_l2": error})
                    if "tikhonov_coils" in arrays:
                        a = np.asarray(arrays["encoding"], np.complex128)
                        y = np.asarray(arrays.get("rofft_corrected", arrays["rofft_original"]), np.complex128)
                        x = np.asarray(arrays["tikhonov_coils"], np.complex128)
                        tik = frame.get("tikhonov", {})
                        alpha = float(tik["lambda_absolute"])
                        lam = float(tik["lambda_relative"])
                        if alpha <= 0 or lam <= 0 or not math.isfinite(alpha + lam):
                            raise ValueError("Nonpositive/nonfinite Tikhonov penalty")
                        if a.ndim != 2 or y.ndim != 3 or x.ndim != 3 or a.shape[0] != y.shape[0] or a.shape[1] != x.shape[0] or x.shape[1:] != y.shape[1:]:
                            raise ValueError("Tikhonov operator/observation/result dimensions disagree")
                        ah = a.conj().T
                        yr, xr = y.reshape(y.shape[0], -1), x.reshape(x.shape[0], -1)
                        normal = ah @ (a @ xr - yr) + alpha * xr
                        residual = float(np.linalg.norm(normal) / max(np.linalg.norm(ah @ yr), 1e-30))
                        key = hashlib.sha256(a.tobytes()).hexdigest()
                        if key not in matrix_cache:
                            matrix_cache[key] = float(np.linalg.svd(a, compute_uv=False)[0])
                        smax = matrix_cache[key]
                        if not np.isclose(alpha, lam * smax**2, rtol=1e-9, atol=0):
                            issue("tikhonov_penalty_normalization_mismatch", cid, fid)
                        if not np.isclose(float(tik["encoding_spectral_norm"]), smax, rtol=1e-9, atol=0):
                            issue("recorded_encoding_spectral_norm_mismatch", cid, fid)
                        diagnostic["tikhonov_recomputed_normal_residual"] = residual
                        numerical["tikhonov_checks"] += 1
                        numerical["max_tikhonov_normal_residual"] = max(numerical["max_tikhonov_normal_residual"], residual)
                        if not np.isfinite(residual) or residual > NORMAL_TOLERANCE:
                            issue("tikhonov_normal_equation_failed", cid, fid, residual)
                        if float(tik.get("relative_normal_residual", math.inf)) > NORMAL_TOLERANCE:
                            issue("recorded_tikhonov_normal_equation_failed", cid, fid)
                except Exception as exc:
                    issue("cannot_recompute_numerical_checks", cid, fid, str(exc))

            if mat_record and raw_frame:
                recorded = frame.get("mat_regression")
                recorded_status = recorded.get("status", "missing") if isinstance(recorded, dict) else "missing"
                mat_recorded[recorded_status] += 1
                for name, comparison in (recorded or {}).get("arrays", {}).items():
                    mat_array_recorded["passed" if comparison.get("passed") else "mismatch"] += 1
                    if comparison.get("passed") and ("relative_l2_error" not in comparison or comparison["relative_l2_error"] >= MAT_TOLERANCE):
                        issue("recorded_mat_pass_violates_original_tolerance", cid, fid, name)
                if counts and frame.get("echo_index") != counts["echoes"] - 1:
                    mat_recomputed["not_applicable"] += 1
                    continue
                checks = {}
                if reference is not None and counts:
                    for ours, old in REFERENCE_ARRAYS.items():
                        if ours not in arrays or old not in reference:
                            checks[old] = {"status": "not_checked", "reason": "Required saved or reference array missing"}
                            continue
                        try:
                            previous = reference[old]
                            target = previous[:, :, 0, :].reshape(*previous.shape[:2], counts["slices"], counts["volumes"], counts["coils"], order="F")
                            target = target[:, :, frame["slice_index"], frame["volume_index"], :]
                            if target.shape != arrays[ours].shape:
                                raise ValueError(f"Reference shape {target.shape} differs from {arrays[ours].shape}")
                            error = relative_l2(arrays[ours], target)
                            checks[old] = {"status": "passed" if error < MAT_TOLERANCE else "mismatch", "relative_l2_error": error}
                            mat_array_recomputed[checks[old]["status"]] += 1
                            prior = (recorded or {}).get("arrays", {}).get(old)
                            if prior and bool(prior.get("passed")) != (error < MAT_TOLERANCE):
                                issue("recorded_mat_verdict_differs_from_saved_arrays", cid, fid, {"array": old, "recomputed_error": error})
                        except Exception as exc:
                            checks[old] = {"status": "not_checked", "reason": str(exc)}
                actual_mat_status = "passed" if len(checks) == 3 and all(x["status"] == "passed" for x in checks.values()) else "mismatch" if any(x["status"] == "mismatch" for x in checks.values()) else "not_checked"
                mat_recomputed[actual_mat_status] += 1
                diagnostic["mat_recomputed_status"] = actual_mat_status
                diagnostic["mat_recomputed_arrays"] = checks
                if recorded_status == "passed" and actual_mat_status != "passed":
                    issue("recorded_mat_frame_pass_not_confirmed", cid, fid)
                if actual_mat_status == "mismatch":
                    mat_mismatches.append({"case_id": cid, "frame_id": fid, "mat_path": mat_record["mat_path"], "recorded_status": recorded_status, "arrays": checks})
                elif actual_mat_status == "not_checked":
                    mat_not_checked.append({"case_id": cid, "frame_id": fid, "arrays": checks})

        if case_number % 50 == 0:
            print(json.dumps({"event": "verification_progress", "cases_checked": case_number,
                              "cases_total": len(records), "referenced_npz_checked": referenced_array_files,
                              "integrity_errors": len(errors)}, ensure_ascii=False), flush=True)

    # Detect a concurrent rerun; a mixed-time snapshot is never final proof.
    if sha(run_dir / "inventory.json") != inventory_sha:
        changed_files.append("inventory.json")
    if sha(data_root / "manifest.json") != mat_manifest_sha:
        changed_files.append("source_mat_manifest.json")
    for relative, digest in source_case_sha.items():
        path = run_dir / relative
        if not path.exists() or sha(path) != digest:
            changed_files.append(relative)
    if changed_files:
        issue("collection_changed_during_verification", detail=sorted(set(changed_files)))
    all_npz = {str(path.relative_to(run_dir)) for path in (run_dir / "cases").rglob("*.npz")}
    unreferenced = sorted(all_npz - referenced_npz)
    if unreferenced:
        warnings.append({"code": "unreferenced_npz_ignored", "count": len(unreferenced),
                         "note": "Only case.json references define current outputs. Old arrays do not count as completed frames."})
    on_disk_cases = {path.parent.name for path in (run_dir / "cases").glob("*/case.json")}
    unknown_cases = sorted(on_disk_cases - set(ids))
    if unknown_cases:
        warnings.append({"code": "case_directories_outside_inventory_ignored", "ids": unknown_cases})
    if full_collection and (checked_nonempty != 751 or unavailable_count != 31):
        issue("not_all_expected_raw_availability_records_checked")
    expected_mat_frames = sum(mat_reference_frame_counts.values())
    if full_collection and (len(mat_case_paths) != 501 or expected_mat_frames != 811):
        issue("historical_mat_coverage_missing", detail={"mat_files": len(mat_case_paths), "expected_frames": expected_mat_frames})
    if full_collection and sum(mat_recomputed[k] for k in ("passed", "mismatch", "not_checked")) != expected_mat_frames:
        issue("historical_mat_frame_checks_incomplete", detail={"checks": dict(mat_recomputed), "expected": expected_mat_frames})
    corrections = [{"case_id": x["id"], "correction": x["dimension_correction"]} for x in results if x.get("dimension_correction")]
    integrity_passed = not errors
    history_passed = not mat_mismatches and not mat_not_checked and mat_recomputed["passed"] == expected_mat_frames
    return {
        "schema_version": 1, "created_at": datetime.now().astimezone().isoformat(),
        "run_dir": str(run_dir), "data_root": str(data_root), "full_collection_checked": full_collection,
        "collection_stable_during_verification": not changed_files,
        "provisional": bool(changed_files or case_statuses.get("running") or not full_collection),
        "integrity_passed": integrity_passed, "historical_mat_all_passed": history_passed,
        "passed": bool(integrity_passed and history_passed and full_collection),
        "inventory_sha256": inventory_sha, "source_mat_manifest_sha256": mat_manifest_sha,
        "source_case_manifest_sha256": source_case_sha,
        "coverage": {"inventory_records": len(ids), "checked_case_manifests": len(source_case_sha),
                     "nonempty_records_checked": checked_nonempty, "unavailable_records_checked": unavailable_count,
                     "source_raw_status_counts": dict(declared_statuses),
                     "case_status_counts": dict(case_statuses), "frame_status_counts": dict(frame_statuses),
                     "declared_expected_nonempty_frames": inventory["summary"]["expected_nonempty_frames"],
                     "decoded_expected_frames": expected_decoded_total, "raw_frames_checked": decoded_total,
                     "referenced_npz_checked": referenced_array_files, "array_values_checked_finite": scanned_array_values},
        "numerical_checks": {**numerical, "sampling_stride": numeric_every,
                             "inva_relative_l2_tolerance": INVA_TOLERANCE,
                             "tikhonov_relative_normal_tolerance": NORMAL_TOLERANCE,
                             "distinct_encoding_spectral_norms_recomputed": len(matrix_cache)},
        "historical_mat_regression": {"reference_files_checked": len(mat_case_paths),
                                      "expected_reference_frames": expected_mat_frames,
                                      "relative_l2_tolerance": MAT_TOLERANCE,
                                      "criterion": "relative_l2_error < 1e-6, unchanged from the export regression",
                                      "recorded_frame_status_counts": dict(mat_recorded),
                                      "recorded_array_verdict_counts": dict(mat_array_recorded),
                                      "recomputed_frame_status_counts": dict(mat_recomputed),
                                      "recomputed_array_verdict_counts": dict(mat_array_recomputed),
                                      "mismatches": mat_mismatches, "not_checked": mat_not_checked},
        "dimension_corrections": corrections,
        "unreferenced_npz": {"count": len(unreferenced), "relative_paths": unreferenced,
                             "ignored_for_completion_and_numerical_checks": True},
        "errors": errors, "warnings": warnings, "cases": results,
        "elapsed_seconds": time.monotonic() - start,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--out", type=Path, help="Default: RUN/verification.json")
    parser.add_argument("--case-id", action="append", help="Development-only subset; output is explicitly provisional")
    parser.add_argument("--numeric-every", type=int, default=1,
                        help="Numerical-check stride; 1 recomputes all InvA/Tikhonov equations. All arrays and MAT references are always inspected.")
    args = parser.parse_args()
    if args.numeric_every < 1:
        parser.error("--numeric-every must be positive")
    result = audit(args.run_dir, data_root=args.data_root, case_ids=set(args.case_id or []), numeric_every=args.numeric_every)
    destination = args.out or args.run_dir / "verification.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    print(json.dumps({"output": str(destination), "integrity_passed": result["integrity_passed"],
                      "historical_mat_all_passed": result["historical_mat_all_passed"],
                      "provisional": result["provisional"], "coverage": result["coverage"],
                      "integrity_error_count": len(result["errors"]),
                      "mat_regression": result["historical_mat_regression"]["recomputed_frame_status_counts"],
                      "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=False, indent=2))
    return 0 if result["integrity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
