#!/usr/bin/env python3
"""Attach readout-support diagnostics without modifying reconstructions.

Usage: python scripts/trajectory_quality.py --run runs/all_raw_260915

Run only when reconstruction workers have stopped. Source method files must
match their saved SHA256. Every case is checked before metadata is written;
existing arrays, reconstruction statuses, and acquisition parameters remain
unchanged. The 25% support threshold is a display warning, not an exclusion
criterion or a claim that the calibration is valid.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT.parent / "spenpy"))
from spenpy._legacy.bruker.param import read_pv_param
from spenpy._legacy.recon.gridding import smooth_trajectory


WARNING_PREFIX = "trajectory_quality: "


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def trajectory_diagnostics(trajectory, target_ro: int, input_ro: int | None = None) -> dict:
    """Describe the preserved regridding operator; never invent a trajectory.

    ``hard_reject`` identifies degenerate inputs / the operator's exact zero
    support. Callers may use it to prevent reconstruction. This module's CLI
    only reports it and leaves existing case/frame statuses unchanged.
    """
    values = np.asarray([] if trajectory is None else trajectory, dtype=np.float64).reshape(-1)
    result = {"source_samples": int(values.size), "target_ro": int(target_ro),
              "edge_zero_rows_each_side": 3, "hard_reject": False,
              "quality_warnings": [], "calibration_validity": "not_established_by_numeric_checks"}
    if target_ro < 1:
        result.update(hard_reject=True, rejection_reason="invalid_target_readout_size")
        return result
    if values.size < 2 or not np.isfinite(values).all():
        result.update(hard_reject=True, rejection_reason="missing_or_nonfinite_trajectory")
        return result
    result["raw_range"] = [float(values.min()), float(values.max())]
    if input_ro is not None and values.size != int(input_ro):
        result.update(hard_reject=True, rejection_reason="trajectory_ADC_sample_count_mismatch")
        return result
    try:
        smoothed, _ = smooth_trajectory(values)
    except (ValueError, TypeError) as exc:
        result.update(hard_reject=True, rejection_reason="trajectory_smoothing_failed", error=str(exc))
        return result
    if not np.isfinite(smoothed).all() or np.ptp(smoothed) <= 0:
        result.update(hard_reject=True, rejection_reason="constant_or_nonfinite_smoothed_trajectory")
        return result
    n_out = int(np.floor(float(smoothed.max()) + 0.5))
    surviving = max(n_out - 6, 0)
    result.update(smoothed_range=[float(smoothed.min()), float(smoothed.max())],
                  n_out=n_out, surviving_readout_rows=surviving,
                  readout_operator_rank_upper_bound=min(surviving, int(target_ro), int(values.size)),
                  surviving_rows_fraction=surviving / int(target_ro),
                  negative_smoothed_steps=int(np.count_nonzero(np.diff(smoothed) < 0)),
                  negative_step_fraction=float(np.mean(np.diff(smoothed) < 0)))
    if n_out <= 6:
        result.update(hard_reject=True,
                      rejection_reason="trajectory_regridding_has_no_support_after_six_edge_rows_are_zeroed")
        result["quality_warnings"].append(
            WARNING_PREFIX + f"读出重采样 n_out={n_out}，前后各 3 行清零后无有效读出；不能视作有效双方法重建。")
    elif surviving / int(target_ro) < 0.25:
        result["quality_warnings"].append(
            WARNING_PREFIX + f"读出支持度低：n_out={n_out}，截边后仅 {surviving}/{int(target_ro)} 行，"
            f"读出算子秩至多 {result['readout_operator_rank_upper_bound']}；图像质量受限，未替换原轨迹。")
    result["warning_policy"] = "surviving_rows_fraction < 0.25 is diagnostic only; no automatic exclusion"
    return result


def merge_warnings(existing, added: list[str]) -> list:
    if existing is None:
        existing = []
    if not isinstance(existing, list):
        existing = [existing]
    return [warning for warning in existing if not str(warning).startswith(WARNING_PREFIX)] + added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--audit-json", type=Path, help="Optional additional audit copy, e.g. workspace/tmp/<name>.json")
    args = parser.parse_args()
    run = args.run.resolve()
    summary_path = run / "summary.json"
    original_summary = json.loads(summary_path.read_text())
    inventory = json.loads((run / "inventory.json").read_text())
    settings = json.loads((run / "settings.json").read_text())
    data_root = Path(settings.get("data_root", original_summary["data_root"])).resolve()
    code_path = Path(__file__).resolve()
    code_sha = digest(code_path)
    snapshot = run / "source" / f"trajectory_quality_{code_sha[:12]}.py"
    created_at = datetime.now(timezone.utc).isoformat()
    pending, audit = [], []
    fallback_sources = None
    cached = {}
    for item in inventory["records"]:
        case_path = run / "cases" / item["id"] / "case.json"
        original_hash = digest(case_path)
        case = json.loads(case_path.read_text())
        relative_scan = case.get("trajectory", {}).get("raw_scan_path")
        if not relative_scan:
            own_scan = Path(case["raw_scan_path"])
            trajectory_id = case.get("trajectory_scan_id")
            relative_scan = str(own_scan.parent / str(trajectory_id) if trajectory_id not in (None, -1) else own_scan)
        source = (data_root / relative_scan / "method").resolve()
        source_relative = str(source.relative_to(data_root))
        expected_hash = case.get("source_sha256", {}).get(source_relative)
        if source.is_file() and expected_hash is None:
            if fallback_sources is None:
                manifest = json.loads((data_root / "raw_manifest.json").read_text())
                fallback_sources = {record["relative_path"]: record["sha256"] for record in manifest["records"]}
            expected_hash = fallback_sources.get(source_relative)
        if source.is_file():
            if not expected_hash:
                raise ValueError(f"No verified source SHA256 for {source_relative}")
            actual_hash = digest(source)
            if actual_hash != expected_hash:
                raise ValueError(f"Source method changed: {source_relative}")
            if actual_hash not in cached:
                cached[actual_hash] = read_pv_param(str(source.parent), "PVM_EpiTrajAdjkx")
            trajectory = cached[actual_hash]
        else:
            actual_hash, trajectory = None, None
        target_ro = int(case.get("parameters", {}).get("matrix_ro_pe", [0])[0])
        # Source-sample count describes the trajectory itself. Raw readers may
        # preserve acquired oversampling, so do not infer ADC size from Matrix.
        quality = trajectory_diagnostics(trajectory, target_ro)
        quality.update(source_method_path=source_relative, source_method_sha256=actual_hash,
                       source_hash_verified=actual_hash is not None,
                       diagnostic_source_path=str(snapshot.relative_to(run)), diagnostic_source_sha256=code_sha)
        case["trajectory_quality"] = quality
        case["quality_warnings"] = merge_warnings(case.get("quality_warnings"), quality["quality_warnings"])
        for frame in case.get("frames", []):
            frame["trajectory_quality"] = quality
            frame["quality_warnings"] = merge_warnings(frame.get("quality_warnings"), quality["quality_warnings"])
        pending.append((case_path, original_hash, case))
        audit.append({"case_id": case["id"], "experiment_name": case["experiment_name"],
                      "scan_id": case["scan_id"], "status": case["status"],
                      "raw_status": case.get("raw_status"), "frame_count": len(case.get("frames", [])),
                      "reference_mat_path": case.get("reference_mat_path"), "trajectory_quality": quality})
    for case_path, original_hash, _ in pending:
        if digest(case_path) != original_hash:
            raise RuntimeError(f"Case changed while diagnosing; stop workers and retry: {case_path}")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists() and digest(snapshot) != code_sha:
        raise ValueError(f"Diagnostic snapshot has unexpected content: {snapshot}")
    if not snapshot.exists():
        shutil.copy2(code_path, snapshot)
    from run_collection import collect_summary, write_json
    for case_path, _, case in pending:
        write_json(case_path, case)
    summary = collect_summary(run, inventory, settings, original_summary.get("status", "finished"))
    counts = {
        "case_count": len(audit),
        "diagnosed_nout_case_count": sum("n_out" in row["trajectory_quality"] for row in audit),
        "quality_warning_case_count": sum(bool(row["trajectory_quality"]["quality_warnings"]) for row in audit),
        "quality_warning_frame_count": sum(row["frame_count"] for row in audit if row["trajectory_quality"]["quality_warnings"]),
        "zero_support_case_count": sum(row["trajectory_quality"].get("n_out", 99) <= 6 for row in audit),
        "low_nonzero_support_case_count": sum(0 < row["trajectory_quality"].get("surviving_rows_fraction", 1) < .25 for row in audit),
    }
    report = {"created_at": created_at, "run_dir": str(run), "data_root": str(data_root),
              "diagnostic_source_sha256": code_sha, "diagnostic_snapshot": str(snapshot.relative_to(run)),
              "arrays_modified": False, "reconstruction_statuses_modified": False, "counts": counts,
              "case_status_counts": dict(Counter(row["status"] for row in audit)), "records": audit}
    write_json(run / "trajectory_quality_audit.json", report)
    if args.audit_json:
        args.audit_json.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.audit_json, report)
    summary["trajectory_quality_audit"] = {**counts, "path": "trajectory_quality_audit.json",
                                            "diagnostic_source_sha256": code_sha}
    write_json(summary_path, summary)
    print(json.dumps({"run_dir": str(run), **counts, "arrays_modified": False, "statuses_modified": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
