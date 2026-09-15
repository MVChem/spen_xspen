"""Independently audit completed latent reconstruction artifacts on the CPU.

Run after assembly and rendering. The default report is
--outroot/verification/final.json. No reconstruction is rerun and no GPU is
used. Finite-budget nonlinear proximal subproblems need not be converged;
their stopping reasons and objective decreases are inspected and reported.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
DEFAULT_REFERENCE = PROJECT / "runs/rodent192_spen2x_260914/figures_260915"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_finite(value, where: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            check_finite(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            check_finite(item, f"{where}[{index}]")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f"Non-finite number at {where}")


def read_json(path: Path):
    result = json.loads(path.read_text(encoding="utf-8"))
    check_finite(result, str(path))
    return result


def read_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    for key, array in arrays.items():
        if np.issubdtype(array.dtype, np.number):
            require(np.isfinite(array).all(), f"Non-finite array: {path}:{key}")
    return arrays


def exact(actual: np.ndarray, expected: np.ndarray, where: str) -> None:
    require(actual.shape == expected.shape and actual.dtype == expected.dtype
            and np.array_equal(actual, expected), f"Array differs from expected bytes/values: {where}")


def display(raw: np.ndarray) -> np.ndarray:
    return ((raw + 1) / 2).clip(0, 1)


def has_reference_metric(value) -> bool:
    if isinstance(value, dict):
        return any(key.lower() in {"psnr", "ssim"} or has_reference_metric(item)
                   for key, item in value.items())
    if isinstance(value, list):
        return any(has_reference_metric(item) for item in value)
    return False


def metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    p, g = display(prediction), display(target)
    mse = float(np.mean((p - g) ** 2))
    mu_p, mu_g = [gaussian_filter(x, 1.5, truncate=3.5) for x in (p, g)]
    var_p = gaussian_filter(p * p, 1.5, truncate=3.5) - mu_p ** 2
    var_g = gaussian_filter(g * g, 1.5, truncate=3.5) - mu_g ** 2
    cov = gaussian_filter(p * g, 1.5, truncate=3.5) - mu_p * mu_g
    ssim = ((2 * mu_p * mu_g + .01 ** 2) * (2 * cov + .03 ** 2)
            / ((mu_p ** 2 + mu_g ** 2 + .01 ** 2) * (var_p + var_g + .03 ** 2)))
    return {"psnr": float(-10 * np.log10(max(mse, 1e-12))),
            "ssim": float(ssim[5:-5, 5:-5].mean())}


def verify_trace(path: Path, outer_steps: int) -> dict:
    diagnostic = read_json(path)
    trace = diagnostic["trace"]
    require(diagnostic["steps"] == diagnostic["total_subproblems"] == len(trace) == outer_steps,
            f"Expected {outer_steps} outer steps: {path}")
    require(diagnostic["initialization_uses_ground_truth"] is False,
            f"Ground truth used for initialization: {path}")
    require(diagnostic["decoder_clamp"] is False, f"Decoder was clipped inside solver: {path}")
    require(diagnostic["exact_proximal"] is False, f"Approximate proximal mislabeled: {path}")
    sigmas = diagnostic["sigmas"]
    require(len(sigmas) == outer_steps + 1 and sigmas[-1] == 0
            and all(a >= b for a, b in zip(sigmas[:-1], sigmas[1:])), f"Invalid schedule: {path}")
    reasons, accepted_total, converged_total, max_increase = Counter(), 0, 0, 0.
    for index, row in enumerate(trace):
        require(row["step"] == index, f"Outer step order changed: {path}")
        before, after = row["objective_before"], row["objective_after"]
        tolerance = 1e-5 + 1e-6 * abs(before)
        require(after <= before + tolerance, f"Outer objective increased: {path}, step {index}")
        max_increase = max(max_increase, after - before)
        require(row["stop_reason"] in {"inner_budget", "gradient_tolerance",
                "line_search_stalled", "objective_stagnation"}, f"Unknown stop reason: {path}")
        inner = row["inner_trace"]
        require(len(inner) <= row["inner_budget"] == diagnostic["inner_steps"],
                f"Inner-step budget mismatch: {path}, step {index}")
        accepted = sum(bool(item["accepted"]) for item in inner)
        require(accepted == row["accepted_steps"], f"Accepted-step count mismatch: {path}")
        for inner_index, item in enumerate(inner):
            require(item["iteration"] == inner_index, f"Inner step order changed: {path}")
            inner_before, inner_after = item["before"]["objective"], item["after"]["objective"]
            require(inner_after <= inner_before + 1e-5 + 1e-6 * abs(inner_before),
                    f"Inner objective increased: {path}, outer {index}, inner {inner_index}")
        reasons[row["stop_reason"]] += 1
        accepted_total += accepted
        converged_total += bool(row["converged"])
    require(accepted_total == diagnostic["accepted_steps"], f"Total accepted steps mismatch: {path}")
    require(converged_total == diagnostic["converged_subproblems"], f"Converged count mismatch: {path}")
    return {"outer_subproblems": outer_steps, "accepted_inner_steps": accepted_total,
            "converged_subproblems": converged_total, "stop_reasons": dict(reasons),
            "maximum_objective_increase": max_increase}


def verify_case(folder: Path, step: int, outer_steps: int, *, real: bool = False) -> tuple[dict, dict, dict]:
    arrays, row = read_npz(folder / "arrays.npz"), read_json(folder / "metrics.json")
    require(row["checkpoint_step"] == step, f"Checkpoint step differs: {folder}")
    require(arrays["prediction_raw"].shape == arrays["initial_raw"].shape == (192, 192),
            f"Wrong decoded image grid: {folder}")
    require(row["parameters"]["steps"] == outer_steps, f"Wrong solver steps: {folder}")
    if real:
        require(not has_reference_metric(row), f"Real-data PSNR/SSIM present: {folder}")
        require(not {"gt", "target"}.intersection(arrays), f"Real ground truth array present: {folder}")
    else:
        measured = metrics(arrays["prediction_raw"], arrays["gt"])
        for key, value in measured.items():
            require(abs(value - row["metrics"][key]) <= 1e-6, f"Saved {key} does not match arrays: {folder}")
    return arrays, row, verify_trace(folder / "solver_trace.json", outer_steps)


def verify(outroot: Path, reference: Path, report: dict, expected_step: int | None,
           outer_steps: int) -> None:
    summary = read_json(outroot / "summary.json")
    frozen = read_json(outroot / "checkpoint_metadata.json")
    rendered = read_json(outroot / "render_metadata.json")
    results = read_npz(outroot / "reconstruction_results.npz")
    old = {kind: read_npz(reference / f"{kind}.npz") for kind in ("simulation", "real")}
    new = {kind: read_npz(outroot / f"{kind}.npz") for kind in old}
    step, digest = frozen["step"], sha256(outroot / "checkpoint.pt")
    require(isinstance(step, int) and step > 0, "Invalid frozen checkpoint step")
    require(expected_step is None or step == expected_step, "Frozen checkpoint differs from --step")
    require(summary["checkpoint_step"] == rendered["checkpoint_step"] == results["checkpoint_step"].item() == step,
            "Summary, renderer, or NPZ checkpoint step differs from frozen metadata")
    require(summary["checkpoint_sha256"] == digest, "Summary SHA256 differs from frozen checkpoint")
    require(rendered["results_npz_sha256"] == sha256(outroot / "reconstruction_results.npz"),
            "Renderer used a different reconstruction NPZ")
    require(not rendered.get("verification_mock", False) and not results.get("verification_mock", False),
            "Mock render cannot pass final verification")
    require(results["value_range"].item() == "display_0_1"
            and results["real_orientation"].item() == "reference_display_rot180", "Display metadata differs")
    report["checkpoint"] = {"step": step, "sha256": digest}
    unchanged = {"simulation": ["target", "degraded", "phase_inva", "tikhonov", "labels",
                                "mask", "noise_sigma", "case_keys"],
                 "real": ["degraded", "phase_inva", "tikhonov", "labels", "fov_mm"]}
    for kind, keys in unchanged.items():
        require(rendered["reference_sha256"][f"{kind}.npz"] == sha256(reference / f"{kind}.npz"),
                f"Reference hash differs for {kind}")
        require(new[kind]["checkpoint_step"].item() == step, f"Merged NPZ step differs: {kind}")
        for key in keys:
            exact(new[kind][key], old[kind][key], f"{kind}.{key}")
        for key in (set(new[kind]) & set(old[kind])) - {"diffusion"}:
            exact(new[kind][key], old[kind][key], f"{kind}.{key}")
        require(not np.array_equal(new[kind]["diffusion"], old[kind]["diffusion"]),
                f"The new prior row is identical to the old prior: {kind}")
    report["unchanged_reference_arrays"] = unchanged
    exact(results["simulation_case_keys"], old["simulation"]["case_keys"], "simulation result order")
    exact(results["real_labels"], old["real"]["labels"], "real result labels")
    exact(results["real_fov_mm"], old["real"]["fov_mm"], "real result FOV")
    records = read_json(reference / "simulation_cases.json")
    keys = {role: [case["key"] for case in records[role]] for role in ("calibration", "report")}
    sources = {role: [case["dataset"] for case in records[role]] for role in keys}
    require(all(len(set(value)) == len(value) == 4 for value in keys.values()), "Expected four unique cases per role")
    require(not set(keys["calibration"]) & set(keys["report"]), "Calibration and report cases overlap")
    require(sources["calibration"] == sources["report"] == old["simulation"]["labels"].tolist()
            and len(set(sources["report"])) == 4, "Expected one case per each of four sources")
    report["simulation_selection"] = {"keys": keys, "sources": sources, "disjoint": True,
                                       "held_out_from_prior_training": False}
    case_checks, category_reasons, category_counts = {}, {}, Counter()

    def record_case(folder: Path, category: str, is_real: bool = False):
        arrays, row, trace = verify_case(folder, step, outer_steps, real=is_real)
        case_checks[str(folder.relative_to(outroot))] = trace
        category_reasons.setdefault(category, Counter()).update(trace["stop_reasons"])
        category_counts[category] += 1
        return arrays, row

    for index, condition in enumerate(("R1", "R2")):
        folder = outroot / condition
        config = read_json(folder / "config.json")
        require(config["checkpoint_step"] == step and config["checkpoint_sha256"] == digest,
                f"Mixed checkpoint in {condition}")
        require(config["reference_npz_sha256"] == sha256(reference / "simulation.npz"),
                f"Simulation worker used different reference: {condition}")
        require(config["calibration_keys"] == keys["calibration"] and config["report_keys"] == keys["report"],
                f"Worker case order changed: {condition}")
        provenance = read_json(folder / "reference_provenance.json")
        require(provenance["training_manifest_sha256"] == frozen["manifest_sha256"], "Training manifest mismatch")
        trials, selected = read_json(folder / "calibration.json"), read_json(folder / "selected.json")
        require([trial["lamb"] for trial in trials] == config["lambda_candidates"], "Calibration trials incomplete")
        require(selected["lamb"] == max(trials, key=lambda trial: trial["mean_psnr"])["lamb"],
                f"Selected parameter did not maximize calibration PSNR: {condition}")
        for trial in trials:
            scores = []
            for case_index, expected_key in enumerate(keys["calibration"]):
                casefolder = folder / "calibration" / f'lambda_{trial["lamb"]:g}' / f"case_{case_index}"
                _, row = record_case(casefolder, "simulation_calibration")
                require(row["key"] == expected_key and row["case_index"] == case_index, "Calibration identity differs")
                require(row == trial["cases"][case_index], "Calibration aggregate differs from case metrics")
                scores.append(row["metrics"]["psnr"])
            require(abs(float(np.mean(scores)) - trial["mean_psnr"]) <= 1e-6, "Calibration mean PSNR differs")
        predictions = read_npz(folder / "predictions.npz")
        exact(predictions["case_keys"], old["simulation"]["case_keys"], f"{condition} predictions order")
        raw = results[f"simulation_diffusion_raw_{condition}"]
        exact(raw, predictions["prediction_raw"], f"{condition} assembled raw")
        exact(results[f"simulation_diffusion_{condition}"], display(raw), f"{condition} clipped display")
        exact(new["simulation"]["diffusion"][index], display(raw), f"{condition} plotted diffusion")
        for case_index, expected_key in enumerate(keys["report"]):
            arrays, row = record_case(folder / "report" / f"case_{case_index}", "simulation_report")
            require(row["key"] == expected_key and row["case_index"] == case_index, "Report identity differs")
            require(row["parameters"]["lamb"] == selected["lamb"], "Report used a different lambda")
            require(row == summary["simulation"][condition]["cases"][case_index], "Summary report case differs")
            exact(arrays["prediction_raw"], raw[case_index], f"{condition} case raw")
            exact(display(arrays["gt"]), old["simulation"]["target"][case_index], f"{condition} GT")
            expected_obs = old["simulation"]["observation"][index, case_index][:, old["simulation"]["mask"][index], :][None]
            exact(arrays["observation"], expected_obs, f"{condition} saved observation")
    require(len(summary["real"]) == 10 and not has_reference_metric(summary["real"]), "Invalid real-data summary")
    real_configs = sorted((outroot / "real").glob("config_*.json"))
    require(real_configs, "Missing real worker configuration")
    configured_indices = []
    for path in real_configs:
        config = read_json(path)
        require(config["checkpoint_step"] == step and config["checkpoint_sha256"] == digest,
                f"Real worker checkpoint mismatch: {path}")
        require(config["no_ground_truth"] is True and not has_reference_metric(config), "Real config claims GT metrics")
        configured_indices.extend(config["case_indices"])
    require(sorted(configured_indices) == list(range(10)), "Real workers did not cover ten distinct cases")
    for index in range(10):
        arrays, row = record_case(outroot / "real" / f"case_{index:02d}", "real", True)
        require(row == summary["real"][index] and row["case_index"] == index, "Real summary/order differs")
        rotated = np.rot90(display(arrays["prediction_raw"]), 2)
        exact(results["real_diffusion"][index], rotated, f"Real {index} display rotation")
        exact(new["real"]["diffusion"][index], rotated, f"Real {index} plotted diffusion")
        meta = row["metadata"]
        require(meta["fov_mm"] == old["real"]["fov_mm"][index]
                and f'Acquisition #{meta["export_index"]}' == old["real"]["labels"][index], "Real acquisition identity differs")
    report["raw_to_display"] = {"simulation": "exact clip((raw+1)/2,0,1)",
                                 "real": "exact rot180(clip((raw+1)/2,0,1))"}
    report["solver"] = {"outer_steps_per_case": outer_steps, "case_counts": dict(category_counts),
                         "stop_reasons_by_category": {key: dict(value) for key, value in category_reasons.items()},
                         "case_traces": case_checks,
                         "objective_tolerance": "1e-5 + 1e-6 * abs(objective_before)",
                         "approximate_subproblems_allowed": True}
    report["figures"] = {}
    for name in ("figure1_simulation.png", "figure2_real.png"):
        with Image.open(outroot / name) as image, Image.open(reference / name) as old_image:
            image.load()
            require(image.size == old_image.size, f"Figure dimensions differ from the reference: {name}")
            dpi = image.info.get("dpi")
            require(dpi is not None and all(abs(value - 300) < .01 for value in dpi), f"Figure DPI differs: {name}")
            report["figures"][name] = {"size": list(image.size), "dpi": list(dpi), "sha256": sha256(outroot / name)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--step", type=int, help="Optional expected checkpoint step; otherwise use frozen metadata")
    parser.add_argument("--outer-steps", type=int, default=60)
    args = parser.parse_args()
    outroot, reference = args.outroot.resolve(), args.reference_dir.resolve()
    report = {"status": "running", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
              "outroot": str(outroot), "reference_dir": str(reference), "gpu_used": False,
              "verifier_sha256": sha256(Path(__file__).resolve())}
    try:
        verify(outroot, reference, report, args.step, args.outer_steps)
        report["status"] = "passed"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    path = outroot / "verification/final.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)
    print(json.dumps({"status": report["status"], "report": str(path), "error": report.get("error")}, ensure_ascii=False))
    if report["status"] != "passed":
        sys.exit(1)


if __name__ == "__main__":
    main()
