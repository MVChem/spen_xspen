"""Repeat the frozen report cases without retuning, then render a local review.

Inference writes only to a new runs directory. Reference baselines and the
original final run are read-only; real data retain their original display turn.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CAMPAIGN = PROJECT / "runs/rodent192_latent_dit_260915"
REFERENCE = PROJECT / "runs/rodent192_spen2x_260914/figures_260915"


def read_json(path):
    return json.loads(Path(path).read_text())


def infer(args):
    import torch
    from evaluate_reconstruction import load_prior, save_json, sha, solve_case
    from reference_cases import load_real_reference, load_simulation_reference
    from verify_reconstruction import verify_case

    root = args.out / args.group
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = True
    checkpoint = args.source_run / "checkpoint.pt"
    model = load_prior(checkpoint, args.device)
    assert model[-1]["step"] == 60000
    sim = args.group in ("R1", "R2")
    chosen = read_json(args.source_run / (args.group if sim else "R1") / "selected.json")
    for key in ("steps", "inner_steps", "sigma_max", "prox_lr"):
        setattr(args, key, chosen[key])
    reference = (load_simulation_reference if sim else load_real_reference)(args.reference_dir, args.device)
    if sim:
        assert reference["provenance"]["training_manifest_sha256"] == model[-1]["manifest_sha256"]
        cases = [c for c in reference["report"] if c["condition"] == args.group]
    else:
        fov = int(args.group.removeprefix("real"))
        cases = [c for c in reference["cases"] if c["fov_mm"] == fov]
    save_json(root / "provenance.json", dict(
        started_at_utc=datetime.now(timezone.utc).isoformat(),
        checkpoint=str(checkpoint), checkpoint_sha256=model[-1]["checkpoint_sha256"],
        checkpoint_step=model[-1]["step"], parameter_source=str(args.source_run),
        selected=chosen, retuned=False, case_keys=[c["key"] for c in cases],
        reference=reference["provenance"],
        code_sha256={p.name: sha(p) for p in [Path(__file__), HERE / "evaluate_reconstruction.py",
                     HERE / "reference_cases.py", HERE / "latent_reconstruction.py",
                     HERE / "dit.py", HERE / "vae_codec.py"]}))
    rows = []
    for case in cases:
        folder = root / f'case_{case["case_index"]:02d}'
        image, row = solve_case(case, model, args, chosen["lamb"], folder)
        _, _, trace = verify_case(folder, 60000, args.steps, real=not sim)
        previous = (args.source_run / args.group / "report" / f'case_{case["case_index"]}'
                    if sim else args.source_run / "real" / f'case_{case["case_index"]:02d}')
        with np.load(previous / "arrays.npz", allow_pickle=False) as old:
            observation_delta = float(np.max(np.abs(old["observation"] - case["observation"].cpu().numpy())))
            delta = image - old["prediction_raw"]
        assert observation_delta <= 2e-5, "Observation changed from the frozen final run"
        old_row = read_json(previous / "metrics.json")
        assert row["key"] == old_row["key"] and row["parameters"] == old_row["parameters"]
        row["repeat_check"] = dict(prediction_raw_max_abs=float(np.abs(delta).max()),
            prediction_raw_rmse=float(np.sqrt(np.mean(delta**2))),
            observation_max_abs=observation_delta, trace_verified=True,
            measurement_nrmse_delta=row["measurement_nrmse"] - old_row["measurement_nrmse"])
        if sim:
            row["repeat_check"]["psnr_delta"] = row["metrics"]["psnr"] - old_row["metrics"]["psnr"]
        rows.append(row)
    save_json(root / "completed.json", dict(status="completed", fresh_inference=True,
        finished_at_utc=datetime.now(timezone.utc).isoformat(), group=args.group, cases=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=["R1", "R2", "real16", "real24"], required=True)
    parser.add_argument("--source-run", type=Path, default=CAMPAIGN / "reconstruction_final_260916")
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.source_run = args.source_run.resolve()
    args.reference_dir = args.reference_dir.resolve()
    infer(args)


if __name__ == "__main__":
    main()
