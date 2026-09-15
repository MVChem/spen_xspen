"""Render the fixed simulation/real comparison with a VAE + DiT prior row.

This CPU-only renderer reuses the approved prior192 layout, metric convention,
and reference arrays. It changes only the prior reconstruction row and its
label. All inputs and outputs use display intensities in [0, 1]; real images
must already have the same 180-degree rotation as the reference figures.

Required keys in --results (an NPZ, loaded without pickle):
  simulation_diffusion_R1: float array (4, 192, 192), full PE
  simulation_diffusion_R2: float array (4, 192, 192), random 50% PE
  real_diffusion: float array (10, 192, 192), reference display orientation
  checkpoint_step: positive integer scalar, read to form the row label
  value_range: string scalar "display_0_1"
  real_orientation: string scalar "reference_display_rot180"

Optional simulation_case_keys, real_labels, and real_fov_mm arrays are checked
against the reference ordering when supplied. An optional --step is an
additional assertion against checkpoint_step, never a replacement for it.
An optional verification_mock=True scalar marks CPU layout checks explicitly
as MOCK in the figure row labels and output metadata.
The output directory receives the two PNGs, their merged plotting NPZs, and
render_metadata.json. Existing files are not overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "prior192"))
import render_rebuilt as reference_renderer


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scalar(arrays: dict[str, np.ndarray], key: str):
    if key not in arrays or arrays[key].ndim != 0:
        raise ValueError(f"{key} must be a scalar in the results NPZ")
    return arrays[key].item()


def _display_array(arrays: dict[str, np.ndarray], key: str,
                   shape: tuple[int, ...]) -> np.ndarray:
    if key not in arrays:
        raise ValueError(f"Missing results NPZ key: {key}")
    image = arrays[key]
    if (image.shape != shape or not np.issubdtype(image.dtype, np.floating)
            or not np.isfinite(image).all()):
        raise ValueError(f"{key} must be finite floating-point values with shape {shape}")
    if image.min() < 0 or image.max() > 1:
        raise ValueError(f"{key} must already be clipped display intensities in [0, 1]")
    return image


def _check_order(results: dict[str, np.ndarray], key: str, expected: np.ndarray) -> None:
    if key in results and not np.array_equal(results[key], expected):
        raise ValueError(f"{key} does not match the reference case order")


def render_results(reference_dir: Path, results_path: Path, out: Path,
                   expected_step: int | None = None) -> dict[str, str]:
    reference_dir, results_path, out = (
        path.expanduser().resolve() for path in (reference_dir, results_path, out))
    if out == reference_dir:
        raise ValueError("Use a new output directory; the reference directory cannot be overwritten")
    workspace = HERE.parents[3]
    if out.is_relative_to(workspace / "experiments"):
        raise ValueError("Generated figures belong in the project runs directory, not experiments")

    output_names = ("figure1_simulation.png", "figure2_real.png", "simulation.npz",
                    "real.npz", "render_metadata.json")
    for name in output_names:
        if (out / name).exists():
            raise FileExistsError(f"Output already exists; use a fresh directory: {out / name}")

    simulation = reference_renderer._load_images(reference_dir / "simulation.npz", simulation=True)
    real = reference_renderer._load_images(reference_dir / "real.npz", simulation=False)
    if simulation["target"].shape != (4, 192, 192) or real["diffusion"].shape != (10, 192, 192):
        raise ValueError("Expected the approved four simulation cases and ten real cases")
    if "fov_mm" not in real or not np.array_equal(real["fov_mm"], [16] * 5 + [24] * 5):
        raise ValueError("Expected five real cases each for FOV 16 mm and 24 mm")
    with np.load(reference_dir / "simulation.npz", allow_pickle=False) as source:
        simulation["case_keys"] = source["case_keys"].copy()
    with np.load(results_path, allow_pickle=False) as source:
        results = {key: source[key] for key in source.files}
    step = _scalar(results, "checkpoint_step")
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError("checkpoint_step must be a positive integer scalar")
    if expected_step is not None and expected_step != step:
        raise ValueError(f"--step {expected_step} disagrees with results checkpoint_step {step}")
    if _scalar(results, "value_range") != "display_0_1":
        raise ValueError("value_range must be 'display_0_1'; convert predictions before rendering")
    if _scalar(results, "real_orientation") != "reference_display_rot180":
        raise ValueError("real_orientation must be 'reference_display_rot180'")
    mock = _scalar(results, "verification_mock") if "verification_mock" in results else False
    if not isinstance(mock, bool):
        raise ValueError("verification_mock must be a boolean scalar when supplied")
    _check_order(results, "simulation_case_keys", simulation["case_keys"])
    _check_order(results, "real_labels", real["labels"])
    _check_order(results, "real_fov_mm", real["fov_mm"])
    simulation["diffusion"] = np.stack([
        _display_array(results, "simulation_diffusion_R1", (4, 192, 192)),
        _display_array(results, "simulation_diffusion_R2", (4, 192, 192)),
    ])
    real["diffusion"] = _display_array(results, "real_diffusion", (10, 192, 192))
    for arrays in (simulation, real):
        arrays["checkpoint_step"] = np.asarray(step, dtype=np.int64)
        arrays["prior_label"] = np.asarray("VAE + DiT prior")

    # Validate all inputs before creating output files. Only display arrays are
    # copied, so old reconstruction traces/weight metadata cannot be misattributed.
    metadata = {
        "checkpoint_step": step,
        "prior_label": (f"VAE + DiT prior\nstep {step:,}"
                        + ("\nMOCK: layout only" if mock else "")),
        "verification_mock": mock,
        "results_npz": str(results_path),
        "results_npz_sha256": _sha256(results_path),
        "reference_dir": str(reference_dir),
        "reference_sha256": {
            name: _sha256(reference_dir / name) for name in ("simulation.npz", "real.npz")
        },
        "render_source_sha256": _sha256(Path(__file__).resolve()),
        "reference_renderer_sha256": _sha256(Path(reference_renderer.__file__)),
        "unchanged_rows": ["target (simulation)", "degraded", "phase_inva", "tikhonov"],
        "display": "Fixed [0,1] grayscale; real arrays already rotated 180 degrees",
        "simulation_grid": [5, 8],
        "real_grid": [4, 10],
        "dpi": 300,
        "simulation_metrics": "PSNR / SSIM; white bold Times New Roman; original reference convention",
        "real_metrics": "No paired HR ground truth; no PSNR/SSIM",
        "outputs": {name: str(out / name) for name in output_names},
    }
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "simulation.npz", **simulation)
    np.savez_compressed(out / "real.npz", **real)
    old_label = reference_renderer.ROW_LABELS["diffusion"]
    reference_renderer.ROW_LABELS["diffusion"] = metadata["prior_label"]
    try:
        reference_renderer.render_simulation(out / "simulation.npz", out / "figure1_simulation")
        reference_renderer.render_real(out / "real.npz", out / "figure2_real")
    finally:
        reference_renderer.ROW_LABELS["diffusion"] = old_label
    (out / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata["outputs"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reference-dir", type=Path, required=True,
                        help="Approved figure array directory containing simulation.npz and real.npz")
    parser.add_argument("--results", type=Path, required=True, help="New reconstruction result NPZ")
    parser.add_argument("--out", type=Path, required=True, help="New output directory under runs")
    parser.add_argument("--step", type=int, help="Assert the checkpoint step stored in --results")
    args = parser.parse_args()
    outputs = render_results(args.reference_dir, args.results, args.out, args.step)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
