"""Read selected public rodent T2 volumes without reconstructing thick slices.

All indices refer to the original file's third axis.  No volume reorientation,
interpolation or intensity window is applied.  Aging images receive a documented
180-degree in-plane display rotation, established by visual source inspection.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np


# These are first-pass candidate windows, not anatomical masks or a QC verdict.
# The exporter may replace them after inspecting source-specific previews.
DATASETS = {
    "ds005236": ("T2w-TurboRARE", (3, 12)),
    "ds005186": ("T2w-TurboRARE", (3, 12)),
    "ds006663": ("T2w-RARE", (8, 25)),
    "ds002868": ("T2w-RARE", (8, 40)),
    "ds002870": ("T2w-RARE", (1, 11)),
    "figshare_aging_28433102": ("T2w-TurboRARE", (3, 13)),
}

EXCLUDED_DATASETS = {
    "ds004644": "FLASH/GRE and complex MP2RAGE/UTE components; outside this T2 batch",
    "figshare_tc1_wt_3258139": "Ex vivo GRE; outside this T2 batch",
    "zenodo5834507": "Analyze T2/MT channel ordering and image spacing remain unresolved",
    "zenodo6844489": "T2WI confirmed, but sequence and source-specific orientation await review",
    "zenodo6379879": "Stroke T2 partial download; requires separate pathology and orientation review",
}


def _public_root(root: Path) -> Path:
    root = Path(root).absolute()
    candidate = root / "public_rodent_mri"
    return candidate if candidate.is_dir() else root


def _image_files(root: Path):
    """Follow the project's source-directory symlinks, omitting provenance."""
    visited = set()
    for directory, subdirs, names in os.walk(root, followlinks=True):
        resolved = Path(directory).resolve()
        if resolved in visited:
            subdirs[:] = []
            continue
        visited.add(resolved)
        subdirs[:] = sorted(d for d in subdirs if d not in {"_provenance", "archives"})
        for name in sorted(names):
            if name.endswith((".nii", ".nii.gz", ".img")) or name == "sub-26_T2W.gz":
                yield Path(directory) / name


def _open_nifti(path: Path):
    if path.name == "sub-26_T2W.gz":
        # Author filename is a gzipped NIfTI without the usual .nii suffix.
        # FileHolder leaves gzip opening to nibabel and keeps ArrayProxy usable.
        return nib.Nifti1Image.from_file_map(
            {"image": nib.FileHolder(filename=str(path))}
        )
    return nib.load(str(path))


def _identity_groups(root: Path) -> dict[str, str]:
    path = root / "_provenance" / "cross_dataset_subject_mapping.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    records = data.get("mappings", []) + data.get("adult_identity_records", [])
    return {
        row["source_subject"]: row["canonical_acquisition_subject"]
        for row in records
        if row.get("source_subject") and row.get("canonical_acquisition_subject")
    }


def _provenance_images(root: Path) -> dict[str, dict]:
    path = root / "provenance_260914.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    return {
        row["local_relative_path"]: row
        for row in data.get("files", [])
        if row.get("role") == "published_mri_image" and row.get("local_relative_path")
    }


def _sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _valid_t2_filename(path: Path) -> bool:
    return path.name.endswith(("_T2w.nii.gz", "_T2W.nii.gz", "_T2_TurboRARE.nii.gz")) or path.name == "sub-26_T2W.gz"


def _geometry(image) -> tuple[np.ndarray, tuple[str, ...]]:
    affine = np.asarray(image.affine, dtype=float)
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise ValueError("Invalid affine")
    basis = affine[:3, :3]
    spacing = np.linalg.norm(basis, axis=0)
    if not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Invalid voxel spacing")
    # A general sheared in-plane lattice cannot be represented by spacing_yx.
    cosine = abs(float(np.dot(basis[:, 0], basis[:, 1]) / (spacing[0] * spacing[1])))
    if cosine > 1e-3:
        raise ValueError("In-plane shear needs an explicit resampling transform")
    return spacing, tuple(nib.aff2axcodes(affine))


def discover_nifti(root: Path) -> tuple[list[dict], list[dict]]:
    """Return supported T2 sources and image-file exclusion records.

    ``root`` may be rodent_mri/ or its public_rodent_mri/ child.  Discovery reads
    headers only.  Duplicate animal/timepoint scans remain separate sources but
    share ``subject_group`` when the existing identity audit provides a match.
    """
    root = _public_root(root)
    identities = _identity_groups(root)
    provenance = _provenance_images(root)
    sources, exclusions = [], []
    used_ids = set()
    for dataset_root in sorted(root.iterdir()):
        if not dataset_root.is_dir() or dataset_root.name.startswith("_"):
            continue
        dataset = dataset_root.name
        for path in _image_files(dataset_root):
            excluded = {"dataset": dataset, "path": str(path)}
            if dataset not in DATASETS:
                exclusions.append({**excluded, "reason": EXCLUDED_DATASETS.get(dataset, "Source not reviewed for this T2 batch")})
                continue
            if not _valid_t2_filename(path):
                exclusions.append({**excluded, "reason": "Not a selected structural T2 volume (for example DWI or mask)"})
                continue
            try:
                image = _open_nifti(path)
                if len(image.shape) != 3:
                    raise ValueError(f"Expected scalar 3D image, got {image.shape}")
                spacing, axis_codes = _geometry(image)
            except Exception as exc:
                exclusions.append({**excluded, "reason": f"Unreadable or unsupported NIfTI: {exc}"})
                continue
            subject = next((part for part in path.parts if part.startswith("sub-")), path.stem)
            session = next((part for part in path.parts if part.startswith("ses-")), "")
            sequence, (start, stop) = DATASETS[dataset]
            if dataset == "ds002870" and image.shape == (144, 144, 64):
                # A separate native horizontal stack, not 64 coronal slices.
                # The exporter's minimum native matrix rule excludes this
                # acquisition from the 192-pixel batch without upsampling it.
                start, stop = (18, 52) if axis_codes == ("R", "A", "S") else (12, 46)
            group_key = f"{dataset}:{subject}"
            group = identities.get(group_key, group_key)
            source_id = "_".join(part for part in (dataset, subject, session) if part)
            relative_path = path.relative_to(root).as_posix()
            if source_id in used_ids:
                source_id += "_" + hashlib.sha256(relative_path.encode()).hexdigest()[:12]
            record = provenance.get(relative_path)
            notes = []
            if "待处理" in path.name:
                actual_hash = _sha256(path)
                if record is None:
                    record = next((row for row in provenance.values() if row.get("dataset_id") == dataset and row.get("sha256") == actual_hash), None)
                if record is None or record.get("sha256") != actual_hash:
                    exclusions.append({**excluded, "reason": "Filename contains pending marker and no matching original-file SHA256 was verified"})
                    continue
                notes.append("Source filename contains 待处理; image bytes match the original-file SHA256 audit")
            used_ids.add(source_id)
            if dataset == "ds002870":
                notes.append("Rat RARE acquisition; preserve native third-axis planes and source slice order")
                if image.shape == (256, 256, 12):
                    notes.append("Native coronal/oblique-coronal 12-plane stack; 0-based candidate window [1,11) reviewed on representative source images")
                elif image.shape == (144, 144, 64):
                    notes.append("Native horizontal 64-plane stack; below this batch's 192x192 minimum native matrix")
                    if axis_codes == ("R", "A", "S"):
                        notes.append("Published sub-074 rows and slice direction differ from the other 144x144 acquisitions; no display flip or slice reorder applied")
                if int(image.header["sform_code"]) == 3:
                    notes.append("Published header uses Talairach transform code 3; affine and spacing retained as published")
            elif dataset == "ds002868":
                notes.append("Native third-axis planes; original acquisition plane not independently established")
                if int(image.header["sform_code"]) == 3:
                    notes.append("Published header uses Talairach transform code 3 and nonuniform spacing")
            elif dataset == "figshare_aging_28433102":
                notes.append("Native third-axis stack; in-plane 180-degree display rotation based on visual inspection; anatomical left/right not independently validated")
            else:
                notes.append("Native third-axis coronal/oblique-coronal stack; preserve thick-slice sampling")
            sources.append({
                "source_id": source_id,
                "dataset": dataset,
                "subject": subject,
                "session": session,
                "subject_group": group,
                "species": "rat" if dataset == "ds002870" else "mouse",
                "sequence": sequence,
                "path": str(path),
                "expected_sha256": record.get("sha256") if record else None,
                "expected_sha256_evidence": "public_rodent_mri/provenance_260914.json" if record else None,
                "source_filename_contains_pending_marker": "待处理" in path.name,
                "slice_axis": 2,
                "inplane_rotation_degrees": 180 if dataset == "figshare_aging_28433102" else 0,
                "slice_start": min(start, image.shape[2]),
                "slice_stop": min(stop, image.shape[2]),
                "candidate_window_note": "Conservative initial index window; not an anatomical mask or QC approval",
                "quality_note": "Later native slices have lower intensity and more visible noise under volume-level scaling; retained because inspected late slices still contain brain tissue" if dataset == "ds002868" else "",
                "native_shape": list(image.shape),
                "spacing_yx": [float(spacing[1]), float(spacing[0])],
                "native_axis_codes": list(axis_codes),
                "orientation_note": "; ".join(notes),
            })
    return sources, exclusions


def load_nifti(source: dict) -> tuple[np.ndarray, dict]:
    """Return all native slices as float32 [slice, row, column] and metadata.

    ``slice_start``/``slice_stop`` are deliberately not applied here: returned
    slice i always corresponds to source third-axis index i.  The exporter owns
    selection, masking, intensity scaling and 192-pixel resampling.
    """
    path = Path(source["path"])
    image = _open_nifti(path)
    if len(image.shape) != 3 or int(source.get("slice_axis", 2)) != 2:
        raise ValueError("Only native third-axis scalar 3D NIfTI images are supported")
    if np.issubdtype(image.get_data_dtype(), np.complexfloating):
        raise ValueError("Complex components require an explicit magnitude reconstruction")
    spacing, axis_codes = _geometry(image)
    data = image.get_fdata(dtype=np.float32)
    stack = data.transpose(2, 1, 0)
    rotation = int(source.get("inplane_rotation_degrees", 180 if source.get("dataset") == "figshare_aging_28433102" else 0))
    if rotation not in (0, 180):
        raise ValueError("Only reviewed 0/180-degree in-plane display rotations are supported")
    if rotation == 180:
        stack = stack[:, ::-1, ::-1]
    stack = np.ascontiguousarray(stack)
    metadata = {
        "spacing_yx": [float(spacing[1]), float(spacing[0])],
        "native_shape": list(image.shape),
        "native_spacing_xyz": [float(value) for value in spacing],
        "header_spacing_xyz": [float(value) for value in image.header.get_zooms()[:3]],
        "spatial_unit": image.header.get_xyzt_units()[0],
        "native_affine": image.affine.tolist(),
        "native_axis_codes": list(axis_codes),
        "qform_code": int(image.header["qform_code"]),
        "sform_code": int(image.header["sform_code"]),
        "native_dtype": str(image.get_data_dtype()),
        "nifti_scaling_slope": float(image.dataobj.slope),
        "nifti_scaling_intercept": float(image.dataobj.inter),
        "slice_axis": 2,
        "inplane_rotation_degrees": rotation,
        "array_transform": "transpose(2,1,0); " + ("flip rows and columns (180-degree in-plane rotation)" if rotation else "no flips") + "; source slice order unchanged",
        "orientation_note": source.get("orientation_note", "Native third-axis planes; no anatomical reorientation"),
        "quality_note": source.get("quality_note", ""),
        "source_hash": _sha256(path),
        "source_hash_algorithm": "sha256",
        "source_bytes": path.stat().st_size,
        "nonfinite_voxels": int(stack.size - np.isfinite(stack).sum()),
    }
    return stack, metadata
