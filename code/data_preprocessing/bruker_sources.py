"""Read the curated laboratory brain RARE reconstructions without altering them.

Only the provenance inventory's brain candidates from RAT and data4_M0427
enter the first batch. Echoes are separate sources. Array axes are native
scanner [slice, row, column]; no anatomical reorientation is inferred.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np


_DTYPES = {"_16BIT_SGN_INT": "i2", "_32BIT_SGN_INT": "i4", "_32BIT_FLOAT": "f4"}
_FIRST_BATCH = {"RAT", "data4_M0427"}


def _params(path: Path) -> dict[str, str]:
    # Comments can occur within parameter arrays, not only between parameters.
    content = "\n".join(line.split("$$", 1)[0] for line in path.read_text().splitlines())
    result = {}
    for match in re.finditer(r"^##\$(\w+)=(.*?)(?=^##|\Z)", content, re.M | re.S):
        value = match.group(2).strip()
        value = re.sub(r"^\(\s*\d+(?:\s*,\s*\d+)*\s*\)\s*", "", value)
        value = re.sub(r"@(\d+)\*\(([^()]*)\)", lambda m: " ".join([m[2]] * int(m[1])), value)
        result[match[1]] = value.strip()
    return result


def _numbers(params: dict, key: str, *, size: int | None = None) -> np.ndarray:
    try:
        value = np.array([float(v) for v in params[key].split()], dtype=np.float64)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"Missing or non-numeric {key}") from exc
    if not value.size or not np.isfinite(value).all() or (size is not None and value.size != size):
        raise ValueError(f"Invalid {key}: expected {size or 'nonempty'} finite values")
    return value


def _integer(params: dict, key: str) -> int:
    value = _numbers(params, key, size=1)[0]
    if value != int(value) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return int(value)


def _metadata(scan: Path) -> dict:
    visu = _params(scan / "pdata/1/visu_pars")
    method = _params(scan / "method")
    if _integer(visu, "VisuCoreDim") != 2 or visu.get("VisuCoreFrameType") != "MAGNITUDE_IMAGE":
        raise ValueError("Only 2D magnitude reconstructions are supported")
    dims = _numbers(visu, "VisuCoreSize", size=2)
    if (dims < 2).any() or not np.equal(dims, np.floor(dims)).all():
        raise ValueError("Invalid integer in-plane dimensions")
    width, height = dims.astype(int)
    frames = _integer(visu, "VisuCoreFrameCount")
    word_type = visu.get("VisuCoreWordType")
    if word_type not in _DTYPES or visu.get("VisuCoreByteOrder") not in {"littleEndian", "bigEndian"}:
        raise ValueError("Unsupported word type or byte order")
    dtype = np.dtype(("<" if visu["VisuCoreByteOrder"] == "littleEndian" else ">") + _DTYPES[word_type])
    expected = frames * int(width) * int(height) * dtype.itemsize
    actual = (scan / "pdata/1/2dseq").stat().st_size
    if actual != expected:
        raise ValueError(f"2dseq size mismatch: {actual} bytes, expected {expected}")
    groups = [(int(n), name) for n, name in re.findall(r"\(\s*(\d+)\s*,\s*<(FG_\w+)>", visu.get("VisuFGOrderDesc", ""))]
    if not groups or any(n < 1 for n, _ in groups) or int(np.prod([n for n, _ in groups])) != frames:
        raise ValueError("Frame groups do not match frame count")
    names = [name for _, name in groups]
    if names not in [["FG_SLICE"], ["FG_ECHO", "FG_SLICE"]]:
        raise ValueError(f"Unsupported frame groups: {names}")
    echoes = groups[0][0] if names[0] == "FG_ECHO" else 1
    slices = groups[-1][0]
    tes = _numbers(method, "EffectiveTE", size=echoes)
    if (tes <= 0).any():
        raise ValueError("Echo times must be positive")
    # This rejects repeated images or multiple packages masquerading as slices.
    orientations = _numbers(visu, "VisuCoreOrientation").reshape(-1, 3, 3)
    positions = _numbers(visu, "VisuCorePosition").reshape(-1, 3)
    if len(orientations) not in {1, slices} or len(positions) != slices:
        raise ValueError("Geometry count does not match slice group")
    ori = orientations[0]
    if not np.allclose(orientations, ori, atol=1e-4) or not np.allclose(ori @ ori.T, np.eye(3), atol=1e-4):
        raise ValueError("Varying or non-orthonormal slice orientations")
    spacing_z = None
    if slices > 1:
        delta = np.diff(positions, axis=0)
        projections = delta @ ori[2]
        if (not np.allclose(delta, projections[:, None] * ori[2], atol=1e-3)
                or not np.allclose(projections, projections[0], atol=1e-3)
                or abs(projections[0]) < 1e-6):
            raise ValueError("Nonuniform or multiple-package slice positions")
        spacing_z = float(abs(projections[0]))
    extent = _numbers(visu, "VisuCoreExtent", size=2)
    if (extent <= 0).any():
        raise ValueError("FOV must be positive")
    slopes = _numbers(visu, "VisuCoreDataSlope")
    offsets = _numbers(visu, "VisuCoreDataOffs")
    if slopes.size not in {1, frames} or offsets.size not in {1, frames} or (slopes <= 0).any():
        raise ValueError("Invalid per-frame slope or offset")
    disk_order = visu.get("VisuCoreDiskSliceOrder", "")
    if disk_order and "reverse" in disk_order.lower():
        raise ValueError("Reverse disk slice order needs explicit geometry handling")
    return {
        "shape": (slices, int(height), int(width)), "frames": frames, "echoes": echoes,
        "tes": tes, "dtype": dtype, "slopes": np.broadcast_to(slopes, (frames,)),
        "offsets": np.broadcast_to(offsets, (frames,)),
        "spacing_yx": [float(extent[1] / height), float(extent[0] / width)],
        "slice_spacing_mm": spacing_z, "fov_xy_mm": extent.tolist(),
        "orientation": ori.tolist(), "positions": positions.tolist(),
        "frame_groups": groups,
    }


def discover_bruker(root: Path) -> tuple[list[dict], list[dict]]:
    """Find deduplicated laboratory sources and report every excluded record.

    ``root`` is the rodent_mri directory (or its lab_mouse child).
    Missing/unsupported data is reported, never silently reconstructed.
    """
    root = Path(root).resolve()
    lab = root if root.name == "lab_mouse" else root / "lab_mouse"
    inventory_path = lab / "provenance/source_inventory_260914.json"
    inventory = json.loads(inventory_path.read_text())
    sources, skipped = [], []
    seen_hashes = set()
    for record in sorted(inventory["records"], key=lambda r: r["scan_path"]):
        remote = record["scan_path"]
        reason = None
        if record["category"] != "brain_candidate":
            reason = "Not a brain candidate (body/tumor/CEST)"
        elif record["source"] not in _FIRST_BATCH:
            reason = "Excluded from first batch: motion, limited coverage, or incompletely screened hs_brain"
        elif record.get("sequence") != "RARE":
            reason = "Sequence is not ordinary RARE"
        if reason:
            skipped.append({"dataset": "lab_mouse", "path": remote, "reason": reason})
            continue
        digest = record["reconstruction_sha256"]
        if digest in seen_hashes:
            skipped.append({"dataset": "lab_mouse", "path": remote, "reason": "Duplicate primary reconstruction SHA256"})
            continue
        scan = lab / "old_server" / remote.lstrip("/")
        try:
            meta = _metadata(scan)
            if not np.allclose(meta["tes"], record["effective_TE_ms"], atol=1e-6):
                raise ValueError("Method echo times differ from provenance inventory")
        except (OSError, ValueError, KeyError) as exc:
            skipped.append({"dataset": "lab_mouse", "path": str(scan), "reason": str(exc)})
            continue
        seen_hashes.add(digest)
        subject_path = scan.parent / "subject"
        subject_params = _params(subject_path) if subject_path.exists() else {}
        subject = subject_params.get("SUBJECT_id", scan.parent.name).strip("<>")
        for echo_index, te in enumerate(meta["tes"]):
            sources.append({
                "source_id": f"lab_{digest[:12]}_e{echo_index + 1}",
                "dataset": "lab_mouse", "collection": record["source"],
                "subject": subject, "study": scan.parent.name,
                "species": "rodent_unspecified",
                "species_note": "Rodent brain provenance; scanner SUBJECT_type is a generic Human placeholder",
                "sequence": f"RARE_TE{float(te):g}ms",
                "native_plane_shape": list(meta["shape"][1:]),
                "spacing_yx": meta["spacing_yx"],
                "path": str(scan / "pdata/1/2dseq"), "scan_path": str(scan),
                "echo_index": echo_index, "echo_time_ms": float(te),
                "reconstruction_sha256": digest,
                "protocol": record.get("protocol", ""),
                "provenance_path": str(inventory_path),
                "source_aliases": record.get("alias_scan_paths", [remote]),
                "quality_note": record.get("notes", ""),
            })
    return sources, skipped


def load_bruker(source: dict) -> tuple[np.ndarray, dict]:
    """Decode one echo as float32 [slice,row,col], preserving native ordering."""
    scan = Path(source["scan_path"])
    metadata = _metadata(scan)
    binary = (scan / "pdata/1/2dseq").read_bytes()
    expected_hash = source.get("reconstruction_sha256")
    if expected_hash and hashlib.sha256(binary).hexdigest() != expected_hash:
        raise ValueError("Primary reconstruction SHA256 differs from inventory")
    raw = np.frombuffer(binary, dtype=metadata["dtype"])
    _, height, width = metadata["shape"]
    raw = raw.reshape(metadata["frames"], height, width)
    echo_index = int(source.get("echo_index", 0))
    if not 0 <= echo_index < metadata["echoes"]:
        raise ValueError("Echo index outside frame group")
    indices = np.arange(echo_index, metadata["frames"], metadata["echoes"])
    volume = raw[indices].astype(np.float64)
    volume *= metadata["slopes"][indices, None, None]
    volume += metadata["offsets"][indices, None, None]
    if not np.isfinite(volume).all() or np.max(np.abs(volume)) > np.finfo(np.float32).max:
        raise ValueError("Scaled reconstruction cannot be represented as finite float32")
    meta = {
        "source_shape": list(metadata["shape"]),
        "spacing_yx": metadata["spacing_yx"],
        "pixel_spacing_mm": metadata["spacing_yx"],
        "slice_spacing_mm": metadata["slice_spacing_mm"],
        "fov_xy_mm": metadata["fov_xy_mm"],
        "orientation_native": metadata["orientation"],
        "slice_positions_native_mm": metadata["positions"],
        "orientation_note": "Native Bruker rows/columns; no rotation, transpose, or flip",
        "array_axis_order": ["slice", "row", "column"],
        "source_frame_indices": indices.tolist(),
        "echo_index": echo_index, "echo_time_ms": float(metadata["tes"][echo_index]),
        "frame_groups": metadata["frame_groups"],
        "scaling": "physical_value = stored_value * VisuCoreDataSlope + VisuCoreDataOffs",
        "frame_slopes": metadata["slopes"][indices].tolist(),
        "frame_offsets": metadata["offsets"][indices].tolist(),
    }
    return volume.astype(np.float32), meta
