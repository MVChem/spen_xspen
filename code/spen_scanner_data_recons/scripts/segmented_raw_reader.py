"""Read Bruker segmented packets without dropping singleton or frame axes.

``read_segmented_raw(scan_dir, params) -> (raw, info)`` returns complex128 data
with [RO, PE, slice, volume, coil, echo] axes. The acquired PE and RO dimensions
come from the EPI acquisition metadata and are checked against the exact raw
byte count. Declared image dimensions are not substituted for acquired lines.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

import numpy as np

SPENPY = Path(__file__).resolve().parents[2] / "spenpy"
if str(SPENPY) not in sys.path:
    sys.path.insert(0, str(SPENPY))
from spenpy._legacy.bruker.param import read_pv_param

AXES = ["RO", "PE", "slice", "volume", "coil", "echo"]


def _param(path, name):
    value = read_pv_param(str(path), name)
    return value.tolist() if isinstance(value, np.ndarray) else value


def _integer(value, name):
    values = np.asarray(value).reshape(-1) if value is not None else []
    if len(values) != 1 or not np.isfinite(float(values[0])) or int(values[0]) != float(values[0]) or int(values[0]) < 1:
        raise ValueError(f"{name} must be an explicit positive integer; got {value!r}")
    return int(values[0])


def _vector(value):
    return [] if value is None else np.asarray(value).reshape(-1).tolist()


def read_segmented_raw(scan_dir, params):
    """Unpack all single/multiple echoes for acquisitions with >1 segments.

    ``params`` supplies matrix_ro_pe, n_segments, slices, volumes, coils,
    echoes, as in raw_frame_core. It is never modified. Volumes may contain an
    independently validated correction made by the caller. No PE/RO zeros,
    frame truncation, or implicit byte padding are introduced by this reader.
    """
    scan_dir = Path(scan_dir).resolve()
    segments, slices, volumes, coils, echoes = (
        _integer(params[key], key) for key in
        ("n_segments", "slices", "volumes", "coils", "echoes"))
    if segments < 2:
        raise ValueError("This reader is for segmented acquisitions (NSegments > 1)")
    matrix = list(params["matrix_ro_pe"])
    if len(matrix) != 2:
        raise ValueError("Only two spatial encoding axes are supported")
    matrix = [_integer(value, "matrix_ro_pe") for value in matrix]
    names = ("PVM_Matrix", "PVM_EncMatrix", "PVM_EpiMatrix", "PVM_EpiNShots",
             "PVM_EpiNEchoes", "PVM_EpiNSamplesPerScan", "PVM_EpiPrefixNavSize",
             "PVM_EncNReceivers", "PVM_NEchoImages", "PVM_SPackArrNSlices",
             "PVM_ObjOrderList", "NSegments", "ACQ_size", "ACQ_jobs", "ACQ_jobs_size",
             "ACQ_word_size", "GO_raw_data_format", "GO_block_size", "BYTORDA")
    evidence = {name: _param(scan_dir, name) for name in names}
    if _integer(evidence["NSegments"], "NSegments") != segments:
        raise ValueError("Caller segment count differs from the acquisition")
    for name, expected in (("PVM_EncNReceivers", coils), ("PVM_NEchoImages", echoes),
                           ("PVM_EpiNShots", segments)):
        if evidence[name] is not None and _integer(evidence[name], name) != expected:
            raise ValueError(f"Caller dimensions disagree with {name}")
    stored_slices = _vector(evidence["PVM_SPackArrNSlices"])
    if stored_slices and sum(stored_slices) != slices:
        raise ValueError("Caller slice count differs from PVM_SPackArrNSlices")
    if evidence["PVM_EpiPrefixNavSize"] not in (None, 0):
        raise ValueError("Prefix navigator packets require their own explicit layout")

    rows_per_shot = _integer(evidence["PVM_EpiNEchoes"], "PVM_EpiNEchoes")
    samples_per_shot = _integer(evidence["PVM_EpiNSamplesPerScan"], "PVM_EpiNSamplesPerScan")
    if samples_per_shot % rows_per_shot:
        raise ValueError("EPI samples per shot are not divisible by EPI readout lines")
    acquired_ro = samples_per_shot // rows_per_shot
    acquired_pe = rows_per_shot * segments
    encoding = _vector(evidence["PVM_EncMatrix"])
    epi_matrix = _vector(evidence["PVM_EpiMatrix"])
    if len(encoding) < 2:
        raise ValueError("PVM_EncMatrix is required to independently verify acquired PE")
    if int(encoding[1]) != acquired_pe:
        raise ValueError("PVM_EncMatrix[1] conflicts with PVM_EpiNEchoes * NSegments")
    if epi_matrix and (len(epi_matrix) < 2 or int(epi_matrix[1]) != acquired_pe):
        raise ValueError("PVM_EpiMatrix[1] conflicts with the acquired PE lines")
    if epi_matrix and int(epi_matrix[0]) != acquired_ro:
        raise ValueError("PVM_EpiMatrix[0] conflicts with the acquired RO samples")

    # PV360 ACQ_size can be a stale adjustment header (e.g. 1024 x 1).
    # A job0 packet descriptor, when present, independently records 2*ADC
    # complex samples per shot. Do not replace EPI geometry with ACQ_size.
    job = evidence["ACQ_jobs"]
    job_first_word_count = None
    if job is not None and evidence["ACQ_jobs_size"] != 0:
        if evidence["ACQ_jobs_size"] not in (None, 1):
            raise ValueError("Multiple acquisition jobs require a job-specific layout")
        match = re.match(r"\s*\(\s*(\d+)\s*,", str(job))
        if match and "job0" in str(job).lower():
            job_first_word_count = int(match.group(1))
            if job_first_word_count != 2 * samples_per_shot:
                raise ValueError("ACQ job0 packet size conflicts with EPI samples per shot")

    word = str(evidence["ACQ_word_size"] or evidence["GO_raw_data_format"] or "").lower()
    if "32" not in word or "float" in word:
        raise ValueError(f"Unsupported/unrecorded raw word format {word!r}")
    endian = str(evidence["BYTORDA"] or "").lower()
    if endian not in ("little", "littleendian", "0"):
        raise ValueError(f"Unsupported/unrecorded byte order {endian!r}")
    payloads = [scan_dir / name for name in ("rawdata.job0", "fid")
                if (scan_dir / name).is_file() and (scan_dir / name).stat().st_size]
    if len(payloads) != 1:
        raise ValueError("Expected exactly one nonempty rawdata.job0/fid payload")
    raw_path = payloads[0]
    expected_bytes = 2 * acquired_ro * acquired_pe * slices * volumes * coils * echoes * 4
    actual_bytes = raw_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(f"Exact raw byte count mismatch: {actual_bytes} != {expected_bytes}; no padding or frame count is guessed")

    words = np.fromfile(raw_path, dtype="<i4")
    # MATLAB acquisition packet order, independently specified by the legacy
    # vendor readers. Echo remains between receiver and slice in the file.
    packed = words.reshape(2, acquired_ro, rows_per_shot, coils, echoes,
                           slices, segments, volumes, order="F")
    complex_packets = packed[0].astype(np.float64) + 1j * packed[1]
    packets = complex_packets.transpose(0, 1, 4, 5, 6, 2, 3)
    raw = np.empty((acquired_ro, acquired_pe, slices, volumes, coils, echoes), np.complex128)
    for shot in range(segments):
        block = packets[:, :, :, shot].copy()
        flip_start = 0 if segments % 2 == 0 or shot % 2 else 1
        block[:, flip_start::2] = block[::-1, flip_start::2].copy()
        raw[:, shot::segments] = block

    order = np.asarray(_vector(evidence["PVM_ObjOrderList"]), dtype=float)
    if order.size != slices or not np.array_equal(np.sort(order), np.arange(slices)):
        raise ValueError("PVM_ObjOrderList is not a complete slice permutation")
    reordered = np.empty_like(raw)
    reordered[:, :, order.astype(int)] = raw
    correction = None
    if [acquired_ro, acquired_pe] != matrix:
        correction = {
            "declared_matrix_ro_pe": matrix,
            "acquired_matrix_ro_pe": [acquired_ro, acquired_pe],
            "declared_pe": matrix[1], "acquired_pe": acquired_pe,
            "pe_lines_per_shot": rows_per_shot, "n_segments": segments,
            "reason": "Read acquisition geometry from PVM_EncMatrix / PVM_EpiMatrix and EPI packet counts instead of substituting PVM_Matrix",
            "evidence": {key: evidence[key] for key in ("PVM_EncMatrix", "PVM_EpiMatrix", "PVM_EpiNShots", "PVM_EpiNEchoes", "PVM_EpiNSamplesPerScan", "ACQ_jobs")},
            "pe_padding_added": 0, "ro_padding_added": 0,
            "note": "Retain all acquired lines; missing declared lines are not fabricated as measured samples",
        }
    info = {
        "reader": "metadata-validated Bruker segmented acquisition packets",
        "axes": AXES, "shape": list(reordered.shape), "raw_payload_path": str(raw_path),
        "raw_payload_bytes": actual_bytes, "expected_raw_payload_bytes": expected_bytes,
        "declared_matrix_ro_pe": matrix, "acquired_matrix_ro_pe": [acquired_ro, acquired_pe],
        "acquired_pe_lines": acquired_pe, "pe_lines_per_shot": rows_per_shot,
        "adc_samples_per_shot": samples_per_shot, "job0_int32_words_per_shot": job_first_word_count,
        "dimension_correction": correction, "source_parameters": evidence,
        "slice_order_applied": order.astype(int).tolist(),
        "padding": {"introduced_ro_samples": 0, "introduced_pe_lines": 0,
                    "unaccounted_source_bytes": 0, "explanation": "Every byte accounted for by acquisition packet dimensions; no synthetic lines or assumed file padding"},
        "reflection_rule": "even NSegments: reverse RO on even zero-based line-within-shot; odd NSegments: reverse when line-within-shot + shot is odd",
    }
    return reordered, info
