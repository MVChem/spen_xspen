"""Packet-order, geometry, and real-collection checks for segmented reading."""
from pathlib import Path
import json

import numpy as np
import pytest

from segmented_raw_reader import read_segmented_raw

PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT.parent / "data/spen_acquired_260915"


def _fixture(path, segments, echoes=1, coils=2, slices=3, volumes=2,
             declared_pe=None, acquired_ro=6, declared_ro=6):
    """Serialize explicit nested vendor acquisition loops and labelled samples."""
    rows = 4
    pe = rows * segments
    order = list(range(slices))[::-1]
    declared_pe = pe if declared_pe is None else declared_pe
    pairs = []
    expected = np.empty((acquired_ro, pe, slices, volumes, coils, echoes), np.complex128)
    for volume in range(volumes):
        for shot in range(segments):
            for acquired_slice in range(slices):
                for echo in range(echoes):
                    for coil in range(coils):
                        for line in range(rows):
                            global_pe = shot + segments * line
                            for ro in range(acquired_ro):
                                value = (1 + ro + 10*line + 100*coil + 1000*echo
                                         + 10000*acquired_slice + 100000*shot + 1000000*volume)
                                pairs += [value, -value - 7]
                                reflected = global_pe % 2 == 1 if segments % 2 else line % 2 == 0
                                sorted_ro = acquired_ro - 1 - ro if reflected else ro
                                expected[sorted_ro, global_pe, order[acquired_slice], volume, coil, echo] = complex(value, -value - 7)
    np.asarray(pairs, dtype="<i4").tofile(path / "fid")
    (path / "method").write_text(f"""##$PVM_Matrix=( 2 )
{declared_ro} {declared_pe}
##$PVM_EncMatrix=( 2 )
{declared_ro} {pe}
##$PVM_EpiMatrix=( 4 )
{acquired_ro} {pe} 0 0
##$PVM_EpiNShots={segments}
##$PVM_EpiNEchoes={rows}
##$PVM_EpiNSamplesPerScan={acquired_ro * rows}
##$PVM_EpiPrefixNavSize=0
##$PVM_EncNReceivers={coils}
##$PVM_NEchoImages={echoes}
##$PVM_SPackArrNSlices=( 1 )
{slices}
##$PVM_ObjOrderList=( {slices} )
{' '.join(map(str, order))}
##$NSegments={segments}
""")
    (path / "acqp").write_text("##$ACQ_word_size=_32_BIT\n##$BYTORDA=little\n##$GO_block_size=continuous\n##$Unused=0\n")
    return {"matrix_ro_pe": [declared_ro, declared_pe], "n_segments": segments,
            "coils": coils, "slices": slices, "volumes": volumes, "echoes": echoes}, expected


@pytest.mark.parametrize("segments,echoes", [(3, 1), (4, 1), (5, 1), (4, 2), (5, 2)])
def test_all_frame_axes_and_reflected_lines_survive_packet_sorting(tmp_path, segments, echoes):
    params, expected = _fixture(tmp_path, segments, echoes=echoes)
    actual, info = read_segmented_raw(tmp_path, params)
    np.testing.assert_array_equal(actual, expected)
    assert info["dimension_correction"] is None
    assert info["padding"]["unaccounted_source_bytes"] == 0


def test_singleton_receiver_echo_slice_dimensions_are_not_squeezed(tmp_path):
    params, expected = _fixture(tmp_path, 5, coils=1, slices=1, volumes=1)
    actual, _ = read_segmented_raw(tmp_path, params)
    assert actual.shape == (6, 20, 1, 1, 1, 1)
    np.testing.assert_array_equal(actual, expected)


def test_declared_but_unacquired_lines_are_not_invented(tmp_path):
    params, expected = _fixture(tmp_path, 5, declared_pe=21,
                                acquired_ro=6, declared_ro=8)
    original_params = dict(params)
    actual, info = read_segmented_raw(tmp_path, params)
    np.testing.assert_array_equal(actual, expected)
    assert info["declared_matrix_ro_pe"] == [8, 21]
    assert info["acquired_matrix_ro_pe"] == [6, 20]
    assert info["dimension_correction"]["acquired_pe"] == 20
    assert params == original_params


def test_truncated_or_padded_file_is_rejected(tmp_path):
    params, _ = _fixture(tmp_path, 5)
    with (tmp_path / "fid").open("ab") as stream:
        stream.write(b"\0\0\0\0")
    with pytest.raises(ValueError, match="Exact raw byte count mismatch"):
        read_segmented_raw(tmp_path, params)


def test_conflicting_encoding_geometry_is_rejected(tmp_path):
    params, _ = _fixture(tmp_path, 5)
    text = (tmp_path / "method").read_text().replace("##$PVM_EncMatrix=( 2 )\n6 20", "##$PVM_EncMatrix=( 2 )\n6 21")
    (tmp_path / "method").write_text(text)
    with pytest.raises(ValueError, match=r"PVM_EncMatrix\[1\] conflicts"):
        read_segmented_raw(tmp_path, params)


def _real_segmented_records():
    manifest = DATA / "raw_manifest.json"
    if not manifest.exists():
        return []
    # The source inventory is independent of run outputs and tmp snapshots.
    import sys
    sys.path.insert(0, str(PROJECT.parent / "spenpy"))
    from spenpy._legacy.bruker.param import read_pv_param
    records = []
    for study in json.loads(manifest.read_text())["experiments"]:
        for scan in study["scans"]:
            if scan["classification"] not in ("spen_imaging", "xspen") or scan["raw_status"] != "nonempty":
                continue
            path = DATA / scan["destination_relative_path"]
            p = lambda name: read_pv_param(str(path), name)
            segments = p("NSegments") or 1
            if segments < 2:
                continue
            diffusion = p("PVM_DwNDiffExp") or p("DwNDiffExp") or 1
            params = {"matrix_ro_pe": p("PVM_Matrix"), "n_segments": int(segments),
                      "slices": int(np.sum(p("PVM_SPackArrNSlices") or 1)),
                      "volumes": int(diffusion) * int(p("PVM_NRepetitions") or 1),
                      "coils": int(p("PVM_EncNReceivers") or 1),
                      "echoes": int(p("PVM_NEchoImages") or 1)}
            records.append((path, params))
    return records


@pytest.mark.parametrize("scan_dir,params", _real_segmented_records(),
                         ids=lambda value: value.name if isinstance(value, Path) else None)
def test_each_real_segmented_scan_keeps_all_samples_and_frames(scan_dir, params):
    actual, info = read_segmented_raw(scan_dir, params)
    assert actual.shape[2:] == tuple(params[k] for k in ("slices", "volumes", "coils", "echoes"))
    source = np.fromfile(info["raw_payload_path"], dtype="<i4").astype(np.float64)
    assert actual.size * 2 == source.size
    # A reorder/reflection may change locations, never raw component values.
    # Exact sorted components avoid reduction-roundoff in large signal power.
    np.testing.assert_array_equal(np.sort(actual.real.reshape(-1)), np.sort(source[0::2]))
    np.testing.assert_array_equal(np.sort(actual.imag.reshape(-1)), np.sort(source[1::2]))
    assert np.count_nonzero(actual) > 0
    if params["echoes"] > 1:
        from raw_frame_core import _read_segmented_multiecho
        np.testing.assert_array_equal(actual, _read_segmented_multiecho(scan_dir, params))
    if params["matrix_ro_pe"] == [96, 96] and params["n_segments"] == 5:
        assert info["acquired_matrix_ro_pe"] == [96, 95]
        assert info["source_parameters"]["PVM_EpiNEchoes"] == 19
        assert info["source_parameters"]["PVM_EpiNSamplesPerScan"] == 1824
