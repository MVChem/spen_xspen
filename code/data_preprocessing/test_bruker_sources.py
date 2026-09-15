"""Numerical checks for binary ordering, echo separation, and rejection rules."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bruker_sources import discover_bruker, load_bruker


def _scan(tmp_path: Path, *, endian="littleEndian") -> tuple[dict, np.ndarray]:
    scan = tmp_path / "scan"
    pdata = scan / "pdata/1"
    pdata.mkdir(parents=True)
    raw = np.arange(24, dtype=np.int16).reshape(4, 2, 3)
    raw.astype("<i2" if endian == "littleEndian" else ">i2").tofile(pdata / "2dseq")
    (scan / "method").write_text("##$EffectiveTE=( 2 )\n20 60\n##END=\n")
    (pdata / "visu_pars").write_text(f"""##$VisuCoreDim=2
##$VisuCoreFrameType=MAGNITUDE_IMAGE
##$VisuCoreSize=( 2 )
3 2
##$VisuCoreFrameCount=4
##$VisuCoreWordType=_16BIT_SGN_INT
##$VisuCoreByteOrder={endian}
##$VisuFGOrderDesc=( 2 )
(2, <FG_ECHO>, <>, 0, 1) (2, <FG_SLICE>, <>, 1, 2)
##$VisuCoreOrientation=( 2, 9 )
1 0 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0 1
##$VisuCorePosition=( 2, 3 )
0 0 0 0 0 0.7
##$VisuCoreExtent=( 2 )
12 6
##$VisuCoreDataSlope=( 4 )
2 3 4 5
##$VisuCoreDataOffs=( 4 )
10 11 12 13
##END=
""")
    return {"scan_path": str(scan), "echo_index": 1}, raw


@pytest.mark.parametrize("endian", ["littleEndian", "bigEndian"])
def test_decode_echo_scaling_and_geometry(tmp_path, endian):
    source, raw = _scan(tmp_path, endian=endian)
    volume, meta = load_bruker(source)
    expected = raw[[1, 3]] * np.array([3, 5])[:, None, None] + np.array([11, 13])[:, None, None]
    np.testing.assert_array_equal(volume, expected)
    assert volume.dtype == np.float32
    assert volume.shape == (2, 2, 3)
    assert meta["spacing_yx"] == [3, 4]
    assert meta["source_frame_indices"] == [1, 3]
    assert meta["echo_time_ms"] == 60
    assert meta["slice_spacing_mm"] == pytest.approx(0.7)


def test_rejects_truncated_binary(tmp_path):
    source, _ = _scan(tmp_path)
    path = Path(source["scan_path"]) / "pdata/1/2dseq"
    path.write_bytes(path.read_bytes()[:-2])
    with pytest.raises(ValueError, match="size mismatch"):
        load_bruker(source)


def test_rejects_mixed_slice_directions(tmp_path):
    source, _ = _scan(tmp_path)
    path = Path(source["scan_path"]) / "pdata/1/visu_pars"
    path.write_text(path.read_text().replace("1 0 0 0 1 0 0 0 1 1 0 0 0 1 0 0 0 1", "1 0 0 0 1 0 0 0 1 1 0 0 0 0 1 0 1 0"))
    with pytest.raises(ValueError, match="orientations"):
        load_bruker(source)


def test_rejects_wrong_hash(tmp_path):
    source, _ = _scan(tmp_path)
    source["reconstruction_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        load_bruker(source)


def test_inventory_excludes_body_motion_and_duplicate_aliases(tmp_path):
    source, _ = _scan(tmp_path / "lab_mouse/old_server/home/example")
    digest = hashlib.sha256((Path(source["scan_path"]) / "pdata/1/2dseq").read_bytes()).hexdigest()
    record = {"scan_path": "/home/example/scan", "category": "brain_candidate", "source": "RAT", "sequence": "RARE", "reconstruction_sha256": digest, "effective_TE_ms": [20, 60]}
    records = [record, dict(record), dict(record, source="data1_motion"), dict(record, category="body_tumor")]
    provenance = tmp_path / "lab_mouse/provenance"
    provenance.mkdir()
    (provenance / "source_inventory_260914.json").write_text(json.dumps({"records": records}))
    sources, skipped = discover_bruker(tmp_path)
    assert len(sources) == 2  # Two echoes from exactly one reconstruction.
    assert len(skipped) == 3
    assert [s["echo_index"] for s in sources] == [0, 1]
    assert {s["sequence"] for s in sources} == {"RARE_TE20ms", "RARE_TE60ms"}
