"""Guard the image-only expansion against held-out leakage and bit-depth loss."""
import numpy as np
from PIL import Image
import pytest

import expand_rodent96 as exporter


def record(dataset, subject, name="old.nii.gz"):
    return dict(dataset=dataset, subject=f"{dataset}:{subject}",
                source=f"/legacy/{name}", view="mouse_fov16")


def test_cross_dataset_identity_and_comr_family_protect_holdout():
    manifest = {"records": {"train": [], "val": [record("ds005236", "sub-27")],
                            "test": [record("ds006663", "sub-COMR221a")]}}
    identities = {"ds005236:sub-27": "Farmer_1588_2f", "ds005186:sub-02": "Farmer_1588_2f"}
    held_out, _, _ = exporter.legacy_index(manifest, identities)
    assert exporter.canonical_group("ds005186", "sub-02", identities) in held_out
    assert exporter.canonical_group("ds006663", "sub-COMR221f", identities) in held_out
    manifest["records"]["train"].append(record("ds005186", "sub-02"))
    with pytest.raises(ValueError, match="span partitions"):
        exporter.legacy_index(manifest, identities)


def test_selection_keeps_low_matrix_lab_and_excludes_reference_and_old_volumes(tmp_path, monkeypatch):
    manifest = {"records": {"train": [record("ds005236", "sub-01", "existing.nii.gz")],
                            "val": [record("ds005236", "sub-27")], "test": []}}
    lab = dict(dataset="lab_mouse", native_plane_shape=[120, 160], subject="lab-animal",
               study="study1", reconstruction_sha256="a" * 64, echo_index=0)
    reference = dict(lab, subject="20180210_mouse_M0427", reconstruction_sha256="b" * 64)
    ghosted = dict(lab, reconstruction_sha256=next(iter(exporter.VISUAL_EXCLUSIONS)))
    public = [dict(dataset="ds005186", subject="sub-02", native_shape=[256, 256, 14], path="heldout.nii.gz"),
              dict(dataset="ds005236", subject="sub-01", native_shape=[256, 256, 14], path="existing.nii.gz")]
    monkeypatch.setattr(exporter, "_identity_groups", lambda _: {
        "ds005236:sub-27": "shared", "ds005186:sub-02": "shared"})
    monkeypatch.setattr(exporter, "discover_bruker", lambda _: ([lab, reference, ghosted], []))
    monkeypatch.setattr(exporter, "discover_nifti", lambda _: (public, []))
    selected, skipped = exporter.select_sources(tmp_path, manifest)
    assert len(selected) == 1 and selected[0]["subject"] == "lab-animal"
    assert sum(skipped.values()) == 4


def test_lossless_legacy_export_and_duplicates_across_partitions(tmp_path):
    old, out = tmp_path / "old", tmp_path / "out"
    old.mkdir()
    manifest = {"records": {}}
    # Exercise values that would be clipped by PIL.convert('L') or an 8-bit cast.
    base = (np.arange(96 * 96, dtype=np.uint16).reshape(96, 96) * 7)
    for i, part in enumerate(exporter.PARTS):
        (out / part).mkdir(parents=True)
        np.save(old / f"{part}.npy", (base + i)[None])
        manifest["records"][part] = [record("ds005236", f"sub-0{i}")]
    seen = {}
    counts = exporter.export_legacy(old, out, manifest, seen)
    assert dict(counts) == {"train": 1, "val": 1, "test": 1}
    assert all(p.suffix == ".png" for p in out.rglob("*") if p.is_file())
    with Image.open(next((out / "val").iterdir())) as image:
        assert np.asarray(image).dtype == np.uint16
        np.testing.assert_array_equal(np.asarray(image), base + 1)
    destination = out / "train" / "duplicate_of_test.png"
    assert not exporter.write_png(destination, base + 2, seen, "train")
    assert not destination.exists()
