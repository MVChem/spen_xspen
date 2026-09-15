"""Coordinate, scaling, selection and unusual-filename regression checks."""

import hashlib
import json

import nibabel as nib
import numpy as np
import pytest

from nifti_sources import discover_nifti, load_nifti


def _write(path, values=None, affine=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if values is None:
        values = np.arange(4 * 5 * 14, dtype=np.int16).reshape(4, 5, 14)
    if affine is None:
        affine = np.array([[0.1, 0, 0, 3], [0, 0, 0.9, 4], [0, -0.2, 0, 5], [0, 0, 0, 1]])
    image = nib.Nifti1Image(values, affine)
    image.header.set_xyzt_units("mm")
    image.header.set_slope_inter(2, 7)
    nib.save(image, path)
    return values


def test_load_preserves_source_indices_and_applies_nifti_scaling(tmp_path):
    path = tmp_path / "example.nii.gz"
    original = _write(path)
    stack, meta = load_nifti({"path": str(path), "slice_start": 3, "slice_stop": 7})
    assert stack.dtype == np.float32
    assert stack.shape == (14, 5, 4)
    assert stack[8, 3, 2] == original[2, 3, 8] * 2 + 7
    assert stack[0, 0, 0] == 7
    assert meta["spacing_yx"] == pytest.approx([0.2, 0.1])
    assert meta["native_axis_codes"] == ["R", "I", "A"]
    assert meta["source_hash"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_author_gzip_filename_is_loaded_without_renaming(tmp_path):
    original_path = tmp_path / "normal.nii.gz"
    original = _write(original_path)
    unusual = tmp_path / "sub-26_T2W.gz"
    unusual.write_bytes(original_path.read_bytes())
    stack, _ = load_nifti({"path": str(unusual)})
    assert stack[11, 2, 1] == original[1, 2, 11] * 2 + 7
    assert unusual.is_file()
    assert not (tmp_path / "sub-26_T2W.nii.gz").exists()


def test_aging_display_rotation_preserves_slice_indices_and_pixel_values(tmp_path):
    path = tmp_path / "aging.nii.gz"
    original = _write(path)
    stack, meta = load_nifti({"path": str(path), "dataset": "figshare_aging_28433102"})
    assert stack[8, 1, 0] == original[3, 3, 8] * 2 + 7
    assert stack[0, 4, 3] == original[0, 0, 0] * 2 + 7
    assert meta["inplane_rotation_degrees"] == 180
    assert "source slice order unchanged" in meta["array_transform"]
    assert meta["spacing_yx"] == pytest.approx([0.2, 0.1])


def test_discovery_follows_source_symlinks_and_keeps_animal_groups(tmp_path):
    public = tmp_path / "rodent_mri" / "public_rodent_mri"
    target = tmp_path / "original" / "sub-01"
    _write(target / "anat" / "sub-01_T2w.nii.gz")
    directory = public / "ds005236"
    directory.mkdir(parents=True)
    (directory / "sub-01").symlink_to(target, target_is_directory=True)
    _write(public / "ds005186" / "sub-02" / "anat" / "sub-02_T2w.nii.gz")
    _write(public / "figshare_aging_28433102" / "sub-01" / "dwi" / "sub-01_dwi.nii.gz")
    _write(public / "zenodo6844489" / "label" / "mask.nii.gz")
    audit = public / "_provenance" / "cross_dataset_subject_mapping.json"
    audit.parent.mkdir()
    audit.write_text(json.dumps({"mappings": [
        {"source_subject": "ds005186:sub-02", "canonical_acquisition_subject": "Farmer_same"}
    ], "adult_identity_records": [
        {"source_subject": "ds005236:sub-01", "canonical_acquisition_subject": "Farmer_same"}
    ]}))
    sources, excluded = discover_nifti(public.parent)
    assert len(sources) == 2
    assert {item["subject_group"] for item in sources} == {"Farmer_same"}
    assert {item["sequence"] for item in sources} == {"T2w-TurboRARE"}
    assert len(excluded) == 2
    assert {item["dataset"] for item in excluded} == {"figshare_aging_28433102", "zenodo6844489"}


def test_four_dimensional_t2_cannot_silently_become_more_slices(tmp_path):
    public = tmp_path / "public_rodent_mri"
    path = public / "ds005236" / "sub-01" / "anat" / "sub-01_T2w.nii.gz"
    _write(path, np.zeros((4, 5, 14, 3), dtype=np.int16))
    sources, excluded = discover_nifti(public)
    assert not sources
    assert "scalar 3D" in excluded[0]["reason"]
    with pytest.raises(ValueError, match="scalar 3D"):
        load_nifti({"path": str(path)})


def test_aging_author_filename_variants_are_discovered(tmp_path):
    public = tmp_path / "public_rodent_mri"
    dataset = public / "figshare_aging_28433102"
    _write(dataset / "sub-27" / "anat" / "sub-27--待处理_T2_TurboRARE.nii.gz")
    path = dataset / "sub-26" / "anat" / "sub-26_T2W.nii.gz"
    _write(path)
    path.rename(path.with_name("sub-26_T2W.gz"))
    unusual = dataset / "sub-27" / "anat" / "sub-27--待处理_T2_TurboRARE.nii.gz"
    digest = hashlib.sha256(unusual.read_bytes()).hexdigest()
    (public / "provenance_260914.json").write_text(json.dumps({"files": [{
        "local_relative_path": unusual.relative_to(public).as_posix(),
        "role": "published_mri_image", "dataset_id": "figshare_aging_28433102", "sha256": digest,
    }]}))
    sources, excluded = discover_nifti(public)
    assert len(sources) == 2
    assert not excluded
    assert {source["subject"] for source in sources} == {"sub-26", "sub-27"}
    marked = next(source for source in sources if source["subject"] == "sub-27")
    assert marked["expected_sha256"] == digest
    assert marked["source_filename_contains_pending_marker"]


def test_pending_marker_without_verified_original_is_excluded(tmp_path):
    public = tmp_path / "public_rodent_mri"
    _write(public / "figshare_aging_28433102" / "sub-27" / "anat" / "sub-27--待处理_T2_TurboRARE.nii.gz")
    sources, excluded = discover_nifti(public)
    assert not sources
    assert "pending marker" in excluded[0]["reason"]


def test_multiple_acquisitions_in_one_subject_have_distinct_ids(tmp_path):
    public = tmp_path / "public_rodent_mri"
    anat = public / "ds005236" / "sub-01" / "anat"
    _write(anat / "sub-01_acq-1_T2w.nii.gz")
    _write(anat / "sub-01_acq-2_T2w.nii.gz")
    sources, excluded = discover_nifti(public)
    assert not excluded
    assert len({source["source_id"] for source in sources}) == 2
    assert len({source["subject_group"] for source in sources}) == 1


def test_sheared_in_plane_grid_requires_explicit_resampling(tmp_path):
    path = tmp_path / "sheared.nii.gz"
    affine = np.diag([0.1, 0.2, 0.5, 1.0])
    affine[0, 1] = 0.08
    _write(path, affine=affine)
    with pytest.raises(ValueError, match="In-plane shear"):
        load_nifti({"path": str(path)})


def test_rat_rare_discovery_keeps_species_native_plane_and_thick_slice_window(tmp_path):
    public = tmp_path / "public_rodent_mri"
    path = public / "ds002870/sub-001/ses-1/anat/sub-001_ses-1_acq-RARE_T2w.nii.gz"
    affine = np.array([[0.1, 0, 0, 3], [0, 0, 1, 4], [0, -0.1, 0, 5], [0, 0, 0, 1]])
    original = np.arange(256 * 256 * 12, dtype=np.int32).reshape(256, 256, 12)
    _write(path, original, affine)

    sources, excluded = discover_nifti(public)
    assert not excluded
    assert len(sources) == 1
    source = sources[0]
    assert source["species"] == "rat"
    assert source["sequence"] == "T2w-RARE"
    assert source["native_shape"] == [256, 256, 12]
    assert (source["slice_start"], source["slice_stop"]) == (1, 11)
    assert source["native_axis_codes"] == ["R", "I", "A"]
    assert source["inplane_rotation_degrees"] == 0
    stack, meta = load_nifti(source)
    assert stack.shape == (12, 256, 256)
    assert stack[8, 3, 2] == original[2, 3, 8] * 2 + 7
    assert meta["native_spacing_xyz"] == pytest.approx([0.1, 0.1, 1])


@pytest.mark.parametrize("affine, expected_window", [
    (np.diag([0.2, -0.2, -0.2, 1.0]), (12, 46)),
    (np.diag([0.2, 0.2, 0.2, 1.0]), (18, 52)),
])
def test_rat_low_matrix_family_keeps_its_native_plane_metadata(tmp_path, affine, expected_window):
    public = tmp_path / "public_rodent_mri"
    _write(public / "ds002870/sub-074/ses-1/anat/sub-074_ses-1_acq-RARE_T2w.nii.gz",
           np.zeros((144, 144, 64), dtype=np.int16), affine)
    sources, excluded = discover_nifti(public)
    assert not excluded
    source = sources[0]
    assert source["species"] == "rat"
    assert (source["slice_start"], source["slice_stop"]) == expected_window
    assert "below this batch's 192x192 minimum" in source["orientation_note"]
    assert source["inplane_rotation_degrees"] == 0
