"""Protect the native-resolution floor and tissue-preserving crop behavior."""

import json

import numpy as np
import pytest
from scipy.ndimage import label

from image_processing import fit_plane
import export_rodent_brain as exporter


@pytest.mark.parametrize("shape", [(128, 256), (256, 128), (191, 192), (192, 191)])
def test_foreground_mode_rejects_either_native_axis_below_192(shape):
    with pytest.raises(ValueError, match="native_matrix_below_output_size"):
        fit_plane(np.ones(shape, np.float32), [0.1, 0.1], 1, crop_mode="foreground")


def test_native_192_plane_retains_every_pixel_without_interpolation():
    y, x = np.mgrid[:192, :192]
    plane = np.where((y - 96) ** 2 + (x - 96) ** 2 < 65 ** 2,
                     100 + (13 * y + 7 * x) % 500, 3).astype(np.float32)
    output, info = fit_plane(plane, [0.1, 0.1], 1024, crop_mode="foreground")
    expected = np.rint(plane / 1024 * 65535).astype(np.uint16)
    np.testing.assert_array_equal(output, expected)
    assert info["direct_native_crop"]
    assert info["interpolation"] == "none; native integer crop"
    assert not info["upsampled"] and not info["added_padding"]
    assert info["offset_yx"] == [0, 0]
    assert info["sampling_yx"] == [1, 1]
    assert info["antialias_sigma_yx"] == [0, 0]


def test_offset_tissue_selects_an_offset_native_crop_and_preserves_tissue_pixels():
    y, x = np.mgrid[:320, :480]
    tissue = ((y - 140) / 43) ** 2 + ((x - 355) / 67) ** 2 <= 1
    plane = np.where(tissue, 100 + x % 25, 0).astype(np.float32)
    output, info = fit_plane(plane, [0.1, 0.1], 150, crop_mode="foreground")
    assert info["direct_native_crop"]
    assert info["crop_center_yx"] == pytest.approx([140, 355], abs=1)
    assert info["crop_center_yx"][1] > plane.shape[1] / 2 + 100
    assert np.count_nonzero(output) == np.count_nonzero(tissue)
    oy, ox = np.rint(info["offset_yx"]).astype(int)
    expected = np.rint(plane[oy:oy + 192, ox:ox + 192] / 150 * 65535).astype(np.uint16)
    np.testing.assert_array_equal(output, expected)
    yy, xx = np.nonzero(output)
    assert min(yy.min(), xx.min()) > 10
    assert max(yy.max(), xx.max()) < 182


def test_anisotropic_crop_keeps_physical_circle_round_without_padding_or_enlarging():
    y, x = np.mgrid[:384, :512]
    circle = ((y - 190) * 1.5) ** 2 + (x - 250) ** 2 <= 70 ** 2
    # Nonzero acquired background makes artificial zero padding observable.
    plane = np.where(circle, 100, 12).astype(np.float32)
    output, info = fit_plane(plane, [1.5, 1], 100, crop_mode="foreground")
    yy, xx = np.nonzero(output > 32767)
    assert abs(np.ptp(yy) - np.ptp(xx)) <= 2
    assert min(output.ravel()) >= round(12 / 100 * 65535) - 1
    sampling = np.asarray(info["sampling_yx"])
    offset = np.asarray(info["offset_yx"])
    assert np.all(sampling >= 1)
    assert np.all(offset >= 0)
    assert np.all(offset + sampling * 191 <= np.asarray(plane.shape) - 1)
    np.testing.assert_allclose(sampling * [1.5, 1], info["output_pixel_spacing"])
    assert not info["upsampled"] and not info["added_padding"]


def test_anisotropic_native_grid_without_unpadded_non_enlarged_square_is_rejected():
    plane = np.ones((192, 384), np.float32)
    # 192 samples at the fine spacing cannot span 192 coarse native samples.
    with pytest.raises(ValueError, match="physical_square_requires_upsampling"):
        fit_plane(plane, [1, 2], 1, crop_mode="foreground")


def test_threshold_split_bilateral_tissue_keeps_both_sides_in_crop():
    y, x = np.mgrid[:384, :384]
    left = (y - 192) ** 2 + (x - 115) ** 2 <= 53 ** 2
    right = (y - 192) ** 2 + (x - 260) ** 2 <= 49 ** 2
    bridge = (abs(y - 192) < 12) & (x > 140) & (x < 235)
    plane = np.where(left, 1, np.where(right, 0.75, np.where(bridge, 0.15, 0))).astype(np.float32)
    output, info = fit_plane(plane, [1, 1], 1, crop_mode="foreground")
    lower, upper = np.asarray(info["crop_bounds_yx"])
    source_tissue = np.argwhere(left | right)
    assert np.all(source_tissue >= lower)
    assert np.all(source_tissue <= upper)
    components, count = label(output > int(0.5 * 65535))
    areas = np.bincount(components.ravel())[1:]
    assert count == 2
    assert min(areas) > 1000
    assert np.max(output[:, :96]) > 60000
    assert np.max(output[:, 96:]) > 45000
    assert info["foreground_retained_fraction"] >= 0.98
    assert not info["crop_detector_is_brain_segmentation"]


@pytest.mark.parametrize("reader", ["public", "lab"])
@pytest.mark.parametrize("low_shape", [(128, 256), (256, 128)])
def test_export_filters_small_native_sources_before_preview_and_voxel_loading(
        tmp_path, monkeypatch, reader, low_shape):
    """A small first candidate must not displace the eligible preview source."""
    project = tmp_path / "preprocessing"
    project.mkdir()
    low_path = tmp_path / "small_source.bin"
    high_path = tmp_path / "eligible_source.bin"
    low_path.write_bytes(b"low source should never be loaded")
    high_path.write_bytes(b"eligible source provenance")
    common = dict(dataset="synthetic", subject="same-subject", sequence="T2w-RARE")
    low = dict(common, source_id="a_low", path=str(low_path))
    high = dict(common, source_id="z_eligible", path=str(high_path))
    if reader == "public":
        # NIfTI discovery's native shape is [column, row, slice].
        low["native_shape"] = [low_shape[1], low_shape[0], 1]
        high["native_shape"] = [256, 256, 1]
    else:
        low["native_plane_shape"] = list(low_shape)
        high["native_plane_shape"] = [256, 256]
    loaded = []

    def load(source):
        loaded.append(source["source_id"])
        assert source["source_id"] == "z_eligible"
        y, x = np.mgrid[:256, :256]
        plane = np.where((y - 127) ** 2 + (x - 127) ** 2 < 65 ** 2, 100, 0).astype(np.float32)
        return plane[None], {"spacing_yx": [0.1, 0.1]}

    kind = "nifti" if reader == "public" else "bruker"
    monkeypatch.setattr(exporter, f"discover_{kind}", lambda _: ([low, high], []))
    monkeypatch.setattr(exporter, f"load_{kind}", load)
    monkeypatch.setattr(exporter, "HERE", project)
    out = project / "runs/check"
    monkeypatch.setattr(exporter.sys, "argv", [
        "export_rodent_brain.py", "--out", str(out), "--sources", reader,
        "--trim-fraction", "0", "--preview-per-dataset", "1",
    ])
    assert exporter.main() == 0
    assert loaded == ["z_eligible"]
    selected = json.loads((out / "selected_sources.json").read_text())
    assert [source["source_id"] for source in selected] == ["z_eligible"]
    exclusions = json.loads((out / "native_resolution_exclusions.json").read_text())
    assert len(exclusions) == 1
    assert exclusions[0]["native_plane_shape"] == list(low_shape)
    assert exclusions[0]["minimum_native_size"] == 192
    assert exclusions[0]["reason"] == "native_matrix_below_minimum"
    record = json.loads((out / "manifest.jsonl").read_text())
    assert record["source_id"] == "z_eligible"
    assert record["transform"]["crop_mode"] == "foreground"
    assert record["transform"]["direct_native_crop"]
    assert not record["transform"]["upsampled"]
    assert not record["transform"]["added_padding"]
