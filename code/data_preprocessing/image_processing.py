"""Deterministic 2D export with physical aspect ratio and no anatomy synthesis."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import affine_transform, binary_opening, gaussian_filter, label


def volume_window(volume):
    values = np.asarray(volume)
    positive = values[np.isfinite(values) & (values > 0)]
    if not positive.size:
        raise ValueError('Volume contains no positive finite signal')
    upper = float(np.percentile(positive, 99.5))
    if upper <= 0:
        raise ValueError('Invalid intensity window')
    return upper


def assess_slice(plane, upper):
    """Signal screening only; this does not identify or segment the brain."""
    plane = np.asarray(plane)
    finite_fraction = float(np.isfinite(plane).mean())
    if finite_fraction < 0.999:
        return False, 'nonfinite_values', {'finite_fraction': finite_fraction}
    scaled = np.clip(np.nan_to_num(plane, nan=0, posinf=0, neginf=0) / upper, 0, 1)
    foreground = scaled > 0.10
    labels, count = label(foreground)
    largest = float(np.bincount(labels.ravel())[1:].max() / plane.size) if count else 0.
    stats = dict(finite_fraction=finite_fraction, foreground_fraction=float(foreground.mean()),
                 largest_component_fraction=largest, normalized_std=float(scaled.std()),
                 normalized_p99=float(np.percentile(scaled, 99)))
    if stats['normalized_p99'] < 0.20 or stats['normalized_std'] < 0.035:
        return False, 'low_signal_or_constant', stats
    if stats['foreground_fraction'] < 0.06 or largest < 0.03:
        return False, 'small_foreground', stats
    return True, 'accepted', stats


def foreground_crop(plane, spacing, upper, size, margin=0.10):
    """Locate a conservative tissue ROI; this is not a brain segmentation.

    The mask only chooses a crop. Original intensities inside it are untouched.
    Keep at least ``size`` native samples on BOTH axes, so no upsampling occurs.
    """
    if min(plane.shape) < size:
        raise ValueError('native_matrix_below_output_size')
    if not np.isfinite(margin) or not 0 <= margin <= 0.5:
        raise ValueError('Invalid crop margin')
    shape = np.array(plane.shape)
    normalized = np.clip(np.nan_to_num(plane, nan=0, posinf=0, neginf=0) / upper, 0, 1)
    sigma = max(0.8, min(shape) / 128)
    smooth = gaussian_filter(normalized, sigma=sigma)
    threshold = float(np.percentile(smooth, 99) * 0.4)
    foreground = binary_opening(smooth > threshold, iterations=max(1, round(min(shape) / 85)))
    labels, count = label(foreground)
    if not count:
        raise ValueError('crop_foreground_not_found')
    areas = np.bincount(labels.ravel())
    areas[0] = 0
    largest = int(areas.argmax())
    # Preserve substantial nearby components: uneven signal may split hemispheres.
    anchor = np.argwhere(labels == largest).mean(axis=0)
    kept = [largest]
    for component in np.flatnonzero(areas >= areas[largest] * 0.20):
        if component == largest:
            continue
        center = np.argwhere(labels == component).mean(axis=0)
        if np.linalg.norm((center - anchor) * spacing) <= 0.4 * min(shape * spacing):
            kept.append(int(component))
    mask = np.isin(labels, kept)
    coordinates = np.argwhere(mask)
    lower, upper_bound = coordinates.min(axis=0), coordinates.max(axis=0) + 1
    center = (lower + upper_bound - 1) / 2
    required_side = float(max((upper_bound - lower) * spacing) * (1 + 2 * margin))
    side = max(required_side, size * max(spacing))
    # Header differences <=0.01% need not interpolate otherwise square native pixels.
    near_isotropic = bool(np.allclose(spacing, spacing[0], rtol=1e-4, atol=0))
    if near_isotropic:
        width = min(int(np.ceil(side / max(spacing) - 1e-8)), int(min(shape)))
        sampling = np.full(2, width / size, dtype=float)
        side = width * float(max(spacing))
    else:
        # Fit the output sample centers within acquired sample centers; no padding.
        maximum_side = float(size * min((shape - 1) * spacing) / (size - 1))
        if maximum_side < size * max(spacing) * (1 - 1e-6):
            raise ValueError('physical_square_requires_upsampling')
        side = min(side, maximum_side)
        sampling = side / size / spacing
    half_span = sampling * (size - 1) / 2
    center = np.clip(center, half_span, shape - 1 - half_span)
    offset = center - half_span
    direct = bool(np.allclose(sampling, 1, rtol=0, atol=1e-8))
    if direct:
        offset = np.clip(np.rint(offset), 0, shape - size)
        center = offset + (size - 1) / 2
    cell_lower = offset - sampling / 2
    cell_upper = offset + sampling * (size - 0.5)
    retained = np.all((coordinates >= cell_lower) & (coordinates <= cell_upper), axis=1).mean()
    if retained < 0.98:
        raise ValueError('foreground_does_not_fit_unpadded_square')
    info = dict(crop='foreground-centered physical square; no artificial padding',
                crop_margin=margin, crop_center_yx=center.tolist(),
                crop_bounds_yx=[cell_lower.tolist(), cell_upper.tolist()],
                foreground_bbox_yx=[lower.tolist(), upper_bound.tolist()],
                crop_detection_threshold=threshold, crop_detection_sigma=sigma,
                crop_detection='slice smooth q99 * 0.4; opening; largest and substantial nearby components',
                foreground_retained_fraction=float(retained),
                crop_detector_is_brain_segmentation=False,
                crop_margin_limited_by_fov=bool(side < required_side - 1e-6),
                added_padding=False, direct_native_crop=direct,
                spacing_near_isotropic_tolerance=1e-4,
                spacing_relative_anisotropy=float(max(spacing) / min(spacing) - 1))
    return sampling, offset, side, info


def fit_plane(plane, spacing_yx, upper, size=192, bit_depth=16, crop_mode='full', crop_margin=0.10):
    """Resample in physical coordinates; foreground mode never enlarges or pads."""
    plane = np.asarray(plane, dtype=np.float32)
    spacing = np.asarray(spacing_yx, dtype=float)
    if plane.ndim != 2 or min(plane.shape) < 2:
        raise ValueError('Expected a two-dimensional plane')
    if spacing.shape != (2,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
        raise ValueError('Invalid in-plane spacing')
    if bit_depth not in (8, 16) or size < 2 or not np.isfinite(upper) or upper <= 0:
        raise ValueError('Invalid export parameters')
    extent = np.array(plane.shape) * spacing
    if crop_mode == 'foreground':
        sampling, offset, side, crop_info = foreground_crop(plane, spacing, upper, size, crop_margin)
    elif crop_mode == 'full':
        side = float(extent.max())
        sampling = side / size / spacing
        offset = (np.array(plane.shape) - 1) / 2 - sampling * (size - 1) / 2
        crop_info = dict(crop='none; full acquired in-plane FOV, zero square padding',
                         added_padding=bool(np.ptp(extent) > 1e-6), direct_native_crop=False)
    else:
        raise ValueError('Unknown crop mode')
    step = side / size
    sigma = np.maximum((sampling - 1) * 0.5, 0)
    clean = np.maximum(np.nan_to_num(plane, nan=0, posinf=0, neginf=0), 0)
    if np.any(sigma > 0):
        clean = gaussian_filter(clean, sigma=sigma, mode='nearest')
    if crop_info['direct_native_crop']:
        y, x = offset.astype(int)
        resized = clean[y:y + size, x:x + size]
    else:
        resized = affine_transform(clean, np.diag(sampling), offset=offset,
                               output_shape=(size, size), order=1, mode='constant',
                               cval=0, prefilter=False)
    scaled = np.clip(resized / upper, 0, 1)
    maximum = (1 << bit_depth) - 1
    pixels = np.rint(scaled * maximum).astype(np.uint16 if bit_depth == 16 else np.uint8)
    return pixels, dict(native_plane_shape=list(plane.shape), native_spacing_yx=spacing.tolist(),
                        native_fov_yx=extent.tolist(), square_fov=side,
                        output_pixel_spacing=step, output_shape=[size, size], bit_depth=bit_depth,
                        output_spacing_yx=(sampling * spacing).tolist(),
                        sampling_yx=sampling.tolist(), offset_yx=offset.tolist(),
                        upsampled=bool(np.any(sampling < 1 - 1e-6)),
                        scale_yx=(1 / sampling).tolist(), antialias_sigma_yx=sigma.tolist(),
                        interpolation='none; native integer crop' if crop_info['direct_native_crop'] else
                                      'linear; Gaussian antialias only on downsampled axes',
                        normalization_window=[0., upper], normalization='per-source positive q99.5',
                        crop_mode=crop_mode, **crop_info)
