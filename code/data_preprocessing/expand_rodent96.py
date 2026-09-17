"""Expand the old 96x96 prior dataset; write only uint16 PNGs in train/val/test.

The old JSON/NPY files are read once to preserve pixels and held-out subjects.
New scans are decoded from the existing raw catalog, never from upscaled PNGs.
No manifest, array cache, augmentation copies, or raw files are written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image

from bruker_sources import discover_bruker, load_bruker
from image_processing import assess_slice, fit_plane, volume_window
from nifti_sources import _identity_groups, discover_nifti, load_nifti


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
PARTS = ("train", "val", "test")
# The original inventory identifies this reconstruction as RAREImage1.mat,
# a scanner reference used in evaluation. Exclude all scans of this study too.
REFERENCE_HASHES = {"89645751281b8be55d33a256e7ad828665c82b9ed65b1caae771017bc4081929"}
REFERENCE_SUBJECT_MARKERS = ("m0427",)
# Reviewed on 2026-09-17: pervasive ghosting/blurring across accepted slices.
# Hashes identify the full primary reconstruction, including both echoes.
VISUAL_EXCLUSIONS = {
    "c2909b6f20c153084b9b3cb15842ca9fe39c6fdd56c62fa41faa1f0e4b656fc1",
    "27027f23071387b9724dd69e05ac8449d6cb0c865c75066149a0ed2a9aa2f3a7",
}
EXPECTED_CROP_REJECTIONS = {
    "crop_foreground_not_found", "physical_square_requires_upsampling",
    "foreground_does_not_fit_unpadded_square",
}


def safe(value):
    return re.sub(r"[^A-Za-z0-9-]+", "-", str(value)).strip("-") or "unknown"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_group(dataset, subject, identities):
    key = subject if subject.startswith(dataset + ":") else f"{dataset}:{subject}"
    if key in identities:
        return identities[key]
    if dataset == "ds006663":
        # Preserve the old conservative COMR family-level grouping.
        match = re.fullmatch(r"ds006663:sub-(COMR\d+)[a-z]*", key)
        if not match:
            raise ValueError(f"Unrecognized COMR identity: {key}")
        return f"ds006663:{match[1]}"
    return key


def legacy_index(manifest, identities):
    groups = defaultdict(set)
    source_names, source_hashes = set(), set()
    for part in PARTS:
        for record in manifest["records"][part]:
            dataset = record["dataset"]
            group = canonical_group(dataset, record["subject"], identities)
            groups[group].add(part)
            source_names.add((dataset, Path(record["source"]).name))
            if record.get("source_sha256"):
                source_hashes.add(record["source_sha256"])
    conflicts = {group: sorted(parts) for group, parts in groups.items() if len(parts) > 1}
    if conflicts:
        raise ValueError(f"Existing canonical subjects span partitions: {conflicts}")
    held_out = {group for group, parts in groups.items() if parts & {"val", "test"}}
    return held_out, source_names, source_hashes


def select_sources(root, manifest):
    identities = _identity_groups(root / "public_rodent_mri")
    if not identities:
        raise ValueError("Missing public cross-dataset identity map; cannot protect old held-out subjects")
    held_out, old_names, old_hashes = legacy_index(manifest, identities)
    lab, lab_skips = discover_bruker(root)
    public, public_skips = discover_nifti(root)
    skipped = Counter()
    for label, rows in (("lab catalog", lab_skips), ("public catalog", public_skips)):
        for row in rows:
            skipped[f"{label}: {row['reason']}"] += 1
    selected = []
    seen_sources = set()
    for source in lab + public:
        is_lab = source["dataset"] == "lab_mouse"
        shape = source.get("native_plane_shape", source.get("native_shape"))
        if min(shape[:2]) < 96:
            skipped["native matrix below 96"] += 1
            continue
        if is_lab:
            identity = (source["subject"] + " " + source["study"]).lower()
            if (source["reconstruction_sha256"] in REFERENCE_HASHES
                    or any(marker in identity for marker in REFERENCE_SUBJECT_MARKERS)):
                skipped["lab scanner evaluation reference (M0427)"] += 1
                continue
            digest = source["reconstruction_sha256"]
            if digest in VISUAL_EXCLUSIONS:
                skipped["lab severe ghosting on visual review (2026-09-17)"] += 1
                continue
            source_key = (digest, source["echo_index"])
        else:
            group = canonical_group(source["dataset"], source["subject"], identities)
            if group in held_out:
                skipped["old validation/test animal or cross-dataset alias"] += 1
                continue
            if (source["dataset"], Path(source["path"]).name) in old_names:
                skipped["volume already represented in old dataset"] += 1
                continue
            digest = file_hash(source["path"])
            if source.get("expected_sha256") and digest != source["expected_sha256"]:
                raise ValueError(f"Source hash mismatch: {source['path']}")
            if digest in old_hashes:
                skipped["source bytes already represented in old dataset"] += 1
                continue
            source_key = (digest, None)
        if source_key in seen_sources:
            skipped["duplicate source reconstruction"] += 1
            continue
        seen_sources.add(source_key)
        selected.append(dict(source, reader="lab" if is_lab else "public"))
    return selected, skipped


def write_png(path, pixels, seen, part):
    if pixels.dtype != np.uint16 or pixels.shape != (96, 96):
        raise ValueError(f"Expected uint16 96x96 pixels: {path}")
    digest = hashlib.sha256(pixels.astype("<u2", copy=False).tobytes()).hexdigest()
    if digest in seen:
        return False
    Image.fromarray(pixels).save(path)
    # Verify actual PNG decoding, including 16-bit precision, as it is written.
    with Image.open(path) as image:
        if image.size != (96, 96) or not np.array_equal(np.asarray(image), pixels):
            raise ValueError(f"PNG roundtrip failed: {path}")
    seen[digest] = part
    return True


def export_legacy(root, out, manifest, seen):
    counts = Counter()
    for part in PARTS:
        array = np.load(root / f"{part}.npy", mmap_mode="r", allow_pickle=False)
        records = manifest["records"][part]
        if array.dtype != np.uint16 or array.shape != (len(records), 96, 96):
            raise ValueError(f"Invalid old {part} array: {array.shape}, {array.dtype}")
        for index, (record, pixels) in enumerate(zip(records, array, strict=True)):
            name = (f"old__{safe(record['dataset'])}__{safe(record['subject'])}__"
                    f"{index:06d}__{safe(record['view'])}.png")
            if not write_png(out / part / name, pixels, seen, part):
                raise ValueError(f"Duplicate legacy pixels; review original partitions: {part}/{index}")
            counts[part] += 1
        print(f"Preserved {part}: {counts[part]:,} PNGs, exact original pixels", flush=True)
    return counts


def export_new(sources, out, seen):
    counts, rejected = Counter(), Counter()
    accepted_sources = 0
    low_matrix_images = 0
    for number, source in enumerate(sources, 1):
        loader = load_bruker if source["reader"] == "lab" else load_nifti
        # Unexpected read/geometry errors stop the run instead of hiding missing data.
        volume, metadata = loader(source)
        upper = volume_window(volume)
        trim = int(len(volume) * 0.12)
        start = max(trim, int(source.get("slice_start", 0)))
        stop = min(len(volume) - trim, int(source.get("slice_stop", len(volume))))
        accepted = 0
        for index, plane in enumerate(volume):
            if not start <= index < stop:
                rejected["stack edge or source window"] += 1
                continue
            ok, reason, _ = assess_slice(plane, upper)
            if not ok:
                rejected[reason] += 1
                continue
            try:
                pixels, transform = fit_plane(plane, metadata["spacing_yx"], upper,
                                              96, 16, "foreground", 0.10)
            except ValueError as error:
                if str(error) not in EXPECTED_CROP_REJECTIONS:
                    raise
                rejected[str(error)] += 1
                continue
            if transform["upsampled"] or transform["added_padding"]:
                raise ValueError(f"Unexpected upsampling/padding: {source['source_id']}")
            label = "lab-RAT" if source["reader"] == "lab" else source["dataset"]
            name = (f"{safe(label)}__{safe(source['subject'])}__{safe(source['source_id'])}__"
                    f"sl{index:03d}__{safe(source['sequence'])}.png")
            destination = out / "train" / name
            if destination.exists():
                raise ValueError(f"Filename collision: {destination}")
            if not write_png(destination, pixels, seen, "train"):
                rejected["duplicate output pixels (including held-out PNGs)"] += 1
                continue
            counts[label] += 1
            accepted += 1
            if source["reader"] == "lab" and min(volume.shape[1:]) < 192:
                low_matrix_images += 1
        accepted_sources += bool(accepted)
        if number % 25 == 0 or number == len(sources):
            print(f"New sources {number}/{len(sources)}; added {sum(counts.values()):,} PNGs", flush=True)
    print(f"Sources with accepted images: {accepted_sources}/{len(sources)}", flush=True)
    print(f"New lab PNGs from matrices below 192: {low_matrix_images}", flush=True)
    for reason, count in sorted(rejected.items()):
        print(f"Rejected slices: {reason}: {count}", flush=True)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy", type=Path, default=DATA / "prior96_0911_260916/mouse_mixed")
    parser.add_argument("--raw", type=Path, default=DATA / "rodent_mri")
    parser.add_argument("--out", type=Path, required=True, help="New image-only folder under code/data/")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source eligibility without writing images")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(DATA) or out == DATA:
        parser.error("--out must be a new subdirectory of code/data/")
    staging = out.with_name(out.name + ".partial")
    if out.exists() or staging.exists():
        parser.error("Output or partial output already exists; choose a new directory")
    manifest = json.loads((args.legacy / "manifest.json").read_text())
    sources, skipped = select_sources(args.raw.resolve(), manifest)
    print("Eligible new sources: " + str(dict(Counter(s["dataset"] for s in sources))), flush=True)
    for reason, count in sorted(skipped.items()):
        print(f"Skipped sources: {reason}: {count}", flush=True)
    if args.dry_run:
        return
    if not sources:
        raise ValueError("No new sources found")
    for part in PARTS:
        (staging / part).mkdir(parents=True)
    seen = {}
    original = export_legacy(args.legacy, staging, manifest, seen)
    added = export_new(sources, staging, seen)
    if not sum(added.values()):
        raise ValueError("No new images accepted")
    staging.rename(out)
    print(f"Complete: {out}", flush=True)
    for label, count in sorted(added.items()):
        print(f"Added {label}: {count:,}", flush=True)
    print(f"train: {original['train']:,} -> {original['train'] + sum(added.values()):,}", flush=True)
    print(f"val: {original['val']:,}; test: {original['test']:,}; both pixel-identical to legacy", flush=True)
    print("Only 96x96 uint16 grayscale PNGs; all output pixels unique; no new split or augmentation copies.", flush=True)


if __name__ == "__main__":
    main()
