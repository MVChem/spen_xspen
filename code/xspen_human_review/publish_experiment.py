"""Place a review in experiments with assets usable in directory-scoped previews.

HTML/JS/CSS are copied. Data files use hard links: the preview server sees real
paths within this folder while the underlying bytes are shared with the run.
An additional source_data symlink retains a direct link to the original run.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGE_FILES = {'index.html', 'viewer.js', 'viewer.css', 'archive_pixels.js',
              'coverage.html', 'legacy_figures.html'}


def hardlink_tree(source, target, counts):
    source = source.resolve()
    if target.is_symlink():
        target.unlink()
    if source.is_dir():
        target.mkdir(exist_ok=True)
        for item in sorted(source.iterdir()):
            if not item.name.startswith('.'):
                hardlink_tree(item, target / item.name, counts)
        return
    if target.exists() and os.path.samefile(source, target):
        counts['hardlinked_files'] += 1
        counts['shared_bytes'] += source.stat().st_size
        return
    staging = target.with_name(target.name + '.link-part')
    if staging.exists():
        staging.unlink()
    os.link(source, staging)
    staging.replace(target)
    counts['hardlinked_files'] += 1
    counts['shared_bytes'] += source.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE/'runs/unified_review_260916')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source, target = args.run.resolve(), args.out.absolute()
    assert source.is_relative_to(HERE/'runs')
    assert target.is_relative_to(ROOT/'experiments') and not target.is_symlink()
    target.mkdir(parents=True, exist_ok=True)
    assert source.stat().st_dev == target.stat().st_dev, 'Hard links require the same filesystem.'
    previous = target/'placement.json'
    if previous.is_file() and not (target/'placement_before_preview_fix.json').exists():
        shutil.copy2(previous, target/'placement_before_preview_fix.json')
    counts = dict(hardlinked_files=0, shared_bytes=0)
    copied = {}
    for item in sorted(source.iterdir()):
        if item.name.startswith('.'):
            continue
        destination = target/item.name
        if item.name in PAGE_FILES:
            # Replace the destination atomically, never edit a shared data inode.
            staging = destination.with_name(destination.name + '.copy-part')
            shutil.copy2(item, staging)
            staging.replace(destination)
            copied[item.name] = hashlib.sha256(item.read_bytes()).hexdigest()
        else:
            hardlink_tree(item, destination, counts)
    raw_link = target/'source_data'
    if raw_link.is_symlink():
        raw_link.unlink()
    raw_link.symlink_to(os.path.relpath(source, target), target_is_directory=True)
    record = dict(placed_at_utc=datetime.now(timezone.utc).isoformat(),
        source_run=os.path.relpath(source, target), entry='index.html',
        copied_page_sha256=copied, data_link_mode='hardlink',
        source_symlink=dict(source_data=os.path.relpath(source, target)),
        preview_compatibility='All browser resource realpaths stay within the experiment folder.',
        summary=json.loads((target/'manifest.json').read_text())['summary'], **counts)
    (target/'placement.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
