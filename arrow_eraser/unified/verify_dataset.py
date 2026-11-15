"""Recheck the copied dataset independently of its original source folders."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from build_dataset import pixel_hash


def main():
    root = Path(__file__).resolve().parent / 'dataset/UnifiedArrow-Seg'
    with (root / 'manifest.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    groups, hashes, parents = defaultdict(set), defaultdict(set), defaultdict(set)
    errors, listed = [], set()
    for row in rows:
        image, mask = root / row['image'], root / row['mask']
        listed.update([image.resolve(), mask.resolve()])
        with Image.open(image) as im:
            shape = (im.height, im.width)
        with Image.open(mask) as im:
            labels = np.asarray(im)
        if labels.shape != shape or not set(np.unique(labels)) <= {0, 1, 2, 255}:
            errors.append('Invalid pair: ' + row['id'])
        for path, mode, field in [(image, 'RGB', 'pixel_sha256'), (mask, 'L', 'mask_pixel_sha256')]:
            if pixel_hash(path, mode) != row[field]:
                errors.append('Changed pixels: ' + str(path))
        for table, key in [(groups, 'group'), (hashes, 'pixel_sha256'), (parents, 'parent_id')]:
            table[row[key]].add(row['split'])
        if row['split'] != 'train' and row['augmentation'] != 'original':
            errors.append('Held-out augmented sample: ' + row['id'])
    for name, table in [('group', groups), ('pixels', hashes), ('parent', parents)]:
        if any(len(splits) > 1 for splits in table.values()):
            errors.append('Cross split overlap: ' + name)
    actual = {p.resolve() for split in ['train', 'val', 'val_synthetic', 'test']
              for kind in ['images', 'masks'] for p in (root / split / kind).glob('*') if p.is_file()}
    if actual != listed:
        errors.append('Unlisted or missing files')
    if len({r['id'] for r in rows}) != len(rows):
        errors.append('Duplicate ids')
    result = {'passed': not errors, 'samples': len(rows),
              'counts': dict(Counter(r['split'] for r in rows)), 'errors': errors,
              'manifest_sha256': hashlib.sha256((root / 'manifest.csv').read_bytes()).hexdigest()}
    (root / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
