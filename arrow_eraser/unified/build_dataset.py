"""Create a portable, source-group-safe unified three-class dataset."""
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def pixel_hash(path, mode):
    with Image.open(path) as im:
        a = np.asarray(im.convert(mode))
    return hashlib.sha256(str(a.shape).encode() + a.tobytes()).hexdigest()


def main():
    output = ROOT / 'dataset' / 'UnifiedArrow-Seg'
    if output.exists():
        raise FileExistsError(f'Output exists; refusing overwrite: {output}')
    sources = {
        'solid': ROOT.parent / 'solid/dataset/SolidArrow-Seg',
        'dashed': ROOT.parent / 'dashed/dataset/DashedArrow-Seg-Augmented',
    }
    rows = []
    for domain, source in sources.items():
        name = 'dataset_manifest.csv' if domain == 'solid' else 'manifest.csv'
        for old in read_csv(source / name):
            row = dict(old)
            row.update(domain=domain, source_id=old['id'],
                       id=domain + '_' + old['id'], source_root=str(source),
                       original_split=old['split'],
                       parent_id=domain + '_' + old.get('parent_id', old['id']),
                       group=domain + ':' + old.get('group', old['id']),
                       augmentation=old.get('augmentation', 'original'),
                       kind=old.get('kind', 'historical_mixed'))
            row['pixel_sha256'] = pixel_hash(source / row['image'], 'RGB')
            rows.append(row)

    # Union by provenance AND decoded image pixels (PNG/JPEG metadata is ignored).
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    seen = {}
    for i, row in enumerate(rows):
        for key in [('parent', row['parent_id']), ('group', row['group']),
                    ('pixels', row['pixel_sha256'])]:
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    rank = {'train': 0, 'val_synthetic': 1, 'val': 2, 'test': 3}
    exported, omitted, moved = [], [], []
    hashes = defaultdict(set)
    for number, members in enumerate(groups.values()):
        split = max((r['original_split'] for r in members), key=rank.__getitem__)
        for row in members:
            if split != 'train' and row['augmentation'] != 'original':
                omitted.append(row['id'])
                continue
            source = Path(row['source_root'])
            src_image, src_mask = source / row['image'], source / row['mask']
            with Image.open(src_mask) as im:
                mask = np.asarray(im)
            with Image.open(src_image) as im:
                shape = (im.height, im.width)
            assert mask.shape == shape, row['id']
            assert set(np.unique(mask)) <= {0, 1, 2, 255}, row['id']
            image = Path(split) / 'images' / (row['id'] + src_image.suffix)
            mask_path = Path(split) / 'masks' / (row['id'] + '.png')
            (output / image).parent.mkdir(parents=True, exist_ok=True)
            (output / mask_path).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_image, output / image)
            shutil.copy2(src_mask, output / mask_path)
            assert pixel_hash(output / image, 'RGB') == row['pixel_sha256']
            assert pixel_hash(output / mask_path, 'L') == pixel_hash(src_mask, 'L')
            if split != row['original_split']:
                moved.append({'id': row['id'], 'from': row['original_split'], 'to': split})
            exported.append({k: row.get(k, '') for k in (
                'id', 'source_id', 'domain', 'kind', 'parent_id', 'augmentation',
                'augmentation_seed', 'augmentation_params', 'original_split', 'pixel_sha256',
                'source_root', 'source_ids') } | {
                    'split': split, 'group': f'group_{number:05d}',
                    'image': image.as_posix(), 'mask': mask_path.as_posix(),
                    'mask_pixel_sha256': pixel_hash(output / mask_path, 'L')})
            hashes[row['pixel_sha256']].add(split)
    assert all(len(splits) == 1 for splits in hashes.values())
    with (output / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    info = {
        'name': 'UnifiedArrow-Seg', 'classes': ['background', 'arrow', 'bond'],
        'ignore_index': 255, 'counts': dict(Counter(r['split'] for r in exported)),
        'domain_counts': dict(Counter(r['split'] + ':' + r['domain'] for r in exported)),
        'train_parents': dict(Counter(domain for domain in sources for _ in
            {r['parent_id'] for r in exported if r['split'] == 'train' and r['domain'] == domain})),
        'split_policy': 'Union provenance and decoded identical images; test > val > val_synthetic > train; omit held-out augmentations',
        'moved': moved, 'omitted_augmented_ids': omitted,
        'cross_split_pixel_duplicates': 0, 'passed': True,
        'limitations': ['Solid provenance is not pure arrow type or exclusively real imagery.',
                       'Dashed has no independent test split. Existing val has been used for model selection.',
                       'Decoded exact duplicate check does not detect resized/cropped near-duplicates.'],
    }
    for name in ['dataset.json', 'quality_report.json']:
        (output / name).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
