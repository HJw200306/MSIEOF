"""Export typed labels from reviewed workspaces, preserving the unified split."""
import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
ARROW = ROOT.parent
NAMES = ['background', 'solid_arrow', 'dashed_arrow', 'bond']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def load_mask(path):
    with Image.open(path) as im:
        return np.array(im.convert('L'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collapse(mask):
    result = mask.copy()
    result[(mask == 1) | (mask == 2)] = 1
    result[mask == 3] = 2
    return result


def typed_consensus(masks):
    stack = np.stack(masks)
    solid = ((stack == 1) | (stack == 3)).sum(0)
    dashed = ((stack == 4) | (stack == 6)).sum(0)
    bond = ((stack == 2) | (stack == 3) | (stack == 6)).sum(0) * 2 >= len(masks)
    arrow = (solid + dashed) * 2 >= len(masks)
    result = np.zeros(stack.shape[1:], dtype=np.uint8)
    result[arrow & (solid >= dashed)] = 1
    result[arrow & (dashed > solid)] = 2
    result[bond] = 3
    result[((stack == 255).sum(0) * 2 >= len(masks)) & ~arrow & ~bond] = 255
    conflict = (solid > 0) & (dashed > 0) & ~bond
    return result, int(conflict.sum())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, default=ROOT/'dataset/UnifiedArrow-Type-Seg-v1')
    p.add_argument('--refresh-export', action='store_true', help='Rebuild only this generated export; preserve source annotations')
    args = p.parse_args()
    out = args.output.resolve()
    if out.exists() and not args.refresh_export:
        raise FileExistsError(f'Refusing overwrite: {out}')
    if out.exists() and args.refresh_export:
        prior = json.loads((out/'quality_report.json').read_text(encoding='utf-8'))
        if prior.get('classes') != NAMES:
            raise ValueError('Not a recognized generated export')
    source = ARROW/'unified/dataset/UnifiedArrow-Seg'
    workspace = ARROW/'solid/annotation/annotation_v1'
    status = json.loads((workspace/'review_status.json').read_text(encoding='utf-8'))
    rows = read_csv(source/'manifest.csv')
    out.mkdir(parents=True, exist_ok=args.refresh_export)
    resolutions = json.loads((ROOT/'type_resolutions.json').read_text(encoding='utf-8'))
    exported, conflicts = [], []
    counts = Counter()
    for row in rows:
        old = load_mask(source/row['mask'])
        if row['domain'] == 'solid':
            masks, hashes = [], {}
            for sid in row['source_ids'].split(';'):
                if status[sid]['status'] != 'reviewed':
                    raise ValueError(f'Unreviewed annotation {sid}')
                path = workspace/'corrected_masks'/f'{sid}.png'
                if not path.exists():
                    path = workspace/'preannotations'/f'{sid}.png'
                m = load_mask(path)
                if not set(np.unique(m)) <= {0, 1, 2, 3, 4, 6, 255}:
                    raise ValueError(f'Invalid workspace mask: {path}')
                masks.append(m)
                hashes[sid] = digest(path)
            typed, nconflict = typed_consensus(masks)
            resolution = resolutions.get(row['id'])
            if resolution:
                if row['pixel_sha256'] != resolution['image_pixel_sha256']:
                    raise ValueError('Resolution image hash mismatch')
                typed[(typed == 1) | (typed == 2)] = resolution['arrow_class']
                nconflict = 0
            if nconflict:
                conflicts.append({'id': row['id'], 'pixels': nconflict})
            provenance = 'reviewed_solid_workspace'
        else:
            # This is the declared semantic class of the dashed dataset,
            # not a classification inferred from the filename.
            typed = old.copy()
            typed[old == 1] = 2
            typed[old == 2] = 3
            hashes = {'source_mask': digest(source/row['mask'])}
            provenance = 'dashed_dataset_declared_semantics'
        if not np.array_equal(collapse(typed), old):
            raise ValueError(f'Collapsed labels differ from existing data: {row["id"]}')
        if typed.shape != old.shape:
            raise ValueError(row['id'])
        dest_image, dest_mask = out/row['image'], out/row['mask']
        three_mask = out/'masks_three_class'/row['split']/(row['id']+'.png')
        for path in (dest_image, dest_mask, three_mask):
            path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/row['image'], dest_image)
        Image.fromarray(typed).save(dest_mask)
        Image.fromarray(collapse(typed)).save(three_mask)
        values, nums = np.unique(typed, return_counts=True)
        counts.update({str(int(k)): int(v) for k, v in zip(values, nums)})
        exported.append(row | {'mask_three_class': three_mask.relative_to(out).as_posix(),
            'mask_sha256': digest(dest_mask), 'image_file_sha256': digest(dest_image),
            'type_provenance': provenance, 'annotation_hashes': json.dumps(hashes),
            'has_solid': bool((typed == 1).any()), 'has_dashed': bool((typed == 2).any())})
    with (out/'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(exported[0]))
        writer.writeheader()
        writer.writerows(exported)
    report = {'classes': NAMES, 'ignore_index': 255, 'samples': len(exported),
        'counts': dict(Counter(r['split']+':'+r['domain'] for r in rows)),
        'class_pixels': dict(counts), 'type_conflicts': conflicts,
        'type_resolutions': resolutions,
        'source_manifest_sha256': digest(source/'manifest.csv'),
        'manifest_sha256': digest(out/'manifest.csv'),
        'collapsed_masks_equal_existing': True,
        'passed': not conflicts,
        'review_note': 'Existing reviewed annotations reused; one documented agent visual type resolution. No new full human review performed.',
        'limitations': ['Dashed real validation has been used for model selection; no independent dashed-source test.',
                       'Source split/group information inherited; near-duplicate and missing-type annotations require human review.',
                       'Existing augmented images/masks preserved, including historical low-resolution masks.']}
    (out/'quality_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if conflicts:
        raise RuntimeError('Type conflicts require review; training is blocked by quality report.')


if __name__ == '__main__':
    main()
