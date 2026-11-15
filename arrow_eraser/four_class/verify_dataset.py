"""Verify exported pixels, paired class mapping, hashes and split isolation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from core import read_rows, file_hash, collapse

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data',type=Path,default=Path(__file__).resolve().parent/'dataset/UnifiedArrow-Type-Seg-v1')
    args = p.parse_args()
    root = args.data.resolve()
    rows = read_rows(root)
    tables = {key:defaultdict(set) for key in ['parent_id','group','pixel_sha256']}
    seen = set()
    for row in rows:
        assert row['id'] not in seen, row['id']
        seen.add(row['id'])
        with Image.open(root/row['image']) as im:
            shape = (im.height,im.width)
        with Image.open(root/row['mask']) as im:
            mask = np.array(im)
        with Image.open(root/row['mask_three_class']) as im:
            three = np.array(im)
        assert mask.shape == shape and set(np.unique(mask)) <= {0,1,2,3,255},row['id']
        assert np.array_equal(collapse(mask),three),row['id']
        assert file_hash(root/row['mask']) == row['mask_sha256'],row['id']
        assert file_hash(root/row['image']) == row['image_file_sha256'],row['id']
        assert row['split']=='train' or row['augmentation']=='original',row['id']
        for key,table in tables.items():
            table[row[key]].add(row['split'])
    for key,table in tables.items():
        assert all(len(v)==1 for v in table.values()),key
    report = json.loads((root/'quality_report.json').read_text(encoding='utf-8'))
    assert report['passed'] and file_hash(root/'manifest.csv') == report['manifest_sha256']
    result = {'passed':True,'samples':len(rows),'paired_mapping':True,'cross_split_exact_duplicates':0,
              'manifest_sha256':file_hash(root/'manifest.csv')}
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    main()
