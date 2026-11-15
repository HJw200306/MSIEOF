"""Extract the dataset archive and verify all published file hashes."""
from pathlib import Path
import csv
import hashlib
import zipfile

root = Path(__file__).resolve().parents[1]
archive = root / 'data/UnifiedArrow-Type-Seg-v1.zip'
with zipfile.ZipFile(archive) as z:
    for member in z.infolist():
        target = (root / member.filename).resolve()
        if not target.is_relative_to(root.resolve()):
            raise ValueError('Archive member escapes the project directory')
    z.extractall(root)
with (root / 'data/release_manifest.csv').open(encoding='utf-8', newline='') as f:
    rows = list(csv.DictReader(f))
for row in rows:
    p = root / row['path']
    if p.stat().st_size != int(row['bytes']) or hashlib.sha256(p.read_bytes()).hexdigest() != row['sha256']:
        raise ValueError(f'Integrity check failed: {row["path"]}')
print(f'PASS: {len(rows)} dataset files verified')
