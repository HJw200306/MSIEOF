"""Paired comparison on identical images; never starts training.

Default: VGG-U-Net + current three-class dashed U-Net++, versus unified U-Net++.
Both independent models and the unified model use three-class segmentation.
Input is the common arrow-stage input (after SR if required).
"""
import argparse
import csv
import gc
import hashlib
import json
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
import segmentation_models_pytorch as smp

try:
    from .comparison_metrics import counts, summarize, fuse_and_erase
except ImportError:  # 兼容直接运行本文件。
    from comparison_metrics import counts, summarize, fuse_and_erase

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
sys.path.insert(0, str(PROJECT))
from nets.unet import Unet
from utils_test.utils import resize_image, preprocess_input


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def load_weights(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    state = checkpoint.get('model_state_dict', checkpoint.get('state_dict', checkpoint))
    state = {k.removeprefix('module.'): v for k, v in state.items()}
    return checkpoint, state


class Predictor:
    def __init__(self, path, device, role):
        checkpoint, state = load_weights(path)
        self.device, self.role = device, role
        if role == 'solid':
            if int(state['final.weight'].shape[0]) != 3:
                raise ValueError(f'{path}: solid model must have three output classes')
            self.model = Unet(num_classes=3, pretrained=False, backbone='vgg')
        else:
            state = {k.removeprefix('model.'): v for k, v in state.items()}
            n = int(state['segmentation_head.0.weight'].shape[0])
            expected = 3
            if n != expected:
                raise ValueError(f'{path}: expected {expected} classes, found {n}')
            names = checkpoint.get('class_names')
            expected_names = ['background', 'arrow', 'bond'] if role == 'unified' else ['background', 'dashed_arrow', 'bond']
            if names is not None and names != expected_names:
                raise ValueError(f'Unexpected class mapping: {names}')
            self.model = smp.UnetPlusPlus(encoder_name=checkpoint.get('encoder', 'resnet101'),
                                         encoder_weights=None, in_channels=3, classes=n)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval()
        self.parameters = sum(p.numel() for p in self.model.parameters())

    @torch.inference_mode()
    def __call__(self, rgb):
        h, w = rgb.shape[:2]
        if self.role == 'solid':
            im, nw, nh = resize_image(Image.fromarray(rgb), (512, 512))
            a = preprocess_input(np.array(im, dtype=np.float32)).transpose(2, 0, 1)
            x = torch.from_numpy(a.copy()).unsqueeze(0).to(self.device)
            prob = self.model(x)[0].softmax(0).permute(1, 2, 0).cpu().numpy()
            prob = prob[(512-nh)//2:(512-nh)//2+nh, (512-nw)//2:(512-nw)//2+nw]
            return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        ph, pw = (-h) % 32, (-w) % 32
        rgb = cv2.copyMakeBorder(rgb, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255,255,255))
        a = (rgb.astype(np.float32) / 255. - np.array([.485,.456,.406], np.float32)) / np.array([.229,.224,.225], np.float32)
        x = torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0).to(self.device)
        return self.model(x)[0,:,:h,:w].softmax(0).permute(1,2,0).cpu().numpy()


def process(rgb, predictors):
    probs = [p(rgb) for p in predictors]
    labels = [p.argmax(2) for p in probs]
    return fuse_and_erase(rgb, labels)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unified', type=Path, help='Explicit best.pth; auto-select only if exactly one exists')
    parser.add_argument('--solid', type=Path, default=PROJECT / 'model/legacy_arrow_models/misp_weights_no_freeze.pth')
    parser.add_argument('--dashed', type=Path, help='Three-class checkpoint; prefer dashed/runs/**/best.pth, otherwise the archived three-class model')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/UnifiedArrow-Seg')
    parser.add_argument('--split', choices=['test','val','val_synthetic'], default='test')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    parser.add_argument('--limit', type=int, default=0, help='0=all; limited runs are diagnostic only')
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--no-visuals', action='store_true')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths/manifests only; no checkpoint loading or inference')
    args = parser.parse_args()
    if args.limit < 0 or args.warmup < 0:
        parser.error('limit/warmup must be nonnegative')
    if args.unified is None:
        candidates = sorted((ROOT / 'runs').glob('*/best.pth'))
        if len(candidates) != 1:
            parser.error('Specify --unified: runs must contain exactly one best.pth for auto-selection')
        args.unified = candidates[0]
    if args.dashed is None:
        candidates = sorted((ROOT.parent / 'dashed/runs').rglob('best.pth'))
        if len(candidates) > 1:
            parser.error('Multiple dashed runs found; explicitly specify --dashed')
        args.dashed = candidates[0] if candidates else PROJECT / 'model/legacy_arrow_models/dashed_arrow_model.pth'
    for path in [args.unified, args.solid, args.dashed]:
        if not path.is_file():
            parser.error(f'Missing checkpoint: {path}')
    manifest = args.data / 'manifest.csv'
    with manifest.open(encoding='utf-8-sig', newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['split'] == args.split]
    rows.sort(key=lambda r:r['id'])
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error('No evaluation samples')
    if len({r['id'] for r in rows}) != len(rows):
        parser.error('Duplicate sample IDs')
    for row in rows:
        if row.get('augmentation', 'original') != 'original':
            parser.error('Held-out set contains augmented samples')
        for key in ['image', 'mask']:
            if not (args.data / row[key]).is_file():
                parser.error(f'Missing {key}: {row[key]}')
        if '/' in row['id'] or '\\' in row['id'] or row['id'] in {'.','..'}:
            parser.error('Unsafe sample ID')
    plan = {'samples': len(rows), 'split':args.split, 'unified':str(args.unified),
            'solid':str(args.solid), 'dashed':str(args.dashed),
            'source_counts':dict(__import__('collections').Counter(r.get('domain','unknown') for r in rows)),
            'methods':['dual_shared', 'unified_shared'],
            'class_mapping': {'solid':['background','arrow','bond'],
                              'dashed':['background','dashed_arrow','bond'],
                              'unified':['background','arrow','bond']}}
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    output = args.output or ROOT / 'comparisons' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | plan | {'torch':torch.__version__, 'device_name':torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        'manifest_sha256':file_hash(manifest),
        'erasure_policy':'three-class-argmax-arrow-union-bond-union-priority-v1',
        'scope':'Arrow-removal stage on identical RGB inputs; shared upstream SR/denoising excluded',
        'precision':'float32', 'timing':'decode excluded, preprocessing+inference+erasure included; synchronized, sequential models per method',
        'checkpoint_hashes':{str(p.resolve()):file_hash(p) for p in [args.solid,args.dashed,args.unified]},
        'source_hashes':{str(p):file_hash(p) for p in [Path(__file__),ROOT/'comparison_metrics.py',PROJECT/'nets/unet.py']}}
    (output/'config.json').write_text(json.dumps(config, default=str, ensure_ascii=False, indent=2),encoding='utf-8')
    specs = [('dual_shared',[(args.solid,'solid'),(args.dashed,'dashed')]),
             ('unified_shared',[(args.unified,'unified')])]
    summaries, paired = {}, defaultdict(dict)
    for method, spec in specs:
        predictors = [Predictor(path, device, role) for path,role in spec]
        with Image.open(args.data/rows[0]['image']) as im:
            first = np.array(im.convert('RGB'))
        for _ in range(args.warmup):
            process(first, predictors)
        sync(device)
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        results, grouped = [], defaultdict(list)
        destination = output/method
        for sub in ['cleaned','predictions','erase_masks']:
            (destination/sub).mkdir(parents=True)
        for index,row in enumerate(rows):
            with Image.open(args.data/row['image']) as im:
                rgb = np.array(im.convert('RGB'))
            with Image.open(args.data/row['mask']) as im:
                target = np.array(im)
            sync(device)
            start = time.perf_counter()
            pred, erased, cleaned = process(rgb,predictors)
            sync(device)
            seconds = time.perf_counter()-start
            record = counts(pred,erased,target) | {'seconds':seconds,'id':row['id'],'domain':row.get('domain','unknown')}
            results.append(record)
            grouped[record['domain']].append(record)
            paired[row['id']][method] = summarize([record])
            for sub, array in [('cleaned',cleaned),('predictions',pred),('erase_masks',erased.astype(np.uint8)*255)]:
                Image.fromarray(array).save(destination/sub/(row['id']+'.png'))
            print(f'{method}: {index+1}/{len(rows)} {row["id"]}',flush=True)
        with (destination/'per_image.jsonl').open('w',encoding='utf-8') as f:
            for record in results:
                f.write(json.dumps(record)+'\n')
        summaries[method] = {'all':summarize(results),'by_source':{k:summarize(v) for k,v in grouped.items()},
                             'parameters':sum(p.parameters for p in predictors),
                             'peak_allocated_cuda_mib':torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None}
        del predictors
        gc.collect()
        if device.type=='cuda':
            torch.cuda.empty_cache()
    (output/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    flat = []
    for row in rows:
        record = {'id':row['id'],'domain':row.get('domain','unknown')}
        for method in plan['methods']:
            for key in ['arrow_iou','bond_iou','arrow_residual_rate','bond_false_erasure_rate','mean_seconds']:
                record[method+'_'+key] = paired[row['id']][method][key]
        for key in ['arrow_iou','bond_iou','arrow_residual_rate','bond_false_erasure_rate','mean_seconds']:
            a,b = record['unified_shared_'+key],record['dual_shared_'+key]
            record['unified_minus_dual_'+key] = a-b if a is not None and b is not None else None
        flat.append(record)
    with (output/'paired.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    if not args.no_visuals:
        (output/'visuals').mkdir()
        palette = np.array([[255,255,255],[255,0,0],[0,150,0]],np.uint8)
        for row in rows:
            with Image.open(args.data/row['image']) as im:
                original = im.convert('RGB')
            with Image.open(args.data/row['mask']) as im:
                gt = np.array(im)
            colors = palette[np.minimum(gt,2)]; colors[gt==255] = (128,128,128)
            panels = [('original',original),('GT: red arrow / green bond',Image.fromarray(colors))]
            for method in plan['methods']:
                with Image.open(output/method/'cleaned'/(row['id']+'.png')) as im:
                    panels.append((method,im.convert('RGB')))
            canvas = Image.new('RGB',(original.width*len(panels),original.height+28),'white')
            draw = ImageDraw.Draw(canvas)
            for i,(name,im) in enumerate(panels):
                canvas.paste(im,(i*original.width,28)); draw.text((i*original.width+3,6),name,fill='black')
            canvas.save(output/'visuals'/(row['id']+'.png'))
    lines = ['# Model comparison', '', f'Split: {args.split}; images: {len(rows)}.', '',
             '| Method | Arrow IoU | Bond IoU | Arrow residual | Bond false erasure | Seconds/image |',
             '|---|---:|---:|---:|---:|---:|']
    for method,summary in summaries.items():
        values=[summary['all'][k] for k in ['arrow_iou','bond_iou','arrow_residual_rate','bond_false_erasure_rate','mean_seconds']]
        lines.append('| '+method+' | '+' | '.join('NA' if v is None else f'{v:.6f}' for v in values)+' |')
    lines.extend(['','Rates are pooled pixel counts; NA means zero denominator.',
                  'Source subsets are not pure arrow-type subsets. Test currently has no independent dashed source.',
                  'Validation results are diagnostic, not independent test evidence. Independent-model training overlap is not certified.',
                  'OCSR accuracy is not measured here; cleaned images are exported for the same downstream recognizer.',
                  'No automatic winner: review both source groups, bond deletion and downstream recognition.'])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Completed: {output}')


if __name__ == '__main__':
    main()
