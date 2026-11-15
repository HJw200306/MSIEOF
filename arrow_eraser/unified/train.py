"""Train or evaluate the unified ResNet-101 U-Net++ model."""
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from training_core import (ArrowBondUNetPlusPlus, SegmentationDataset,
                           CLASS_NAMES, IGNORE_INDEX, run_epoch)

ROOT = Path(__file__).resolve().parent


class BalancedParents(Dataset):
    """Draw equally from source domains, uniformly over parents then views."""
    def __init__(self, base, rows, length):
        self.base, self.length = base, length
        indices = {p.stem: i for i, p in enumerate(base.files)}
        groups = {'solid': defaultdict(list), 'dashed': defaultdict(list)}
        for row in rows:
            if row['split'] == 'train':
                groups[row['domain']][row['parent_id']].append(indices[row['id']])
        self.parents = {key: list(value.values()) for key, value in groups.items()}
        if not all(self.parents.values()):
            raise ValueError('Both source domains must have training parents')

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        domain = 'solid' if index % 2 == 0 else 'dashed'
        views = random.choice(self.parents[domain])
        return self.base[random.choice(views)]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def evaluation_loaders(data, rows, split, args):
    base = SegmentationDataset(data / split / 'images', data / split / 'masks', args.size, False)
    domains = {r['id']: r['domain'] for r in rows if r['split'] == split}
    result = {}
    for domain in ['solid', 'dashed']:
        ids = [i for i, p in enumerate(base.files) if domains[p.stem] == domain]
        if ids:
            result[domain] = DataLoader(Subset(base, ids), batch_size=1,
                                       num_workers=args.workers)
    return result


def main():
    parser = argparse.ArgumentParser(description='统一箭头三分类训练 / 独立评估')
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/UnifiedArrow-Seg')
    parser.add_argument('--output', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch-samples', type=int, default=0,
                        help='Even number of draws per epoch; default twice larger domain parent count')
    parser.add_argument('--encoder-weights', choices=['imagenet', 'none'], default='imagenet')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--smoke-test', action='store_true', help='One training batch and validation batch, no pretrained download')
    parser.add_argument('--evaluate', type=Path, help='Checkpoint to evaluate; does not train')
    parser.add_argument('--split', choices=['val', 'val_synthetic', 'test'], default='test')
    args = parser.parse_args()
    if args.size < 32 or args.size % 32 or args.batch_size < 1:
        parser.error('size must be >=32 and divisible by32; batch-size must be positive')
    if args.epoch_samples < 0 or args.epoch_samples % 2:
        parser.error('epoch-samples must be a nonnegative even number')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    data = args.data.resolve()
    report = json.loads((data / 'quality_report.json').read_text(encoding='utf-8'))
    if not report['passed']:
        raise ValueError('Dataset quality checks failed')
    with (data / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    weights = None if args.smoke_test or args.evaluate or args.encoder_weights == 'none' else 'imagenet'
    model = ArrowBondUNetPlusPlus('resnet101', encoder_weights=weights).to(device)
    ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([.20, 1., 1.35], device=device), ignore_index=IGNORE_INDEX)
    amp = device.type == 'cuda' and not args.no_amp
    if args.evaluate:
        checkpoint = torch.load(args.evaluate, map_location=device, weights_only=False)
        if checkpoint['class_names'] != CLASS_NAMES:
            raise ValueError('Checkpoint classes do not match unified labels')
        model.load_state_dict(checkpoint['model_state_dict'])
        results = {}
        for domain, loader in evaluation_loaders(data, rows, args.split, args).items():
            loss, values = run_epoch(model, loader, device, None, ce, .5, use_amp=amp)
            results[domain] = dict(loss=loss, **values)
        destination = args.output or args.evaluate.parent / ('evaluation_' + args.split)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
        print(json.dumps(results, indent=2))
        return
    if args.smoke_test:
        args.size = 64
        args.batch_size = 2
    base = SegmentationDataset(data / 'train/images', data / 'train/masks', args.size, True)
    count = args.epoch_samples or 2 * max(report['train_parents'].values())
    train_set = BalancedParents(base, rows, 2 if args.smoke_test else count)
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(args.seed),
                        pin_memory=device.type == 'cuda')
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    if args.smoke_test:
        loss, values = run_epoch(model, loader, device, optimizer, ce, .5, scaler=scaler, use_amp=amp)
        if not np.isfinite(loss):
            raise RuntimeError('Non-finite training loss')
        val_loss, _ = run_epoch(model, loader, device, None, ce, .5, use_amp=amp)
        if not np.isfinite(val_loss):
            raise RuntimeError('Non-finite validation loss')
        print(json.dumps({'smoke_test': 'passed', 'device': str(device),
                          'train_loss': loss, 'eval_loss': val_loss,
                          'output_classes': CLASS_NAMES, 'metrics': values}, indent=2))
        return
    validation = evaluation_loaders(data, rows, 'val', args)
    if set(validation) != {'solid', 'dashed'}:
        raise ValueError('Need both source validation subsets')
    output = args.output or ROOT / 'runs' / datetime.now().strftime('unetpp_r101_%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    config = vars(args) | {'class_names': CLASS_NAMES, 'device': str(device),
        'encoder': 'resnet101', 'normalization': 'ImageNet mean/std',
        'epoch_samples_actual': count, 'sampling': '50% solid / 50% dashed, uniform parent then view',
        'selection': 'mean of solid-val and dashed-val foreground mIoU',
        'manifest_sha256': hashlib.sha256((data / 'manifest.csv').read_bytes()).hexdigest()}
    (output / 'config.json').write_text(json.dumps(config, indent=2, default=str), encoding='utf-8')
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=.5, patience=4)
    best, stale = -1., 0
    with (output / 'history.jsonl').open('w', encoding='utf-8') as history:
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(model, loader, device, optimizer, ce, .5, scaler=scaler, use_amp=amp)
            result = {'epoch': epoch, 'train_loss': train_loss}
            for domain, val_loader in validation.items():
                loss, values = run_epoch(model, val_loader, device, None, ce, .5, use_amp=amp)
                result[domain] = dict(loss=loss, **values)
            score = sum(result[d]['foreground_miou'] for d in validation) / 2
            scheduler.step(score)
            result.update(selection_score=score, lr=optimizer.param_groups[0]['lr'])
            history.write(json.dumps(result) + '\n')
            history.flush()
            print(json.dumps(result), flush=True)
            checkpoint = {'model_state_dict': model.state_dict(), 'encoder': 'resnet101',
                          'num_classes': 3, 'class_names': CLASS_NAMES, 'ignore_index': IGNORE_INDEX,
                          'epoch': epoch, 'metrics': result, 'config': json.loads(json.dumps(config, default=str)),
                          'optimizer_state_dict': optimizer.state_dict(),
                          'scheduler_state_dict': scheduler.state_dict(), 'scaler_state_dict': scaler.state_dict()}
            torch.save(checkpoint, output / 'last.pth')
            if score > best:
                best, stale = score, 0
                torch.save(checkpoint, output / 'best.pth')
            else:
                stale += 1
            if stale >= args.patience:
                break
    print(f'Best selection score: {best:.6f}; checkpoint: {output / "best.pth"}')


if __name__ == '__main__':
    main()
