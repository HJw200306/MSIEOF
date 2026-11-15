"""Portable ResNet-101 U-Net++ four-class training and paired three-class control."""
import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from core import ArrowDataset, CLASS_NAMES, Metrics, loss_fn, file_hash

ROOT = Path(__file__).resolve().parent


def evaluate(model, loader, device, classes):
    model.eval()
    metrics = Metrics()
    per_image = []
    with torch.inference_mode():
        for x, _, truth, ids in loader:
            pred = model(x.to(device)).argmax(1).cpu().numpy()
            for p,t,sid in zip(pred,truth.numpy(),ids):
                metrics.update(p,t,classes)
                one = Metrics()
                one.update(p,t,classes)
                per_image.append({'id':sid, **one.result()})
    return metrics.result(), per_image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'dataset/UnifiedArrow-Type-Seg-v1')
    p.add_argument('--output',type=Path)
    p.add_argument('--classes',type=int,choices=[3,4],default=4)
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--epoch-samples',type=int,default=2934)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--lr',type=float,default=8e-5)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-bond-erasure',type=float,default=.003)
    p.add_argument('--encoder-weights',choices=['imagenet','none'],default='imagenet')
    p.add_argument('--cpu',action='store_true')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--smoke-size',type=int,default=64)
    p.add_argument('--evaluate',type=Path)
    p.add_argument('--resume',type=Path)
    p.add_argument('--split',choices=['val','test','val_synthetic'],default='val')
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32 or args.size % 32 or args.epoch_samples < 2 or args.epoch_samples % 2:
        p.error('Positive epochs/batch; size multiple of 32; epoch-samples positive even required')
    if args.evaluate and args.resume:
        p.error('evaluate and resume are exclusive')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    data = args.data.resolve()
    report = json.loads((data/'quality_report.json').read_text(encoding='utf-8'))
    if not report['passed'] or file_hash(data/'manifest.csv') != report['manifest_sha256']:
        raise ValueError('Dataset audit failed or manifest changed')
    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    checkpoint = None
    if args.evaluate or args.resume:
        checkpoint = torch.load(args.evaluate or args.resume,map_location='cpu',weights_only=False)
        if checkpoint['num_classes'] != args.classes:
            raise ValueError('Specify the checkpoint class count with --classes')
    model = smp.UnetPlusPlus(encoder_name='resnet101',encoder_weights=(None if checkpoint or args.smoke_test or args.encoder_weights=='none' else 'imagenet'),in_channels=3,classes=args.classes).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'],strict=True)
    if args.evaluate:
        dataset = ArrowDataset(data,args.split,args.classes)
        values, items = evaluate(model,DataLoader(dataset,batch_size=1,num_workers=args.workers),device,args.classes)
        out = args.output or args.evaluate.parent/('evaluation_'+args.split)
        out.mkdir(parents=True,exist_ok=False)
        (out/'metrics.json').write_text(json.dumps(values,indent=2),encoding='utf-8')
        (out/'per_image.json').write_text(json.dumps(items,indent=2),encoding='utf-8')
        (out/'config.json').write_text(json.dumps({'checkpoint_sha256':file_hash(args.evaluate),'manifest_sha256':report['manifest_sha256'],'split':args.split,'classes':args.classes},indent=2),encoding='utf-8')
        print(json.dumps(values,indent=2))
        return
    if args.smoke_test:
        if args.smoke_size < 64 or args.smoke_size % 32:
            p.error('smoke-size must be a multiple of 32 and at least 64')
        args.size,args.epoch_samples,args.epochs = args.smoke_size,2,1
    train = ArrowDataset(data,'train',args.classes,args.size,args.seed,args.epoch_samples)
    loader = DataLoader(train,batch_size=args.batch_size,num_workers=args.workers,shuffle=False)
    val = ArrowDataset(data,'val',args.classes)
    validation = DataLoader(val,batch_size=1,num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
    start,best = 0,-1.
    out = args.output or ROOT/'runs'/datetime.now().strftime(f'unetpp_r101_{args.classes}class_%Y%m%d_%H%M%S')
    config = vars(args) | {'manifest_sha256':report['manifest_sha256'],'class_names':CLASS_NAMES[args.classes],
        'device':str(device),'sampling':'per-index RNG(seed,epoch,index); 50/50 source, uniform parent/view',
        'lr_schedule':'cosine over fixed epochs; min_lr=1e-6','crop':'50% arrow,25% arrow/bond vicinity,25% random',
        'selection':'mean type-conditioned removal IoU subject to bond erasure <= configured limit',
        'code_hashes':{f:file_hash(ROOT/f) for f in ['train.py','core.py']}}
    if args.resume:
        for k in ['manifest_sha256','classes','epochs','batch_size','size','epoch_samples','seed','lr','max_bond_erasure','code_hashes']:
            if checkpoint['config'][k] != config[k]:
                raise ValueError(f'Resume config mismatch: {k}')
        opt.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        start,best = checkpoint['epoch'],checkpoint['best_score']
        out = args.resume.parent
        torch.set_rng_state(checkpoint['torch_rng'])
        if device.type == 'cuda' and checkpoint.get('cuda_rng'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    if not args.smoke_test:
        out.mkdir(parents=True,exist_ok=bool(args.resume))
        (out/'config.json').write_text(json.dumps(config,indent=2,default=str),encoding='utf-8')
    for epoch in range(start,args.epochs):
        train.epoch = epoch
        lr = 1e-6 + (args.lr-1e-6)*(1+math.cos(math.pi*epoch/max(1,args.epochs-1)))/2
        for group in opt.param_groups:
            group['lr'] = lr
        model.train()
        total, samples = 0.,0
        for step,(x,y,_,_) in enumerate(loader):
            x,y = x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=device.type=='cuda'):
                loss = loss_fn(model(x),y)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item()*len(x)
            samples += len(x)
            if step % 100 == 0:
                print(f'epoch {epoch+1}/{args.epochs} step {step+1}/{len(loader)} loss={loss.item():.5f}',flush=True)
        if args.smoke_test:
            model.eval()
            with torch.inference_mode():
                logits = model(x)
            assert logits.shape[1] == args.classes and torch.isfinite(logits).all()
            print(json.dumps({'smoke_test':'passed','classes':args.classes,'shape':list(logits.shape),'train_loss':total/samples,'device':str(device)}))
            return
        values,_ = evaluate(model,validation,device,args.classes)
        eligible = values['bond_false_erasure_rate'] is not None and values['bond_false_erasure_rate'] <= args.max_bond_erasure
        scores = [values[t]['iou'] for t in ['solid','dashed']]
        score = sum(scores)/2 if eligible and all(v is not None for v in scores) else -1.
        improved = score > best
        best = max(best,score)
        record = {'epoch':epoch+1,'train_loss':total/samples,'lr':lr,'selection_eligible':eligible,'selection_score':score,'val':values}
        with (out/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(record)+'\n')
        payload = {'model_state_dict':model.state_dict(),'num_classes':args.classes,'class_names':CLASS_NAMES[args.classes],
            'encoder':'resnet101','ignore_index':255,'epoch':epoch+1,'best_score':best,'metrics':record,
            'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'config':config,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else None}
        for name in (['last.pth','best.pth'] if improved else ['last.pth']):
            temp = out/(name+'.tmp')
            torch.save(payload,temp)
            temp.replace(out/name)
        print(json.dumps(record),flush=True)
    print(f'Completed: {out}; best eligible score={best}. If -1, no eligible best checkpoint exists.')


if __name__ == '__main__':
    main()
