"""Fixed, threshold-free fusion ablations on real dashed validation images."""
import csv
import argparse
import gc
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from compare_models import Predictor, file_hash, ROOT, PROJECT
from comparison_metrics import counts, summarize, fuse_and_erase


def main():
    data = ROOT / 'dataset/UnifiedArrow-Seg'
    rows = [r for r in csv.DictReader((data/'manifest.csv').open(encoding='utf-8-sig'))
            if r['split'] == 'val' and r['domain'] == 'dashed']
    assert len(rows) == 20
    parser=argparse.ArgumentParser()
    parser.add_argument('--resume',type=Path)
    parser.add_argument(
        '--dashed-checkpoint', type=Path,
        default=PROJECT/'model/legacy_arrow_models/dashed_arrow_model.pth',
        help='Dashed three-class checkpoint to evaluate.',
    )
    args=parser.parse_args()
    out = args.resume or ROOT/'comparisons'/('dashed_fusion_diagnostic_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    out.mkdir(exist_ok=bool(args.resume))
    paths = {'solid': PROJECT/'model/legacy_arrow_models/misp_weights_no_freeze.pth',
             'dashed': args.dashed_checkpoint,
             'unified': ROOT/'runs/unetpp_r101_20261027_235511/best.pth'}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {'images':20, 'split':'val', 'domain':'dashed', 'device':str(device),
              'checkpoint_hashes':{k:file_hash(v) for k,v in paths.items()},
              'manifest_sha256':file_hash(data/'manifest.csv'),
              'script_sha256':file_hash(Path(__file__)), 'threshold_tuning':False,
              'note':'Diagnostic validation only; not an independent test. No timing comparison.'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    for role,path in paths.items():
        if (args.resume and role != 'dashed'
                and all((out/role/(r['id']+'.png')).exists() for r in rows)):
            continue
        model = Predictor(path,device,role)
        dest = out/role
        dest.mkdir(exist_ok=role == 'dashed')
        for i,row in enumerate(rows):
            rgb = np.array(Image.open(data/row['image']).convert('RGB'))
            pred = model(rgb).argmax(2).astype(np.uint8)
            Image.fromarray(pred).save(dest/(row['id']+'.png'))
            print(role,i+1,len(rows),flush=True)
        del model
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
    results = defaultdict(list)
    attribution = []
    for row in rows:
        rgb = np.array(Image.open(data/row['image']).convert('RGB'))
        gt = np.array(Image.open(data/row['mask']))
        labels = {k:np.array(Image.open(out/k/(row['id']+'.png'))) for k in paths}
        s,d = labels['solid'],labels['dashed']
        methods = {k:fuse_and_erase(rgb,[v]) for k,v in labels.items()}
        methods['dual_union'] = fuse_and_erase(rgb,[s,d])
        # Isolate adding solid arrows from adding solid bond protection.
        for name,arrow,bond in [
            ('arrow_union_dashed_bond', (s==1)|(d==1), d==2),
            ('dashed_arrow_bond_union', d==1, (s==2)|(d==2)),
            ('arrow_intersection_bond_union', (s==1)&(d==1), (s==2)|(d==2))]:
            p=np.zeros(gt.shape,np.uint8); p[arrow]=1; p[bond]=2
            methods[name]=fuse_and_erase(rgb,[p])
        for name,(pred,erase,clean) in methods.items():
            results[name].append(counts(pred,erase,gt)|{'id':row['id']})
            dest=out/name/'cleaned'; dest.mkdir(parents=True,exist_ok=True)
            Image.fromarray(clean).save(dest/(str(rows.index(row)).zfill(3)+'.png'))
        de=methods['dashed'][1]; ue=methods['dual_union'][1]
        attribution.append({'id':row['id'],
            'added_background_erasure':int(((gt==0)&ue&~de).sum()),
            'added_arrow_recovery':int(((gt==1)&ue&~de).sum()),
            'blocked_correct_arrow':int(((gt==1)&de&~ue).sum()),
            'prevented_bond_deletion':int(((gt==2)&de&~ue).sum())})
    summary={k:summarize(v) for k,v in results.items()}
    for v in summary.values():
        cm=np.array(v['confusion']); tp=cm[1,1]
        v['arrow_precision']=float(tp/max(1,cm[:,1].sum()))
        v['arrow_recall']=float(tp/max(1,cm[1].sum()))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'per_image.json').write_text(json.dumps(dict(results),indent=2),encoding='utf-8')
    (out/'attribution.json').write_text(json.dumps(attribution,indent=2),encoding='utf-8')
    lines=['# Real dashed validation fusion diagnosis','',
           '| Method | Arrow IoU | Precision | Recall | Residual | Bond deletion | Background erasure |',
           '|---|---:|---:|---:|---:|---:|---:|']
    keys=['arrow_iou','arrow_precision','arrow_recall','arrow_residual_rate','bond_false_erasure_rate','background_erasure_rate']
    for k,v in summary.items():
        lines.append('| '+k+' | '+' | '.join(f'{v[x]:.6f}' for x in keys)+' |')
    lines+=['','20 validation images used previously for selection. Diagnostic evidence only.',
            'Fusion rules were fixed before evaluation. No thresholds were tuned.']
    (out/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    print('\n'.join(lines),flush=True)
    print('OUTPUT',out,flush=True)


if __name__ == '__main__':
    main()
