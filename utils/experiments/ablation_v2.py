"""运行 MSIEOF 端到端消融实验，并比较超分辨率与箭头去除的执行顺序。

实验流程：
1. 从 E 盘统一评测清单 ``PageToSMILES_approved.jsonl`` 读取页面、分子真值和化学结构。
2. 对整页图像执行 YOLO 实例提取，并按配置启用连通域残片清理和候选筛选。
3. 对候选分子图按配置执行选择性/统一/关闭超分辨率，以及实线、虚线双模型并集箭头去除。
4. ``sr_then_arrow`` 执行“超分辨率→箭头去除”，``arrow_then_sr`` 执行反向顺序。
5. 使用 MolScribe 识别结构，并按旧表A冻结规则计算 CEM、GEM 和 Tanimoto。

旧表A评分规则：
- 删除游离的 ``.C~C`` 图形伪影，其他 ``~`` 键按普通单键处理。
- dummy/Markush/超原子标签统一为 ``*``，但保留通配符数量和连接位置。
- CEM 保留立体信息；GEM 仅额外忽略立体信息。
- CEM/GEM 以全部 GT 为分母，漏检和无效预测记为 0。

运行与输出：
- 每次运行都会重新执行所选页面和配置，不读取历史逐页结果。
- 超分、箭头去除和 MolScribe 均重新推理，不跨配置共享预测。
- 中间图像和掩膜只作为本次运行的可追溯产物保存。

主要输出（默认位于 ``outputs/ablation_各功能模块的消融实验_双模型并集_YYYYMMDD/``）：
- ``<config>/pages/*.json``：逐页明细。
- ``<config>/summary.json``：配置的累计指标。
- ``table_A.json``：全部配置汇总。
- ``order_comparison.json``：两种图像优化顺序的指标及差值。
- ``status.json``：运行状态，可用于中断后续跑。
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from dataclasses import asdict, dataclass

# 允许在 PyCharm 中直接运行本文件，不要求把工作目录手工设为项目根目录。
PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from utils.base.paths import PROJECT_ROOT as ROOT
import cv2
import numpy as np
from PIL import Image
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from utils.base.chemical_graph import convert_graph, SCHEMA_VERSION
from utils.evaluation.evaluate_page2smiles_full import FullPipelineEvaluator
from arrow_eraser.unified.compare_models import Predictor
from arrow_eraser.common.visualize_dual_union_on_pages import (
    refine_dashed_arrow,
    refine_solid_arrow,
)
import torch

RUN_DATE=datetime.now().strftime('%Y%m%d')
DEFAULT_OUT=ROOT/f'outputs/ablation_各功能模块的消融实验_双模型并集_{RUN_DATE}'
SEG_ROOT=Path(r'E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal')
DEFAULT_GT=SEG_ROOT/'PageToSMILES_approved.jsonl'
TEST_SET_ROOT=SEG_ROOT/'images'/'PageToSMILES_approved'
LEGACY_SCORING_VERSION='table-a-wave-wildcard-stereo-only-v1'
ARROW_PIPELINE_VERSION='dual-model-postprocessed-arrow-union-bond-union-priority-v2'
DASHED_ARROW_POSTPROCESS='v4_25'
SOLID_ARROW_MODEL=ROOT/'model/legacy_arrow_models/misp_weights_no_freeze.pth'
DASHED_ARROW_MODEL=ROOT/'model/legacy_arrow_models/dashed_arrow_model.pth'

# 正式完整实验：PyCharm 直接运行时执行全部消融配置，不复用历史结果。
DEFAULT_CONFIGS=[
    'full','no_cc','no_filter','no_arrow','no_sr','no_optim','uniform_sr',
    'arrow_then_sr','sr_then_arrow','full_unprotected',
]


def legacy_scoring_mol(smiles):
    """按旧表A规则规范化一个SMILES，失败时返回 ``None``。"""
    if not isinstance(smiles,str) or not smiles.strip():return None
    mol=Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms()==0:return None
    remove=set()
    for fragment in Chem.GetMolFrags(mol):
        if len(fragment)==2:
            u,v=fragment;bond=mol.GetBondBetweenAtoms(u,v)
            if (bond is not None and bond.GetBondType()==Chem.BondType.UNSPECIFIED
                    and all(mol.GetAtomWithIdx(i).GetAtomicNum()==6 for i in fragment)):
                remove.update(fragment)
    rw=Chem.RWMol(mol)
    for idx in sorted(remove,reverse=True):rw.RemoveAtom(idx)
    mol=rw.GetMol()
    if mol.GetNumAtoms()==0:return None
    for bond in mol.GetBonds():
        if bond.GetBondType()==Chem.BondType.UNSPECIFIED:
            bond.SetBondType(Chem.BondType.SINGLE)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum()==0:
            atom.SetIsotope(0)
            for prop in ('atomLabel','molFileAlias','_displayLabel','_displayLabelW'):
                if atom.HasProp(prop):atom.ClearProp(prop)
    Chem.SanitizeMol(mol);mol=Chem.RemoveHs(mol);Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol,cleanIt=True,force=True)
    return mol


def legacy_keys(smiles):
    """返回旧规则的 ``(CEM键, GEM键)``。"""
    try:
        mol=legacy_scoring_mol(smiles)
        if mol is None:return None,None
        cem=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        Chem.RemoveStereochemistry(mol)
        gem=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
        return cem,gem
    except Exception:return None,None


def legacy_fingerprint(smiles):
    """返回旧规则使用的无立体Morgan指纹。"""
    try:
        mol=legacy_scoring_mol(smiles)
        if mol is None:return None
        Chem.RemoveStereochemistry(mol)
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,2,nBits=2048,useChirality=False)
    except Exception:return None


def legacy_compare(gt_smiles,pred_smiles):
    """使用截图所对应的旧表A规则比较GT和预测。"""
    gc,gg=legacy_keys(gt_smiles);pc,pg=legacy_keys(pred_smiles)
    valid=pc is not None and pg is not None
    cem=bool(valid and gc is not None and pc==gc)
    gem=bool(valid and gg is not None and pg==gg)
    if cem and not gem:raise AssertionError('CEM成功必须同时GEM成功')
    a,b=legacy_fingerprint(gt_smiles),legacy_fingerprint(pred_smiles)
    tan=float(DataStructs.TanimotoSimilarity(a,b)) if a is not None and b is not None else 0.0
    return {'cem':cem,'gem':gem,'scoring_valid':valid,'tanimoto':tan}


@dataclass(frozen=True)
class AblationConfig:
    """定义一个端到端消融配置。

    ``cc`` 控制连通域残片清理，``filter`` 控制候选筛选，``arrow`` 控制
    箭头去除；``sr`` 可为 ``selective``、``uniform`` 或 ``off``；``order``
    可为 ``sr_first`` 或 ``arrow_first``。
    """

    name: str
    cc: bool
    filter: bool
    arrow: bool
    sr: str
    order: str


# 保留旧版的八个配置，并增加名称明确的
# sr_then_arrow。它与 full 完全相同，但单独落盘，便于直接做顺序对照；
# 正式实验不跨配置共享模型预测，每个配置均独立执行完整流程。
ABLATION_CONFIGS = {
    'full': AblationConfig('full', True, True, True, 'selective', 'sr_first'),
    'no_cc': AblationConfig('no_cc', False, True, True, 'selective', 'sr_first'),
    'no_filter': AblationConfig('no_filter', True, False, True, 'selective', 'sr_first'),
    'no_arrow': AblationConfig('no_arrow', True, True, False, 'selective', 'sr_first'),
    'no_sr': AblationConfig('no_sr', True, True, True, 'off', 'sr_first'),
    'no_optim': AblationConfig('no_optim', True, True, False, 'off', 'sr_first'),
    'uniform_sr': AblationConfig('uniform_sr', True, True, True, 'uniform', 'sr_first'),
    'arrow_then_sr': AblationConfig(
        'arrow_then_sr', True, True, True, 'selective', 'arrow_first'),
    'sr_then_arrow': AblationConfig(
        'sr_then_arrow', True, True, True, 'selective', 'sr_first'),
}


class AblationEvaluator(FullPipelineEvaluator):
    """复用正式端到端评估器的页面预处理和 IoU 匹配能力。

    这里只增加 v2 使用的兼容方法名；模型加载、YOLO 候选提取、连通域
    残片清理、U-Net++ 候选筛选和匹配逻辑均以正式评估器为准。
    """

    def __init__(self, config, test_set_root, label_table=None, batch_size=1,
                 device='cuda'):
        super().__init__(test_set_root=test_set_root, device=device,
                         batch_size=batch_size)
        self.config=config
        self.label_table=label_table or {}

    def _denoise_candidates(self, candidates, out_dir, page_rgb=None):
        """执行连通域残片清理；方法名为兼容已有 v2 输出而保留。"""
        return self._cleanup_fragments(candidates, out_dir, page_rgb=page_rgb)

    def _match(self, predictions, ground_truths):
        """按正式评估口径，用未扩边候选框与真值框进行 IoU 匹配。"""
        return self.match_predictions_to_ground_truth(predictions, ground_truths)

# 图像优化顺序对照：sr_then_arrow = 超分辨率→箭头去除；
# arrow_then_sr = 箭头去除→超分辨率。full 仍保留原有含义和输出目录。
ORDER_COMPARISON_CONFIGS = {
    'sr_then_arrow': 'sr_then_arrow',
    'arrow_then_sr': 'arrow_then_sr',
}

def save_json(path,data):
    """先写临时文件再替换目标文件，避免中断留下不完整 JSON。"""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8');temp.replace(path)

def image_key(rgb):
    """由数组形状和 RGB 字节生成内容哈希，用作产物标识。"""
    h=hashlib.sha256();h.update(str(rgb.shape).encode());h.update(rgb.tobytes());return h.hexdigest()

def serial_conversion():
    """把 MolScribe 图转 SMILES 固定为串行转换，保证 Windows 运行稳定。"""
    import molscribe.interface as iface
    from molscribe.chemistry import _convert_graph_to_smiles
    def convert(coords,symbols,edges,images=None,num_workers=1):
        args=zip(coords,symbols,edges) if images is None else zip(coords,symbols,edges,images)
        results=[_convert_graph_to_smiles(*x) for x in args]
        smi,mol,ok=zip(*results)
        return smi,mol,float(np.mean(ok))
    iface.convert_graph_to_smiles=convert


class Runner:
    """管理共享模型、本次运行产物、图像优化、识别和逐页评分。"""

    def __init__(self,out):
        self.out=out;self.artifacts=out/'artifacts';self.artifacts.mkdir(parents=True,exist_ok=True)
        self.evaluator=AblationEvaluator(ABLATION_CONFIGS['full'],TEST_SET_ROOT,{},batch_size=1)
        # 正式评估器默认构造统一四分类模型；本实验确定使用旧版双模型并集，
        # 因此释放统一模型，只保留同一对象中的 HAT 超分模型。
        self.evaluator.arrow_sr.unified_model=None
        if torch.cuda.is_available():torch.cuda.empty_cache()
        device=self.evaluator.arrow_sr.device
        self.arrow_predictors=[
            Predictor(SOLID_ARROW_MODEL,device,'solid'),
            Predictor(DASHED_ARROW_MODEL,device,'dashed'),
        ]
        self.predict_calls=0

    def write_image(self,rgb):
        """保存本次运行的 RGB 产物，并返回内容哈希和文件路径。"""
        key=image_key(rgb);path=self.artifacts/'images'/f'{key}.png'
        path.parent.mkdir(exist_ok=True);Image.fromarray(rgb).save(path)
        return key,str(path)

    def recognize(self,rgb):
        """每次重新运行 MolScribe，并转换为统一的化学图表示。"""
        key,path=self.write_image(rgb);target=self.artifacts/'predictions'/f'{key}.json'
        pred=self.evaluator.img2smiles.predict_batch([rgb],batch_size=1)[0]
        save_json(target,pred);self.predict_calls+=1
        chem=convert_graph(pred.get('atoms',[]),pred.get('bonds',[]),image_size=[rgb.shape[1],rgb.shape[0]],coordinates_normalized=True)
        return pred,chem,key,path

    def sr(self,rgb):
        """每次重新运行 HAT 超分辨率。"""
        key=image_key(rgb);meta=self.artifacts/'sr'/f'{key}.json'
        bgr=self.evaluator.arrow_sr.sr_model.predict(cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
        out=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);_,path=self.write_image(out)
        save_json(meta,{'path':path});return out

    def erase(self,rgb,mode):
        """运行双模型及后处理，并以化学键并集保护最终擦除区域。"""
        key=image_key(rgb)
        artifact_mode=f'{ARROW_PIPELINE_VERSION}_{mode}'
        target=self.artifacts/'arrow'/artifact_mode/f'{key}.json'
        probs=[predictor(rgb) for predictor in self.arrow_predictors]
        labels=[prob.argmax(2).astype(np.uint8) for prob in probs]
        solid_arrow=refine_solid_arrow(rgb,labels[0])
        dashed_arrow=refine_dashed_arrow(
            labels[1],probs[1][...,1],DASHED_ARROW_POSTPROCESS)
        solid_bond=labels[0]==2;dashed_bond=labels[1]==2
        arrow_union=solid_arrow|dashed_arrow
        bond_union=solid_bond|dashed_bond
        protected_erased=arrow_union&~bond_union
        if mode=='unprotected':
            erased=arrow_union
        else:
            erased=protected_erased
        out=rgb.copy();out[erased]=255
        arrays={
            'solid_proposed':solid_arrow,
            'dashed_proposed':dashed_arrow,
            'arrow_union':arrow_union,
            'solid_bond':solid_bond,
            'dashed_bond':dashed_bond,
            'bond':bond_union,
            'erased':erased,
        }
        meta={'version':ARROW_PIPELINE_VERSION,'protection_mode':mode,
              'solid_postprocess':'hsv_halo_3x3_iterations_3_saturation_gt_30',
              'dashed_postprocess':DASHED_ARROW_POSTPROCESS,
              'solid_model':str(SOLID_ARROW_MODEL),
              'dashed_model':str(DASHED_ARROW_MODEL)}
        _,path=self.write_image(out)
        target.parent.mkdir(parents=True,exist_ok=True)
        masks=target.with_suffix('.npz');np.savez_compressed(masks,**arrays)
        save_json(target,{**meta,'path':path,'input_key':key,'masks':str(masks),
            'erased_ink_pixels':int((arrays['erased'] & (cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)<200)).sum())})
        return out,str(target)

    def optimize(self,cand,cfg,name):
        """按配置决定是否超分，并执行指定的超分/去箭头顺序。"""
        rgb=np.asarray(Image.open(cand['crop_path']).convert('RGB'))
        _,crop_image=self.write_image(rgb)
        is_low=False
        if cfg.sr=='selective':is_low=bool(self.evaluator.res_classifier.predict(str(cand['crop_path'])).get('is_low_res',False))
        do_sr=cfg.sr=='uniform' or (cfg.sr=='selective' and is_low)
        mode='unprotected' if name=='full_unprotected' else 'protected'
        arrow_meta=None
        if cfg.order=='sr_first':
            if do_sr:rgb=self.sr(rgb)
            _,before=self.write_image(rgb)
            if cfg.arrow:rgb,arrow_meta=self.erase(rgb,mode)
        else:
            _,before=self.write_image(rgb)
            if cfg.arrow:rgb,arrow_meta=self.erase(rgb,mode)
            if do_sr:rgb=self.sr(rgb)
        return rgb,{'is_low_res':is_low,'is_sr':do_sr,'crop_image':crop_image,'before_arrow_image':before,'arrow_meta':arrow_meta}

    def evaluate(self,page,candidates,cfg,name):
        """识别一页候选，与冻结真值匹配并生成逐实例比较结果。"""
        predictions=[]
        for cand in candidates:
            rgb,meta=self.optimize(cand,cfg,name)
            pred,chem,key,path=self.recognize(rgb)
            predictions.append({'bbox':cand['raw_bbox'],'exp_bbox':cand['exp_bbox'],
                'confidence':cand['confidence'],'fg_ratio':cand.get('fg_ratio'),**meta,
                'image_key':key,'image':path,'raw_prediction':pred,'chemistry':chem,
                'smiles_raw':pred.get('smiles',''),'smiles_recon':chem['smiles'],'is_valid':chem['valid']})
        gt=page['molecules']
        matched,fp,fn=self.evaluator._match(predictions,gt)
        comparisons=[]
        for pi,gi,iou in matched:
            pred=predictions[pi];g=gt[gi]
            score=legacy_compare(g['smiles'],pred['smiles_recon'])
            comparisons.append({'pred_idx':pi,'gt_idx':gi,'iou':iou,'gt_molecule_id':g['molecule_id'],
                'recognition_eligible':g.get('recognition_eligible',True),
                'canonical_match':score['cem'],'graph_match':score['gem'],
                'pred_valid':score['scoring_valid'],'pred_strict_valid':pred['is_valid'],
                'gt_valid':legacy_keys(g['smiles'])[0] is not None,'tanimoto':score['tanimoto'],
                'is_sr':pred['is_sr'],'is_low_res':pred['is_low_res']})
        return {'page_id':page['page_id'],'page_file':page['page_file'],'n_ground_truth':len(gt),
            'n_predictions':len(predictions),'n_matched':len(matched),'n_false_positives':len(fp),'n_false_negatives':len(fn),
            'n_super_resolved':sum(p['is_sr'] for p in predictions),'predictions':predictions,
            'ground_truths':gt,'matched_results':comparisons,'unmatched_gt_indices':fn,'unmatched_pred_indices':fp,
            'scoring_version':LEGACY_SCORING_VERSION,'cem_gem_denominator':'all_ground_truth',
            'arrow_pipeline_version':ARROW_PIPELINE_VERSION}


def rescore_page_result(result):
    """将已有逐页预测原地迁移到旧表A评分，无需重新执行模型推理。"""
    if result.get('scoring_version')==LEGACY_SCORING_VERSION:
        return result,False
    predictions=result['predictions'];ground_truths=result['ground_truths']
    for row in result['matched_results']:
        pred=predictions[row['pred_idx']];truth=ground_truths[row['gt_idx']]
        pred_smiles=pred.get('smiles_recon') or pred.get('chemistry',{}).get('smiles','')
        score=legacy_compare(truth['smiles'],pred_smiles)
        row.update(canonical_match=score['cem'],graph_match=score['gem'],
            pred_valid=score['scoring_valid'],pred_strict_valid=pred.get('is_valid',False),
            gt_valid=legacy_keys(truth['smiles'])[0] is not None,tanimoto=score['tanimoto'],
            recognition_eligible=truth.get('recognition_eligible',True))
    result['scoring_version']=LEGACY_SCORING_VERSION
    result['cem_gem_denominator']='all_ground_truth'
    return result,True


def metrics(results):
    """汇总检测与识别指标。

    P/R/F1/Coverage 使用全部GT；Validity和Tan.使用匹配对；CEM、GEM和
    Tan@all-GT使用全部GT作为分母，因此漏检和无效预测自动贡献0。
    """
    gt=sum(r['n_ground_truth'] for r in results);pred=sum(r['n_predictions'] for r in results)
    matches=[m for r in results for m in r['matched_results']];n=len(matches)
    p=n/pred if pred else 0;r=n/gt if gt else 0
    n_valid=sum(m['pred_valid'] for m in matches)
    n_cem=sum(m['canonical_match'] for m in matches)
    n_gem=sum(m['graph_match'] for m in matches)
    tan_sum=sum(m.get('tanimoto') or 0 for m in matches)
    return {'P':p,'R':r,'F1':2*p*r/(p+r) if p+r else 0,'Cov.':r,
        'Val.':n_valid/n if n else 0,'CEM':n_cem/gt if gt else 0,
        'GEM':n_gem/gt if gt else 0,'Tan.':tan_sum/n if n else 0,
        'Tan@all-GT':tan_sum/gt if gt else 0,
        'n_gt':gt,'n_pred':pred,'n_matched':n,'n_eligible_gt':gt,'n_eligible_matched':n,
        'n_valid':n_valid,'n_cem':n_cem,'n_gem':n_gem,'n_gem_only':n_gem-n_cem,
        'n_sr':sum(r['n_super_resolved'] for r in results),'pending_gt':0,
        'chemistry_schema':SCHEMA_VERSION,'scoring_version':LEGACY_SCORING_VERSION,
        'cem_gem_denominator':'all_ground_truth'}


def build_order_comparison(summaries):
    """比较两种处理顺序，并计算“超分→去箭头”减去反向顺序的差值。"""
    required=set(ORDER_COMPARISON_CONFIGS.values())
    if not required.issubset(summaries):
        return None
    sr_then_arrow=summaries[ORDER_COMPARISON_CONFIGS['sr_then_arrow']]
    arrow_then_sr=summaries[ORDER_COMPARISON_CONFIGS['arrow_then_sr']]
    metric_names=('Val.','CEM','GEM','Tan.','Tan@all-GT')
    return {
        'comparison':'image_optimization_order',
        'rows':[
            {
                'config':'arrow_then_sr',
                'execution_order':'arrow_removal_then_super_resolution',
                **{key:arrow_then_sr[key] for key in metric_names},
            },
            {
                'config':'sr_then_arrow',
                'execution_order':'super_resolution_then_arrow_removal',
                **{key:sr_then_arrow[key] for key in metric_names},
            },
        ],
        'delta_sr_then_arrow_minus_arrow_then_sr':{
            key:sr_then_arrow[key]-arrow_then_sr[key] for key in metric_names
        },
        'n_gt':sr_then_arrow['n_gt'],
        'n_eligible_gt':sr_then_arrow['n_eligible_gt'],
        'chemistry_schema':SCHEMA_VERSION,
        'scoring_version':LEGACY_SCORING_VERSION,
        'cem_gem_denominator':'all_ground_truth',
    }


def main():
    """按页面运行所选配置，持续写入中间结果并生成最终汇总。"""
    parser=argparse.ArgumentParser(
        description='运行 MSIEOF v2 消融实验及超分辨率/箭头去除顺序对比')
    parser.add_argument(
        '--output',type=Path,default=DEFAULT_OUT,
        help='结果、中间产物和运行状态目录')
    parser.add_argument(
        '--gt',type=Path,default=DEFAULT_GT,
        help='由 E 盘 PageToSMILES_approved 原始标注生成的页面级真值 JSONL')
    parser.add_argument(
        '--configs',nargs='+',default=DEFAULT_CONFIGS,
        help=('要运行的配置；默认运行全部消融配置，并且不复用历史结果'))
    parser.add_argument(
        '--max-pages',type=int,
        help='仅运行前 N 页，用于快速检查；不设置则运行全部页面')
    parser.add_argument(
        '--resume',action='store_true',
        help='从当前输出目录中已完整写出的逐页结果继续；仅用于中断恢复')
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    valid_configs=set(ABLATION_CONFIGS)|{'full_unprotected'}
    unknown_configs=[name for name in args.configs if name not in valid_configs]
    if unknown_configs:
        parser.error(f'未知配置: {", ".join(unknown_configs)}')
    if not args.gt.is_file():
        parser.error(f'缺少页面级真值 JSONL: {args.gt}')
    if not TEST_SET_ROOT.is_dir():
        parser.error(f'缺少页面图像目录: {TEST_SET_ROOT}')
    for model_path in (SOLID_ARROW_MODEL,DASHED_ARROW_MODEL):
        if not model_path.is_file():parser.error(f'缺少双模型权重: {model_path}')
    pages=[json.loads(x) for x in args.gt.read_text(encoding='utf8').splitlines() if x.strip()]
    if args.max_pages:pages=pages[:args.max_pages]
    start=time.time()
    torch.manual_seed(0);np.random.seed(0);serial_conversion();runner=Runner(out)
    cache_policy=('explicit-resume-existing-page-results' if args.resume else 'disabled')
    save_json(out/'status.json',{'state':'loading','completed_pages':0,'total_pages':len(pages),
        'configs':args.configs,'scoring_version':LEGACY_SCORING_VERSION,
        'cache_policy':cache_policy,'needs_model_inference':True})
    all_results={name:[] for name in args.configs}
    for page_index,page in enumerate(pages):
        pid=page['page_id']
        existing={}
        if args.resume:
            for name in args.configs:
                result_path=out/name/'pages'/f'{pid}.json'
                if result_path.is_file():
                    try:
                        existing[name]=json.loads(result_path.read_text(encoding='utf8'))
                    except (OSError,json.JSONDecodeError):
                        pass
        if len(existing)==len(args.configs):
            for name in args.configs:all_results[name].append(existing[name])
            save_json(out/'status.json',{'state':'running','completed_pages':page_index+1,
                'total_pages':len(pages),'page':pid,'config':'resume-skip-complete-page',
                'elapsed_seconds':time.time()-start,'scoring_version':LEGACY_SCORING_VERSION,
                'cache_policy':cache_policy})
            print(f'{page_index+1}/{len(pages)} resume {pid}: complete page reused',flush=True)
            continue
        save_json(out/'status.json',{'state':'running','completed_pages':page_index,'total_pages':len(pages),
            'page':pid,'config':'preprocess','elapsed_seconds':time.time()-start,
            'scoring_version':LEGACY_SCORING_VERSION,'cache_policy':cache_policy})
        work=out/'work'/pid
        rgb=np.asarray(Image.open(TEST_SET_ROOT/page['page_file']).convert('RGB'))
        raw=runner.evaluator._extract_candidates(rgb,work/'crop',pid)
        denoised=runner.evaluator._denoise_candidates(
            copy.deepcopy(raw),work/'denoised',page_rgb=rgb)
        filtered=runner.evaluator._filter_candidates(copy.deepcopy(denoised),work/'kept')
        raw_filtered=(runner.evaluator._filter_candidates(
            copy.deepcopy(raw),work/'raw_kept') if 'no_cc' in args.configs else [])
        for name in args.configs:
            if name in existing:
                print(f'{page_index+1}/{len(pages)} resume {name} {pid}: result reused',flush=True)
                continue
            cfg=ABLATION_CONFIGS['full' if name=='full_unprotected' else name]
            candidates=raw_filtered if name=='no_cc' else (denoised if not cfg.filter else filtered)
            save_json(out/'status.json',{'state':'running','completed_pages':page_index,'total_pages':len(pages),
                'page':pid,'config':name,'new_model_predictions':runner.predict_calls,
                'elapsed_seconds':time.time()-start,'scoring_version':LEGACY_SCORING_VERSION,
                'cache_policy':cache_policy})
            result=runner.evaluate(page,candidates,cfg,name)
            result['config']=asdict(cfg);result['config']['name']=name
            save_json(out/name/'pages'/f'{pid}.json',result)
            print(f'{page_index+1}/{len(pages)} {name} {pid}: {len(result["matched_results"])} matches',flush=True)
        for name in args.configs:
            all_results[name].append(json.loads((out/name/'pages'/f'{pid}.json').read_text(encoding='utf8')))
            save_json(out/name/'summary.json',metrics(all_results[name]))
        save_json(out/'partial_table.json',{name:metrics(rows) for name,rows in all_results.items()})
        torch.cuda.empty_cache()
    summaries={}
    for name,rows in all_results.items():
        save_json(out/name/'results.json',rows);summaries[name]=metrics(rows)
    save_json(out/'table_A.json',summaries)
    order_comparison=build_order_comparison(summaries)
    if order_comparison is not None:
        save_json(out/'order_comparison.json',order_comparison)
    save_json(out/'status.json',{'state':'complete','completed_pages':len(pages),'total_pages':len(pages),
        'configs':args.configs,'new_model_predictions':runner.predict_calls,
        'cache_policy':cache_policy,'scoring_version':LEGACY_SCORING_VERSION,
        'cem_gem_denominator':'all_ground_truth','elapsed_seconds':time.time()-start})
    print(json.dumps(summaries,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
