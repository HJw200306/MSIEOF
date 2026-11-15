"""
改进虚线箭头去除的后处理
目标：减少视觉残留，同时提升SMILES识别准确度
"""
import os
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from molscribe import MolScribe
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import segmentation_models_pytorch as smp
from scipy import ndimage

# ==================== 配置 ====================
INPUT_IMAGE = r"E:\MSIREOT_DATA\journal\images\JNP_2024_04_1059-1066_004.png"
OUTPUT_DIR = r"C:\Users\Admin\PycharmProjects\msirost\outputs\improved"
MOLSCRIBE_MODEL = r"D:\workspace\Model\swin_base_char_aux_1m680k.pth"
MODEL_3CLASS = str(Path(__file__).resolve().parents[2] / "model" / "legacy_arrow_models" / "bond_arrow_best.pth")

# ==================== 模型定义 ====================
class ArrowUNetPlusPlus(torch.nn.Module):
    def __init__(self, encoder="resnet101", num_classes=3):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=3,
            classes=num_classes,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)

# ==================== SMILES工具 ====================
def normalize_smiles(smiles_str):
    if not smiles_str or smiles_str.strip() == "":
        return ""
    parts = smiles_str.split('.')
    filtered_parts = []
    for part in parts:
        if '~' not in part:
            filtered_parts.append(part)
        elif len(part) > 3:
            filtered_parts.append(part.replace('~', ''))
    normalized = '.'.join(filtered_parts)
    normalized = re.sub(r'\[R\d*\]', '*', normalized)
    return normalized

def compute_tanimoto(smiles1, smiles2):
    try:
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

def compute_tanimoto_gem(smiles1, smiles2):
    try:
        s1 = re.sub(r'[@/\\]', '', smiles1)
        s2 = re.sub(r'[@/\\]', '', smiles2)
        mol1 = Chem.MolFromSmiles(s1)
        mol2 = Chem.MolFromSmiles(s2)
        if mol1 is None or mol2 is None:
            return 0.0
        fp1 = AllChem.GetMorganFingerprint(mol1, 2)
        fp2 = AllChem.GetMorganFingerprint(mol2, 2)
        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    except:
        return 0.0

# ==================== 图像处理工具 ====================
def preprocess_image(image_np):
    """预处理，确保尺寸是32的倍数"""
    h, w = image_np.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    if pad_h > 0 or pad_w > 0:
        image_np = cv2.copyMakeBorder(
            image_np, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor, (h, w), image_np.shape[:2]

def get_mask_prediction(img_pil, model, device):
    """获取模型预测掩膜"""
    img_np = np.array(img_pil)
    img_tensor, orig_hw, padded_hw = preprocess_image(img_np)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)

    pred_np = pred.squeeze(0).cpu().numpy()
    pred_np = pred_np[:orig_hw[0], :orig_hw[1]]

    # 同时返回概率图用于后处理
    probs_np = probs.squeeze(0).cpu().numpy()
    probs_np = probs_np[:, :orig_hw[0], :orig_hw[1]]

    return pred_np, probs_np

# ==================== 后处理策略 ====================
def postprocess_v1_dilate(pred_mask, dilation_kernel=3):
    """策略1: 膨胀虚线箭头掩膜"""
    mask_dashed = (pred_mask == 1).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel, dilation_kernel))
    mask_dashed_dilated = cv2.dilate(mask_dashed, kernel, iterations=1)
    return mask_dashed_dilated.astype(bool)

def postprocess_v2_morphology(pred_mask, close_kernel=5, dilate_kernel=3):
    """策略2: 形态学闭运算 + 膨胀"""
    mask_dashed = (pred_mask == 1).astype(np.uint8)

    # 闭运算填补小洞
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    mask_closed = cv2.morphologyEx(mask_dashed, cv2.MORPH_CLOSE, kernel_close)

    # 再膨胀
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    mask_final = cv2.dilate(mask_closed, kernel_dilate, iterations=1)

    return mask_final.astype(bool)

def postprocess_v3_confidence_threshold(pred_mask, probs, conf_threshold=0.3):
    """策略3: 降低置信度阈值，捕获更多虚线箭头"""
    # 原始预测（argmax）
    mask_dashed = (pred_mask == 1).astype(bool)

    # 低置信度的虚线箭头区域（概率>阈值但未被argmax选中）
    prob_dashed = probs[1]  # class 1的概率
    mask_low_conf = (prob_dashed > conf_threshold) & (~mask_dashed)

    # 合并
    mask_final = mask_dashed | mask_low_conf
    return mask_final

def postprocess_v4_hybrid(pred_mask, probs, conf_threshold=0.25, dilate_kernel=3):
    """策略4: 混合方法 - 置信度阈值 + 形态学"""
    # 先用置信度阈值扩展
    mask_dashed = (pred_mask == 1).astype(bool)
    prob_dashed = probs[1]
    mask_low_conf = (prob_dashed > conf_threshold) & (~mask_dashed)
    mask_expanded = mask_dashed | mask_low_conf

    # 形态学闭运算
    mask_uint8 = mask_expanded.astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)

    # 轻微膨胀
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    mask_final = cv2.dilate(mask_closed, kernel_dilate, iterations=1)

    return mask_final.astype(bool)

def postprocess_v5_bond_aware(pred_mask, probs, conf_threshold=0.25, dilate_kernel=3, bond_buffer=2):
    """策略5: 避免损伤化学键的混合方法"""
    # 先用混合方法
    mask_dashed = (pred_mask == 1).astype(bool)
    prob_dashed = probs[1]
    mask_low_conf = (prob_dashed > conf_threshold) & (~mask_dashed)
    mask_expanded = mask_dashed | mask_low_conf

    # 形态学处理
    mask_uint8 = mask_expanded.astype(np.uint8)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_kernel, dilate_kernel))
    mask_dilated = cv2.dilate(mask_closed, kernel_dilate, iterations=1)

    # 保护化学键区域：从膨胀后的掩膜中减去化学键缓冲区
    mask_bond = (pred_mask == 2).astype(np.uint8)
    if bond_buffer > 0:
        kernel_bond = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (bond_buffer, bond_buffer))
        mask_bond_buffered = cv2.dilate(mask_bond, kernel_bond, iterations=1)
    else:
        mask_bond_buffered = mask_bond

    # 从虚线掩膜中减去化学键保护区
    mask_final = mask_dilated & (~mask_bond_buffered.astype(bool))

    return mask_final

def apply_mask_to_image(img_np, mask):
    """应用掩膜去除箭头"""
    img_removed = img_np.copy()
    img_removed[mask] = [255, 255, 255]
    return img_removed

# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("改进虚线箭头去除 - 后处理实验")
    print("=" * 80)
    print(f"输入: {INPUT_IMAGE}")
    print(f"输出: {OUTPUT_DIR}\n")

    # 加载图片
    img_bgr = cv2.imread(INPUT_IMAGE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 加载模型
    print("⏳ 加载模型...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(MODEL_3CLASS, map_location=device)
    model = ArrowUNetPlusPlus(
        encoder=checkpoint.get('encoder', 'resnet101'),
        num_classes=checkpoint.get('num_classes', 3)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ 模型加载完成\n")

    # 获取预测
    print("⏳ 获取模型预测...")
    pred_mask, probs = get_mask_prediction(img_pil, model, device)
    print("✅ 预测完成\n")

    # 定义后处理策略
    strategies = [
        ("baseline", "基线（无后处理）", lambda: (pred_mask == 1)),
        ("v1_dilate3", "膨胀3x3", lambda: postprocess_v1_dilate(pred_mask, 3)),
        ("v1_dilate5", "膨胀5x5", lambda: postprocess_v1_dilate(pred_mask, 5)),
        ("v2_morph", "形态学(close5+dilate3)", lambda: postprocess_v2_morphology(pred_mask, 5, 3)),
        ("v3_conf30", "置信度阈值0.30", lambda: postprocess_v3_confidence_threshold(pred_mask, probs, 0.30)),
        ("v3_conf25", "置信度阈值0.25", lambda: postprocess_v3_confidence_threshold(pred_mask, probs, 0.25)),
        ("v4_hybrid25", "混合方法(conf0.25+dilate3)", lambda: postprocess_v4_hybrid(pred_mask, probs, 0.25, 3)),
        ("v4_hybrid30", "混合方法(conf0.30+dilate3)", lambda: postprocess_v4_hybrid(pred_mask, probs, 0.30, 3)),
        ("v5_bond_aware", "键保护混合(conf0.25+bond2)", lambda: postprocess_v5_bond_aware(pred_mask, probs, 0.25, 3, 2)),
    ]

    # 加载MolScribe
    print("⏳ 加载MolScribe...")
    molscribe = MolScribe(MOLSCRIBE_MODEL, device=device)
    print("✅ MolScribe加载完成\n")

    # 测试所有策略
    results = []

    # 原图
    print("=" * 80)
    print("测试原图（无去箭头）")
    print("=" * 80)
    img_np = np.array(img_pil)
    try:
        pred = molscribe.predict_images([img_np], return_atoms_bonds=False)[0]
        smiles_orig = pred.get('smiles', '')
        smiles_orig_norm = normalize_smiles(smiles_orig)
        print(f"原始SMILES: {smiles_orig[:100]}...")
        print(f"归一化SMILES: {smiles_orig_norm[:100]}...\n")
    except Exception as e:
        print(f"❌ 识别失败: {e}\n")
        smiles_orig_norm = ""

    # 测试各策略
    for strategy_id, strategy_name, strategy_func in strategies:
        print("=" * 80)
        print(f"测试策略: {strategy_name} ({strategy_id})")
        print("=" * 80)

        # 应用后处理
        mask_final = strategy_func()

        # 去除箭头
        img_removed = apply_mask_to_image(img_np, mask_final)

        # 保存图片
        img_path = os.path.join(OUTPUT_DIR, f"{strategy_id}.png")
        cv2.imwrite(img_path, cv2.cvtColor(img_removed, cv2.COLOR_RGB2BGR))

        # 保存掩膜可视化
        mask_vis = np.zeros_like(img_np)
        mask_vis[mask_final] = [255, 0, 0]
        mask_path = os.path.join(OUTPUT_DIR, f"{strategy_id}_mask.png")
        cv2.imwrite(mask_path, cv2.cvtColor(mask_vis, cv2.COLOR_RGB2BGR))

        print(f"  掩膜像素数: {mask_final.sum()}")
        print(f"  掩膜覆盖率: {mask_final.sum() / mask_final.size * 100:.2f}%")

        # SMILES识别
        try:
            pred = molscribe.predict_images([img_removed], return_atoms_bonds=False)[0]
            smiles = pred.get('smiles', '')
            smiles_norm = normalize_smiles(smiles)

            print(f"  SMILES: {smiles[:80]}...")

            # 计算相似度
            if smiles_orig_norm and smiles_norm:
                cem = compute_tanimoto(smiles_orig_norm, smiles_norm)
                gem = compute_tanimoto_gem(smiles_orig_norm, smiles_norm)
            else:
                cem, gem = 0.0, 0.0

            print(f"  CEM Tanimoto: {cem:.4f}")
            print(f"  GEM Tanimoto: {gem:.4f}")

            results.append({
                'strategy_id': strategy_id,
                'strategy_name': strategy_name,
                'mask_pixels': int(mask_final.sum()),
                'mask_coverage': float(mask_final.sum() / mask_final.size),
                'smiles_raw': smiles,
                'smiles_normalized': smiles_norm,
                'tanimoto_cem': cem,
                'tanimoto_gem': gem
            })

        except Exception as e:
            print(f"  ❌ SMILES识别失败: {e}")
            results.append({
                'strategy_id': strategy_id,
                'strategy_name': strategy_name,
                'mask_pixels': int(mask_final.sum()),
                'mask_coverage': float(mask_final.sum() / mask_final.size),
                'smiles_raw': '',
                'smiles_normalized': '',
                'tanimoto_cem': 0.0,
                'tanimoto_gem': 0.0
            })

        print()

    # 保存结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "postprocess_comparison.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    print("=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(df[['strategy_name', 'mask_pixels', 'mask_coverage', 'tanimoto_cem', 'tanimoto_gem']].to_string(index=False))

    print("\n" + "=" * 80)
    print("🎉 实验完成！")
    print("=" * 80)
    print(f"结果保存在: {OUTPUT_DIR}")
    print(f"  - <strategy_id>.png: 各策略处理后的图片")
    print(f"  - <strategy_id>_mask.png: 各策略的掩膜可视化")
    print(f"  - postprocess_comparison.csv: 对比结果")

if __name__ == "__main__":
    main()
