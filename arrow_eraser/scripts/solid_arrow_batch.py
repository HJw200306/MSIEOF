"""
使用正确的实线箭头模型（VGG16-U-Net）批量测试6张图片
包含完整的可视化对比
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from molscribe import MolScribe
from rdkit import Chem
from rdkit.Chem import AllChem
import re

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from split_tools.unet import Unet

# ==================== 配置 ====================
IMAGE_DIR = r"E:\MSIREOT_DATA\journal\images"
OUTPUT_DIR = r"C:\Users\Admin\PycharmProjects\msirost\outputs\solid_arrow_vgg16"
MOLSCRIBE_MODEL = r"D:\workspace\Model\swin_base_char_aux_1m680k.pth"
MODEL_PATH = str(PROJECT_ROOT / "model" / "legacy_arrow_models" / "misp_weights_no_freeze.pth")

TEST_IMAGES = [
    "JNP_2024_02_266-275_004.png",
    "JNP_2024_02_297-303_002.png",
    "JNP_2024_02_365-370_003.png",
    "JNP_2024_02_396-403_002.png",
    "JNP_2024_04_1059-1066_004.png",
    "Phyt_2024_217_113926_006.png"
]

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

# ==================== 箭头去除 ====================
def remove_arrows(img_pil, model, config):
    """使用VGG16-UNet去除箭头"""
    # 预测
    pred_mask = model.detect_image(
        img_pil,
        count=False,
        name_classes=config['name_classes'],
        mix_type=1
    )

    # 转换为numpy
    img_np = np.array(img_pil)
    pred_np = np.array(pred_mask)

    # 提取箭头掩膜（红色区域）
    arrow_mask = (pred_np[:, :, 0] == 255) & \
                 (pred_np[:, :, 1] == 0) & \
                 (pred_np[:, :, 2] == 0)

    # 去除箭头
    img_removed = img_np.copy()
    img_removed[arrow_mask] = [255, 255, 255]

    # 统计
    arrow_pixels = arrow_mask.sum()
    bond_mask = (pred_np[:, :, 0] == 0) & \
                (pred_np[:, :, 1] == 128) & \
                (pred_np[:, :, 2] == 0)
    bond_pixels = bond_mask.sum()

    return Image.fromarray(img_removed), pred_mask, arrow_pixels, bond_pixels

def create_visualization(original, removed, mask, output_path):
    """创建三图对比可视化"""
    orig_np = np.array(original)
    removed_np = np.array(removed)
    mask_np = np.array(mask)

    # 水平拼接
    combined = np.hstack([orig_np, removed_np, mask_np])

    # 保存
    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("实线箭头模型（VGG16-U-Net）批量测试")
    print("=" * 80)
    print(f"模型: misp_weights_no_freeze.pth")
    print(f"测试图片: {len(TEST_IMAGES)}")
    print(f"输出: {OUTPUT_DIR}\n")

    # 配置
    config = {
        'num_classes': 3,
        'name_classes': ["_background_", "arrow", "bond"]
    }

    # 加载模型
    print("⏳ 加载实线箭头模型...")
    model = Unet(
        MODEL_PATH,
        num_classes=config['num_classes'],
        input_shape=[512, 512]
    )
    print("✅ 模型加载成功\n")

    # 加载MolScribe
    print("⏳ 加载MolScribe...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    molscribe = MolScribe(MOLSCRIBE_MODEL, device=device)
    print("✅ MolScribe加载完成\n")

    # 处理每张图片
    all_results = []

    for img_name in TEST_IMAGES:
        print("=" * 80)
        print(f"处理: {img_name}")
        print("=" * 80)

        img_path = os.path.join(IMAGE_DIR, img_name)
        if not os.path.exists(img_path):
            print(f"⚠️ 图片不存在，跳过\n")
            continue

        # 加载图片
        img_pil = Image.open(img_path).convert("RGB")
        img_np = np.array(img_pil)

        # 创建输出目录
        img_output_dir = os.path.join(OUTPUT_DIR, img_name.replace('.png', ''))
        os.makedirs(img_output_dir, exist_ok=True)

        # 保存原图
        img_pil.save(os.path.join(img_output_dir, "1_original.png"))

        # 识别原图SMILES
        print("  识别原图...")
        try:
            pred = molscribe.predict_images([img_np], return_atoms_bonds=False)[0]
            smiles_orig = pred.get('smiles', '')
            smiles_orig_norm = normalize_smiles(smiles_orig)
            print(f"    原图SMILES: {smiles_orig[:60]}...")
        except Exception as e:
            print(f"    ❌ 识别失败: {e}")
            smiles_orig_norm = ""

        # 去除箭头
        print("  去除箭头...")
        img_removed, mask_vis, arrow_pixels, bond_pixels = remove_arrows(img_pil, model, config)

        print(f"    检测到箭头: {arrow_pixels} 像素")
        print(f"    检测到化学键: {bond_pixels} 像素")

        # 保存处理后的图片和掩膜
        img_removed.save(os.path.join(img_output_dir, "2_removed_arrows.png"))
        mask_vis.save(os.path.join(img_output_dir, "3_mask_visualization.png"))

        # 创建对比可视化
        vis_path = os.path.join(img_output_dir, "4_comparison.png")
        create_visualization(img_pil, img_removed, mask_vis, vis_path)
        print(f"    ✅ 可视化保存: 4_comparison.png")

        # 识别去箭头后的SMILES
        print("  识别去箭头后...")
        try:
            img_removed_np = np.array(img_removed)
            pred = molscribe.predict_images([img_removed_np], return_atoms_bonds=False)[0]
            smiles_removed = pred.get('smiles', '')
            smiles_removed_norm = normalize_smiles(smiles_removed)
            print(f"    去箭头SMILES: {smiles_removed[:60]}...")

            # 计算相似度
            if smiles_orig_norm and smiles_removed_norm:
                cem = compute_tanimoto(smiles_orig_norm, smiles_removed_norm)
                gem = compute_tanimoto_gem(smiles_orig_norm, smiles_removed_norm)
            else:
                cem, gem = 0.0, 0.0

            print(f"    CEM Tanimoto: {cem:.4f}")
            print(f"    GEM Tanimoto: {gem:.4f}")

            all_results.append({
                'image': img_name,
                'arrow_pixels': arrow_pixels,
                'bond_pixels': bond_pixels,
                'arrow_coverage': arrow_pixels / (img_np.shape[0] * img_np.shape[1]),
                'smiles_original': smiles_orig,
                'smiles_removed': smiles_removed,
                'smiles_original_norm': smiles_orig_norm,
                'smiles_removed_norm': smiles_removed_norm,
                'tanimoto_cem': cem,
                'tanimoto_gem': gem
            })

        except Exception as e:
            print(f"    ❌ 识别失败: {e}")
            all_results.append({
                'image': img_name,
                'arrow_pixels': arrow_pixels,
                'bond_pixels': bond_pixels,
                'arrow_coverage': arrow_pixels / (img_np.shape[0] * img_np.shape[1]),
                'smiles_original': smiles_orig,
                'smiles_removed': '',
                'smiles_original_norm': smiles_orig_norm,
                'smiles_removed_norm': '',
                'tanimoto_cem': 0.0,
                'tanimoto_gem': 0.0
            })

        print()

    # 保存结果
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')

    # 打印汇总
    print("=" * 80)
    print("测试结果汇总")
    print("=" * 80)
    for _, row in df.iterrows():
        print(f"\n{row['image']}:")
        print(f"  箭头检测: {row['arrow_pixels']} 像素 ({row['arrow_coverage']*100:.3f}%)")
        print(f"  化学键检测: {row['bond_pixels']} 像素")
        print(f"  CEM Tanimoto: {row['tanimoto_cem']:.4f}")
        print(f"  GEM Tanimoto: {row['tanimoto_gem']:.4f}")

    # 有效性评估
    print("\n" + "=" * 80)
    print("有效性评估")
    print("=" * 80)
    for _, row in df.iterrows():
        if row['arrow_pixels'] == 0:
            status = "⚠️ 未检测到箭头"
        elif row['tanimoto_cem'] >= 0.8:
            status = "✅ 高度有效"
        elif row['tanimoto_cem'] >= 0.5:
            status = "✅ 有效"
        elif row['tanimoto_cem'] >= 0.1:
            status = "⚠️ 部分有效"
        else:
            status = "❌ 无效或破坏结构"

        print(f"{row['image']}: {status} (箭头={row['arrow_pixels']}px, CEM={row['tanimoto_cem']:.4f})")

    print("\n" + "=" * 80)
    print("🎉 批量测试完成！")
    print("=" * 80)
    print(f"结果保存在: {OUTPUT_DIR}")
    print("每张图片包含:")
    print("  1_original.png - 原图")
    print("  2_removed_arrows.png - 去箭头后")
    print("  3_mask_visualization.png - 分割掩膜")
    print("  4_comparison.png - 三图对比（原图|去箭头|掩膜）")
    print(f"\n汇总结果: results_summary.csv")

if __name__ == "__main__":
    main()
