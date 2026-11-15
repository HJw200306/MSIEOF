"""
对比去虚线箭头模型对SMILES匹配的影响
使用新的三分类模型 bond_arrow_best.pth (UNet++ with ResNet101)
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

# ==================== 配置区域 ====================
INPUT_IMAGE = r"E:\MSIREOT_DATA\journal\images\JNP_2024_04_1059-1066_004.png"
OUTPUT_DIR = r"C:\Users\Admin\PycharmProjects\msirost\outputs"
MOLSCRIBE_MODEL = r"D:\workspace\Model\swin_base_char_aux_1m680k.pth"

# 新的三分类模型（包含虚线箭头、化学键）
MODEL_3CLASS = str(Path(__file__).resolve().parents[2] / "model" / "legacy_arrow_models" / "bond_arrow_best.pth")

# ==================== 模型定义 ====================
class ArrowUNetPlusPlus(torch.nn.Module):
    """基于segmentation_models_pytorch的UNet++模型"""
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

# ==================== SMILES归一化函数 ====================
def normalize_smiles(smiles_str):
    """
    应用波浪线归一化规则：
    1. 游离 .C~C 整体删除
    2. 其他连接型 ~ 转为普通单键
    3. R基、超原子和缩写统一折叠为 *
    """
    if not smiles_str or smiles_str.strip() == "":
        return ""

    # 处理波浪线
    parts = smiles_str.split('.')
    filtered_parts = []
    for part in parts:
        if '~' not in part:
            filtered_parts.append(part)
        elif len(part) > 3:  # 不是简单的C~C形式
            filtered_parts.append(part.replace('~', ''))

    normalized = '.'.join(filtered_parts)

    # 处理R基和超原子：将[R]、[R1]等替换为*
    normalized = re.sub(r'\[R\d*\]', '*', normalized)

    return normalized

def compute_tanimoto(smiles1, smiles2):
    """计算两个SMILES的Tanimoto相似度（CEM - 保留立体化学）"""
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
    """计算Tanimoto相似度（GEM - 忽略立体化学）"""
    try:
        # 移除立体化学标记
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

# ==================== 箭头去除函数 ====================
def preprocess_image(image_np, window_size=512):
    """预处理图片用于模型推理，确保尺寸是32的倍数"""
    h, w = image_np.shape[:2]

    # 计算padding使得尺寸是32的倍数
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32

    # 使用白色padding
    if pad_h > 0 or pad_w > 0:
        image_np = cv2.copyMakeBorder(
            image_np, 0, pad_h, 0, pad_w,
            cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

    # Normalize
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, (h, w), image_np.shape[:2]

def remove_dashed_arrows(img_pil, model, device, window_size=512):
    """
    使用三分类模型去除虚线箭头
    模型输出: ['background', 'dashed_arrow', 'bond']
    """
    # 转为numpy (RGB)
    img_np = np.array(img_pil)
    original_shape = img_np.shape[:2]

    # 预处理（包含padding）
    img_tensor, orig_hw, padded_hw = preprocess_image(img_np, window_size)
    img_tensor = img_tensor.to(device)

    # 推理
    with torch.no_grad():
        logits = model(img_tensor)  # [B, C, H, W]
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1)  # [B, H, W]

    # 转回CPU和numpy
    pred_np = pred.squeeze(0).cpu().numpy()  # [H, W]

    # 裁剪掉padding部分，恢复到原始尺寸
    pred_np = pred_np[:orig_hw[0], :orig_hw[1]]

    # 创建可视化掩膜
    vis_mask = np.zeros((*orig_hw, 3), dtype=np.uint8)
    vis_mask[pred_np == 1] = [255, 0, 0]  # 虚线箭头 - 红色
    vis_mask[pred_np == 2] = [0, 255, 0]  # 化学键 - 绿色

    # 去除虚线箭头 (class 1)
    mask_dashed = (pred_np == 1)
    img_removed = img_np.copy()
    img_removed[mask_dashed] = [255, 255, 255]

    return Image.fromarray(img_removed), Image.fromarray(vis_mask)

# ==================== 可视化函数 ====================
def create_comparison_visualization(images_dict, masks_dict, output_path):
    """创建对比可视化图"""
    imgs = [np.array(img) for img in images_dict.values()]
    masks = [np.array(mask) for mask in masks_dict.values()]

    # 2行N列
    row1 = np.hstack(imgs)
    row2 = np.hstack(masks)
    combined = np.vstack([row1, row2])

    cv2.imwrite(output_path, cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
    print(f"✅ 可视化对比图已保存: {output_path}")

# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 检查输入图片
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 输入图片不存在: {INPUT_IMAGE}")
        return

    print(f"📷 输入图片: {INPUT_IMAGE}")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("=" * 80)

    # 加载图片
    img_bgr = cv2.imread(INPUT_IMAGE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)

    # 保存原图
    original_path = os.path.join(OUTPUT_DIR, "0_original.png")
    img_pil.save(original_path)

    # ==================== 1. 加载模型 ====================
    print("\n" + "=" * 80)
    print("步骤 1: 加载三分类箭头模型")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    checkpoint = torch.load(MODEL_3CLASS, map_location=device)
    print(f"模型信息:")
    print(f"  - Classes: {checkpoint.get('num_classes')}")
    print(f"  - Class names: {checkpoint.get('class_names')}")
    print(f"  - Encoder: {checkpoint.get('encoder')}")

    # 创建模型
    model = ArrowUNetPlusPlus(
        encoder=checkpoint.get('encoder', 'resnet101'),
        num_classes=checkpoint.get('num_classes', 3)
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("✅ 模型加载成功")

    # ==================== 2. 去除虚线箭头 ====================
    print("\n" + "=" * 80)
    print("步骤 2: 去除虚线箭头")
    print("=" * 80)

    print("⏳ 处理中...")
    img_no_dashed, mask_vis = remove_dashed_arrows(img_pil, model, device)

    path_no_dashed = os.path.join(OUTPUT_DIR, "1_removed_dashed_arrow.png")
    img_no_dashed.save(path_no_dashed)
    print(f"✅ 保存去箭头图: {path_no_dashed}")

    path_mask = os.path.join(OUTPUT_DIR, "1_mask_visualization.png")
    mask_vis.save(path_mask)
    print(f"✅ 保存掩膜可视化: {path_mask}")

    # 创建对比可视化
    print("\n⏳ 创建对比可视化...")
    create_comparison_visualization(
        {'原图': img_pil, '去虚线箭头': img_no_dashed},
        {'原图': img_pil, '检测掩膜': mask_vis},
        os.path.join(OUTPUT_DIR, "visualization_comparison.png")
    )

    # ==================== 3. MolScribe识别SMILES ====================
    print("\n" + "=" * 80)
    print("步骤 3: 使用MolScribe识别SMILES")
    print("=" * 80)

    print("⏳ 加载MolScribe模型...")
    molscribe = MolScribe(MOLSCRIBE_MODEL, device=device)
    print("✅ MolScribe加载成功")

    test_images = [
        ("原图", img_pil),
        ("去虚线箭头", img_no_dashed)
    ]

    results = []
    for name, img in test_images:
        print(f"\n⏳ 识别: {name}")
        img_rgb_array = np.array(img)

        try:
            pred = molscribe.predict_images([img_rgb_array], return_atoms_bonds=False)[0]
            smiles = pred.get('smiles', '')
            print(f"   原始SMILES: {smiles}")

            norm_smiles = normalize_smiles(smiles)
            print(f"   归一化SMILES: {norm_smiles}")

            results.append({
                'variant': name,
                'smiles_raw': smiles,
                'smiles_normalized': norm_smiles
            })
        except Exception as e:
            print(f"   ❌ 识别失败: {e}")
            results.append({
                'variant': name,
                'smiles_raw': '',
                'smiles_normalized': ''
            })

    # ==================== 4. 计算相似度 ====================
    print("\n" + "=" * 80)
    print("步骤 4: 计算SMILES相似度（以原图为参考）")
    print("=" * 80)

    if results[0]['smiles_normalized']:
        ref_smiles = results[0]['smiles_normalized']
        print(f"参考SMILES（原图归一化）: {ref_smiles}\n")

        for i, res in enumerate(results):
            if i == 0:
                res['tanimoto_cem'] = 1.0
                res['tanimoto_gem'] = 1.0
            else:
                test_smiles = res['smiles_normalized']
                if test_smiles:
                    cem = compute_tanimoto(ref_smiles, test_smiles)
                    gem = compute_tanimoto_gem(ref_smiles, test_smiles)
                    res['tanimoto_cem'] = cem
                    res['tanimoto_gem'] = gem
                    print(f"{res['variant']}:")
                    print(f"  CEM Tanimoto: {cem:.4f}")
                    print(f"  GEM Tanimoto: {gem:.4f}")
                else:
                    res['tanimoto_cem'] = 0.0
                    res['tanimoto_gem'] = 0.0
                    print(f"{res['variant']}: SMILES为空，相似度=0")
    else:
        print("⚠️ 原图SMILES识别失败，无法计算相似度")
        for res in results:
            res['tanimoto_cem'] = 0.0
            res['tanimoto_gem'] = 0.0

    # ==================== 5. 保存结果 ====================
    print("\n" + "=" * 80)
    print("步骤 5: 保存结果")
    print("=" * 80)

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "comparison_results.csv")
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"✅ 结果已保存: {csv_path}")

    print("\n" + "=" * 80)
    print("汇总结果")
    print("=" * 80)
    print(df.to_string(index=False))

    print("\n" + "=" * 80)
    print("🎉 所有处理完成！")
    print("=" * 80)
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("包含文件:")
    print("  - 0_original.png                    # 原图")
    print("  - 1_removed_dashed_arrow.png        # 去虚线箭头")
    print("  - 1_mask_visualization.png          # 掩膜可视化（红=虚线箭头，绿=化学键）")
    print("  - visualization_comparison.png      # 对比可视化")
    print("  - comparison_results.csv            # SMILES对比结果")

if __name__ == "__main__":
    main()
