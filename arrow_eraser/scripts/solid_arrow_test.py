"""
使用正确的实线箭头模型（VGG16-U-Net 三类分割）测试
模型: misp_weights_no_freeze.pth
"""
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# 添加路径
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from split_tools.unet import Unet

# ==================== 配置 ====================
INPUT_IMAGE = r"E:\MSIREOT_DATA\journal\images\JNP_2024_04_1059-1066_004.png"
OUTPUT_DIR = r"C:\Users\Admin\PycharmProjects\msirost\outputs\solid_arrow_test"
MODEL_PATH = str(PROJECT_ROOT / "model" / "legacy_arrow_models" / "misp_weights_no_freeze.pth")

# ==================== 主流程 ====================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("实线箭头模型（VGG16-U-Net）测试")
    print("=" * 80)
    print(f"模型: {MODEL_PATH}")
    print(f"输入: {INPUT_IMAGE}")
    print(f"输出: {OUTPUT_DIR}\n")

    # 检查图片
    if not os.path.exists(INPUT_IMAGE):
        print(f"❌ 图片不存在: {INPUT_IMAGE}")
        return

    # 加载图片
    img = Image.open(INPUT_IMAGE).convert("RGB")
    print(f"✅ 图片加载成功: {img.size}")

    # 配置模型（3类分割）
    config = {
        'num_classes': 3,
        'name_classes': ["_background_", "arrow", "bond"]  # 需要确认
    }

    print(f"\n⏳ 加载实线箭头模型...")
    print(f"   类别数: {config['num_classes']}")
    print(f"   类别名: {config['name_classes']}")

    try:
        model = Unet(
            MODEL_PATH,
            num_classes=config['num_classes'],
            input_shape=[512, 512]
        )
        print("✅ 模型加载成功\n")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 预测
    print("⏳ 进行预测...")
    try:
        pred_mask = model.detect_image(
            img,
            count=False,
            name_classes=config['name_classes'],
            mix_type=1  # 只返回分割结果
        )
        print("✅ 预测完成")

        # 保存掩膜可视化
        mask_path = os.path.join(OUTPUT_DIR, "mask_visualization.png")
        pred_mask.save(mask_path)
        print(f"✅ 掩膜保存: {mask_path}")

        # 分析掩膜
        pred_np = np.array(pred_mask)
        print(f"\n掩膜分析:")
        print(f"  形状: {pred_np.shape}")

        # 统计各类像素
        colors_count = {}
        for color_name, color_value in [
            ("背景(黑色)", (0, 0, 0)),
            ("箭头(红色)", (255, 0, 0)),
            ("化学键(绿色)", (0, 128, 0)),
        ]:
            mask = (pred_np[:, :, 0] == color_value[0]) & \
                   (pred_np[:, :, 1] == color_value[1]) & \
                   (pred_np[:, :, 2] == color_value[2])
            count = mask.sum()
            colors_count[color_name] = count
            print(f"  {color_name}: {count} 像素 ({count/pred_np.size*100:.3f}%)")

        # 去除箭头（涂白红色区域）
        print(f"\n⏳ 去除箭头...")
        img_np = np.array(img)
        arrow_mask = (pred_np[:, :, 0] == 255) & \
                     (pred_np[:, :, 1] == 0) & \
                     (pred_np[:, :, 2] == 0)

        img_removed = img_np.copy()
        img_removed[arrow_mask] = [255, 255, 255]

        removed_path = os.path.join(OUTPUT_DIR, "removed_arrows.png")
        Image.fromarray(img_removed).save(removed_path)
        print(f"✅ 去箭头图片保存: {removed_path}")

        # 保存原图
        img.save(os.path.join(OUTPUT_DIR, "original.png"))

        print("\n" + "=" * 80)
        print("🎉 测试完成！")
        print("=" * 80)
        print(f"输出目录: {OUTPUT_DIR}")
        print("  - original.png: 原图")
        print("  - mask_visualization.png: 分割掩膜")
        print("  - removed_arrows.png: 去箭头后的图片")

    except Exception as e:
        print(f"❌ 预测失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
