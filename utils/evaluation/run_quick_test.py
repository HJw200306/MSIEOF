"""
快速启动评估脚本 - 评估前 10 页用于快速验证

直接在 PyCharm 中运行此文件即可开始快速测试
"""

from pathlib import Path
import sys

from utils.base.paths import DATA_DIR, PROJECT_ROOT
from utils.evaluation.evaluate_page2smiles import evaluate_dataset

if __name__ == "__main__":
    # 配置参数
    TEST_SET_ROOT = DATA_DIR / "MolMole_Patent300"
    TEST_SET_JSONL = TEST_SET_ROOT / "01_MolMole_Patent300" / "derived" / "pages_ordinary_complete.jsonl"
    OUTPUT_DIR = PROJECT_ROOT / "evaluation_results_快速测试"

    # 快速测试：只评估前 10 页
    MAX_PAGES = 10

    print("="*70)
    print("快速测试模式 - 评估前 10 页")
    print("="*70)
    print(f"测试集: {TEST_SET_JSONL.name}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*70 + "\n")

    # 检查测试集是否存在
    if not TEST_SET_JSONL.exists():
        print(f"❌ 错误: 测试集文件不存在")
        print(f"   路径: {TEST_SET_JSONL}")
        print(f"\n请确认测试集位置是否正确")
        sys.exit(1)

    # 开始评估
    try:
        summary = evaluate_dataset(
            test_set_jsonl=TEST_SET_JSONL,
            test_set_root=TEST_SET_ROOT,
            output_dir=OUTPUT_DIR,
            iou_threshold=0.5,
            conf_threshold=0.25,
            morgan_radius=2,
            max_pages=MAX_PAGES,
            device="cuda"
        )

        print(f"\n✓ 快速测试完成！")
        print("✓ 如需完整评估，请运行: python -m utils.evaluation.evaluate_page2smiles")

    except Exception as e:
        print(f"\n❌ 评估失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
