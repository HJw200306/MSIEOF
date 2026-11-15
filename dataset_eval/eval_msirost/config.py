r"""评估配置：所有路径 / 阈值 集中在这里。

- MSIROST_ROOT：你的项目根，用于 import extract / post_segment 及其模型权重
- PATCID_RESTORED_ROOT：recover_pages.py 恢复出的 GT 根目录
                        (<subset>/<office>/<stem>.{png, mask.png, yolo.txt})
- EVAL_ROOT：本次评估所有产物的输出根（不动 msirost 的 crop_dataset/）

评估口径参数
-----------
- RECALL_THRESHOLD = 0.60
    操作 = |P_i ∩ G_j| / |G_j|（预测像素覆盖 GT 像素的比例），
    单位 = 无量纲比例 [0, 1]，
    几何尺度 = 单个 GT 分子的前景像素集合内部占比，
    调大（例 0.95） → 判定更严，msirost 只要漏掉一点点原子标签 / 短取代基就不算完整；
    调小（例 0.80） → 判定更宽松，允许"骨架都在、但角落文字漏了"也算完整。

- GT 前景像素提取
    完全迁移自 D:\workspace\MPOCSR-master\utils\process_patent.py 的
    get_refined_mask_and_bbox：全图 THRESH_BINARY_INV @ 220 → per-instance ROI
    (pad=5) → fillPoly ∩ binary → 局部去噪（8连通，只有当 area<5 的小 CC ≥ 10
    个时才丢弃它们）→ 3x3 膨胀 1 次 → MST 桥接 2px 线。与训练 GT 分布完全对齐。

- POST_SEG_THRESHOLD = 0.15
    继承 msirost.utils.post_segment.FG_RATIO_THRESHOLD 的默认值，
    这里显式传入 classify_image 以便复现实验时可调。

- EXTRACT_CONF = 0.25
    继承 extract.py 默认置信度阈值。
"""

from pathlib import Path

# --- 项目路径 ---
MSIROST_ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost")
PATCID_RESTORED_ROOT = Path(
    r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored"
)
# 输出根：用户指定的 E:\其他文章及数据集\eval
EVAL_ROOT = Path(r"E:\其他文章及数据集\eval")
CROPS_TMP_DIR = EVAL_ROOT / "_crops_tmp"     # extract/post_segment 的临时 crop 输出
RESULTS_DIR = EVAL_ROOT / "results"
VIZ_DIR = RESULTS_DIR / "viz"                # 每页 4-in-1 overlay 图（可选）

# collect_failures.py 的输出（"倾向 A：整体分布 + 只看错的"两份）
# 全部页面 extract/post_seg 可视化 → 看整体分布
VIS_ALL_ROOT = EVAL_ROOT / "vis_all"
VIS_ALL_EXTRACT = VIS_ALL_ROOT / "extract"
VIS_ALL_POST = VIS_ALL_ROOT / "post_seg"
# 覆盖率<0.90 的页面 extract/post_seg 可视化（vis_all 的子集副本）
VIS_FAIL_ROOT = EVAL_ROOT / "vis_fail"
VIS_FAIL_EXTRACT = VIS_FAIL_ROOT / "extract"
VIS_FAIL_POST = VIS_FAIL_ROOT / "post_seg"
# 页级 coverage<0.90 的整页原图目录
BELOW_090_PAGES_DIR = EVAL_ROOT / "below_090_pages"
# 分子级 recall<0.90 的失败分子裁图目录（保持旧命名）
BELOW_090_MOLECULES_DIR = EVAL_ROOT / "below_090_molecules"

# --- 评估口径 ---
RECALL_THRESHOLD = 0.60

# GT 前景提取参数（保留符号以兼容旧引用；build_gt.py 用的是 MPOCSR 迁移版硬编码值）
GT_FG_BINARY_METHOD = "fixed"      # 实际用固定阈值 220（THRESH_BINARY_INV）
GT_FG_FIXED_THR = 220
GT_FG_CLOSE_R = 3                  # 兼容占位，不再直接使用
GT_FG_MIN_AREA_PX = 200            # 兼容占位

# --- msirost 模块参数（透传给 extract / post_segment）---
EXTRACT_CONF = 0.25
POST_SEG_THRESHOLD = 0.01  # 与 post_segment.FG_RATIO_THRESHOLD 对齐（0.15 过高会误杀真分子）

# --- 颜色感知前景提取（仅自己验证集模式启用；PatCID 流程保持旧灰度口径）---
# 见 fg_utils.color_aware_ink_mask：背景(纸张) = V>=V_BG_THR 且 S<=S_BG_THR，前景=取反。
#   COLOR_AWARE_V_BG_THR：HSV V 明度阈值 [0,255]，与旧灰度阈值 220 对齐（V>=221 ⇔ gray>220）；
#       调大→淡像素也算前景 GT 虚胖数字被压低，调小→淡骨架线丢失。
#   COLOR_AWARE_S_BG_THR：HSV S 饱和度阈值 [0,255]；调大→淡色墨迹被当纸张丢失，调小→色边也算前景。
COLOR_AWARE_V_BG_THR = 221
COLOR_AWARE_S_BG_THR = 40

# --- 数据枚举 ---
SUBSETS = ("benchmark_D2C-RND", "benchmark_D2C-UNI")
OFFICES = ("cn", "eu", "jp", "kr", "us")

# --- 自己的验证集（patent + journal 语义分割数据集）---
# 结构：<root>/images/<split>/*.png、<root>/json/<split>/*.json（LabelMe polygon）、
#       <root>/masks/<split>/*.png（回退用二值 mask）。
MYVAL_DATA_ROOT = Path(
    r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal"
)
# 自己验证集评估的输出根（与 PatCID 的 EVAL_ROOT 完全隔离，不互相覆盖）。
MYVAL_EVAL_ROOT = Path(r"E:\MSIREOT_DATA\eval_myval")
MYVAL_CROPS_TMP_DIR = MYVAL_EVAL_ROOT / "_crops_tmp"
MYVAL_RESULTS_DIR = MYVAL_EVAL_ROOT / "results"
MYVAL_VIZ_DIR = MYVAL_RESULTS_DIR / "viz"

# --- DECIMER vs MSIEOF 对比评估的输出根 ---
# compare_decimer_msirost.py 用：per_page_{method}.csv + comparison.csv/.md 都写这里。
COMPARE_RESULTS_DIR = Path(r"E:\其他文章及数据集\eval_compare\results")
