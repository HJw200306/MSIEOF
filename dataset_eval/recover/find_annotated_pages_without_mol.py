# -*- coding: utf-8 -*-
"""找出"页面上没有任何分子结构、但 mask 标注非空"的恢复页面，删除其假标注。

背景
----
recover_pages.py 按 benchmark seg/boxes 标注把 PDF 页面渲染成
restored/<subset>/<office>/<stem>.png（页面）+ <stem>.mask.png（标注掩码）。
但部分标注对不上实际页面内容（如 EP18877299NWA1_18）：mask 里画了分子区域，
页面对应位置却是纯文本/空白，页面上根本没有分子结构。这类"假标注页"会
污染 eval_msirost 的检测评测（召回率被无中生有的 GT 拉低），需要整体筛出
来，并把 restored 里的这些假标注删掉。

区分办法（复用 utils/post_segment.py 的 UNet++ 二分类思路，分母口径改用
eval_msirost/build_gt.py 的前景提取）
----
post_segment.py 用已训好的 UNet++(resnet101, classes=2) 预测"2D 分子骨架
像素"前景。post_segment 场景的门控以"标注区域面积"做分母（其 YOLO 多边形
紧贴骨架，真分子 25%~50%，文本/空白 < 10%）；但本脚本面对的是 PatCID 的
矩形标注框，框普遍比骨架松散，框面积做分母会把单个真分子的占比稀释到与
文本难分。故本脚本把分母换成"标注框内提取出的页面前景（墨迹）像素"：

    fg_ratio = |UNet++ 前景 ∩ 页面墨迹 ∩ 标注区域| / |页面墨迹 ∩ 标注区域|

"页面墨迹"的提取采用 eval_msirost/build_gt.py 的 GT 前景口径（其自身迁移
自 MPOCSR utils/process_patent.py）：灰度图 cv2.threshold(gray, 220, 255,
THRESH_BINARY_INV) 再与标注区域求交，即 build_gt.py 中 binary_global ∩
mask_poly 得到 content_binary_roi 的那一步（阈值常量 GLOBAL_BIN_THR 直接
从 build_gt.py 导入）。这样比值的含义变为"框内墨迹中被 UNet++ 判为分子
骨架的比例"，与框松紧无关：真分子区域墨迹以骨架为主 → 比值高；文本/
表格区域墨迹几乎不被判为骨架 → 比值接近 0；框内无墨迹（纯空白）按 0 处理。
分子端与墨迹求交是为排除 UNet++ 在空白处的零星误报，保证比值 ∈ [0, 1]。

本脚本把该门控逐页应用：对 mask 的每个标注连通域裁剪出对应页面区域跑
UNet++，若【所有】标注区域的 fg_ratio 都低于阈值 → 该页"有标注但没有
分子结构"，把 <stem>.png / <stem>.mask.png（以及 <stem>.yolo.txt，若有）
一起复制到输出目录（保持 <subset>/<office> 子目录结构），并写出一份
report.csv 记录每页的判定与逐区域 ratio，便于人工抽查。

复制留档后，命中页在 restored 里的假标注被删除（默认开启，--no-delete
可改为只找不删）：<stem>.mask.png 覆写为同尺寸全零掩码——页面图保留，
该页变成与 recover_pages.py no_box 情况一致的"无标注页"，eval 时作为
无 GT 的负样本参与检测评测；<stem>.yolo.txt（若有）整个删除，避免残留
假框被 GT 构建读到。删除严格发生在复制之后，输出目录里的副本就是删除
前的完整备份，误删可从那里拷回。已清空的页 mask 全空，再跑本脚本会被
直接跳过，重复运行安全。

命中页默认同时生成 <stem>.vis.png 整页叠加可视化，染色逻辑与
post_segment.py 的 _save_classify_overlay 完全一致（UNet++ 前景绿色
alpha 0.35、fg_ratio 分母区域（标注框内墨迹）黄色 alpha 0.20，标题写
ratio/阈值/判定），另以红框+数字标出每个参与判定连通域的 fg_ratio。
前景与分母掩码均来自与算 fg_ratio 同一次推理/提取（各区域裁剪结果拼回
整页），"看见的就是判定用的"，
便于排查"post_segment 判分子、本脚本判无分子"的分歧原因。可视化文件
与页面/掩码放在输出目录同一 <subset>/<office> 子目录下（存放逻辑与
数据集一致），仅命中页输出；--no-visualize 可关闭。

mask 全空（recover_pages.py 的 no_box 情况）不属于目标情形，直接跳过。

运行
----
PyCharm 直接 Run 本文件即可（所有参数均有默认值，含删除；只想复查判定、
不动 restored 时加 --no-delete）；也可命令行覆盖：
    python delete_fake_annotated_pages_without_mol.py --device cpu --threshold 0.15
依赖与 utils/post_segment.py 相同（torch / smp / albumentations / cv2），
模型权重沿用 model/post_seg_unetpp.pth。
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

# 项目根目录 = .../msirost（本文件位于 dataset_eval/recover/ 下两层）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from utils.post_segment import (  # noqa: E402
    MODEL_PATH,
    _image_to_tensor,
    _load_model,
    _whole_image_inference,
)

# fg_ratio 分母的墨迹提取阈值，直接采用 eval_msirost/build_gt.py 的
# GLOBAL_BIN_THR（=220，其迁移自 MPOCSR utils/process_patent.py），保证
# "标注框内前景像素"与 eval GT 的前景口径完全一致。
#   操作 = cv2.threshold(gray, 220, 255, THRESH_BINARY_INV)，灰度 ≤ 220 的
#          像素视为墨迹；本脚本在各标注区域的裁剪图（可能已按 MAX_SIDE
#          缩放）上执行，使分母与 UNet++ 看到同一张输入图；
#   单位 = 灰度值 [0, 255]；
#   几何尺度 = 单像素灰度；
#   调整后果见 build_gt.py 中该常量的注释（调大拉进浅色底纹 → 分母变大、
#   fg_ratio 偏低；调小漏掉浅印骨架）。如需调整应改 build_gt.py 一处。
from dataset_eval.eval_msirost.build_gt import GLOBAL_BIN_THR  # noqa: E402

# 标注区域内的前景占比达标下限（本脚本专用）。
#   操作 = 单个标注连通域内 |UNet++ 前景 ∩ 页面墨迹 ∩ 标注区域| /
#          |页面墨迹 ∩ 标注区域|，与阈值比较，任一区域 ≥ 阈值即认为页面
#          存在分子结构；墨迹提取见 GLOBAL_BIN_THR 导入处注释；
#   单位 = 无量纲比例 [0, 1]；
#   几何尺度 = 分母是标注框内的墨迹像素（骨架/文字笔画，300DPI 下 1~3 px
#              宽的线条），不含框内空白，故与矩形框松紧无关。真分子区域
#              墨迹以骨架为主，比值近似 UNet++ 对骨架墨迹的召回（偏高段）；
#              框内混入编号/取代基文字时被文字墨迹稀释到中段；文本/表格
#              区域墨迹几乎全是文字，比值应接近 0；
#   调大（如 0.5）→ 判"存在分子"更严，骨架极细（MAX_SIDE 缩放后断裂、
#          UNet++ 召回下降）或框内文字墨迹多的真分子页会被误判命中；
#   调小（如 0.1）→ 文本墨迹上零星的 UNet++ 误报前景即可让空页"达标"
#          而漏检；
#   默认 0.30 = 由旧口径（分母为框面积，默认 0.35）折算：旧口径下松框
#               真分子被稀释到 0.12~0.15（如 A__2005017446_95），按框内
#               墨迹密度约 1/3 折算，换墨迹分母后约落到 0.4 上下，而文本
#               页应 < 0.1，取 0.30 给双向留余量；命中页均输出 .vis.png
#               （红框标逐区域 ratio），若发现真分子页被误判命中，可据
#               可视化回查并下调。
PAGE_REGION_FG_THRESHOLD = 0.15

RESTORED_ROOT = Path(r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored")
OUTPUT_ROOT = Path(r"E:\其他文章及数据集\eval")

# 标注连通域的最小面积门槛。
#   操作 = 对 mask 做 8 连通域分析后，面积（像素数）小于该值的连通域不参与
#          fg_ratio 判定；
#   单位 = 像素数（px²）；
#   几何尺度 = 页面为 A4@300DPI（约 2480×3509），一个正常分子标注框边长
#              数百像素、面积 10⁴~10⁶ px²，400 px² 相当于 20×20 的小斑点，
#              远小于任何真实标注；
#   调大（如 5000）→ 更多小标注被当噪声忽略，可能把"页面只有一个小分子
#                    标注"的页错误归入"无有效标注"而漏检/误判；
#   调小（如 0）  → mask 里的孤立噪点也会参与判定，噪点区域 fg_ratio 必然
#                    极低，会稀释不了"任一区域达标即保留"的逻辑（本脚本取
#                    any 判定，故影响有限），但白白多跑推理。
MIN_REGION_AREA = 400

# 裁剪标注区域时向四周外扩的边距。
#   操作 = 以连通域外接矩形为基础，四边各向外扩这么多像素后再从页面裁剪
#          （出界处截断到页面边缘）；
#   单位 = 像素（px）；
#   几何尺度 = 300DPI 下 16 px ≈ 1.4 mm，约为分子骨架一条键长的 1/3~1/5，
#              只为给 UNet++ 提供少量上下文、避免骨架贴边被截断；
#   调大（如 64）→ 上下文更充分，但裁剪区可能引入邻近文本，UNet++ 若在
#                  文本上误报前景会轻微抬高 fg_ratio（分子只统计标注区域
#                  内的交集，故影响很小）；
#   调小（如 0）→ 骨架贴紧标注边缘时边界像素预测变差，真分子 ratio 略降。
CROP_MARGIN = 16

# 裁剪区域送入 UNet++ 前的最长边上限。
#   操作 = 若裁剪区长边超过该值，对图像与区域 mask 按同一比例双线性/最近邻
#          缩小到长边等于该值后再推理（fg_ratio 是比例量，对等比缩放近似
#          不变）；
#   单位 = 像素（px）；
#   几何尺度 = 整页标注最大可接近整页 2480×3509，直接推理约需数 GB 显存；
#              1024 px 时 UNet++(resnet101) 单张推理显存 < 2 GB；
#   调大（如 2048）→ 细骨架保留更完整，fg_ratio 更准，但显存/耗时上升，
#                    大区域可能 OOM；
#   调小（如 512）→ 快且省显存，但极细的骨架线条缩小后可能断裂，真分子
#                    的 fg_ratio 被低估，增大把"少数真分子页"误判进 eval
#                    的风险。
MAX_SIDE = 1024

IMAGE_SUFFIX = ".png"
MASK_SUFFIX = ".mask.png"


def iter_page_mask_pairs(root: Path) -> List[Tuple[Path, Path]]:
    """递归收集 (页面.png, 页面.mask.png) 成对文件。

    排除 *.mask.png / *.vis.png 本身被当作页面图。
    """
    pairs: List[Tuple[Path, Path]] = []
    for png in sorted(root.rglob(f"*{IMAGE_SUFFIX}")):
        name = png.name
        if name.endswith(MASK_SUFFIX) or name.endswith(".vis.png"):
            continue
        mask_path = png.with_name(f"{png.stem}{MASK_SUFFIX}")
        if mask_path.exists():
            pairs.append((png, mask_path))
    return pairs


def _crop_with_margin(
    x: int, y: int, w: int, h: int, W: int, H: int, margin: int
) -> Tuple[int, int, int, int]:
    """连通域外接矩形外扩 margin 后截断到页面范围，返回 (x1, y1, x2, y2)。"""
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(W, x + w + margin)
    y2 = min(H, y + h + margin)
    return x1, y1, x2, y2


def _region_fg_ratio(
    model,
    device: torch.device,
    page_rgb: np.ndarray,
    region_mask: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """对单个标注区域裁剪图跑 UNet++，返回 (fg_ratio, UNet++ 前景掩码, 分母掩码)。

    page_rgb: 裁剪后的页面区域 (H, W, 3) RGB；
    region_mask: 与之同尺寸的该连通域二值掩码 (H, W)，{0,1}。

    fg_ratio = |UNet++ 前景 ∩ 墨迹 ∩ 区域| / |墨迹 ∩ 区域|。墨迹用
    build_gt.py 口径（THRESH_BINARY_INV @ GLOBAL_BIN_THR）在送入 UNet++
    的同一张输入图（含 MAX_SIDE 缩放后）上提取，分子/分母经历同样的缩放
    退化，比值口径自洽；区域内无墨迹（纯空白标注）时按 0.0 处理。分子端
    与墨迹求交是为排除 UNet++ 在空白处的零星误报，保证比值 ∈ [0, 1]。

    返回的两个掩码均为 uint8 {0,1}、与传入裁剪区同尺寸：推理若因超过
    MAX_SIDE 被缩小，则按最近邻放大回原尺寸后返回。UNet++ 前景掩码与
    分母掩码（墨迹 ∩ 区域）都出自计算 fg_ratio 的同一次推理/提取，供
    可视化拼回整页——保证"图上看到的"就是"参与判定的"（含缩放导致的
    细骨架断裂等真实退化），可视化才有诊断价值。
    """
    h0, w0 = region_mask.shape
    page_in, region_in = page_rgb, region_mask
    long_side = max(h0, w0)
    if long_side > MAX_SIDE:
        scale = MAX_SIDE / float(long_side)
        new_wh = (max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale))))
        page_in = cv2.resize(page_rgb, new_wh, interpolation=cv2.INTER_AREA)
        region_in = cv2.resize(region_mask, new_wh, interpolation=cv2.INTER_NEAREST)

    # 墨迹提取（build_gt.py 的 binary_global ∩ mask_poly 口径，逐区域执行）
    gray_in = cv2.cvtColor(page_in, cv2.COLOR_RGB2GRAY)
    _, ink_in = cv2.threshold(gray_in, GLOBAL_BIN_THR, 1, cv2.THRESH_BINARY_INV)
    denom_bool = np.logical_and(ink_in.astype(bool), region_in.astype(bool))

    tensor = _image_to_tensor(page_in, device)
    logits = _whole_image_inference(model, tensor)
    pred = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
    unet_fg = (pred == 1).astype(np.uint8)

    denom = int(denom_bool.sum())
    if denom <= 0:
        ratio = 0.0
    else:
        numer = int(np.logical_and(unet_fg.astype(bool), denom_bool).sum())
        ratio = float(numer) / float(denom)

    denom_mask = denom_bool.astype(np.uint8)
    if unet_fg.shape != (h0, w0):
        unet_fg = cv2.resize(unet_fg, (w0, h0), interpolation=cv2.INTER_NEAREST)
        denom_mask = cv2.resize(denom_mask, (w0, h0), interpolation=cv2.INTER_NEAREST)
    return ratio, unet_fg, denom_mask


def classify_page(
    model,
    device: torch.device,
    page_path: Path,
    mask_path: Path,
    threshold: float,
    collect_vis: bool = False,
) -> Tuple[Optional[bool], List[float], Optional[dict]]:
    """单页判定。

    Returns:
        (is_annotated_but_empty, per_region_ratios, vis)
        is_annotated_but_empty:
            True  → mask 有标注且所有区域 fg_ratio < threshold（目标情形）；
            False → 至少一个区域达标（页面上存在分子结构）；
            None  → mask 全空或读图失败，跳过。
        vis: collect_vis=True 且页面参与判定时的可视化素材 dict，键：
            page_rgb  — 整页 RGB (H, W, 3)；
            unet_fg   — 各区域裁剪推理的前景预测拼回整页的 {0,1} 掩码
                        （margin 重叠处按位或），与 fg_ratio 同源；
            denom_bin — fg_ratio 分母掩码 {0,1}（各区域"墨迹 ∩ 标注区域"
                        拼回整页，margin 重叠处按位或），与 fg_ratio 同源；
            regions   — [(x, y, w, h, ratio), ...] 参与判定连通域的外接
                        矩形（不含 margin）及其 fg_ratio。
        collect_vis=False 或页面被跳过时 vis 为 None。
    """
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        return None, [], None
    mask_bin = (mask_img > 127).astype(np.uint8)
    if int(mask_bin.sum()) == 0:
        return None, [], None  # 无标注（no_box 页），不属于目标情形

    page_bgr = cv2.imread(str(page_path))
    if page_bgr is None:
        return None, [], None
    if page_bgr.shape[:2] != mask_bin.shape[:2]:
        # 尺寸对不上视为数据异常，保守跳过并提示
        tqdm.write(f"[WARN] 页面与 mask 尺寸不一致，跳过: {page_path.name}")
        return None, [], None
    page_rgb = cv2.cvtColor(page_bgr, cv2.COLOR_BGR2RGB)
    H, W = mask_bin.shape

    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)

    ratios: List[float] = []
    any_molecule = False
    unet_fg_page = np.zeros((H, W), dtype=np.uint8) if collect_vis else None
    denom_page = np.zeros((H, W), dtype=np.uint8) if collect_vis else None
    vis_regions: List[Tuple[int, int, int, int, float]] = []
    for lb in range(1, n_labels):  # 0 是背景
        x, y, w, h, area = stats[lb]
        if area < MIN_REGION_AREA:
            continue
        x1, y1, x2, y2 = _crop_with_margin(x, y, w, h, W, H, CROP_MARGIN)
        crop_rgb = page_rgb[y1:y2, x1:x2]
        crop_region = (labels[y1:y2, x1:x2] == lb).astype(np.uint8)
        ratio, crop_fg, crop_denom = _region_fg_ratio(model, device, crop_rgb, crop_region)
        ratios.append(ratio)
        if ratio >= threshold:
            any_molecule = True
        if unet_fg_page is not None:
            # 相邻区域的 margin 可能重叠，按位或拼回整页
            unet_fg_page[y1:y2, x1:x2] |= crop_fg
            denom_page[y1:y2, x1:x2] |= crop_denom
            vis_regions.append((int(x), int(y), int(w), int(h), float(ratio)))

    vis: Optional[dict] = None
    if collect_vis:
        vis = {
            "page_rgb": page_rgb,
            "unet_fg": unet_fg_page,
            "denom_bin": denom_page,
            "regions": vis_regions,
        }

    if not ratios:
        # 标注全是小于 MIN_REGION_AREA 的噪点：等价于"有标注但无有效分子"
        return True, ratios, vis
    return (not any_molecule), ratios, vis


def _save_page_overlay(
    vis: dict,
    threshold: float,
    stem: str,
    out_path: Path,
) -> None:
    """命中页整页叠加图，染色逻辑与 post_segment._save_classify_overlay 一致：
    UNet++ 前景（绿，alpha 0.35）+ fg_ratio 分母区域（黄，alpha 0.20；本脚本
    分母 = 标注框内墨迹，非整个标注矩形——黄色只覆盖笔画，矩形范围由红框
    表达），标题写 max_fg_ratio / threshold / 判定结果；此外用红框+数字标出
    每个参与判定连通域的外接矩形与其 fg_ratio，便于逐区域排查"为何没达标"
    （post_segment 场景单图单 ratio 写在标题即可，本脚本一页多区域，
    必须落到框上才能对上号）。仅命中页调用，判定恒为 NO-MOL。
    """
    from utils.viz_utils import (  # 与 post_segment 同一套配色/保存入口
        COLOR_BBOX as COLOR_ANNO,
        COLOR_FG,
        save_overlay_figure,
    )

    regions = vis["regions"]
    max_ratio = max((reg[4] for reg in regions), default=0.0)

    overlays = [
        (vis["unet_fg"].astype(bool), COLOR_FG, 0.35),      # 同 post_segment：UNet++ 前景
        (vis["denom_bin"].astype(bool), COLOR_ANNO, 0.20),  # 同 post_segment：分母掩码（墨迹∩标注）
    ]
    legend_items = [
        ("UNet++ fg", COLOR_FG),
        ("ink∩annotation (denom)", COLOR_ANNO),
    ]

    def _extras(ax) -> None:
        from matplotlib.patches import Rectangle

        for x, y, w, h, ratio in regions:
            ax.add_patch(Rectangle((x, y), w, h, fill=False,
                                   edgecolor="red", linewidth=1.2))
            ax.text(x, max(0, y - 10), f"{ratio:.4f}", color="red", fontsize=9,
                    bbox=dict(facecolor="white", alpha=0.7,
                              edgecolor="none", pad=1))

    save_overlay_figure(
        img_rgb=vis["page_rgb"],
        overlays=overlays,
        title=f"[page_cls] {stem}  max_fg_ratio={max_ratio:.4f}  "
              f"thr={threshold:.4f}  → NO-MOL",
        legend_items=legend_items,
        out_path=out_path,
        extras=_extras,
    )


def _delete_page_annotations(page_path: Path, mask_path: Path) -> None:
    """删除命中页在 restored 里的假标注（页面图本身保留）。

    - <stem>.mask.png 覆写为同尺寸全零掩码：该页变成与 recover_pages.py
      no_box 情况一致的"无标注页"；不删 mask 文件本身，保持"每页必有
      成对 mask"的数据集结构，eval 时该页作为无 GT 的负样本参与评测；
    - <stem>.yolo.txt（若有）整个删除：其内容只剩假框，留着会被 GT 构建
      读到。

    只允许在 run() 已把该页完整复制到输出目录之后调用——副本即删除前
    的备份，误删可从输出目录拷回。
    """
    mask_img = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask_img is None:
        tqdm.write(f"[WARN] 重读 mask 失败，未删标注: {mask_path}")
        return
    cv2.imwrite(str(mask_path), np.zeros_like(mask_img))
    yolo_path = page_path.with_name(f"{page_path.stem}.yolo.txt")
    if yolo_path.exists():
        yolo_path.unlink()


def run(
    input_root: Path,
    output_root: Path,
    device: str,
    threshold: float,
    visualize: bool = True,
    delete: bool = True,
) -> None:
    pairs = iter_page_mask_pairs(input_root)
    if not pairs:
        print(f"⚠️ 未在 {input_root} 下找到任何 (页面, mask) 成对文件")
        return

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"正在加载 UNet++ 分类模型 ({dev}) ...")
    model = _load_model(dev)
    print(f"✓ 权重: {MODEL_PATH.name}")
    print(f"✓ fg_ratio 阈值: {threshold:.4f}")
    print(f"✓ 待检查页面: {len(pairs)}\n")

    output_root.mkdir(parents=True, exist_ok=True)
    report_rows: List[dict] = []
    n_flagged = 0
    n_skipped = 0

    for page_path, mask_path in tqdm(pairs, desc="逐页判定"):
        flagged, ratios, vis = classify_page(
            model, dev, page_path, mask_path, threshold, collect_vis=visualize,
        )
        rel = page_path.relative_to(input_root)
        ratio_str = ";".join(f"{r:.4f}" for r in ratios)

        if flagged is None:
            n_skipped += 1
            continue

        report_rows.append({
            "page": str(rel),
            "annotated_but_empty": int(flagged),
            "n_regions": len(ratios),
            "max_fg_ratio": f"{max(ratios):.4f}" if ratios else "",
            "region_fg_ratios": ratio_str,
        })

        if flagged:
            n_flagged += 1
            dst_dir = output_root / rel.parent
            dst_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(page_path, dst_dir / page_path.name)
            shutil.copy2(mask_path, dst_dir / mask_path.name)
            yolo_src = page_path.with_name(f"{page_path.stem}.yolo.txt")
            if yolo_src.exists():
                shutil.copy2(yolo_src, dst_dir / yolo_src.name)
            if visualize and vis is not None:
                # 仅命中页输出可视化，与页面/掩码同目录（存放逻辑一致），
                # 命名 <stem>.vis.png（iter_page_mask_pairs 已排除该后缀，
                # 对 eval 目录再跑本脚本不会把它误当页面图）
                _save_page_overlay(
                    vis, threshold, stem=page_path.stem,
                    out_path=dst_dir / f"{page_path.stem}.vis.png",
                )
            if delete:
                # 复制/可视化都完成后才删，输出目录副本即删除前备份
                _delete_page_annotations(page_path, mask_path)
            tqdm.write(
                f"[FLAG] {rel}  regions={len(ratios)}  "
                f"max_ratio={max(ratios):.4f}" if ratios else f"[FLAG] {rel}  仅噪点标注"
            )

    report_path = output_root / "report.csv"
    with report_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["page", "annotated_but_empty", "n_regions",
                        "max_fg_ratio", "region_fg_ratios"],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    print("\n" + "=" * 50)
    print(f"检查页面（mask 非空）: {len(report_rows)}")
    action = "已复制+已删除原标注" if delete else "已复制，--no-delete 未删"
    print(f"有标注但无分子（{action}）: {n_flagged}")
    print(f"跳过（mask 全空/读图失败）: {n_skipped}")
    print(f"\n👉 命中页面+掩码保存至: {output_root}")
    print(f"👉 逐页判定明细: {report_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="用 UNet++ fg_ratio 门控找出\"mask 有标注但页面无分子结构\"的"
                    "恢复页面，复制页面+掩码到 eval 目录留档后，删除 restored 里"
                    "的假标注（mask 清零、yolo.txt 删除，页面图保留）。"
    )
    ap.add_argument("--input", default=str(RESTORED_ROOT),
                    help="restored 根目录（递归扫描 <subset>/<office>/*.png）")
    ap.add_argument("--output", default=str(OUTPUT_ROOT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threshold", type=float, default=PAGE_REGION_FG_THRESHOLD,
                    help=f"单个标注区域的 fg_ratio 达标下限（默认 {PAGE_REGION_FG_THRESHOLD}，"
                         "取值依据见文件顶部 PAGE_REGION_FG_THRESHOLD 注释；"
                         "任一区域达标即认为页面存在分子结构）。")
    ap.add_argument("--visualize", dest="visualize", action="store_true",
                    help="为命中页生成 <stem>.vis.png 整页叠加图（默认开启；"
                         "染色逻辑与 utils/post_segment.py 一致：UNet++ 前景绿、"
                         "分母掩码（标注框内墨迹）黄，另加红框标注各区域 fg_ratio）")
    ap.add_argument("--no-visualize", dest="visualize", action="store_false",
                    help="关闭可视化输出")
    ap.set_defaults(visualize=True)
    ap.add_argument("--delete", dest="delete", action="store_true",
                    help="删除命中页在 restored 里的假标注（默认开启）：mask 覆写为"
                         "同尺寸全零掩码（页面图保留，该页变成 no_box 无标注页），"
                         ".yolo.txt 整个删除；删除发生在复制到 --output 之后，"
                         "输出目录副本即删除前备份。")
    ap.add_argument("--no-delete", dest="delete", action="store_false",
                    help="只找出并复制命中页，不改动 restored（复查判定阈值时用）。")
    ap.set_defaults(delete=True)
    args = ap.parse_args()
    run(
        input_root=Path(args.input),
        output_root=Path(args.output),
        device=args.device,
        threshold=args.threshold,
        visualize=args.visualize,
        delete=args.delete,
    )


if __name__ == "__main__":
    main()
