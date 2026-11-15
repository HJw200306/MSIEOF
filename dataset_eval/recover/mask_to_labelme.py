# -*- coding: utf-8 -*-
"""
benchmark_D2C-RND / benchmark_D2C-UNI 掩码页与 X-AnyLabeling json 的双向转换。

两种模式（由下方 MODE 常量控制）：
  "mask2json": 把每个 *.mask.png（黑底白框）转成 X-AnyLabeling/LabelMe 格式
               的 *.json，放在对应原始页 png 旁边。
  "json2mask": 把每个 *.json（人工修正后）重新渲染为黑底白框掩码，
               覆盖写回同名 *.mask.png。

用法：直接在 PyCharm 里运行本文件即可（无参数，按 MODE 执行）。
     也可命令行指定： python mask_to_labelme.py [mask2json|json2mask] [root1 root2 ...]

依赖：opencv-python, numpy, pillow
     pip install opencv-python numpy pillow
"""

import json
import os
import sys

import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    sys.exit(
        "缺少 opencv-python，请先安装：\n    pip install opencv-python numpy pillow"
    )

# ----------------------------- 可调参数 -----------------------------
# 运行模式："mask2json" 掩码->json；"json2mask" json->掩码(覆盖 *.mask.png)。
MODE = "json2mask"

# 待扫描的根目录（会递归查找其下所有 *.mask.png 或 *.json）
DEFAULT_ROOTS = [
    r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored\benchmark_D2C-RND",
    r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored\benchmark_D2C-UNI",
]

# 每个白色区域写入 json 时使用的标签名。PatCID 的 YOLO 标注全是单类(class 0)，
# 因此统一用 "mol"（分子结构）。如需其它名称改这里即可。
LABEL = "mol"

# 形状类型："polygon" 沿掩码轮廓生成多边形；"rectangle" 用外接矩形(两点)。
# 掩码本身是填充矩形框，两种都可以；polygon 对不规则掩码更忠实。
SHAPE_MODE = "polygon"

# 二值化阈值：灰度 > THRESHOLD 视为前景(白)。
THRESHOLD = 127

# 过滤噪点：面积(像素)小于该值的连通域忽略。
MIN_AREA = 50

# polygon 简化强度：epsilon = POLY_EPS_RATIO * 轮廓周长。越大点越少。
# 设为 0 则不简化（保留全部轮廓点）。
POLY_EPS_RATIO = 0.002

# 已存在同名 json 时是否覆盖。
OVERWRITE = True

# 写入 json 的版本号，与样例保持一致。
JSON_VERSION = "3.3.5"
# -------------------------------------------------------------------


def find_original_image(mask_path):
    """由 <stem>.mask.png 找到对应原始页图片 <stem>.png（或其它常见后缀）。"""
    base = mask_path[: -len(".mask.png")]
    for ext in (".png", ".PNG", ".jpg", ".jpeg", ".tif", ".tiff"):
        cand = base + ext
        if os.path.exists(cand):
            return cand
    return None


def load_mask_binary(mask_path):
    """读掩码为单通道二值图 (uint8, 0/255)。用 PIL 读以兼容中文路径。"""
    with Image.open(mask_path) as im:
        arr = np.array(im.convert("L"))
    _, binary = cv2.threshold(arr, THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def contour_to_points(contour):
    """cv2 轮廓 -> [[x, y], ...]，可选做多边形简化。"""
    if POLY_EPS_RATIO > 0:
        eps = POLY_EPS_RATIO * cv2.arcLength(contour, True)
        contour = cv2.approxPolyDP(contour, eps, True)
    pts = contour.reshape(-1, 2)
    return [[float(x), float(y)] for x, y in pts]


def mask_to_shapes(binary):
    """从二值掩码提取每个白色连通域，生成 LabelMe shape 列表。"""
    contours, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    shapes = []
    group_id = 1
    for contour in contours:
        if cv2.contourArea(contour) < MIN_AREA:
            continue

        if SHAPE_MODE == "rectangle":
            x, y, w, h = cv2.boundingRect(contour)
            points = [[float(x), float(y)], [float(x + w), float(y + h)]]
            shape_type = "rectangle"
        else:
            points = contour_to_points(contour)
            if len(points) < 3:
                continue
            shape_type = "polygon"

        shapes.append(
            {
                "label": LABEL,
                "points": points,
                "group_id": group_id,
                "description": "",
                "shape_type": shape_type,
                "flags": {},
            }
        )
        group_id += 1
    return shapes


def build_labelme_dict(shapes, image_name, height, width):
    return {
        "version": JSON_VERSION,
        "flags": {},
        "shapes": shapes,
        "imagePath": image_name,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "description": "",
    }


def convert_one(mask_path):
    """转换单个掩码文件，返回 (状态字符串, shape 数量)。"""
    original = find_original_image(mask_path)
    if original is None:
        return "skip-no-image", 0

    out_path = original.rsplit(".", 1)[0] + ".json"
    if os.path.exists(out_path) and not OVERWRITE:
        return "skip-exists", 0

    binary = load_mask_binary(mask_path)
    mask_h, mask_w = binary.shape[:2]

    # 用原图尺寸写入 json；若掩码与原图尺寸不同，按比例缩放坐标。
    with Image.open(original) as im:
        img_w, img_h = im.size
    if (mask_w, mask_h) != (img_w, img_h):
        binary = cv2.resize(
            binary, (img_w, img_h), interpolation=cv2.INTER_NEAREST
        )

    shapes = mask_to_shapes(binary)
    data = build_labelme_dict(shapes, os.path.basename(original), img_h, img_w)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return "ok", len(shapes)


def iter_mask_files(roots):
    for root in roots:
        if not os.path.isdir(root):
            print(f"[警告] 目录不存在，跳过：{root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".mask.png"):
                    yield os.path.join(dirpath, name)


# ======================= json -> mask（反向转换） =======================

def shape_to_contour(shape):
    """LabelMe shape -> cv2 填充用的整数轮廓数组；不支持的形状返回 None。"""
    points = shape.get("points") or []
    shape_type = shape.get("shape_type", "polygon")

    if shape_type == "rectangle":
        # rectangle 常见两点(对角)，也可能是四点，统一取外接范围。
        if len(points) < 2:
            return None
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x0, x1 = min(xs), max(xs)
        y0, y1 = min(ys), max(ys)
        points = [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    elif shape_type != "polygon":
        return None
    elif len(points) < 3:
        return None

    return np.array(
        [[round(x), round(y)] for x, y in points], dtype=np.int32
    ).reshape(-1, 1, 2)


def json_to_mask(json_path):
    """把单个 json 渲染为黑底白框掩码，覆盖写回同名 *.mask.png。

    返回 (状态字符串, 写入的形状数量)。
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    height = data.get("imageHeight")
    width = data.get("imageWidth")
    if not height or not width:
        return "skip-no-size", 0

    mask = np.zeros((int(height), int(width)), dtype=np.uint8)
    n_drawn = 0
    n_skipped = 0
    for shape in data.get("shapes", []):
        contour = shape_to_contour(shape)
        if contour is None:
            n_skipped += 1
            continue
        cv2.fillPoly(mask, [contour], 255)
        n_drawn += 1
    if n_skipped:
        print(f"[警告] {os.path.basename(json_path)}: "
              f"{n_skipped} 个形状类型不支持或点数不足，未绘制")

    mask_path = json_path[: -len(".json")] + ".mask.png"
    # 用 PIL 保存以兼容中文路径（cv2.imwrite 对非 ASCII 路径不可靠）。
    Image.fromarray(mask).save(mask_path)
    return "ok", n_drawn


def iter_json_files(roots):
    for root in roots:
        if not os.path.isdir(root):
            print(f"[警告] 目录不存在，跳过：{root}")
            continue
        for dirpath, _, filenames in os.walk(root):
            for name in filenames:
                if name.lower().endswith(".json"):
                    yield os.path.join(dirpath, name)


def run_mask2json(roots):
    masks = list(iter_mask_files(roots))
    if not masks:
        print("没有找到任何 *.mask.png 文件。请检查根目录设置。")
        return

    total_shapes = 0
    counts = {"ok": 0, "skip-no-image": 0, "skip-exists": 0}
    for mask_path in masks:
        status, n = convert_one(mask_path)
        counts[status] = counts.get(status, 0) + 1
        total_shapes += n
        if status == "ok":
            print(f"[OK] {os.path.basename(mask_path)} -> {n} 个标注")
        elif status == "skip-no-image":
            print(f"[跳过] 找不到原图：{mask_path}")

    print("\n==== 完成 (mask2json) ====")
    print(f"掩码文件总数 : {len(masks)}")
    print(f"成功转换     : {counts['ok']}")
    print(f"跳过(无原图) : {counts['skip-no-image']}")
    print(f"跳过(已存在) : {counts['skip-exists']}")
    print(f"标注总数     : {total_shapes}")


def run_json2mask(roots):
    jsons = list(iter_json_files(roots))
    if not jsons:
        print("没有找到任何 *.json 文件。请检查根目录设置。")
        return

    total_shapes = 0
    counts = {"ok": 0, "skip-no-size": 0, "error": 0}
    for json_path in jsons:
        try:
            status, n = json_to_mask(json_path)
        except (json.JSONDecodeError, OSError, KeyError, TypeError) as e:
            counts["error"] += 1
            print(f"[出错] {json_path}: {e}")
            continue
        counts[status] = counts.get(status, 0) + 1
        total_shapes += n
        if status == "ok":
            print(f"[OK] {os.path.basename(json_path)} -> 掩码含 {n} 个白框")
        else:
            print(f"[跳过] 缺 imageHeight/imageWidth：{json_path}")

    print("\n==== 完成 (json2mask) ====")
    print(f"json 文件总数 : {len(jsons)}")
    print(f"成功写回掩码  : {counts['ok']}")
    print(f"跳过(缺尺寸)  : {counts['skip-no-size']}")
    print(f"出错          : {counts['error']}")
    print(f"白框总数      : {total_shapes}")


def main():
    args = sys.argv[1:]
    mode = MODE
    if args and args[0] in ("mask2json", "json2mask"):
        mode = args[0]
        args = args[1:]
    roots = args if args else DEFAULT_ROOTS

    if mode == "json2mask":
        run_json2mask(roots)
    else:
        run_mask2json(roots)


if __name__ == "__main__":
    main()
