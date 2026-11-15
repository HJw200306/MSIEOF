"""YoDe-Segmentation 预测模块（官方 get_molecular(0.5) 后处理口径）。

实现论文表4所用的 YoDe 官方掩码后处理口径。Complete、Partial 和
Coverage Rate 由统一评分器按论文规定的 0.90 阈值计算：
    YOLOv5 检测 → bbox 扩边 20px crop → Deeplabv3 resize520 argmax(class1)
    → get_molecular(0.5)：8 连通域标记，只保留面积 > 0.5×最大连通域的部分，
      每个保留连通域拆成独立实例（丢弃的小连通域不计入）。

与 predict_yode.py（完整 class-1 掩码）的**唯一区别**就是后处理这一步：
predict_yode.py 把整块 class-1 掩码当一个实例；本模块按 YoDe 官方
Image_Processing/biggest_shadow.py 的 get_molecular 逻辑做连通域过滤与拆分。

仅在 YoDe_Seg 环境可用（yolov5 + deeplab_v3 + torch），惰性加载。
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from unittest.mock import patch
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

CONF = 0.25
IOU = 0.2
EXPAND = 20.0
IMGSZ = 640
CC_AREA_RATIO = 0.5  # get_molecular 的面积比阈值（>0.5×最大连通域保留）

@lru_cache(maxsize=1)
def _get_ctx(repo: Path):
    yolo_w = repo / "weights" / "best.pt"
    dl_w = repo / "weights" / "model.pth"
    data_yaml = repo / "yolov5" / "paper_data" / "myvoc.yaml"
    for path in (yolo_w, dl_w, data_yaml):
        if not path.is_file():
            raise FileNotFoundError(f"缺少 YoDe 文件: {path}")
    # Legacy YOLOv5 uses top-level imports (``from utils...``).  Keep its
    # directory ahead of MSIEOF's identically named ``utils`` package.
    import_roots = (repo / "yolov5", repo)
    for import_root in import_roots:
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass
    sys.path[:0] = [str(path) for path in import_roots]
    import torch
    from torchvision import transforms

    # Importing models.common also imports plots.py, whose class defaults try
    # to download Arial.ttf immediately when the configured copy is absent.
    # Evaluation does not need that download; bind YOLOv5 to a local Windows
    # font before plots.py is imported.
    from yolov5.utils import general as yolo_general

    system_font = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / "arial.ttf"
    configured_font = Path(yolo_general.CONFIG_DIR) / "Arial.ttf"
    local_yolo_font = next(
        (path for path in (configured_font, system_font) if path.is_file()), None
    )
    if local_yolo_font is None:
        raise FileNotFoundError(
            "缺少本地 Arial 字体，已禁止 YOLOv5 自动下载。"
            f"已检查: {configured_font}, {system_font}"
        )
    yolo_general.FONT = local_yolo_font

    def local_only_check_font(font=None):
        requested = Path(font) if font is not None else local_yolo_font
        candidates = (
            requested,
            Path(yolo_general.CONFIG_DIR) / requested.name,
            system_font,
        )
        local_font = next((path for path in candidates if path.is_file()), None)
        if local_font is None:
            raise FileNotFoundError(
                "缺少本地 Arial 字体，已禁止 YOLOv5 自动下载。"
                f"已检查: {', '.join(map(str, candidates))}"
            )
        return local_font

    yolo_general.check_font = local_only_check_font
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
    from yolov5.utils.torch_utils import select_device
    from deeplab_v3.src import deeplabv3_resnet50

    device = select_device("")
    # 本地 YoDe 的旧 YOLOv5 权重含模型对象，需要使用可信的原始权重。
    # PyTorch 新版的 weights_only 默认值与旧版不同，只在加载这个模型时适配。
    original_load = torch.load

    def load_local_checkpoint(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    with patch.object(torch, "load", load_local_checkpoint):
        yolo = DetectMultiBackend(str(yolo_w), device=device, dnn=False, data=str(data_yaml))
    imgsz = check_img_size((IMGSZ, IMGSZ), s=yolo.stride)

    dl = deeplabv3_resnet50(aux=False, num_classes=2)
    wd = torch.load(dl_w, map_location="cpu", weights_only=False)["model"]
    wd = {k: v for k, v in wd.items() if "aux" not in k}
    dl.load_state_dict(wd)
    dl.to(device).eval()

    resize = transforms.Resize(520)
    totensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    return dict(
        torch=torch, yolo=yolo, dl=dl, device=device, imgsz=imgsz,
        resize=resize, totensor=totensor,
        non_max_suppression=non_max_suppression, scale_coords=scale_coords,
    )


def _get_molecular_components(mask: np.ndarray, ratio: float = CC_AREA_RATIO) -> List[np.ndarray]:
    """复刻 YoDe 官方 get_molecular：8 连通域标记 + 面积比过滤。

    mask: (h, w) uint8，前景=1（class-1 掩码）。
    返回：每个保留连通域的 bool 掩码（shape 同 mask）。
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask == 1).astype(np.uint8), connectivity=8,
    )
    if num <= 1:
        return []
    areas = stats[1:, cv2.CC_STAT_AREA].astype(float)  # 排除 label 0（背景）
    max_area = areas.max()
    if max_area <= 0:
        return []
    keep = np.where(areas > ratio * max_area)[0] + 1
    return [labels == int(k) for k in keep]


def predict_yode_official_for_page(page_png: Path, repo: Path) -> List[Dict]:
    """对一张页面跑 YoDe（官方 get_molecular 口径），返回 score.py 口径实例列表。"""
    ctx = _get_ctx(repo)
    torch = ctx["torch"]
    yolo = ctx["yolo"]
    dl = ctx["dl"]
    device = ctx["device"]

    from yolov5.utils.datasets import LoadImages

    instances: List[Dict] = []
    dataset = LoadImages(str(page_png), img_size=ctx["imgsz"], stride=yolo.stride, auto=yolo.pt)
    for path, im, im0s, _vid_cap, _s in dataset:
        im = torch.from_numpy(im).to(device)
        im = im.half() if yolo.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]
        pred = yolo(im)
        pred = ctx["non_max_suppression"](pred, CONF, IOU, None, True, max_det=1000)

        im0 = im0s.copy()
        H, W = im0.shape[:2]
        for det in pred:
            if len(det) == 0:
                continue
            det[:, :4] = ctx["scale_coords"](im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, _conf, _cls in reversed(det):
                raw = [float(x) for x in xyxy]
                x1 = int(max(0, raw[0] - EXPAND))
                y1 = int(max(0, raw[1] - EXPAND))
                x2 = int(min(W, raw[2] + EXPAND))
                y2 = int(min(H, raw[3] + EXPAND))
                crop_bgr = im0[y1:y2, x1:x2]
                if crop_bgr.size == 0:
                    continue

                ch, cw = crop_bgr.shape[:2]
                pil = _PIL_from_bgr(crop_bgr)
                pil_r = ctx["resize"](pil)
                t = ctx["totensor"](pil_r).unsqueeze(0).to(device)
                with torch.no_grad():
                    mask = dl(t)["out"].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

                mask_crop = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)

                # —— 与 predict_yode.py 的唯一差异：官方 get_molecular 连通域过滤 + 拆分 ——
                for comp in _get_molecular_components(mask_crop):
                    full = np.zeros((H, W), dtype=np.uint8)
                    full[y1:y2, x1:x2] = comp.astype(np.uint8)
                    ys, xs = np.nonzero(full)
                    if xs.size == 0:
                        continue
                    bx1, bx2 = int(xs.min()), int(xs.max()) + 1
                    by1, by2 = int(ys.min()), int(ys.max()) + 1
                    inst_mask = full[by1:by2, bx1:bx2] > 0
                    instances.append({
                        "bbox": (bx1, by1, bx2, by2),
                        "mask": inst_mask,
                        "area": int(inst_mask.sum()),
                    })
        break  # 单页只迭代一次

    return instances


def _PIL_from_bgr(crop_bgr: np.ndarray):
    from PIL import Image
    return Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))


def free_gpu_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
