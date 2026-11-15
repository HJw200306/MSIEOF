"""YoDe-Segmentation 提取表5所需的逐分子 crop。"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLOV5_CONFIG_DIR", str(PROJECT_ROOT / "outputs" / "yolov5_config"))

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

_common_spec = importlib.util.spec_from_file_location(
    "msieof_table5_common", Path(__file__).with_name("table5_common.py"))
_common = importlib.util.module_from_spec(_common_spec)
_common_spec.loader.exec_module(_common)
read_jsonl = _common.read_jsonl
validate_dataset = _common.validate_dataset
write_jsonl = _common.write_jsonl
load_page_checkpoints = _common.load_page_checkpoints
write_page_checkpoint = _common.write_page_checkpoint

# PyTorch 2.6 起 ``torch.load`` 默认 weights_only=True，而 YoDe/YOLOv5 的旧版
# checkpoint 含有模型对象。权重来自本地固定仓库，因此在该工作进程内恢复旧行为。
_torch_load = torch.load


def _trusted_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load(*args, **kwargs)


torch.load = _trusted_torch_load


def main() -> None:
    parser = argparse.ArgumentParser(description="表5 YoDe 分割工作进程")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.2)
    args = parser.parse_args()

    # YoDe vendors YOLOv5 as ``<repo>/yolov5``.  That checkout uses absolute
    # imports such as ``from utils.autoanchor import ...``; therefore the
    # YOLOv5 directory itself must precede this project's own ``utils`` package
    # on sys.path.  Adding only ``repo`` makes ``import yolov5`` work but sends
    # those legacy absolute imports to MSIEOF's unrelated ``utils`` package.
    yolo_root = args.repo / "yolov5"
    for import_root in (yolo_root, args.repo):
        try:
            sys.path.remove(str(import_root))
        except ValueError:
            pass
    sys.path[:0] = [str(yolo_root), str(args.repo)]
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
    # plots.py 会在导入时把 FONT 固化为默认参数，必须提前改成绝对路径。
    yolo_general.FONT = local_yolo_font

    def local_only_check_font(font=None):
        """YOLOv5 的字体仅用于绘图；只检查本机字体，绝不联网下载。"""
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

    # check_dataset() 在模型初始化时会调用此函数；必须在导入模型模块前替换。
    yolo_general.check_font = local_only_check_font
    from yolov5.models.common import DetectMultiBackend
    from yolov5.utils.datasets import LoadImages
    from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
    from yolov5.utils.torch_utils import select_device
    from deeplab_v3.src import deeplabv3_resnet50

    yolo_w = args.repo / "weights" / "best.pt"
    deeplab_w = args.repo / "weights" / "model.pth"
    data_yaml = args.repo / "yolov5" / "paper_data" / "myvoc.yaml"
    for required in (yolo_w, deeplab_w, data_yaml):
        if not required.is_file():
            parser.error(f"缺少 YoDe 文件: {required}")

    pages = read_jsonl(args.gt)
    if args.max_pages:
        pages = pages[:args.max_pages]
    validate_dataset(pages, allow_subset=bool(args.max_pages))
    wanted = {x["page_id"]: x for x in pages}
    completed = load_page_checkpoints(args.output) if args.resume else {}
    if set(wanted).issubset(completed):
        write_jsonl(args.output, [completed[page["page_id"]] for page in pages])
        print(f"YoDe 断点续跑: {len(pages)} 页已全部完成，直接复用", flush=True)
        return

    device = select_device("")
    yolo = DetectMultiBackend(str(yolo_w), device=device, dnn=False, data=str(data_yaml))
    image_size = check_img_size((640, 640), s=yolo.stride)
    segmenter = deeplabv3_resnet50(aux=False, num_classes=2)
    weights = torch.load(deeplab_w, map_location="cpu")["model"]
    segmenter.load_state_dict({k: v for k, v in weights.items() if "aux" not in k})
    segmenter.to(device).eval()
    resize = transforms.Resize(520)
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    crops_root = args.output.parent / "crops"
    dataset = LoadImages(str(args.images), img_size=image_size, stride=yolo.stride, auto=yolo.pt)
    for path, tensor, original, _capture, _status in dataset:
        page_id = Path(path).stem
        if page_id not in wanted:
            continue
        if page_id in completed:
            print(f"resume {page_id}", flush=True)
            continue
        x = torch.from_numpy(tensor).to(device)
        x = x.half() if yolo.fp16 else x.float()
        x /= 255.0
        if x.ndim == 3:
            x = x[None]
        detections = non_max_suppression(yolo(x), args.conf, args.iou, None, True,
                                         max_det=1000)
        instances = []
        for detection in detections:
            if not len(detection):
                continue
            detection[:, :4] = scale_coords(x.shape[2:], detection[:, :4], original.shape).round()
            for idx, (*xyxy, _confidence, _cls) in enumerate(reversed(detection)):
                raw = [int(float(v)) for v in xyxy]
                x1, y1 = max(0, raw[0] - 20), max(0, raw[1] - 20)
                x2, y2 = min(original.shape[1], raw[2] + 20), min(original.shape[0], raw[3] + 20)
                crop_bgr = original[y1:y2, x1:x2]
                if not crop_bgr.size:
                    continue
                pil = resize(Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)))
                with torch.no_grad():
                    mask = segmenter(to_tensor(pil).unsqueeze(0).to(device))["out"].argmax(1)
                mask = mask.squeeze(0).cpu().numpy().astype(bool)
                final = np.where(mask[:, :, None], np.asarray(pil), 255).astype(np.uint8)
                crop_file = crops_root / page_id / f"{page_id}_{idx:03d}.png"
                crop_file.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(final).save(crop_file)
                instances.append({"bbox": raw, "crop_file": str(crop_file.resolve())})
        row = {"page_id": page_id, "page_file": wanted[page_id]["page_file"],
               "instances": instances}
        completed[page_id] = row
        write_page_checkpoint(args.output, row)
        print(f"{page_id}: {len(instances)}", flush=True)
    missing = [p["page_id"] for p in pages if p["page_id"] not in completed]
    if missing:
        raise RuntimeError(f"YoDe 未处理到页面: {missing}")
    rows = [completed[p["page_id"]] for p in pages]
    write_jsonl(args.output, rows)


if __name__ == "__main__":
    main()
