"""DECIMER-Segmentation 提取 crop，并可同时运行 DECIMER OCSR。

本文件在 DECIMER/TensorFlow 环境运行。输出是通用实例清单，之后由主项目的
MolScribe/评分阶段处理。
"""
from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image

from utils.evaluation.table5_common import (load_page_checkpoints, read_jsonl,
                                            validate_dataset, write_jsonl,
                                            write_page_checkpoint)


def require_local_segmentation_weights() -> Path:
    """在导入 decimer_segmentation 前确认本地权重，禁止其自动下载。"""
    package_dir = Path(
        importlib.metadata.distribution("decimer-segmentation")
        .locate_file("decimer_segmentation")
    )
    weights = package_dir / "mask_rcnn_molecule.h5"
    if not weights.is_file():
        raise FileNotFoundError(
            "缺少本地 DECIMER-Segmentation 权重，已禁止自动下载。"
            f"请先把 mask_rcnn_molecule.h5 放到: {weights}"
        )
    print(f"使用本地 DECIMER-Segmentation 权重: {weights}", flush=True)
    return weights


def load_printed_decimer_predictor():
    """只加载表5需要的印刷体 DECIMER 模型，跳过无关的手绘模型下载。"""
    import tensorflow as tf

    package_dir = Path(importlib.metadata.distribution("decimer").locate_file("DECIMER"))

    def load_source(name: str, filename: str):
        spec = importlib.util.spec_from_file_location(name, package_dir / filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    pre_process = load_source("table5_decimer_pre_process", "pre_process.py")
    decimer_utils = load_source("table5_decimer_utils", "utils.py")
    model_dir = Path(os.environ.get(
        "DECIMER_MODEL_DIR", str(Path.home() / ".data" / "DECIMER-V2" / "DECIMER_model")))
    tokenizer_path = model_dir / "assets" / "tokenizer_SMILES.pkl"
    saved_model = model_dir / "saved_model.pb"
    if not tokenizer_path.is_file() or not saved_model.is_file():
        raise FileNotFoundError(
            f"缺少印刷体DECIMER模型: {model_dir}。请先初始化DECIMER或设置DECIMER_MODEL_DIR")
    with tokenizer_path.open("rb") as stream:
        tokenizer = pickle.load(stream)
    model = tf.saved_model.load(str(model_dir))

    def predict(image_input):
        chemical_structure = pre_process.decode_image(image_input)
        predicted_tokens, _confidence = model(tf.constant(chemical_structure))
        tokens = [tokenizer.index_word[i] for i in predicted_tokens[0].numpy()]
        text = "".join(map(str, tokens)).replace("<start>", "").replace("<end>", "")
        return decimer_utils.decoder(text)

    return predict


def main() -> None:
    parser = argparse.ArgumentParser(description="表5 DECIMER 分割/识别工作进程")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--with-ocsr", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    pages = read_jsonl(args.gt)
    if args.max_pages:
        pages = pages[:args.max_pages]
    validate_dataset(pages, allow_subset=bool(args.max_pages))
    completed = load_page_checkpoints(args.output) if args.resume else {}
    wanted_ids = {page["page_id"] for page in pages}
    if wanted_ids.issubset(completed):
        write_jsonl(args.output, [completed[page["page_id"]] for page in pages])
        print(f"DECIMER 断点续跑: {len(pages)} 页已全部完成，直接复用", flush=True)
        return

    # 该第三方包在导入时就会加载模型，并在权重缺失时联网下载；因此必须先检查。
    require_local_segmentation_weights()
    from decimer_segmentation import get_expanded_masks
    predict_smiles = None
    if args.with_ocsr:
        predict_smiles = load_printed_decimer_predictor()

    crops_root = args.output.parent / "crops"
    for page_no, page in enumerate(pages, 1):
        if page["page_id"] in completed:
            print(f"[{page_no}/{len(pages)}] resume {page['page_id']}", flush=True)
            continue
        image_path = args.images / page["page_file"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        masks = get_expanded_masks(image)
        if masks is None or masks.ndim != 3:
            masks = np.zeros((*image.shape[:2], 0), dtype=bool)
        instances = []
        page_dir = crops_root / page["page_id"]
        page_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(masks.shape[2]):
            mask = masks[:, :, idx].astype(bool)
            ys, xs = np.nonzero(mask)
            if not xs.size:
                continue
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            rgb = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
            masked = np.where(mask[y1:y2, x1:x2, None], rgb, 255).astype(np.uint8)
            crop = page_dir / f"{page['page_id']}_{idx:03d}.png"
            Image.fromarray(masked).save(crop)
            pred = ""
            if predict_smiles is not None:
                try:
                    pred = predict_smiles(str(crop)) or ""
                except Exception as exc:
                    print(f"DECIMER OCSR failed: {crop.name}: {exc}", flush=True)
            instances.append({"bbox": [x1, y1, x2, y2],
                              "crop_file": str(crop.resolve()), "pred_smiles": pred})
        row = {"page_id": page["page_id"], "page_file": page["page_file"],
               "instances": instances}
        completed[page["page_id"]] = row
        write_page_checkpoint(args.output, row)
        print(f"[{page_no}/{len(pages)}] {page['page_id']}: {len(instances)}", flush=True)
    write_jsonl(args.output, [completed[page["page_id"]] for page in pages])


if __name__ == "__main__":
    main()
