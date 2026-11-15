"""对表5实例清单执行 MolScribe（可选）并按消融实验口径评分。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from utils.evaluation.table5_common import (read_jsonl, score_manifest,
                                            load_page_checkpoints, validate_dataset,
                                            write_json, write_jsonl,
                                            write_page_checkpoint)


def _prediction_page_reusable(cached, source) -> bool:
    if not isinstance(cached, dict) or cached.get("page_id") != source.get("page_id"):
        return False
    old_instances = cached.get("instances", [])
    new_instances = source.get("instances", [])
    if len(old_instances) != len(new_instances):
        return False
    return all(
        old.get("crop_file") == new.get("crop_file")
        and old.get("bbox") == new.get("bbox")
        and "pred_smiles" in old
        for old, new in zip(old_instances, new_instances)
    )


def add_molscribe_predictions(rows, batch_size: int, output: Path,
                               resume: bool = False) -> None:
    completed = load_page_checkpoints(output) if resume else {}
    pending = []
    for row in rows:
        cached = completed.get(row["page_id"])
        if _prediction_page_reusable(cached, row):
            print(f"MolScribe resume {row['page_id']}", flush=True)
        else:
            completed.pop(row["page_id"], None)
            pending.append(row)
    if not pending:
        rows[:] = [completed[row["page_id"]] for row in rows]
        write_jsonl(output, rows)
        print("MolScribe 断点续跑: 所有页已完成，直接复用", flush=True)
        return

    from utils.base.chemical_graph import convert_graph
    from utils.pipeline.img2smiles import Img2SmilesProcessor
    import molscribe.chemistry as chemistry
    import molscribe.interface as interface

    # 与 ablation_v2.serial_conversion 使用同一策略。Windows 的 spawn 子进程会
    # 重新导入 MolScribe/timm，既容易版本冲突，也会让每个 worker 重复占用大量
    # 内存。表5在主进程串行完成图结构转 SMILES，模型推理仍保持批处理。
    def serial_convert(coords, symbols, edges, images=None, num_workers=1):
        args = (zip(coords, symbols, edges) if images is None
                else zip(coords, symbols, edges, images))
        converted = [chemistry._convert_graph_to_smiles(*item) for item in args]
        if not converted:
            return (), (), 0.0
        smiles, molblocks, success = zip(*converted)
        return smiles, molblocks, float(np.mean(success))

    interface.convert_graph_to_smiles = serial_convert
    chemistry.convert_graph_to_smiles = serial_convert

    processor = Img2SmilesProcessor(device="cuda")
    for page_no, row in enumerate(pending, 1):
        refs = [(instance, Path(instance["crop_file"]))
                for instance in row.get("instances", [])]
        for start in range(0, len(refs), batch_size):
            chunk = refs[start:start + batch_size]
            images = [np.asarray(Image.open(path).convert("RGB")) for _, path in chunk]
            predictions = processor.predict_batch(images, batch_size=batch_size)
            for (instance, _path), image, pred in zip(chunk, images, predictions):
                chemical = convert_graph(pred.get("atoms", []), pred.get("bonds", []),
                                         image_size=[image.shape[1], image.shape[0]],
                                         coordinates_normalized=True)
                instance["pred_smiles"] = chemical.get("smiles", "")
                instance["molscribe_raw_smiles"] = pred.get("smiles", "")
                instance["molscribe_valid"] = chemical.get("valid", False)
            print(f"MolScribe {row['page_id']} "
                  f"{min(start + batch_size, len(refs))}/{len(refs)}", flush=True)
        completed[row["page_id"]] = row
        write_page_checkpoint(output, row)
        print(f"MolScribe 页面完成 {page_no}/{len(pending)}: {row['page_id']}", flush=True)
    rows[:] = [completed[row["page_id"]] for row in rows]
    write_jsonl(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="表5统一 MolScribe/评分工作进程")
    parser.add_argument("--gt", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--recognizer", choices=("molscribe", "existing"), required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    gt_rows = read_jsonl(args.gt)
    if args.max_pages:
        gt_rows = gt_rows[:args.max_pages]
    validate_dataset(gt_rows, allow_subset=bool(args.max_pages))
    rows = read_jsonl(args.instances)
    wanted = {x["page_id"] for x in gt_rows}
    rows = [x for x in rows if x["page_id"] in wanted]
    if args.recognizer == "molscribe":
        add_molscribe_predictions(rows, args.batch_size,
                                  args.output_dir / "predictions.jsonl", args.resume)
    result = score_manifest(args.method, gt_rows, rows)
    write_json(args.output_dir / "pages.json", result["pages"])
    write_json(args.output_dir / "summary.json", result["summary"])
    print(result["summary"], flush=True)


if __name__ == "__main__":
    main()
