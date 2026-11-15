"""枚举、运行并汇总表4。每个方法可使用独立的 Python 环境。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

from .config import OFFICES, RECALL_THRESHOLD, SUBSETS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "table4_coverage"
DEFAULT_YODE = Path.home() / "Desktop" / "seg_molscribe" / "YoDe-Segmentation"


@dataclass(frozen=True)
class Dataset:
    name: str
    split: str | None
    mol_only: bool
    pages: int
    gt: int


DATASETS = {
    "pjchem-test": Dataset("PJChemPage-Test", "val", True, 450, 5704),
    "patcid-repage": Dataset("PatCID-RePage", None, False, 449, 3407),
    "pjchem-page-to-smiles": Dataset("PJChemPageToSMILES", "PageToSMILES_approved", False, 50, 562),
}
METHODS = {
    "msieof": "MSIEOF",
    "decimer": "DECIMER-Segmentation",
    "yode": "YoDe-Segmentation",
}
COUNTS = ("n_gt", "n_pred", "n_complete", "n_missed", "n_false_positive", "n_partial")
COLUMNS = ("Dataset", "Method", "Pages", "GT", "Pred", "Complete", "Missed",
           "False Positive", "Partial", "Coverage Rate (%)")
# 旧表数字来自 0.60 阈值，仅用于定位阈值修正带来的变化。
# 当前论文口径由 RECALL_THRESHOLD=0.90 控制，正式表格由逐页记录求和。
LEGACY_TABLE_AT_060 = {
    "pjchem-test": [(5595, 5552, 150, 41, 2), (5573, 5469, 222, 91, 13), (5305, 4434, 429, 30, 841)],
    "patcid-repage": [(3435, 3310, 53, 81, 44), (3399, 3300, 60, 52, 47), (3206, 2629, 256, 55, 522)],
    "pjchem-page-to-smiles": [(561, 551, 6, 5, 5), (553, 543, 17, 8, 2), (553, 468, 13, 4, 81)],
}

# ``predict_yode_official.py`` changed only in its import-path/font bootstrap:
# the detector, segmenter and mask post-processing are unchanged.  This exact
# migration lets completed checkpoints survive that repair without weakening
# provenance checks for future algorithm changes.
_YODE_RUNTIME_ONLY_MIGRATION = {
    "a235e0d29778f041926dff78c9a8c1684493065bb4f793ec9d5fd088c2abee45",
    "24087396dc60cf8242ac997932fb0d6a052aaedc3aaf33f3a1fd2aa8f6e525de",
}

_METHOD_CODE_SUFFIXES = {
    "msieof": (
        "eval_coverage/build_gt.py", "eval_coverage/config.py",
        "eval_coverage/fragment_cleanup_integration.py", "eval_coverage/image_io.py",
        "eval_coverage/predict.py", "eval_coverage/score.py",
        "eval_msirost/score.py", "eval_msirost/fg_utils.py",
        "utils/pipeline/extract.py", "utils/pipeline/post_segment.py",
        "utils/pipeline/fragment_cleanup.py",
    ),
    "decimer": (
        "eval_coverage/build_gt.py", "eval_coverage/config.py",
        "eval_coverage/image_io.py", "eval_coverage/predict_decimer.py",
        "eval_coverage/score.py", "eval_msirost/score.py", "eval_msirost/fg_utils.py",
    ),
    "yode": (
        "eval_coverage/build_gt.py", "eval_coverage/config.py",
        "eval_coverage/image_io.py", "eval_coverage/predict_yode_official.py",
        "eval_coverage/score.py", "eval_msirost/score.py", "eval_msirost/fg_utils.py",
        "yolov5/models/common.py", "yolov5/paper_data/myvoc.yaml",
    ),
}


@dataclass(frozen=True)
class Page:
    key: str
    image: Path
    mask: Path
    annotation: Path


def list_pages(dataset: Dataset, args) -> list[Page]:
    if dataset.split is not None:
        if args.data_root is None:
            raise ValueError("请用 --data-root 指定含 images、json、masks 的自建集根目录。")
        root = args.data_root.resolve()
        split = dataset.split
        image_dir = root / "images" / split
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        pages = []
        for image in sorted(image_dir.glob("*.png")):
            annotation = root / "json" / split / (image.stem + ".json")
            if not annotation.is_file():
                annotation = image.with_suffix(".json")
            pages.append(Page(image.name, image, root / "masks" / split / image.name, annotation))
        return pages

    if args.patcid_root is None:
        raise ValueError("请用 --patcid-root 指定包含 benchmark_D2C-RND/UNI 的 restored 根目录。")
    root = args.patcid_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    pages = []
    for subset in SUBSETS:
        for office in OFFICES:
            for image in sorted((root / subset / office).glob("*.png")):
                if image.name.endswith((".mask.png", ".vis.png")):
                    continue
                mask = image.with_suffix(".mask.png")
                if mask.is_file():
                    pages.append(Page(image.relative_to(root).as_posix(), image, mask, image.with_suffix(".json")))
    return pages


def build_page_gt(page: Page, dataset: Dataset):
    from .build_gt import build_gt_for_page

    return build_gt_for_page(
        page.image, page.mask, json_path=page.annotation, mol_only=dataset.mol_only,
        color_aware=dataset.split is not None,
    )


def predict_page(method, page, dataset, shape, repo, crops_root):
    if method == "msieof":
        from . import predict

        records = predict.predict_for_page(
            page.image, "page", color_aware=dataset.split is not None,
            crops_root=crops_root, fragment_cleanup=True, use_page_lines=False,
        )
        return predict.build_page_pred_masks(records, shape)
    if method == "decimer":
        from .predict_decimer import predict_decimer_for_page

        return predict_decimer_for_page(page.image)
    from .predict_yode_official import predict_yode_official_for_page

    return predict_yode_official_for_page(page.image, repo)


def release_cache(method):
    module = {"msieof": "predict", "decimer": "predict_decimer", "yode": "predict_yode_official"}[method]
    loaded = sys.modules.get(f"{__package__}.{module}")
    if loaded is not None:
        loaded.free_gpu_memory()


def file_stamp(path: Path):
    if not path.is_file():
        return None
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def make_manifest(pages):
    gt_code = [Path(__file__).with_name(name) for name in ("build_gt.py", "config.py", "image_io.py")]
    gt_code.append(PROJECT_ROOT / "dataset_eval" / "eval_msirost" / "fg_utils.py")
    return digest({
        "pages": [[p.key, file_stamp(p.image), file_stamp(p.mask), file_stamp(p.annotation)] for p in pages],
        "gt_code": [hashlib.sha256(p.read_bytes()).hexdigest() for p in gt_code],
    })


def provenance(method, repo, manifest, pages):
    code_files = list(Path(__file__).parent.glob("*.py"))
    code_files += [PROJECT_ROOT / "dataset_eval" / "eval_msirost" / name
                   for name in ("score.py", "fg_utils.py")]
    weights = []
    if method == "msieof":
        weights = [PROJECT_ROOT / "model" / name for name in ("YOLOv11_best.pt", "post_seg_unetpp.pth")]
        code_files += [PROJECT_ROOT / "utils" / "pipeline" / name
                       for name in ("extract.py", "post_segment.py", "fragment_cleanup.py")]
    elif method == "yode":
        weights = [repo / "weights" / name for name in ("best.pt", "model.pth")]
        code_files += [repo / name for name in ("yolov5/models/common.py", "yolov5/paper_data/myvoc.yaml")]
    for path in weights:
        if not path.is_file():
            raise FileNotFoundError(f"缺少 {METHODS[method]} 权重: {path}")
    from importlib.metadata import PackageNotFoundError, version

    packages = {}
    for name in ("numpy", "scipy", "opencv-python", "opencv-python-headless",
                 "torch", "torchvision", "ultralytics", "tensorflow", "decimer-segmentation"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return {
        "manifest": manifest, "pages": [p.key for p in pages], "threshold": RECALL_THRESHOLD,
        "python": sys.executable, "python_version": sys.version, "packages": packages,
        "weights": [file_stamp(p) for p in weights],
        "yode_repo": str(repo) if method == "yode" else None,
        "code": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(code_files)},
    }


def atomic_text(path, content):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", delete=False,
                                     dir=path.parent, prefix=path.name + ".") as stream:
        temp_path = Path(stream.name)
        stream.write(content)
    try:
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def write_json(path, data):
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def write_csv(path, columns, rows):
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, "\ufeff" + buffer.getvalue())


def _relevant_code(config):
    """Return only code that can affect this method's numerical result."""
    suffixes = _METHOD_CODE_SUFFIXES[config["method"]]
    selected = {}
    for filename, checksum in config.get("code", {}).items():
        normalized = filename.replace("\\", "/").lower()
        for suffix in suffixes:
            if normalized.endswith(suffix.lower()):
                selected[suffix] = checksum
                break
    return selected


def compatible_checkpoint_config(old, new):
    """Accept checkpoints when only unrelated orchestration code changed."""
    old_base = {key: value for key, value in old.items() if key != "code"}
    new_base = {key: value for key, value in new.items() if key != "code"}
    if old_base != new_base:
        return False
    old_code, new_code = _relevant_code(old), _relevant_code(new)
    if old_code == new_code:
        return True
    if new["method"] != "yode" or old_code.keys() != new_code.keys():
        return False
    differences = {
        key for key in old_code if old_code[key] != new_code[key]
    }
    yode_file = "eval_coverage/predict_yode_official.py"
    return (
        differences == {yode_file}
        and {old_code[yode_file], new_code[yode_file]} == _YODE_RUNTIME_ONLY_MIGRATION
    )


@contextmanager
def run_lock(folder):
    # Windows 文件锁随进程退出释放，崩溃后可以用原参数续跑。
    import msvcrt

    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".run.lock").open("a+b") as lock:
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise ValueError(f"已有任务正在写入 {folder}") from exc
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def validate_counts(row):
    if any(type(row[key]) is not int or row[key] < 0 for key in COUNTS):
        raise ValueError("逐页结果存在非法计数。")
    matched = row["n_complete"] + row["n_partial"]
    if row["n_gt"] != matched + row["n_missed"] or row["n_pred"] != matched + row["n_false_positive"]:
        raise ValueError("逐页结果不满足 GT/Pred 计数关系。")


def audit(args):
    selected = [args.dataset] if args.dataset else list(DATASETS)
    ok = True
    for key in selected:
        dataset = DATASETS[key]
        pages = list_pages(dataset, args)
        gt = sum(len(build_page_gt(page, dataset)[0]) for page in pages)
        matched = (len(pages), gt) == (dataset.pages, dataset.gt)
        ok = ok and matched
        print(f"{dataset.name}: Pages={len(pages)}/{dataset.pages}, GT={gt}/{dataset.gt}"
              f"  {'一致' if matched else '不一致'}", flush=True)
    return 0 if ok else 1


def run(args):
    from .score import score_page

    dataset = DATASETS[args.dataset]
    all_pages = list_pages(dataset, args)
    sample = bool(args.offset or args.limit)
    if not sample and len(all_pages) != dataset.pages:
        raise ValueError(f"{dataset.name}: 找到 {len(all_pages)} 页，表4要求 {dataset.pages} 页。")
    pages = all_pages[args.offset:]
    if args.limit:
        pages = pages[:args.limit]
    if not pages:
        raise ValueError("没有选中的页面。请检查数据目录、--offset 和 --limit。")

    output = args.output_dir
    if sample:
        output = output / "smoke" / f"offset-{args.offset}_limit-{args.limit}"
    folder = output / args.dataset / args.method
    manifest = make_manifest(all_pages)
    run_config = provenance(args.method, args.yode_repo.resolve(), manifest, pages)
    run_config.update(dataset=args.dataset, method=args.method, sample=sample)
    with run_lock(folder):
        return run_pages(args, dataset, pages, folder, run_config, score_page)


def prepare_backend(method, yode_repo):
    """将依赖和权重加载错误与单页数据错误分开报告。"""
    if method == "msieof":
        from . import predict

        predict.prepare()
    elif method == "decimer":
        from .predict_decimer import _get_expand_fn

        _get_expand_fn()
    else:
        from .predict_yode_official import _get_ctx

        _get_ctx(yode_repo)


def run_pages(args, dataset, pages, folder, run_config, score_page):
    checkpoint = folder / "checkpoint.json"
    if checkpoint.is_file():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state["config"] != run_config:
            if compatible_checkpoint_config(state["config"], run_config):
                print("检测到仅运行适配层发生变化，保留已有逐页结果并更新运行记录。", flush=True)
                state["config"] = run_config
                write_json(checkpoint, state)
            elif not state.get("rows") and not state.get("errors"):
                print("检测到未产生逐页结果的旧 checkpoint，按当前配置重新初始化。", flush=True)
                state = {"config": run_config, "rows": {}, "errors": {}}
                write_json(checkpoint, state)
            else:
                raise ValueError(
                    "已有逐页结果使用了不同的数据、代码或环境，不能混合续跑。"
                    f"请保留旧结果并指定新的 --output-dir。旧结果在 {folder}"
                )
    else:
        state = {"config": run_config, "rows": {}, "errors": {}}
        write_json(checkpoint, state)
    write_json(folder / "run_config.json", run_config)

    rows, errors = state["rows"], state["errors"]
    if any(page.key not in rows for page in pages):
        print(f"正在加载 {METHODS[args.method]}，解释器: {sys.executable}", flush=True)
        try:
            prepare_backend(args.method, args.yode_repo.resolve())
        except Exception as exc:
            raise ValueError(f"{METHODS[args.method]} 初始化失败: {type(exc).__name__}: {exc}") from exc
    for index, page in enumerate(pages, 1):
        if page.key in rows:
            continue
        started = perf_counter()
        try:
            gt, label = build_page_gt(page, dataset)
            if args.method == "msieof":
                with tempfile.TemporaryDirectory(prefix="table4_crops_", dir=folder) as temp:
                    pred = predict_page(args.method, page, dataset, label.shape, args.yode_repo, Path(temp))
            else:
                pred = predict_page(args.method, page, dataset, label.shape, args.yode_repo.resolve(), None)
            result = score_page(gt, pred, threshold=RECALL_THRESHOLD)
            row = {name: result[name] for name in COUNTS}
            validate_counts(row)
            rows[page.key] = {"page": page.key, **row, "coverage_rate": result["coverage_rate"],
                              "elapsed_s": round(perf_counter() - started, 2)}
            errors.pop(page.key, None)
            print(f"[{index}/{len(pages)}] {page.key} GT={row['n_gt']} Pred={row['n_pred']} "
                  f"Complete={row['n_complete']}", flush=True)
        except Exception as exc:
            errors[page.key] = f"{type(exc).__name__}: {exc}"
            print(f"[{index}/{len(pages)}] {page.key} 失败: {errors[page.key]}", flush=True)
        finally:
            write_json(checkpoint, state)
            release_cache(args.method)

    ordered = [rows[p.key] for p in pages if p.key in rows]
    write_csv(folder / "per_page.csv", ("page", *COUNTS, "coverage_rate", "elapsed_s"), ordered)
    write_json(folder / "per_page.json", ordered)
    write_json(folder / "errors.json", errors)
    total_gt = sum(row["n_gt"] for row in ordered)
    print(f"{dataset.name} / {METHODS[args.method]}: {len(ordered)}/{len(pages)} 页，"
          f"GT={total_gt}，失败 {len(errors)} 页。输出: {folder}")
    if not run_config["sample"] and total_gt != dataset.gt:
        print(f"表4要求 GT={dataset.gt}，当前结果无法纳入正式汇总。")
        return 1
    return int(bool(errors))


def summarize(folder, dataset_key, method):
    path = folder / "checkpoint.json"
    if not path.is_file():
        raise ValueError(f"缺少全量结果: {folder}")
    state = json.loads(path.read_text(encoding="utf-8"))
    config = state["config"]
    dataset = DATASETS[dataset_key]
    if (config["sample"] or config["dataset"] != dataset_key or config["method"] != method
            or config["threshold"] != RECALL_THRESHOLD):
        raise ValueError(f"结果的评测配置不匹配: {folder}")
    rows = state["rows"]
    keys = config["pages"]
    if state["errors"] or len(keys) != dataset.pages or len(rows) != dataset.pages or set(keys) != rows.keys():
        raise ValueError(f"结果未覆盖全部页面或有失败记录，请续跑: {folder}")
    for row in rows.values():
        validate_counts(row)
    totals = [sum(row[field] for row in rows.values()) for field in COUNTS]
    if totals[0] != dataset.gt:
        raise ValueError(f"{folder} 的 GT={totals[0]}，表4要求 {dataset.gt}。")
    rate = 100 * totals[2] / totals[0]
    record = dict(zip(COLUMNS, (dataset.name, METHODS[method], len(keys), *totals, f"{rate:.2f}")))
    return record, (config["manifest"], {key: rows[key]["n_gt"] for key in keys})


def markdown_table(columns, rows):
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(row[key]) for key in columns) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def aggregate(args):
    output = args.output_dir
    rows, comparison, errors = [], [], []
    for dataset_key, dataset in DATASETS.items():
        baseline = None
        for index, method in enumerate(METHODS):
            try:
                row, manifest = summarize(output / dataset_key / method, dataset_key, method)
                if baseline is not None and manifest != baseline:
                    raise ValueError(f"{dataset.name} / {METHODS[method]} 的页面或逐页 GT 与其他方法不一致。")
                baseline = manifest
            except (ValueError, KeyError, OSError) as exc:
                errors.append(str(exc))
                continue
            rows.append(row)
            expected = LEGACY_TABLE_AT_060[dataset_key][index]
            deltas = {column: row[column] - value for column, value in zip(COLUMNS[4:9], expected)}
            expected_pct = float(f"{100 * expected[1] / dataset.gt:.2f}")
            comparison.append({"Dataset": dataset.name, "Method": METHODS[method], **deltas,
                               "Coverage 差值（百分点）": f"{float(row[COLUMNS[-1]]) - expected_pct:+.2f}"})
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    title = "表4 不同页面评测集上的完整分子结构提取性能对比"
    write_csv(output / "table4.csv", COLUMNS, rows)
    write_json(output / "table4.json", rows)
    atomic_text(output / "table4.md", f"# {title}\n\n" + markdown_table(COLUMNS, rows))
    atomic_text(output / "table4_reproduction_report.md",
                "# 表4阈值修正差异\n\n当前结果使用论文规定的0.90阈值。"
                "各列为本次结果减旧0.60阈值表4。\n\n"
                + markdown_table(tuple(comparison[0]), comparison))
    print(f"已汇总 9 组结果: {output / 'table4.md'}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description="论文表4：页面级完整分子结构提取评测")
    commands = parser.add_subparsers(dest="command", required=True)
    audit_parser = commands.add_parser("audit", help="只计算页面数和 GT 总数")
    audit_parser.add_argument("--dataset", choices=DATASETS, help="省略时审计三个数据集")
    run_parser = commands.add_parser("run", help="运行一个数据集和一种方法，自动续跑")
    run_parser.add_argument("--dataset", choices=DATASETS, required=True)
    run_parser.add_argument("--method", choices=METHODS, required=True)
    run_parser.add_argument("--limit", type=int, default=0)
    run_parser.add_argument("--offset", type=int, default=0)
    run_parser.add_argument("--yode-repo", type=Path, default=DEFAULT_YODE)
    aggregate_parser = commands.add_parser("aggregate", help="从九组完整结果生成表4")
    for command in (audit_parser, run_parser):
        command.add_argument("--data-root", type=Path, help="自建集根目录（含 images、json、masks）")
        command.add_argument("--patcid-root", type=Path, help="PatCID-RePage restored 目录")
    for command in (run_parser, aggregate_parser):
        command.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)

    args = parser.parse_args(argv)
    if args.command == "run" and (args.offset < 0 or args.limit < 0):
        parser.error("--offset 和 --limit 必须大于等于 0。")
    if args.command == "run" and args.method == "yode" and "utils" in sys.modules:
        # python -m utils.experiments.Coverage 会预载本项目的 utils。
        # 旧 YOLOv5 checkpoint 需要自己的同名包，改由直接脚本入口启动独立进程。
        import subprocess

        entry = PROJECT_ROOT / "utils" / "experiments" / "Coverage.py"
        return subprocess.call([sys.executable, str(entry), *(sys.argv[1:] if argv is None else argv)])
    try:
        return {"audit": audit, "run": run, "aggregate": aggregate}[args.command](args)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"表4: {exc}\n")
