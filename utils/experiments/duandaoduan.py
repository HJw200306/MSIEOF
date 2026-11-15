"""一键运行并汇总论文表5的四种端到端方法和人工裁剪识别上限。

这是 PyCharm 的总入口。不同方法依赖互相冲突，因此本程序用独立解释器依次
运行 DECIMER、YoDe 和 MolScribe/MSIEOF；所有评分最终都复用
``ablation_v2.legacy_compare``，并强制校验 50 页 / 562 GT。所有计算阶段
默认使用断点续跑：复用已完成页面，并从第一个未完成页面继续。
显式使用 ``--overwrite`` 时才会清理所选阶段的历史结果并完整重算。
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.evaluation.table5_common import read_jsonl, validate_dataset, write_json

DEFAULT_DATASET = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal")
DEFAULT_EXTERNAL = Path.home() / "Desktop" / "seg_molscribe"
RUN_DATE = datetime.now().strftime("%Y%m%d")
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / f"table5_end2end_{RUN_DATE}"


def conda_root() -> Path | None:
    executable = Path(sys.executable).resolve()
    for parent in (executable.parent, *executable.parents):
        if (parent / "conda.exe").is_file() or (parent / "Scripts" / "conda.exe").is_file():
            return parent
    fallback = Path(r"D:\miniconda3")
    return fallback if fallback.is_dir() else None


def discover_python(kind: str) -> Path | None:
    root = conda_root()
    candidates = []
    if kind == "msirost":
        if root:
            candidates.extend(root / "envs" / name / "python.exe"
                              for name in ("msirost", "MSIROST"))
        candidates.append(Path(sys.executable))
    elif kind == "yode" and root:
        candidates.extend(root / "envs" / name / "python.exe"
                          for name in ("YoDe_Seg", "yo_de_seg", "yode_seg"))
    elif kind == "decimer" and root:
        candidates.extend((root / "envs" / "base_decimer" / "python.exe", root / "python.exe"))
    return next((x.resolve() for x in candidates if x.is_file()), None)


def run(command, label: str) -> None:
    print(f"\n===== {label} =====", flush=True)
    print(subprocess.list2cmdline([str(x) for x in command]), flush=True)
    subprocess.run([str(x) for x in command], cwd=PROJECT_ROOT, check=True)


def missing_modules(python: Path, modules) -> list[str]:
    """不导入大型框架，只检查指定解释器是否能找到模块。"""
    code = ("import importlib.util,json; mods=" + repr(list(modules)) +
            "; print(json.dumps([m for m in mods if importlib.util.find_spec(m) is None]))")
    try:
        completed = subprocess.run([str(python), "-c", code], check=True,
                                   capture_output=True, text=True)
        return json.loads(completed.stdout.strip().splitlines()[-1])
    except (subprocess.SubprocessError, json.JSONDecodeError, IndexError) as exc:
        return [f"解释器检查失败: {exc}"]


def require(path: Path, description: str, parser) -> None:
    if not path.exists():
        parser.error(f"缺少{description}: {path}")


def remove_output(path: Path, parser) -> None:
    """删除明确的输出目标，并防止误删项目或磁盘根目录。"""
    target = path.resolve()
    protected = {PROJECT_ROOT.resolve(), PROJECT_ROOT.parent.resolve(), Path(target.anchor)}
    if target in protected:
        parser.error(f"拒绝清理不安全的输出路径: {target}")
    if target.is_dir():
        shutil.rmtree(target)
    elif target.exists():
        target.unlink()


def prepare_overwrite(args, parser) -> None:
    """只清理本次要重算的阶段及其直接下游结果。"""
    if args.stage == "all":
        remove_output(args.output, parser)
        return
    folders = {
        "decimer": ("decimer_shared", "decimer", "decimer_seg_molscribe"),
        "yode": ("yode", "yode_seg_molscribe"),
        "manual": ("manual_crops", "manual_crop_molscribe"),
        "score": ("decimer", "decimer_seg_molscribe", "yode_seg_molscribe",
                  "manual_crop_molscribe"),
        "msieof": ("msieof",),
    }
    for folder in folders.get(args.stage, ()):
        remove_output(args.output / folder, parser)
    if args.stage == "table":
        for filename in ("table5.json", "table5.md"):
            remove_output(args.output / filename, parser)


def read_summary(path: Path, method: str) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    # ablation_v2 和表5 worker 的字段相同，但都保留兼容兜底。
    return {
        "method": method,
        "dataset": "PJChemPageToSMILES",
        "GT": data.get("n_ground_truth_total", data.get("n_gt")),
        "CEM": 100.0 * data["CEM"],
        "GEM": 100.0 * data["GEM"],
        "source": str(path.resolve()),
    }


def export_table(rows, output: Path) -> None:
    write_json(output / "table5.json", rows)
    lines = [
        "# 表5 不同方法在 PJChemPageToSMILES 测试集上的端到端转换性能",
        "", "| 方法 | 数据集 | GT | CEM(%) | GEM(%) |", "|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(f"| {row['method']} | {row['dataset']} | {row['GT']} | "
                     f"{row['CEM']:.3f} | {row['GEM']:.3f} |")
    (output / "table5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="表5五种比较流程总程序（PyCharm入口）")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gt", type=Path, help="默认使用 dataset-root/PageToSMILES_approved.jsonl")
    parser.add_argument("--msieof-gt", type=Path,
                        help="MSIEOF真值；默认与表5其他方法使用同一个50页/562 GT文件")
    parser.add_argument("--decimer-python", type=Path)
    parser.add_argument("--yode-python", type=Path)
    parser.add_argument("--msirost-python", type=Path)
    parser.add_argument(
        "--stage",
        choices=("all", "decimer", "yode", "manual", "score", "msieof", "table"),
        default="all",
        help="运行阶段；默认按断点续跑策略运行全部阶段并生成表5",
    )
    parser.add_argument("--max-pages", type=int, help="冒烟测试；正式表5不要设置")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--manual-crop-padding", type=int, default=20,
                        help="人工标注紧包围框四周的裁剪扩边像素，默认20")
    parser.add_argument("--manual-mask-margin", type=int, default=3,
                        help="人工多边形掩膜向外保留像素，默认3")
    policy = parser.add_mutually_exclusive_group()
    policy.add_argument(
        "--resume", dest="run_policy", action="store_const", const="resume",
        help="断点续跑：复用已完成页面（默认）")
    policy.add_argument(
        "--overwrite", dest="run_policy", action="store_const",
        const="overwrite", help="覆盖重跑：清理所选阶段的旧结果后完整重算")
    parser.set_defaults(run_policy="resume")
    args = parser.parse_args()

    gt = args.gt or args.dataset_root / "PageToSMILES_approved.jsonl"
    msieof_gt = args.msieof_gt or gt
    images = args.dataset_root / "images" / "PageToSMILES_approved"
    annotations = args.dataset_root / "json" / "PageToSMILES_approved"
    yode_repo = args.external_root / "YoDe-Segmentation"
    require(gt, "表5真值", parser)
    require(images, "表5页面目录", parser)
    if args.stage in ("all", "manual"):
        require(annotations, "人工标注JSON目录", parser)
    rows = read_jsonl(gt)
    if args.max_pages:
        rows = rows[:args.max_pages]
    validate_dataset(rows, allow_subset=bool(args.max_pages))
    print(f"数据检查通过: {len(rows)} 页 / {sum(len(x['molecules']) for x in rows)} GT")
    if args.run_policy == "overwrite":
        print(f"运行策略: 覆盖重跑 ({args.stage})", flush=True)
    else:
        print("运行策略: 断点续跑（默认）", flush=True)

    decimer_python = args.decimer_python or discover_python("decimer")
    yode_python = args.yode_python or discover_python("yode")
    msirost_python = args.msirost_python or discover_python("msirost")
    max_pages = ["--max-pages", str(args.max_pages)] if args.max_pages else []
    resume = ["--resume"] if args.run_policy == "resume" else []
    decimer_manifest = args.output / "decimer_shared" / "instances.jsonl"
    yode_manifest = args.output / "yode" / "instances.jsonl"
    manual_manifest = args.output / "manual_crops" / "instances.jsonl"
    if args.stage == "all":
        preflight_errors = []
        if decimer_python is None:
            preflight_errors.append("找不到 DECIMER Python")
        else:
            missing = missing_modules(decimer_python, ("tensorflow", "decimer_segmentation", "DECIMER"))
            if missing:
                preflight_errors.append(f"DECIMER环境缺少 {missing}")
        if yode_python is None:
            preflight_errors.append("找不到 YoDe Python")
        else:
            missing = missing_modules(yode_python, ("torch", "torchvision", "cv2"))
            if missing:
                preflight_errors.append(f"YoDe环境缺少 {missing}")
        if msirost_python is None:
            preflight_errors.append("找不到 msirost Python")
        else:
            missing = missing_modules(msirost_python, ("torch", "molscribe", "rdkit", "cv2"))
            if missing:
                preflight_errors.append(f"msirost环境缺少 {missing}")
        if preflight_errors:
            parser.error("环境预检未通过:\n- " + "\n- ".join(preflight_errors) +
                         f"\n环境文件目录: {args.external_root / 'environments'}")
        if args.run_policy == "overwrite":
            prepare_overwrite(args, parser)

    if args.stage in ("all", "decimer"):
        if decimer_python is None:
            env = args.external_root / "environments" / "base_decimer.yml"
            parser.error(f"找不到 DECIMER Python。请用 {env} 创建环境，或设置 --decimer-python")
        missing = missing_modules(decimer_python, ("tensorflow", "decimer_segmentation", "DECIMER"))
        if missing:
            env = args.external_root / "environments" / "base_decimer.yml"
            parser.error(f"DECIMER环境缺少 {missing}。请用 {env} 创建环境，或设置 --decimer-python")
        if args.run_policy == "overwrite" and args.stage == "decimer":
            prepare_overwrite(args, parser)
        run([decimer_python, "-m", "utils.evaluation.table5_decimer_worker",
             "--gt", gt, "--images", images, "--output", decimer_manifest,
             "--with-ocsr", *max_pages, *resume], "DECIMER-Seg + DECIMER OCSR")

    if args.stage in ("all", "yode"):
        require(yode_repo, "YoDe-Segmentation仓库", parser)
        if yode_python is None:
            env = args.external_root / "environments" / "yo_de_seg.yml"
            parser.error(f"找不到 YoDe Python。请用 {env} 创建环境，或设置 --yode-python")
        missing = missing_modules(yode_python, ("torch", "torchvision", "cv2"))
        if missing:
            env = args.external_root / "environments" / "yo_de_seg.yml"
            parser.error(f"YoDe环境缺少 {missing}。请用 {env} 创建环境，或设置 --yode-python")
        if args.run_policy == "overwrite" and args.stage == "yode":
            prepare_overwrite(args, parser)
        run([yode_python, PROJECT_ROOT / "utils" / "evaluation" / "table5_yode_worker.py",
             "--gt", gt, "--images", images, "--repo", yode_repo,
              "--output", yode_manifest, *max_pages, *resume], "YoDe-Segmentation")

    if args.stage in ("all", "manual"):
        if msirost_python is None:
            parser.error("找不到 msirost Python，请设置 --msirost-python")
        missing = missing_modules(msirost_python, ("PIL",))
        if missing:
            parser.error(f"人工裁剪环境缺少 {missing}，请设置 --msirost-python")
        if args.run_policy == "overwrite" and args.stage == "manual":
            prepare_overwrite(args, parser)
        run([msirost_python, "-m", "utils.evaluation.table5_manual_crop_worker",
             "--gt", gt, "--images", images, "--annotations", annotations,
             "--output", manual_manifest,
             "--padding", str(args.manual_crop_padding),
              "--mask-margin", str(args.manual_mask_margin), *max_pages, *resume],
            "人工标注裁剪图")
        if args.stage == "manual":
            run([msirost_python, "-m", "utils.evaluation.table5_score_worker",
                 "--gt", gt, "--instances", manual_manifest,
                 "--output-dir", args.output / "manual_crop_molscribe",
                 "--method", "人工标注裁剪图 + MolScribe",
                 "--recognizer", "molscribe", "--batch-size", str(args.batch_size),
                  *max_pages, *resume], "评分: 人工标注裁剪图 + MolScribe")

    if args.stage in ("all", "score"):
        if msirost_python is None:
            env = args.external_root / "environments" / "msirost.yml"
            parser.error(f"找不到 msirost Python。请用 {env} 创建环境，或设置 --msirost-python")
        missing = missing_modules(msirost_python, ("torch", "molscribe", "rdkit", "cv2"))
        if missing:
            env = args.external_root / "environments" / "msirost.yml"
            parser.error(f"msirost环境缺少 {missing}。请用 {env} 创建环境，或设置 --msirost-python")
        if args.run_policy == "overwrite" and args.stage == "score":
            prepare_overwrite(args, parser)
        for manifest, method, recognizer, folder in (
            (decimer_manifest, "DECIMER", "existing", "decimer"),
            (decimer_manifest, "DECIMER-Segmentation + MolScribe", "molscribe", "decimer_seg_molscribe"),
            (yode_manifest, "YoDe-Segmentation + MolScribe", "molscribe", "yode_seg_molscribe"),
            (manual_manifest, "人工标注裁剪图 + MolScribe", "molscribe", "manual_crop_molscribe"),
        ):
            require(manifest, f"{method}实例清单", parser)
            run([msirost_python, "-m", "utils.evaluation.table5_score_worker",
                 "--gt", gt, "--instances", manifest,
                 "--output-dir", args.output / folder, "--method", method,
                 "--recognizer", recognizer, "--batch-size", str(args.batch_size),
                  *max_pages, *resume], f"评分: {method}")

    if args.stage in ("all", "msieof"):
        if msirost_python is None:
            parser.error("找不到 msirost Python，请设置 --msirost-python")
        missing = missing_modules(msirost_python, ("torch", "molscribe", "rdkit", "cv2"))
        if missing:
            parser.error(f"msirost环境缺少 {missing}，请设置 --msirost-python")
        require(msieof_gt, "MSIEOF真值", parser)
        msieof_rows = read_jsonl(msieof_gt)
        validate_dataset(msieof_rows, allow_subset=bool(args.max_pages))
        if args.run_policy == "overwrite" and args.stage == "msieof":
            prepare_overwrite(args, parser)
        msieof_out = args.output / "msieof"
        # ablation_v2 已支持逐页结果恢复；覆盖模式则已在上方清理该目录。
        run([msirost_python, "-m", "utils.experiments.ablation_v2",
             "--gt", msieof_gt, "--output", msieof_out, "--configs", "full",
              *max_pages, *resume], "MSIEOF（双模型并集）")

    if args.stage in ("all", "table"):
        specifications = (
            ("DECIMER", args.output / "decimer" / "summary.json"),
            ("YoDe-Segmentation + MolScribe", args.output / "yode_seg_molscribe" / "summary.json"),
            ("DECIMER-Segmentation + MolScribe", args.output / "decimer_seg_molscribe" / "summary.json"),
            ("人工标注裁剪图 + MolScribe", args.output / "manual_crop_molscribe" / "summary.json"),
            ("MSIEOF", args.output / "msieof" / "full" / "summary.json"),
        )
        table_rows = []
        for method, path in specifications:
            require(path, f"{method}汇总结果", parser)
            table_rows.append(read_summary(path, method))
        if args.run_policy == "overwrite" and args.stage == "table":
            prepare_overwrite(args, parser)
        export_table(table_rows, args.output)
        print(f"\n表5已生成: {args.output / 'table5.md'}")


if __name__ == "__main__":
    main()
