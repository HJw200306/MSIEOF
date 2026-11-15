"""论文表4中三个页面评测集 × 三种方法的 PyCharm 总入口。

在 PyCharm 中直接运行本文件且不填写 Parameters 时，会依次评测：
1. PJChemPage-Test（450页）
2. PatCID-RePage（449页）
3. PJChemPageToSMILES（50页）

每个数据集都会运行 MSIEOF、DECIMER-Segmentation 和
YoDe-Segmentation，共 9 组正式评测。三种方法的依赖互相冲突，
因此本入口会自动调用各自的 Conda 环境，全部完成后生成表4汇总文件。

评测采用论文规定的0.90完整提取阈值。九组任务使用彼此独立的 checkpoint，
中断后再次运行会从各自的断点续跑，已完成的页面不会重新执行模型推理。
"""

from __future__ import annotations

import sys
import subprocess
from pathlib import Path

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_eval.eval_coverage.runner import (
    COLUMNS,
    DATASETS,
    DEFAULT_OUTPUT,
    METHODS,
    main,
    summarize,
)

METHOD_PYTHONS = {
    "msieof": Path(r"D:\miniconda3\envs\msirost\python.exe"),
    "decimer": Path(r"D:\miniconda3\envs\base_decimer\python.exe"),
    "yode": Path(r"D:\miniconda3\envs\YoDe_Seg\python.exe"),
}
DATA_ROOT = r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal"
PATCID_ROOT = r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored"

DATASET_ARGUMENTS = {
    "pjchem-test": ["--data-root", DATA_ROOT],
    "patcid-repage": ["--patcid-root", PATCID_ROOT],
    "pjchem-page-to-smiles": ["--data-root", DATA_ROOT],
}
DEFAULT_JOBS = [
    ["run", "--dataset", dataset, "--method", method, *DATASET_ARGUMENTS[dataset]]
    for dataset in DATASETS
    for method in METHODS
]


def run_default_jobs() -> int:
    """在独立环境中依次运行九组评测，并生成完整表4。"""
    missing = [f"{METHODS[method]}: {python}" for method, python in METHOD_PYTHONS.items()
               if not python.is_file()]
    if missing:
        print("找不到以下方法的 Python 解释器：")
        print("\n".join(f"- {item}" for item in missing))
        return 1

    total = len(DEFAULT_JOBS)
    failures = []
    for index, arguments in enumerate(DEFAULT_JOBS, start=1):
        dataset = arguments[arguments.index("--dataset") + 1]
        method = arguments[arguments.index("--method") + 1]
        print(f"\n{'=' * 72}")
        print(f"[{index}/{total}] 开始评测 {DATASETS[dataset].name} × {METHODS[method]}")
        print(f"{'=' * 72}", flush=True)
        command = [str(METHOD_PYTHONS[method]), str(Path(__file__).resolve()), *arguments]
        return_code = subprocess.call(command, cwd=PROJECT_ROOT)
        if return_code != 0:
            failures.append(f"{DATASETS[dataset].name} × {METHODS[method]}")
            print(f"该组评测未正常完成，将继续尝试其余任务。", flush=True)

    if failures:
        print("\n以下评测未正常完成，可修复问题后直接重新运行以从断点续跑：")
        print("\n".join(f"- {item}" for item in failures))
        return 1

    if main(["aggregate"]) != 0:
        return 1

    print("\n全部 9 组评测结果：")
    print("\t".join(COLUMNS))
    for arguments in DEFAULT_JOBS:
        dataset = arguments[arguments.index("--dataset") + 1]
        method = arguments[arguments.index("--method") + 1]
        checkpoint_dir = DEFAULT_OUTPUT / dataset / method
        row, _ = summarize(checkpoint_dir, dataset, method)
        print("\t".join(str(row[column]) for column in COLUMNS))
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if arguments:
        raise SystemExit(main(arguments))
    raise SystemExit(run_default_jobs())
