from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ARROW_ERASER_ROOT = Path(__file__).resolve().parent
ANNOTATOR = ARROW_ERASER_ROOT / "solid" / "annotation" / "seg_annotator" / "run.py"
ANNOTATION_DATA = ARROW_ERASER_ROOT / "solid" / "annotation" / "annotation_v1"
PREFERRED_PYTHON = Path(r"D:\miniconda3\envs\msirost\python.exe")


def main() -> int:
    python_executable = PREFERRED_PYTHON if PREFERRED_PYTHON.is_file() else Path(sys.executable)
    command = [
        str(python_executable),
        str(ANNOTATOR),
        "--data-dir",
        str(ANNOTATION_DATA),
        *sys.argv[1:],
    ]
    print(f"Python: {python_executable}")
    print(f"标注数据: {ANNOTATION_DATA}")
    print("功能: 实线箭头 / 虚线箭头 / 化学键；支持总体、需复查、已审核和虚线样本筛选")
    try:
        return subprocess.call(command, cwd=ARROW_ERASER_ROOT)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
