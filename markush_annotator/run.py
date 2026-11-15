import os
import sys
import webbrowser
import threading
import time
import argparse
import importlib.util
from pathlib import Path

# 确保当前目录与项目根目录在 sys.path 中
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(APP_DIR))

# 可选的外部 MarkushReco 工程。未配置时，标注器的基础功能仍可正常使用。
MARKUSH_RECO_ROOT = os.environ.get("MARKUSH_RECO_ROOT", "")
if MARKUSH_RECO_ROOT and Path(MARKUSH_RECO_ROOT).is_dir():
    sys.path.insert(0, str(Path(MARKUSH_RECO_ROOT).resolve()))


# ============ 配置区域 - PyCharm直接运行时修改这里 ============
# 默认使用项目内的数据；也可通过 --data-dir 或 ANNOTATOR_DATA_DIR 覆盖。
DATA_DIR = APP_DIR / "data" / "seg_page2smiles"

# 服务器配置
HOST = "127.0.0.1"
PORT = 8765
# ============================================================

URL = f"http://{HOST}:{PORT}"


def _configure_console() -> None:
    """Windows 控制台尽量使用 UTF-8；被重定向时也不要因此启动失败。"""
    if sys.platform != "win32":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure:
            reconfigure(encoding="utf-8")


def _check_runtime(data_dir: Path) -> list[str]:
    errors = []
    for module in ("fastapi", "uvicorn", "pydantic", "cv2", "numpy", "rdkit"):
        if importlib.util.find_spec(module) is None:
            errors.append(f"缺少 Python 依赖: {module}")
    if not data_dir.is_dir():
        errors.append(f"数据目录不存在: {data_dir}")
    elif not (data_dir / "manifest.jsonl").is_file():
        errors.append(f"缺少数据清单: {data_dir / 'manifest.jsonl'}")
    if not (APP_DIR / "frontend" / "index.html").is_file():
        errors.append(f"前端文件不存在: {APP_DIR / 'frontend' / 'index.html'}")
    return errors

def main():
    parser = argparse.ArgumentParser(description="Markush Visual Annotator")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser")
    parser.add_argument("--host", default=HOST, help="Host address")
    parser.add_argument("--port", type=int, default=PORT, help="Port number")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data directory (sets ANNOTATOR_DATA_DIR env var)")
    parser.add_argument("--check", action="store_true",
                        help="检查依赖和数据，不启动服务器")
    args, _ = parser.parse_known_args()

    _configure_console()

    # 设置自定义数据目录环境变量
    # 优先级: 命令行参数 > 脚本配置的DATA_DIR
    data_dir = Path(
        args.data_dir or os.environ.get("ANNOTATOR_DATA_DIR") or DATA_DIR
    ).expanduser().resolve()
    # 必须在导入 backend 之前设置环境变量。
    os.environ["ANNOTATOR_DATA_DIR"] = str(data_dir)
    print(f"[INFO] 使用数据目录: {data_dir}")

    errors = _check_runtime(data_dir)
    if errors:
        print("\n[ERROR] 运行环境检查失败:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        print("\n请使用 PyCharm 项目解释器安装 requirements-annotator.txt。", file=sys.stderr)
        return 2

    if args.check:
        # 导入应用可同时验证后端初始化和数据清单读取。
        from backend.app import app  # noqa: F401
        print("[OK] 依赖、数据目录和后端应用检查通过")
        return 0

    host = args.host
    port = args.port
    url = f"http://{host}:{port}"

    print("=" * 70)
    print("步骤2: Markush Visual Annotator - 图结构标注")
    print("=" * 70)
    print(f"  服务器地址: {url}")
    print(f"  数据目录: {data_dir}")
    print("=" * 70)
    print("\n标注说明:")
    print("  1. 在浏览器中打开上述URL")
    print("  2. 查看MolScribe预标注结果")
    print("  3. 精修原子位置和化学键")
    print("  4. 保存标注")
    print("  5. 完成所有标注后，关闭服务器")
    print("  6. 运行 export_seg_results.py 导出结果")
    print("=" * 70)
    
    if not args.no_browser:
        def open_browser_func():
            time.sleep(1.2)
            try:
                print(f"\n[INFO] Opening Markush Visual Annotator: {url}")
                webbrowser.open(url)
            except Exception:
                pass
        threading.Thread(target=open_browser_func, daemon=True).start()
    
    import uvicorn

    uvicorn.run(
        "backend.app:app",
        host=host,
        port=port,
        reload=not args.no_reload,
        reload_dirs=[str(APP_DIR)] if not args.no_reload else None,
        log_level="info"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
