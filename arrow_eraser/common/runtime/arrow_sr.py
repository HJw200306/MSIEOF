"""统一四分类箭头运行时的兼容入口。

生产实现只维护在 ``utils.pipeline.arrow_sr``，避免归档副本与主项目发生漂移。
"""

from utils.pipeline.arrow_sr import (  # noqa: F401
    CROP_DIR,
    HAT_MODEL_PATH,
    OUTPUT_DIR,
    UNIFIED_ARROW_MODEL_PATH,
    UNIFIED_CLASS_NAMES,
    ArrowSRProcessor,
    run,
)

__all__ = [
    "ArrowSRProcessor",
    "CROP_DIR",
    "HAT_MODEL_PATH",
    "OUTPUT_DIR",
    "UNIFIED_ARROW_MODEL_PATH",
    "UNIFIED_CLASS_NAMES",
    "run",
]


if __name__ == "__main__":
    run()
