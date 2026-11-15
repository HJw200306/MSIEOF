"""兼容入口；实际实现统一维护在 :mod:`utils.pipeline.arrow_sr`。"""

from pathlib import Path

from utils.pipeline.arrow_sr import CROP_DIR, OUTPUT_DIR, ArrowSRProcessor, run

__all__ = ["ArrowSRProcessor", "run"]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="对 post_segment 输出按高低清分流执行超分与统一四分类箭头去除"
    )
    parser.add_argument("--input", default=str(CROP_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--threshold", type=float, default=0.10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualize", action="store_true")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    run(
        crop_dir=Path(args.input),
        output_dir=Path(args.output),
        low_threshold=args.threshold,
        device=args.device,
        visualize=args.visualize,
    )
