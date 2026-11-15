"""在 Windows 上读取含中文路径的 PNG。"""

from pathlib import Path

import cv2
import numpy as np


def read_image(path: Path, flags: int) -> np.ndarray:
    data = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    if not data.size:
        raise ValueError(f"空图像文件: {path}")
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"无法解码图像: {path}")
    return image
