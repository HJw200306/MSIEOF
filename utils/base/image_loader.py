"""用于加载图像 (从 D:\\workspace\\pythonProject\\R_Groups_Segment\\utils\\image_loader.py 复制)。"""
from io import BytesIO
from pathlib import Path
from typing import Any, Union

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

InputType = Union[str, np.ndarray, bytes, Path, Image.Image]
# isinstance 接受的具体类型元组 (Python 3.9 不能把 Union 直接喂给 isinstance)。
_RUNTIME_INPUT_TYPES = (str, np.ndarray, bytes, Path, Image.Image)


class ImageLoader:
    def __init__(self):
        pass

    def load_cv_img(self, img: InputType) -> np.ndarray:
        if not isinstance(img, _RUNTIME_INPUT_TYPES):
            raise LoadImageError(
                f"The img type {type(img)} does not in {_RUNTIME_INPUT_TYPES}"
            )

        origin_img_type = type(img)
        img = self._load_img(img)
        img = self.convert_img(img, origin_img_type)
        return img

    def load_image(self, img: InputType) -> Image.Image:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = Image.open(img).convert('RGB')
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = Image.open(BytesIO(img))
            return img

        if isinstance(img, np.ndarray):
            return Image.fromarray(img)

        if isinstance(img, Image.Image):
            return img

        raise LoadImageError(f"{type(img)} is not supported!")

    def _load_img(self, img: InputType) -> np.ndarray:
        if isinstance(img, (str, Path)):
            self.verify_exist(img)
            try:
                img = self.img_to_ndarray(Image.open(img))
            except UnidentifiedImageError as e:
                raise LoadImageError(f"cannot identify image file {img}") from e
            return img

        if isinstance(img, bytes):
            img = self.img_to_ndarray(Image.open(BytesIO(img)))
            return img

        if isinstance(img, np.ndarray):
            return img

        if isinstance(img, Image.Image):
            return self.img_to_ndarray(img)

        raise LoadImageError(f"{type(img)} is not supported!")

    def img_to_ndarray(self, img: Image.Image) -> np.ndarray:
        if img.mode == "1":
            img = img.convert("L")
            return np.array(img)
        return np.array(img)

    def convert_img(self, img: np.ndarray, origin_img_type: Any) -> np.ndarray:
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if img.ndim == 3:
            channel = img.shape[2]
            if channel == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

            if channel == 2:
                return self.cvt_two_to_three(img)

            if channel == 3:
                if issubclass(origin_img_type, (str, Path, bytes, Image.Image)):
                    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                return img

            if channel == 4:
                return self.cvt_four_to_three(img)

            raise LoadImageError(
                f"The channel({channel}) of the img is not in [1, 2, 3, 4]"
            )

        raise LoadImageError(f"The ndim({img.ndim}) of the img is not in [2, 3]")

    @staticmethod
    def cvt_two_to_three(img: np.ndarray) -> np.ndarray:
        """gray + alpha → BGR"""
        img_gray = img[..., 0]
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        img_alpha = img[..., 1]
        not_a = cv2.bitwise_not(img_alpha)
        not_a = cv2.cvtColor(not_a, cv2.COLOR_GRAY2BGR)

        new_img = cv2.bitwise_and(img_bgr, img_bgr, mask=img_alpha)
        new_img = cv2.add(new_img, not_a)
        return new_img

    @staticmethod
    def cvt_four_to_three(img: np.ndarray) -> np.ndarray:
        """RGBA → RGB with white background"""
        r, g, b, a = cv2.split(img)
        a = a.astype(np.float32) / 255.0
        white_bg = np.ones_like(r) * 255
        r = (r * a + white_bg * (1 - a)).astype(np.uint8)
        g = (g * a + white_bg * (1 - a)).astype(np.uint8)
        b = (b * a + white_bg * (1 - a)).astype(np.uint8)
        new_img = cv2.merge((b, g, r))
        return new_img

    @staticmethod
    def verify_exist(file_path: Union[str, Path]):
        if not Path(file_path).exists():
            raise LoadImageError(f"{file_path} does not exist.")


class LoadImageError(Exception):
    pass
