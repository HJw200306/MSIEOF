import os

import cv2
from PIL import Image
from tqdm import tqdm


def otsu_threshold(image_path, save_path):
    # 读取图像
    image = cv2.imread(image_path, 0)

    # 应用Otsu's二值化
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    file_name = os.path.basename(image_path)
    cv2.imwrite(f"{save_path}B{file_name}", binary_image)


if __name__ == "__main__":
    filelist = os.listdir("../result")
    progress_bar = tqdm(total=len(filelist), desc='Processing')
    # 遍历文件夹中的所有文件
    for filename in filelist:
        if filename.endswith(".jpg"):
            otsu_threshold("../result/" + filename, "../result/")
        # 更新进度条
        progress_bar.update(1)
    progress_bar.close()
