import os
from PIL import Image
from split_tools.convert_to_img import convert_to_8bit


def resize_image(img, t_size, color=(255, 255, 255)):
    # 计算调整后的尺寸和偏移量
    width, height = img.size
    target_width, target_height = t_size
    ratio = min(target_width / width, target_height / height)
    new_width = int(width * ratio)
    new_height = int(height * ratio)
    x_offset = (target_width - new_width) // 2
    y_offset = (target_height - new_height) // 2

    # 调整图像大小并填充
    resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    new_image = Image.new("RGB", t_size, color)
    new_image.paste(resized_img, (x_offset, y_offset))

    return new_image


def resize_all_img(path_dir, output_dir, target_size=(512, 512), color=(0, 0, 0), to_8bit=False, endswith=".jpg"):
    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)
    filelist = os.listdir(path_dir)
    # progress_bar = tqdm(total=len(filelist), desc='Processing')
    # 遍历文件夹中的所有文件
    for filename in filelist:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 加载原始图像
            image = Image.open(os.path.join(path_dir, filename))

            # 调整图像大小并填充
            resized_image = resize_image(image, target_size, color)
            if to_8bit:
                resized_image = convert_to_8bit(resized_image)
            file_name = filename.replace('.jpg', '.png') if endswith == ".png" else filename.replace('.png', '.jpg')
            # 保存调整后的图像
            output_filename = os.path.join(output_dir, file_name)
            resized_image.save(output_filename)
        # 更新进度条
        # progress_bar.update(1)

    # progress_bar.close()


if __name__ == '__main__':
    resize_all_img("../result/test", "test", (512, 512), (255, 255, 255), endswith='.png')
