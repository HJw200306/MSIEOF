import os
from PIL import Image
import os.path as osp
from tqdm import tqdm

# 定义颜色列表
color_list = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
              (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0),
              (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0),
              (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12), (255, 0, 0), ]


def find_png_files(directory):
    png_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(os.path.join(root, file))
    return png_files


def get_had_files(directory):
    had_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            had_files.append(file.lower())
    return had_files


def split_img(path_a, path_b, save_path, del_red=True):
    # 打开图像A和B并转换为RGB模式
    image_a = Image.open(path_a).convert('RGB')
    image_b = Image.open(path_b).convert('RGB')

    pixels_a = image_a.load()
    pixels_b = image_b.load()
    width, height = image_b.size
    for y in range(height):
        for x in range(width):
            if del_red and pixels_a[x, y] == (255, 0, 0):
                pixels_a[x, y] = (0, 0, 0)
            if pixels_b[x, y] == (0, 0, 0):  # 背景
                continue
            else:
                pixels_a[x, y] = pixels_b[x, y] if pixels_b[x, y] != (128, 128, 0) else (0, 0, 0)
            # if pixels_b[x, y] == (0, 128, 0):
            #     pixels_a[x, y] = (0, 128, 0)  # 调整为绿色
    image = convert_to_8bit(image_a)
    image.save(save_path)


def copy_img(save_path):
    progress_bar = tqdm(total=len(range(1, 95)), desc='Processing')
    for i in range(1, 95):
        try:
            image = Image.open(save_path + '/' + str(i) + '.png')

            for j in range(94, 377, 94):
                image.save(save_path + '/' + str(i + j) + '.png')

            # 左右翻转
            image_flipped = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_flipped.save(save_path + '/' + str(i + 470) + '.png')

            # 上下翻转
            image_flipped = image.transpose(Image.FLIP_TOP_BOTTOM)
            image_flipped.save(save_path + '/' + str(i + 564) + '.png')

            for j in range(847, 1036, 188):
                image.save(save_path + '/' + str(i + j) + '.png')
        except FileNotFoundError:
            pass

        # 更新进度条
        progress_bar.update(1)

    progress_bar.close()


def convert_to_8bit(image):
    # 创建调色板图像
    palette_image = Image.new('P', (1, 1))
    palette_image.putpalette(sum(color_list, ()))

    # 转换为8位颜色图像，使用自定义调色板
    image = image.quantize(colors=len(color_list), palette=palette_image)

    return image


def add_patch(mark_dir, patch_path, save_dir, del_res=True):
    png_files = find_png_files(patch_path)
    progress_bar = tqdm(total=len(png_files), desc='Processing')
    for png_file in png_files:
        f_name = osp.basename(png_file)
        split_img(mark_dir + f_name, png_file, save_dir + f_name, del_res)

        # 更新进度条
        progress_bar.update(1)

    # 关闭进度条
    progress_bar.close()


def delete_red(mark_path, save_dir):
    had_files = get_had_files(save_dir)
    png_files = find_png_files(mark_path)
    progress_bar = tqdm(total=len(png_files), desc='Processing')
    for png_file in png_files:
        # 更新进度条
        progress_bar.update(1)
        f_name = osp.basename(png_file)
        if f_name in had_files:
            continue
        image = Image.open(png_file).convert('RGB')
        width, height = image.size
        for y in range(height):
            for x in range(width):
                if image.getpixel((x, y)) == (255, 0, 0):
                    image.putpixel((x, y), (0, 0, 0))
        image.save(osp.join(save_dir, f_name))

    # 关闭进度条
    progress_bar.close()


if __name__ == '__main__':
    add_patch("pre_mark/", "../datasets/patch_class/", "tmp/")  # 添加补丁
    copy_img("tmp/")  # 数据增强的数据进行复制
    delete_red("pre_mark/", "tmp/")  # 剩余标签加入目录
    add_patch("../datasets/SegmentationClass/", "tmp/", "res_mark/", False)  # 获得最后的标签
