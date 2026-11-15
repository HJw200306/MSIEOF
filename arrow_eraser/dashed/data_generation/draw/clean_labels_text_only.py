"""
将三分类 YOLO-OBB 数据集清洗为单类文本检测数据集。

输入 (由 draw/mol_obb_gen.py 产生):
    cls 0  atom_text
    cls 1  bond
    cls 2  other_text

输出:
    cls 0  text     (= atom_text ∪ other_text, bond 已丢弃)

行为:
  - 标签文件 (*.txt) 逐行重写: 丢掉 bond, 其余统一写为 0
  - 图片用硬链接到输出目录 (NTFS 同卷免拷贝零额外占用), 失败回退 shutil.copy2
  - 顺便写一份与新数据集匹配的 yolo yaml 配置

用法:
    python draw/clean_labels_text_only.py
    python draw/clean_labels_text_only.py --in-root dataset/yolo_mol_obb --out-root dataset/yolo_mol_obb_text
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 三分类 -> 单类映射: bond 直接丢, 其余统一为 0
CLASS_REMAP = {0: 0, 2: 0}  # cls 1 (bond) 不在表里 -> 被过滤
DROP_CLASSES = {1}


def remap_label_text(content):
    """逐行重写标签内容。
    每行格式: 'cls x1 y1 x2 y2 x3 y3 x4 y4' (空白分隔, 归一化坐标)。
    返回 (新内容, kept_count, dropped_count, malformed_count)。
    """
    kept, dropped, bad = 0, 0, 0
    out_lines = []
    for line in content.splitlines():
        s = line.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 9:  # OBB 至少 1 类 + 8 坐标
            bad += 1
            continue
        try:
            cls = int(parts[0])
        except ValueError:
            bad += 1
            continue
        if cls in DROP_CLASSES:
            dropped += 1
            continue
        new_cls = CLASS_REMAP.get(cls)
        if new_cls is None:
            # 未在映射表里的未知类: 保守丢弃, 避免污染
            dropped += 1
            continue
        parts[0] = str(new_cls)
        out_lines.append(' '.join(parts))
        kept += 1
    # 末尾不加换行, 与 mol_obb_gen.py 写出风格一致
    return '\n'.join(out_lines), kept, dropped, bad


def _link_or_copy(src, dst):
    """优先硬链接 (同卷下零拷贝), 跨卷/权限失败回退到 shutil.copy2。
    dst 已存在则跳过, 保证可重入。
    """
    if dst.exists():
        return 'skip'
    try:
        os.link(src, dst)
        return 'link'
    except OSError:
        shutil.copy2(src, dst)
        return 'copy'


def clean_split(in_root, out_root, split, link_images=True, image_exts=('.png', '.jpg', '.jpeg', '.bmp')):
    in_lbl = in_root / split / 'labels'
    in_img = in_root / split / 'images'
    out_lbl = out_root / split / 'labels'
    out_img = out_root / split / 'images'
    out_lbl.mkdir(parents=True, exist_ok=True)
    if link_images:
        out_img.mkdir(parents=True, exist_ok=True)

    if not in_lbl.exists():
        print(f"[{split}] skip: {in_lbl} not found")
        return None

    txt_files = sorted(in_lbl.glob('*.txt'))
    stats = {
        'files': 0,
        'empty_after_clean': 0,  # 清洗后没有任何 text 框的文件 (可能是只有 bond 的图)
        'kept_lines': 0,
        'dropped_lines': 0,
        'malformed_lines': 0,
        'img_link': 0,
        'img_copy': 0,
        'img_skip': 0,
        'img_missing': 0,
    }
    for txt in txt_files:
        content = txt.read_text(encoding='utf-8')
        new_content, kept, dropped, bad = remap_label_text(content)
        stats['files'] += 1
        stats['kept_lines'] += kept
        stats['dropped_lines'] += dropped
        stats['malformed_lines'] += bad
        if kept == 0:
            stats['empty_after_clean'] += 1
        (out_lbl / txt.name).write_text(new_content, encoding='utf-8')

        if link_images:
            stem = txt.stem
            linked = False
            for ext in image_exts:
                img_src = in_img / f"{stem}{ext}"
                if img_src.exists():
                    action = _link_or_copy(img_src, out_img / img_src.name)
                    stats[f'img_{action}'] += 1
                    linked = True
                    break
            if not linked:
                stats['img_missing'] += 1

    return stats


YAML_TEMPLATE = """# YOLO OBB 单类文本检测数据集
# 由 draw/clean_labels_text_only.py 从三分类数据集 ({src}) 清洗而来:
#   - cls 1 (bond) 已丢弃
#   - cls 0 (atom_text) 与 cls 2 (other_text) 合并为 cls 0 (text)
path: {path}
train: train/images
val: val/images
# test: test/images

nc: 1
names: ['text']
"""


def write_text_config(yaml_path, rel_path, src_name):
    yaml_path.write_text(YAML_TEMPLATE.format(path=rel_path, src=src_name), encoding='utf-8')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in-root', default=str(ROOT_DIR / 'dataset' / 'yolo_mol_obb'),
                    help='输入三分类数据集根目录')
    ap.add_argument('--out-root', default=str(ROOT_DIR / 'dataset' / 'yolo_mol_obb_text'),
                    help='输出单类数据集根目录')
    ap.add_argument('--splits', nargs='+', default=['train', 'val'],
                    help='要处理的 split 子目录, 默认 train/val')
    ap.add_argument('--no-images', action='store_true',
                    help='不在输出目录创建图片链接 (适用于自己处理图片路径的场景)')
    ap.add_argument('--no-yaml', action='store_true',
                    help='不写出新的 yaml 配置文件')
    args = ap.parse_args()

    in_root = Path(args.in_root).resolve()
    out_root = Path(args.out_root).resolve()
    if not in_root.exists():
        raise SystemExit(f"input root does not exist: {in_root}")
    if in_root == out_root:
        raise SystemExit("in-root 与 out-root 不能相同, 避免原地覆盖。请指定不同的 --out-root。")
    out_root.mkdir(parents=True, exist_ok=True)

    link_images = not args.no_images
    print(f"[in ] {in_root}")
    print(f"[out] {out_root}")
    print(f"[link images] {link_images}")

    grand = {}
    for split in args.splits:
        s = clean_split(in_root, out_root, split, link_images=link_images)
        if s is None:
            continue
        print(f"[{split}] files={s['files']} kept={s['kept_lines']} dropped={s['dropped_lines']} "
              f"empty_after={s['empty_after_clean']} bad={s['malformed_lines']} "
              f"| img link={s['img_link']} copy={s['img_copy']} skip={s['img_skip']} miss={s['img_missing']}")
        for k, v in s.items():
            grand[k] = grand.get(k, 0) + v

    if grand:
        print(f"[total] files={grand['files']} kept={grand['kept_lines']} dropped={grand['dropped_lines']} "
              f"empty_after={grand['empty_after_clean']} bad={grand['malformed_lines']}")

    if not args.no_yaml:
        # 与原 yaml 风格一致: 把 yaml 放在 dataset/ 目录, path 写成相对路径
        yaml_path = out_root.parent / f"{out_root.name}_config.yaml"
        try:
            rel_path = f"../{out_root.relative_to(yaml_path.parent.parent).as_posix()}"
        except ValueError:
            rel_path = str(out_root)
        write_text_config(yaml_path, rel_path, in_root.name)
        print(f"[yaml] {yaml_path}")


if __name__ == '__main__':
    main()
