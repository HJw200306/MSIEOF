"""
下载 MolMole Patent300 数据集的图像文件
"""
from huggingface_hub import snapshot_download
from pathlib import Path
from utils.base.paths import DATA_DIR

if __name__ == "__main__":
    # 目标目录
    local_dir = DATA_DIR / "MolMole_Patent300"

    print("开始下载 MolMole_Patent300 数据集...")
    print(f"目标目录: {local_dir}")
    print("\n这可能需要几分钟时间，请耐心等待...\n")

    # 下载整个数据集
    snapshot_download(
        repo_id="doxa-friend/MolMole_Patent300",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=["01_MolMole_Patent300/original/images_300/*.png"]
    )

    print("\n✓ 下载完成!")

    # 验证文件
    images_dir = local_dir / "01_MolMole_Patent300" / "original" / "images_300"
    png_files = list(images_dir.glob("*.png"))

    # 检查文件大小
    valid_files = [f for f in png_files if f.stat().st_size > 0]

    print(f"\n验证结果:")
    print(f"  PNG 文件总数: {len(png_files)}")
    print(f"  有效文件数: {len(valid_files)}")
    print(f"  空文件数: {len(png_files) - len(valid_files)}")

    if len(valid_files) == len(png_files):
        print("\n✓ 所有文件下载成功!")
    else:
        print("\n⚠ 警告: 仍有空文件，可能需要重新下载")
