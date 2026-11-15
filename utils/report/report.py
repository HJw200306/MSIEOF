"""最终结果报告生成器：SMILES + 期刊名/印刷页码 + 分子图的网格卡片图。

用法
----
python -m utils.report.report \
    --pdf input/Prof. Wang-Natural products chemistry study.pdf \
    --smiles crop_dataset_smiles/results.csv \
    --crop crop_dataset_processed/ \
    --out vis/report/ \
    --cards-per-page 12
"""

import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image

matplotlib.use("Agg")

from utils.base.paths import PROJECT_ROOT


def parse_results_csv(csv_path: Path) -> List[dict]:
    """读 results.csv，从 name 反推 page_idx 和 mol_idx。

    Returns: [{"name": ..., "smiles": ..., "confidence": ...,
               "page_idx": 4, "mol_idx": 0, ...}, ...]
    """
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            # 从 "..._p004_mol000.png" 提取 page=4, mol=0
            match = re.search(r"_p(\d+)_mol(\d+)\.png$", name)
            if match:
                page_idx = int(match.group(1))
                mol_idx = int(match.group(2))
            else:
                page_idx = mol_idx = -1
            records.append({
                "name": name,
                "smiles": row["smiles"],
                "confidence": row.get("confidence", ""),
                "n_atoms": row.get("n_atoms", ""),
                "n_bonds": row.get("n_bonds", ""),
                "status": row.get("status", ""),
                "page_idx": page_idx,
                "mol_idx": mol_idx,
            })
    return records


def build_page_meta(pdf_path: Path) -> Dict[int, dict]:
    """从 PDF 页眉提取期刊名 + 卷期年 + 印刷页码。

    Returns: {page_idx: {"journal": "Journal of Advanced Research",
                         "volume": "24", "year": "2020",
                         "pages_range": "69–78", "printed_page": "70"}, ...}
    匹配失败时返回 {"journal": <pdf_stem>, "printed_page": f"p{page_idx}"}。
    """
    doc = fitz.open(str(pdf_path))
    page_meta: Dict[int, dict] = {}

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        # 获取页面文字块（dict 格式）
        text_dict = page.get_text("dict")
        lines = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:  # 非文字块
                continue
            for line in block.get("lines", []):
                y = round(line["bbox"][1])
                x = round(line["bbox"][0])
                txt = "".join(s["text"] for s in line["spans"]).strip()
                if txt:
                    lines.append((y, x, txt))

        # 页眉：y 坐标小（靠上），通常 y < 100
        header_lines = [txt for y, x, txt in lines if y < 100]

        # 尝试匹配 running head: "... / Journal of Advanced Research 24 (2020) 69–78"
        journal = volume = year = pages_range = printed_page = None
        for h in header_lines:
            # 期刊名 + 卷期年 + 页码范围模式
            m = re.search(
                r"([\w\s]+)\s+(\d+)\s*\((\d{4})\)\s+([\d–\-]+)",
                h
            )
            if m:
                journal = m.group(1).strip()
                volume = m.group(2)
                year = m.group(3)
                pages_range = m.group(4)

            # 印刷页码：孤立数字（避免误匹配年份/卷号，限制 1-4 位数）
            # 优先匹配"独立行"的数字
            if re.match(r"^\d{1,4}$", h):
                printed_page = h

        if journal:
            page_meta[page_idx] = {
                "journal": journal,
                "volume": volume or "",
                "year": year or "",
                "pages_range": pages_range or "",
                "printed_page": printed_page or f"p{page_idx}",
            }
        else:
            # 回退：用文件名 stem + 索引
            page_meta[page_idx] = {
                "journal": pdf_path.stem,
                "volume": "",
                "year": "",
                "pages_range": "",
                "printed_page": f"p{page_idx}",
            }

    doc.close()
    return page_meta


def render_single_card(
    record: dict,
    page_meta: Dict[int, dict],
    crop_dir: Path,
    out_path: Path,
    card_index: int,
) -> None:
    """渲染单张分子卡片图（类似截图风格的界面布局）。

    设计风格：
        - 顶部：编号 # N + "分子信息" 标题，右侧显示"分子指代待"
        - 中部：SMILES 信息（展开和折叠两种状态）
        - 底部：分子结构图
        - 整体采用白底灰边框，简洁清爽
    """
    # 配置中文字体
    import matplotlib.font_manager as fm

    # Windows 系统尝试使用微软雅黑或黑体
    chinese_fonts = ['Microsoft YaHei', 'SimHei', 'STSong', 'SimSun']
    font_found = None

    for font_name in chinese_fonts:
        try:
            # 尝试找到字体
            font_list = [f.name for f in fm.fontManager.ttflist]
            if font_name in font_list:
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                font_found = font_name
                break
        except:
            continue

    if not font_found:
        # 回退方案：使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']

    # 加载分子图
    crop_path = crop_dir / record["name"]
    if not crop_path.exists():
        print(f"  ⚠️ 图片缺失: {record['name']}")
        return

    img = Image.open(crop_path)
    img_w, img_h = img.size

    # 创建画布：16:10 比例
    fig_w = 14.0
    fig_h = 10.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor="white")

    # === 顶部区域：编号 + 标题 + 按钮 + 状态 ===
    # 左侧编号 "# 14"
    ax.text(
        0.05, 0.97, f"# {card_index}",
        fontsize=24, color="#666666",
        va="top", ha="left", weight="normal"
    )

    # 标题 "分子信息"
    ax.text(
        0.05, 0.92, "分子信息",
        fontsize=28, color="#000000",
        va="top", ha="left", weight="bold"
    )

    # 右侧图标按钮区域（使用文本模拟按钮）
    button_y = 0.95
    button_spacing = 0.06

    # 复制按钮
    ax.text(0.95 - button_spacing * 2, button_y, "□", fontsize=16, color="#999999", va="center", ha="center")
    # 下载按钮
    ax.text(0.95 - button_spacing, button_y, "↓", fontsize=16, color="#999999", va="center", ha="center")
    # 查看按钮
    ax.text(0.95, button_y, "◎", fontsize=16, color="#999999", va="center", ha="center")


    # 添加虚线下划线
    ax.text(
        0.95, 0.875, "——",
        fontsize=18, color="#cccccc",
        va="top", ha="right"
    )

    # 分隔线
    ax.plot([0.05, 0.95], [0.87, 0.87], color="#e0e0e0", linewidth=1.5, zorder=1)

    # === SMILES 区域 ===
    smiles = record["smiles"]

    # SMILES 标签和复制图标
    ax.text(
        0.05, 0.84, "SMILES",
        fontsize=18, color="#333333",
        va="top", ha="left", weight="normal"
    )

    # 复制图标（使用方框符号代替emoji）
    ax.text(
        0.92, 0.84, "□",
        fontsize=14, color="#999999",
        va="top", ha="right"
    )

    # SMILES内容背景框
    from matplotlib.patches import Rectangle
    smiles_bg = Rectangle(
        (0.05, 0.74),
        0.90, 0.07,
        linewidth=1,
        edgecolor="#e8e8e8",
        facecolor="#f8f9fa",
        zorder=1
    )
    ax.add_patch(smiles_bg)

    # 折叠状态的 SMILES（截断显示）
    smiles_collapsed = smiles[:35] + "..." if len(smiles) > 35 else smiles

    ax.text(
        0.06, 0.79, "折叠",
        fontsize=14, color="#888888",
        va="top", ha="left"
    )
    ax.text(
        0.12, 0.79, smiles_collapsed,
        fontsize=13, color="#000000", family="monospace",
        va="top", ha="left"
    )

    # === Source 区域 ===
    meta = page_meta.get(record["page_idx"], {})
    journal = meta.get("journal", "")
    year = meta.get("year", "")

    # 简化来源信息：只显示"期刊简称 年份"
    if journal and year:
        source = f"{journal} ({year})"
    elif journal:
        source = journal
    else:
        source = record["name"]

    # Source 标签
    ax.text(
        0.05, 0.67, "Source",
        fontsize=18, color="#333333",
        va="top", ha="left", weight="normal"
    )

    # Source 内容
    ax.text(
        0.05, 0.63, source,
        fontsize=15, color="#555555",
        va="top", ha="left"
    )

    # === 分子结构图区域 ===
    # 图片占据底部 55% 的空间
    img_left = 0.05
    img_right = 0.95
    img_bottom = 0.08
    img_top = 0.58

    # 浅灰色边框
    border = Rectangle(
        (img_left, img_bottom),
        img_right - img_left, img_top - img_bottom,
        linewidth=1.5,
        edgecolor="#e0e0e0",
        facecolor="white",
        zorder=2
    )
    ax.add_patch(border)

    # 分子图
    ax.imshow(img, extent=[img_left, img_right, img_bottom, img_top],
              aspect="auto", zorder=3, interpolation='lanczos')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white", pad_inches=0.1)
    plt.close(fig)


def render_cards(
    records: List[dict],
    page_meta: Dict[int, dict],
    crop_dir: Path,
    out_dir: Path,
    cards_per_page: int = 12,  # 此参数已废弃，保留以兼容 CLI
) -> None:
    """渲染单张分子卡片图（一个分子一张独立图片）。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按 page_idx, mol_idx 排序（保持原始顺序）
    records = sorted(records, key=lambda r: (r["page_idx"], r["mol_idx"]))

    n_total = len(records)
    print(f"共 {n_total} 个分子，将生成 {n_total} 张独立卡片图")

    for idx, rec in enumerate(records, start=1):
        # 输出文件名：molecule_001.png, molecule_002.png, ...
        out_path = out_dir / f"molecule_{idx:03d}.png"
        render_single_card(rec, page_meta, crop_dir, out_path, card_index=idx)
        if idx % 10 == 0 or idx == n_total:
            print(f"  ✓ 已完成 {idx}/{n_total} 张")

    print(f"\n所有卡片图已保存至: {out_dir}")


def export_full_csv(
    records: List[dict],
    page_meta: Dict[int, dict],
    out_path: Path,
) -> None:
    """导出带 journal/page 列的完整 CSV（顺带产物，便于核验）。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name", "smiles", "confidence", "n_atoms", "n_bonds", "status",
            "page_idx", "mol_idx", "journal", "volume", "year",
            "pages_range", "printed_page"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rec in records:
            meta = page_meta.get(rec["page_idx"], {})
            writer.writerow({
                "name": rec["name"],
                "smiles": rec["smiles"],
                "confidence": rec["confidence"],
                "n_atoms": rec["n_atoms"],
                "n_bonds": rec["n_bonds"],
                "status": rec["status"],
                "page_idx": rec["page_idx"],
                "mol_idx": rec["mol_idx"],
                "journal": meta.get("journal", ""),
                "volume": meta.get("volume", ""),
                "year": meta.get("year", ""),
                "pages_range": meta.get("pages_range", ""),
                "printed_page": meta.get("printed_page", ""),
            })

    print(f"✓ 完整 CSV 已导出: {out_path}")


def run(
    pdf_path: Path,
    smiles_csv: Path,
    crop_dir: Path,
    out_dir: Path,
    cards_per_page: int = 12,
) -> None:
    """主流程：读 CSV + 提取 PDF 页眉 + 渲染卡片图 + 导出完整 CSV。"""
    print("=" * 60)
    print("最终结果报告生成器")
    print("=" * 60)

    print(f"\n[1/4] 读取 SMILES 结果: {smiles_csv}")
    records = parse_results_csv(smiles_csv)
    print(f"  共 {len(records)} 条记录")

    print(f"\n[2/4] 提取 PDF 页眉元数据: {pdf_path.name}")
    page_meta = build_page_meta(pdf_path)
    print(f"  共解析 {len(page_meta)} 页元数据")
    # 抽查几页
    sample_pages = sorted(page_meta.keys())[:3]
    for p in sample_pages:
        m = page_meta[p]
        print(f"    页 {p}: {m.get('journal', '')} · p.{m.get('printed_page', '')}")

    print(f"\n[3/4] 渲染网格卡片图")
    render_cards(records, page_meta, crop_dir, out_dir, cards_per_page)

    print(f"\n[4/4] 导出完整 CSV（含 journal/page 列）")
    export_full_csv(records, page_meta, out_dir / "report_full.csv")

    print("\n" + "=" * 60)
    print("✓ 全部完成！")
    print("=" * 60)
    print(f"输出目录: {out_dir}")
    print(f"  - molecule_001.png ~ molecule_{len(records):03d}.png (每个分子一张独立卡片)")
    print(f"  - report_full.csv (带 journal/page 的完整表)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="生成 SMILES + 期刊名/印刷页码 + 分子图的网格卡片报告"
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=PROJECT_ROOT / "input" / "Prof. Wang-Natural products chemistry study.pdf",
        help="输入 PDF（用于提取页眉元数据）"
    )
    parser.add_argument(
        "--smiles",
        type=Path,
        default=PROJECT_ROOT / "crop_dataset_smiles" / "results.csv",
        help="SMILES 结果 CSV"
    )
    parser.add_argument(
        "--crop",
        type=Path,
        default=PROJECT_ROOT / "crop_dataset_processed",
        help="分子图目录"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "vis" / "report",
        help="输出目录"
    )
    parser.add_argument(
        "--cards-per-page",
        type=int,
        default=12,
        help="每页卡片数（默认 12 = 4 列 × 3 行）"
    )

    args = parser.parse_args()
    run(
        pdf_path=args.pdf,
        smiles_csv=args.smiles,
        crop_dir=args.crop,
        out_dir=args.out,
        cards_per_page=args.cards_per_page,
    )
