"""交互式分子信息报告生成器：HTML格式，支持展开/折叠、复制SMILES等功能。

用法
----
python -m utils.report.report_interactive \
    --pdf input/Prof. Wang-Natural products chemistry study.pdf \
    --smiles crop_dataset_smiles/results.csv \
    --crop crop_dataset_processed/ \
    --out vis/report_interactive/
"""

import base64
import csv
import re
from io import BytesIO
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
from PIL import Image

from utils.base.paths import PROJECT_ROOT


def parse_results_csv(csv_path: Path) -> List[dict]:
    """读 results.csv，从 name 反推 page_idx 和 mol_idx。"""
    records = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
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
    """从 PDF 页眉提取期刊名 + 卷期年 + 印刷页码。"""
    doc = fitz.open(str(pdf_path))
    page_meta: Dict[int, dict] = {}

    for page_idx in range(doc.page_count):
        page = doc[page_idx]
        text_dict = page.get_text("dict")
        lines = []
        for block in text_dict.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                y = round(line["bbox"][1])
                x = round(line["bbox"][0])
                txt = "".join(s["text"] for s in line["spans"]).strip()
                if txt:
                    lines.append((y, x, txt))

        header_lines = [txt for y, x, txt in lines if y < 100]
        journal = volume = year = pages_range = printed_page = None

        for h in header_lines:
            m = re.search(r"([\w\s]+)\s+(\d+)\s*\((\d{4})\)\s+([\d–\-]+)", h)
            if m:
                journal = m.group(1).strip()
                volume = m.group(2)
                year = m.group(3)
                pages_range = m.group(4)
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
            page_meta[page_idx] = {
                "journal": pdf_path.stem,
                "volume": "",
                "year": "",
                "pages_range": "",
                "printed_page": f"p{page_idx}",
            }

    doc.close()
    return page_meta


def image_to_base64(img_path: Path) -> str:
    """将图片转换为 base64 编码字符串，用于嵌入HTML。"""
    try:
        with Image.open(img_path) as img:
            # 转换为RGB（防止RGBA导致问题）
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background

            buffer = BytesIO()
            img.save(buffer, format='PNG')
            img_str = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"  ⚠️ 图片处理失败 {img_path}: {e}")
        return ""


def generate_html_report(
    records: List[dict],
    page_meta: Dict[int, dict],
    crop_dir: Path,
    out_path: Path,
) -> None:
    """生成交互式HTML报告，类似截图中的界面风格。"""

    # HTML 头部和样式
    html_header = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>分子信息报告</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
            line-height: 1.6;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .molecule-card {
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 24px;
            padding: 24px;
        }

        .card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
            padding-bottom: 12px;
            border-bottom: 2px solid #e0e0e0;
        }

        .card-title {
            font-size: 24px;
            font-weight: 600;
            color: #333;
        }

        .card-index {
            color: #666;
            font-size: 18px;
        }

        .info-section {
            margin-bottom: 20px;
        }

        .section-title {
            font-size: 14px;
            font-weight: 600;
            color: #666;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .smiles-container {
            background: #f8f9fa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 12px;
            position: relative;
        }

        .smiles-content {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: #000;
            word-break: break-all;
            padding-right: 120px;
        }

        .smiles-collapsed {
            max-height: 24px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .smiles-actions {
            position: absolute;
            right: 12px;
            top: 12px;
            display: flex;
            gap: 8px;
        }

        .btn {
            padding: 6px 12px;
            border: 1px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            transition: all 0.2s;
        }

        .btn:hover {
            background: #f0f0f0;
            border-color: #bbb;
        }

        .btn-icon {
            padding: 6px 8px;
        }

        .status-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 12px;
            font-weight: 500;
        }

        .status-pending {
            background: #fff3cd;
            color: #856404;
        }

        .status-done {
            background: #d4edda;
            color: #155724;
        }

        .molecule-image {
            margin-top: 16px;
            text-align: center;
        }

        .molecule-image img {
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .metadata {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 12px;
            margin-top: 16px;
            padding: 16px;
            background: #f8f9fa;
            border-radius: 6px;
        }

        .metadata-item {
            display: flex;
            flex-direction: column;
        }

        .metadata-label {
            font-size: 12px;
            color: #666;
            margin-bottom: 4px;
        }

        .metadata-value {
            font-size: 14px;
            color: #333;
            font-weight: 500;
        }

        .toast {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #333;
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            display: none;
            z-index: 1000;
        }

        .toast.show {
            display: block;
            animation: fadeIn 0.3s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1 style="text-align: center; margin-bottom: 32px; color: #333;">分子信息报告</h1>
"""

    html_footer = """
    </div>

    <div id="toast" class="toast"></div>

    <script>
        function toggleSMILES(id) {
            const element = document.getElementById('smiles-' + id);
            const btn = document.getElementById('toggle-' + id);

            if (element.classList.contains('smiles-collapsed')) {
                element.classList.remove('smiles-collapsed');
                btn.textContent = '折叠';
            } else {
                element.classList.add('smiles-collapsed');
                btn.textContent = '展开';
            }
        }

        function copySMILES(smiles) {
            navigator.clipboard.writeText(smiles).then(() => {
                showToast('SMILES 已复制到剪贴板');
            }).catch(err => {
                showToast('复制失败: ' + err);
            });
        }

        function showToast(message) {
            const toast = document.getElementById('toast');
            toast.textContent = message;
            toast.classList.add('show');
            setTimeout(() => {
                toast.classList.remove('show');
            }, 2000);
        }
    </script>
</body>
</html>
"""

    # 按 page_idx, mol_idx 排序
    records = sorted(records, key=lambda r: (r["page_idx"], r["mol_idx"]))

    # 生成分子卡片HTML
    cards_html = []
    for idx, rec in enumerate(records, start=1):
        smiles = rec["smiles"]
        meta = page_meta.get(rec["page_idx"], {})

        # 获取图片的base64编码
        crop_path = crop_dir / rec["name"]
        img_base64 = image_to_base64(crop_path) if crop_path.exists() else ""

        # 分子指代状态（这里默认为"待标注"，可根据实际情况调整）
        status_class = "status-pending"
        status_text = "分子指代待标注"
        if rec.get("status") == "done":
            status_class = "status-done"
            status_text = "分子指代已完成"

        # 构建来源信息
        source_parts = []
        if meta.get("journal"):
            source_parts.append(meta["journal"])
        if meta.get("volume"):
            source_parts.append(f"Vol. {meta['volume']}")
        if meta.get("year"):
            source_parts.append(f"({meta['year']})")
        if meta.get("printed_page"):
            source_parts.append(f"p. {meta['printed_page']}")
        source = " ".join(source_parts) if source_parts else rec["name"]

        # 转义SMILES中的单引号（避免f-string中的反斜杠问题）
        smiles_escaped = smiles.replace("'", "&#39;")

        card_html = f"""
        <div class="molecule-card">
            <div class="card-header">
                <h2 class="card-title">分子信息</h2>
                <span class="card-index"># {idx}</span>
            </div>

            <div class="info-section">
                <div class="section-title">SMILES</div>
                <div class="smiles-container">
                    <div id="smiles-{idx}" class="smiles-content smiles-collapsed">{smiles}</div>
                    <div class="smiles-actions">
                        <button class="btn btn-icon" onclick="copySMILES('{smiles_escaped}')">📋</button>
                        <button id="toggle-{idx}" class="btn" onclick="toggleSMILES({idx})">展开</button>
                    </div>
                </div>
            </div>

            <div class="info-section">
                <span class="status-badge {status_class}">{status_text}</span>
            </div>
"""

        # 添加分子图片（如果存在）
        if img_base64:
            card_html += f"""
            <div class="molecule-image">
                <img src="{img_base64}" alt="Molecule {idx}">
            </div>
"""

        # 添加元数据
        card_html += f"""
            <div class="metadata">
                <div class="metadata-item">
                    <span class="metadata-label">来源</span>
                    <span class="metadata-value">{source}</span>
                </div>
"""

        if rec.get("n_atoms"):
            card_html += f"""
                <div class="metadata-item">
                    <span class="metadata-label">原子数</span>
                    <span class="metadata-value">{rec['n_atoms']}</span>
                </div>
"""

        if rec.get("n_bonds"):
            card_html += f"""
                <div class="metadata-item">
                    <span class="metadata-label">键数</span>
                    <span class="metadata-value">{rec['n_bonds']}</span>
                </div>
"""

        if rec.get("confidence"):
            card_html += f"""
                <div class="metadata-item">
                    <span class="metadata-label">置信度</span>
                    <span class="metadata-value">{rec['confidence']}</span>
                </div>
"""

        card_html += """
            </div>
        </div>
"""

        cards_html.append(card_html)

        if idx % 10 == 0 or idx == len(records):
            print(f"  ✓ 已生成 {idx}/{len(records)} 张卡片")

    # 组装完整HTML
    full_html = html_header + "\n".join(cards_html) + html_footer

    # 保存文件
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"\n✓ HTML报告已生成: {out_path}")
    print(f"  在浏览器中打开此文件即可查看交互式报告")


def run(
    pdf_path: Path,
    smiles_csv: Path,
    crop_dir: Path,
    out_dir: Path,
) -> None:
    """主流程：读 CSV + 提取 PDF 页眉 + 生成HTML报告。"""
    print("=" * 60)
    print("交互式分子信息报告生成器")
    print("=" * 60)

    print(f"\n[1/3] 读取 SMILES 结果: {smiles_csv}")
    records = parse_results_csv(smiles_csv)
    print(f"  共 {len(records)} 条记录")

    print(f"\n[2/3] 提取 PDF 页眉元数据: {pdf_path.name}")
    page_meta = build_page_meta(pdf_path)
    print(f"  共解析 {len(page_meta)} 页元数据")

    print(f"\n[3/3] 生成交互式HTML报告")
    out_path = out_dir / "report.html"
    generate_html_report(records, page_meta, crop_dir, out_path)

    print("\n" + "=" * 60)
    print("✓ 全部完成！")
    print("=" * 60)
    print(f"输出文件: {out_path}")
    print(f"请在浏览器中打开该文件查看交互式报告")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="生成交互式HTML格式的分子信息报告"
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
        default=PROJECT_ROOT / "vis" / "report_interactive",
        help="输出目录"
    )

    args = parser.parse_args()
    run(
        pdf_path=args.pdf,
        smiles_csv=args.smiles,
        crop_dir=args.crop,
        out_dir=args.out,
    )
