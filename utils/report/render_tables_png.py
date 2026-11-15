"""把表A/表B渲染成高清 PNG 三线表（booktabs 风格），便于贴 Word/PPT。

数据与 tables_paper.tex 完全一致（表A 8 配置消融 + 表B 三系统检测对比）。
纯英文表头避免中文字体依赖。
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from utils.base.paths import PROJECT_ROOT



def three_line_table(col_labels, rows, out_path, col_widths=None,
                     bold_cells=None, fig_width=11.0, fontsize=10):
    ncols = len(col_labels)
    nrows = len(rows)
    if col_widths is None:
        col_widths = [1.0] * ncols
    bold_cells = bold_cells or set()

    fig, ax = plt.subplots(figsize=(fig_width, 0.42 * (nrows + 1) + 0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    top, bottom = 0.97, 0.03
    total_h = top - bottom
    header_h = total_h * 0.2
    row_h = (total_h - header_h) / nrows
    y_header = top - header_h / 2
    y_mid = top - header_h
    y_bottom = y_mid - row_h * nrows

    for y, lw in ((top, 1.4), (y_mid, 0.8), (y_bottom, 1.4)):
        ax.axhline(y, xmin=0.01, xmax=0.99, color="black", linewidth=lw)

    total_w = sum(col_widths)
    edges = [0.01]
    for w in col_widths:
        edges.append(edges[-1] + w / total_w * 0.98)
    centers = [(edges[i] + edges[i + 1]) / 2 for i in range(ncols)]

    for c, lab in enumerate(col_labels):
        ax.text(centers[c], y_header, lab, ha="center", va="center",
                fontsize=fontsize, fontweight="bold")
    for i, row in enumerate(rows):
        for c, val in enumerate(row):
            fw = "bold" if (i, c) in bold_cells else "normal"
            ax.text(centers[c], y_mid - row_h * (i + 0.5), str(val),
                    ha="center", va="center", fontsize=fontsize, fontweight=fw)

    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


def main():
    labels_a = ["Config", "P", "R", "F1", "Cov.", "Val.", "CEM",
                "GEM", "Tan.", "Tan@all-GT"]
    rows_a = [
        ["full", "99.1", "97.2", "98.2", "97.2", "100", "35.5", "35.5", "0.795", "0.7734"],
        ["no_cc", "99.1", "97.2", "98.2", "97.2", "100", "35.5", "35.5", "0.795", "0.7734"],
        ["no_filter", "99.1", "97.2", "98.2", "97.2", "100", "35.5", "35.5", "0.795", "0.7734"],
        ["no_arrow", "99.1", "97.2", "98.2", "97.2", "100", "39.5", "39.5", "0.812", "0.7899"],
        ["no_sr", "99.1", "97.2", "98.2", "97.2", "100", "33.1", "33.1", "0.765", "0.7438"],
        ["no_optim", "99.1", "97.2", "98.2", "97.2", "100", "38.8", "38.8", "0.795", "0.7734"],
        ["uniform_sr", "99.1", "97.2", "98.2", "97.2", "99.8", "33.6", "33.6", "0.791", "0.7673"],
        ["arrow_then_sr", "99.1", "97.2", "98.2", "97.2", "100", "35.1", "35.1", "0.791", "0.7691"],
    ]
    bold_a = {(0, 6), (3, 6)}
    three_line_table(labels_a, rows_a, PROJECT_ROOT / "tableA_ablation.png",
                     col_widths=[1.5] + [1] * 9, bold_cells=bold_a, fig_width=11.5)

    labels_b = ["Method", "GT", "Pred", "Matched", "P (%)", "R (%)",
                "F1 (%)", "Coverage (%)"]
    rows_b = [
        ["Ours (full)", "469", "460", "456", "99.1", "97.2", "98.2", "97.2"],
        ["DECIMER-Seg", "469", "449", "432", "96.2", "92.1", "94.1", "92.1"],
        ["YoDe-Seg", "—", "—", "—", "—", "—", "—", "—"],
    ]
    three_line_table(labels_b, rows_b, PROJECT_ROOT / "tableB_detection.png",
                     col_widths=[1.6] + [1] * 7, fig_width=9.0)


if __name__ == "__main__":
    main()
