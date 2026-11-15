# -*- coding: utf-8 -*-
"""临时诊断：统计写回掩码的 8 连通域数量，与 json 的 shape 数对比。

用途：定位"一个 GT 实例包含多个分子"是 json 还是掩码/连通域阶段引入的。
"""
import sys

import numpy as np
import cv2
from PIL import Image

PATHS = [
    r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored\benchmark_D2C-UNI\us\US20210149301A1_82.mask.png",
    r"E:\其他文章及数据集\PatCID_chemical_structures_in_patent_documents\restored\benchmark_D2C-UNI\us\09682983_33.mask.png",
]

for p in PATHS:
    arr = np.array(Image.open(p).convert("L"))
    binary = (arr > 127).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    print("=" * 70)
    print(p)
    print("连通域数(不含背景):", n - 1)
    for ci in range(1, n):
        x, y, w, h, a = stats[ci]
        print(f"  cc{ci:02d} bbox=({x},{y})-({x+w},{y+h}) area={a}")
