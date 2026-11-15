"""临时验证：DECIMER 端到端（Seg 分割 + OCSR 识别）在一页上能否跑通。"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

PAGE = r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\images\PageToSMILES_approved\CN102863967B_038.png"


def main():
    # 1) DECIMER-Seg 分割
    from decimer_segmentation import get_expanded_masks
    page = cv2.imread(PAGE)
    print(f"page shape: {page.shape}")
    masks = get_expanded_masks(page)
    print(f"masks shape: {masks.shape}")

    # 2) DECIMER OCSR
    from DECIMER import predict_SMILES

    n = masks.shape[2]
    print(f"n structures: {n}")
    for i in range(min(n, 5)):
        m = masks[:, :, i].astype(bool)
        ys, xs = np.nonzero(m)
        if xs.size == 0:
            continue
        x1, x2 = int(xs.min()), int(xs.max()) + 1
        y1, y2 = int(ys.min()), int(ys.max()) + 1
        crop = page[y1:y2, x1:x2]
        tmp = f"_tmp_crop_{i}.png"
        cv2.imwrite(tmp, crop)
        try:
            smiles = predict_SMILES(tmp)
            print(f"[{i}] bbox=({x1},{y1},{x2},{y2})  size={x2-x1}x{y2-y1}  SMILES={smiles}")
        except Exception as e:
            print(f"[{i}] ERR: {e}")
        os.remove(tmp)


if __name__ == "__main__":
    main()
