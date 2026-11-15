"""化学分子结构 → SMILES（流水线终点）。

输入
----
arrow_sr.py 的输出 `crop_dataset_processed/`：每张图都是单分子、抹白背景、
（必要时）超分过、去过实线/虚线箭头的干净 crop。

模型
----
MolScribe（Yujie Qian et al., MIT），Swin-Base 编码器 + Transformer 解码器，
字符级 + 坐标双 head。权重 `model/swin_base_char_aux_1m680k.pth`（官方 1.68M
样本预训练 release）。本模块通过 pip 包 `molscribe` 调用，输入 BGR/RGB ndarray
或文件路径都行，模块内部统一走 ndarray batch 接口 (`predict_images`) 以便上
ImageNet 归一化和分 batch 上 GPU。

兼容性 shim
-----------
`molscribe==1.1.1` 的 `transformer/swin_transformer.py:24` 仍然从
`timm.models.helpers` 导入 `overlay_external_default_cfg`，而 timm ≥ 0.6.x
已把它删除（本地是 1.0.27）。模块顶部在 `import molscribe` 之前注入一个等价
的 5 行 helper，避免改 site-packages。

输出
----
crop_dataset_smiles/results.csv
    name, smiles, confidence, n_atoms, n_bonds, status, smiles_raw, raw_status, schema
    - smiles：与标注、评测共用的严格 CXSMILES，保留 Markush 标签
    - status：OK / REVIEW（立体信息待复核） / INVALID / READ_FAIL
    - smiles_raw / raw_status：原模型字符串及独立校验结果，供追溯
crop_dataset_smiles/<stem>.json
    原模型输出及统一转换后的图、校验原因；无效图保留，但不输出有效 SMILES
crop_dataset_smiles/<stem>.mol
    MolScribe 直接返回的 molblock 字符串，便于下游 RDKit / 反应解析直接 load

可视化（visualize=True）
------------------------
vis/molscribe/<stem>_overlay.png
    在原 crop 上叠加 MolScribe 推出的原子点 + 键线段 + 元素标签
    （C 不标，按 RDKit / IUPAC 习惯）；标题写 confidence / n_atoms / n_bonds，
    解析失败的图标题打 INVALID。
"""

from pathlib import Path
from typing import List, Optional, Tuple
import csv
import json
import sys

from utils.base.paths import PROJECT_ROOT
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.base.chemical_graph import convert_graph, SCHEMA_VERSION

import cv2
import numpy as np
from tqdm import tqdm


# --- timm 兼容 shim：必须在 import molscribe 之前注入 ----------------------
def _patch_timm_for_molscribe() -> None:
    """补回 molscribe 1.1.1 还在用、但新版 timm（本地 1.0.27）已删除的两个 helper。

    1) `timm.models.helpers.overlay_external_default_cfg`
       molscribe `swin_transformer.py:24` 导入。等价实现从 timm 0.5.4 源码搬过来：
       把外部传入的 `external_default_cfg` 合并进 `default_cfg`，并把会被覆盖的
       url/hf_hub 清掉。
    2) `timm.models.vision_transformer._init_vit_weights`
       molscribe `swin_transformer.py:27` 导入。这是新建模型时 `module.apply(...)`
       用的 random init；MolScribe 立刻 `safe_load` 覆盖整套权重，所以这里给一
       个等价的 ViT 初始化（Linear/LN/Conv2d 都按 timm 0.5.4 的 `_init_vit_weights`
       做 trunc_normal / zeros / ones），保证 import 通过且不影响推理结果。
    """
    try:
        import timm.models.helpers as _h
    except Exception:
        return
    if not hasattr(_h, "overlay_external_default_cfg"):
        def overlay_external_default_cfg(default_cfg, kwargs):
            external_default_cfg = kwargs.pop("external_default_cfg", None)
            if external_default_cfg:
                default_cfg.pop("url", None)
                default_cfg.pop("hf_hub", None)
                default_cfg.update(external_default_cfg)
        _h.overlay_external_default_cfg = overlay_external_default_cfg

    try:
        import torch.nn as _nn
        from timm.layers import trunc_normal_, lecun_normal_
        import timm.models.vision_transformer as _vt
    except Exception:
        return
    if not hasattr(_vt, "_init_vit_weights"):
        def _init_vit_weights(module, name: str = "", head_bias: float = 0.0,
                              jax_impl: bool = False):
            if isinstance(module, _nn.Linear):
                if name.startswith("head"):
                    _nn.init.zeros_(module.weight)
                    _nn.init.constant_(module.bias, head_bias)
                elif name.startswith("pre_logits"):
                    lecun_normal_(module.weight)
                    _nn.init.zeros_(module.bias)
                else:
                    if jax_impl:
                        _nn.init.xavier_uniform_(module.weight)
                        if module.bias is not None:
                            if "mlp" in name:
                                _nn.init.normal_(module.bias, std=1e-6)
                            else:
                                _nn.init.zeros_(module.bias)
                    else:
                        trunc_normal_(module.weight, std=0.02)
                        if module.bias is not None:
                            _nn.init.zeros_(module.bias)
            elif jax_impl and isinstance(module, _nn.Conv2d):
                lecun_normal_(module.weight)
                if module.bias is not None:
                    _nn.init.zeros_(module.bias)
            elif isinstance(module, (_nn.LayerNorm, _nn.GroupNorm,
                                     _nn.BatchNorm2d)):
                _nn.init.zeros_(module.bias)
                _nn.init.ones_(module.weight)
        _vt._init_vit_weights = _init_vit_weights


_patch_timm_for_molscribe()

import torch  # noqa: E402
from molscribe import MolScribe  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit import RDLogger  # noqa: E402


# --- MolScribe 后处理 worker 数封顶 ----------------------------------------
# `predict_images` 内部唯一用到 multiprocessing 的地方是 chemistry.py 的
# `convert_graph_to_smiles`，默认 num_workers=16。Windows 下 spawn 会起 16 个子
# 进程，每个都重新 import molscribe+torch（单进程 ~1GB），15.8GB 内存会直接 OOM。
# 这里封到 4（12 核 / 16GB 下安全：4 个 worker ≈ 4GB，主进程还要驻留 Swin-Base）。
# 注意 interface.py 用 `from .chemistry import convert_graph_to_smiles` 在模块级
# 另存了一份绑定，predict_images 调的是 interface 里的那份，所以两处都要换。
from functools import partial as _partial
import molscribe.chemistry as _ms_chem
import molscribe.interface as _ms_iface

_orig_c2s = _ms_chem.convert_graph_to_smiles
_ms_chem.convert_graph_to_smiles = _partial(_orig_c2s, num_workers=4)
_ms_iface.convert_graph_to_smiles = _partial(_orig_c2s, num_workers=4)

# 关闭 RDKit 的 "WARNING: not removing hydrogen atom..." 之类的噪声日志
RDLogger.DisableLog("rdApp.*")

MODEL_PATH = PROJECT_ROOT / "model" / "swin_base_char_aux_1m680k.pth"
INPUT_DIR = PROJECT_ROOT / "crop_dataset_processed"
OUTPUT_DIR = PROJECT_ROOT / "crop_dataset_smiles"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# MolScribe 内部 batch 大小（predict_images 的 batch_size 参数）。
#   操作 = 一次喂给 encoder/decoder 的图片数；
#   单位 = 张；
#   几何尺度 = GPU 显存压力（Swin-Base 384² + Transformer decoder，单张约
#              0.8–1.2 GB 峰值）；
#   调大（如 32）→ 吞吐略升，显存峰值近似线性增长，单张延迟基本不变；
#   调小（如 4） → 显存压力骤降，吞吐打折，CPU/GPU 切换开销凸显；
#   默认 16 = 官方接口默认值，10–12 GB 显存下稳。
BATCH_SIZE = 16


class Img2SmilesProcessor:
    """封装 MolScribe 推理：加载一次模型，批量处理 crop。"""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        device: str = "cuda",
    ):
        print("正在初始化 MolScribe ...")
        dev_str = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.device = torch.device(dev_str)
        self.model = MolScribe(str(model_path), device=self.device)
        print(f"MolScribe 权重: {model_path.name}  ({self.device})")

    def predict_batch(self, images_rgb: List[np.ndarray],
                      batch_size: int = BATCH_SIZE) -> List[dict]:
        """对一批 RGB ndarray 跑 MolScribe，返回每张图的 dict
        （含 smiles / molfile / confidence / atoms / bonds）。"""
        return self.model.predict_images(
            images_rgb,
            return_atoms_bonds=True,
            return_confidence=True,
            batch_size=batch_size,
        )


def _canonicalize_smiles(raw: str) -> Tuple[str, str]:
    """把 MolScribe 输出的 SMILES 走一次 RDKit 规范化。

    Returns (smiles_out, status)
        status = "OK"      → RDKit 能解析，smiles_out 是规范化后的串
        status = "INVALID" → RDKit 解析失败，smiles_out 维持 raw（便于 debug）
        status = "EMPTY"   → raw 为空 / None
    """
    if not raw:
        return "", "EMPTY"
    try:
        mol = Chem.MolFromSmiles(raw)
    except Exception:
        mol = None
    if mol is None:
        return raw, "INVALID"
    try:
        return Chem.MolToSmiles(mol), "OK"
    except Exception:
        return raw, "INVALID"


def standardize_prediction(pred: dict, image_size) -> dict:
    """Keep the model output intact and convert its graph with the GT contract."""
    chemistry = convert_graph(pred.get("atoms", []), pred.get("bonds", []),
                              image_size=image_size, coordinates_normalized=True)
    raw = pred.get("smiles") or ""
    _, raw_status = _canonicalize_smiles(raw)
    status = ("INVALID" if not chemistry["valid"] else
              "REVIEW" if chemistry["warnings"] else "OK")
    return {"smiles": chemistry["smiles"], "status": status,
            "smiles_raw": raw, "raw_status": raw_status,
            "schema": SCHEMA_VERSION, "chemistry": chemistry}


def _save_overlay(
    img_rgb: np.ndarray,
    pred: dict,
    smiles_canon: str,
    status: str,
    out_path: Path,
) -> None:
    """在原图上叠加 MolScribe 推出的原子点 + 键线段 + 非碳元素标签。

    MolScribe 的 atom 坐标 (x, y) 是 [0, 1] 归一化到输入图，乘 (W, H) 即得像素。
    键按类型上色 / 加粗（参见 viz_utils 里的 COLOR_MS_BOND_*）。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except Exception as e:
        print(f"可视化保存失败 [{out_path.name}]: {e}")
        return

    from utils.base.viz_utils import (
        COLOR_MS_ATOM,
        COLOR_MS_BOND_SINGLE,
        COLOR_MS_BOND_DOUBLE,
        COLOR_MS_BOND_TRIPLE,
        COLOR_MS_BOND_AROMATIC,
        COLOR_MS_BOND_WEDGE_SOLID,
        COLOR_MS_BOND_WEDGE_DASH,
        ensure_dir,
    )

    bond_style = {
        "single":       (COLOR_MS_BOND_SINGLE,       1.2, "-"),
        "double":       (COLOR_MS_BOND_DOUBLE,       2.0, "-"),
        "triple":       (COLOR_MS_BOND_TRIPLE,       2.6, "-"),
        "aromatic":     (COLOR_MS_BOND_AROMATIC,     1.6, "--"),
        "solid wedge":  (COLOR_MS_BOND_WEDGE_SOLID,  2.4, "-"),
        "dashed wedge": (COLOR_MS_BOND_WEDGE_DASH,   2.0, ":"),
    }

    h, w = img_rgb.shape[:2]
    atoms = pred.get("atoms") or []
    bonds = pred.get("bonds") or []
    conf = pred.get("confidence")

    xs = [a["x"] * w for a in atoms]
    ys = [a["y"] * h for a in atoms]
    symbols = [a["atom_symbol"] for a in atoms]

    fig_w = max(6.0, min(14.0, w / 150.0))
    fig_h = fig_w * h / max(w, 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.imshow(img_rgb)
    ax.set_axis_off()

    # 键：按类型上色，先画粗线、再画细线，让交叉时上层视觉清晰
    seen_types = set()
    for b in bonds:
        i, j = b["endpoint_atoms"]
        if i >= len(atoms) or j >= len(atoms):
            continue
        btype = b["bond_type"]
        seen_types.add(btype)
        color, lw, ls = bond_style.get(
            btype, (COLOR_MS_BOND_SINGLE, 1.2, "-")
        )
        col_unit = (color[0] / 255, color[1] / 255, color[2] / 255)
        ax.plot([xs[i], xs[j]], [ys[i], ys[j]],
                color=col_unit, linewidth=lw, linestyle=ls,
                solid_capstyle="round", zorder=2)

    # 原子点 + 元素标签（C 不标，省得遮挡骨架）
    atom_unit = (COLOR_MS_ATOM[0] / 255, COLOR_MS_ATOM[1] / 255, COLOR_MS_ATOM[2] / 255)
    for x, y, sym in zip(xs, ys, symbols):
        ax.plot(x, y, "o", color=atom_unit, markersize=3.5, zorder=3)
        if sym and sym not in ("C", "c"):
            ax.text(x, y, sym, fontsize=8, color="white",
                    ha="center", va="center", zorder=4,
                    bbox=dict(facecolor=atom_unit, edgecolor="none",
                              alpha=0.9, pad=0.8))

    conf_str = f"{conf:.4f}" if isinstance(conf, (int, float)) else "n/a"
    if status == "OK":
        title = (f"[molscribe] {out_path.stem.replace('_overlay', '')}  "
                 f"conf={conf_str}  atoms={len(atoms)}  bonds={len(bonds)}\n"
                 f"SMILES: {smiles_canon}")
        title_color = "black"
    else:
        title = (f"[molscribe] {out_path.stem.replace('_overlay', '')}  "
                 f"conf={conf_str}  atoms={len(atoms)}  bonds={len(bonds)}  "
                 f"→ {status}\nRAW: {smiles_canon}")
        title_color = "red"
    ax.set_title(title, fontsize=9, color=title_color)

    legend_handles = []
    if atoms:
        legend_handles.append(Patch(facecolor=atom_unit, edgecolor="black",
                                    label="atom (non-C)"))
    for btype in sorted(seen_types):
        color, _, _ = bond_style.get(btype, (COLOR_MS_BOND_SINGLE, 1.2, "-"))
        col_unit = (color[0] / 255, color[1] / 255, color[2] / 255)
        legend_handles.append(Patch(facecolor=col_unit, edgecolor="black",
                                    label=f"bond: {btype}"))
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper right",
                  fontsize=8, framealpha=0.85)

    ensure_dir(out_path.parent)
    fig.savefig(str(out_path), bbox_inches="tight", dpi=120)
    plt.close(fig)


def _load_rgb(img_path: Path) -> Optional[np.ndarray]:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def run(
    input_dir: Path = INPUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    device: str = "cuda",
    batch_size: int = BATCH_SIZE,
    visualize: bool = False,
) -> None:
    """读取 input_dir 下所有图片，跑 MolScribe，把结果写到 output_dir。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    if not input_dir.exists():
        print(f"输入目录不存在: {input_dir}")
        return

    files = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        print(f"输入目录无图片: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    processor = Img2SmilesProcessor(device=device)

    csv_path = output_dir / "results.csv"
    fieldnames = ["name", "smiles", "confidence", "n_atoms", "n_bonds", "status",
                  "smiles_raw", "raw_status", "schema"]
    csv_f = csv_path.open("w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    writer.writeheader()

    n_in = n_ok = n_invalid = n_read_fail = 0

    # 按 batch_size 分块读图 + 推理；读图失败的留位占行，CSV 里标 READ_FAIL。
    for start in tqdm(range(0, len(files), batch_size), desc="MolScribe"):
        chunk = files[start:start + batch_size]
        images: List[np.ndarray] = []
        valid_idx: List[int] = []
        for k, f in enumerate(chunk):
            img_rgb = _load_rgb(f)
            if img_rgb is None:
                writer.writerow({"name": f.name, "smiles": "",
                                 "confidence": "", "n_atoms": 0, "n_bonds": 0,
                                 "status": "READ_FAIL"})
                n_in += 1
                n_read_fail += 1
                tqdm.write(f"[FAIL ] {f.name}  (imread 失败)")
                continue
            images.append(img_rgb)
            valid_idx.append(k)

        if not images:
            continue

        preds = processor.predict_batch(images, batch_size=batch_size)

        for img_rgb, pred, k in zip(images, preds, valid_idx):
            f = chunk[k]
            n_in += 1
            raw_smiles = pred.get("smiles") or ""
            result = standardize_prediction(pred, (img_rgb.shape[1], img_rgb.shape[0]))
            smiles_canon, status = result["smiles"], result["status"]
            conf = pred.get("confidence")
            atoms = pred.get("atoms") or []
            bonds = pred.get("bonds") or []
            (output_dir / f"{f.stem}.json").write_text(
                json.dumps({"raw_prediction": pred, **result}, ensure_ascii=False, indent=2),
                encoding="utf-8")

            mol_block = pred.get("molfile") or ""
            if mol_block:
                (output_dir / f"{f.stem}.mol").write_text(mol_block,
                                                          encoding="utf-8")

            writer.writerow({
                "name": f.name,
                "smiles": smiles_canon,
                "confidence": f"{conf:.6f}" if isinstance(conf, (int, float)) else "",
                "n_atoms": len(atoms),
                "n_bonds": len(bonds),
                "status": status,
                "smiles_raw": result["smiles_raw"],
                "raw_status": result["raw_status"],
                "schema": result["schema"],
            })

            if status == "OK":
                n_ok += 1
                conf_text = f"{conf:.4f}" if isinstance(conf, (int, float)) else "n/a"
                tqdm.write(f"[OK   ] {f.name}  conf="
                           + conf_text
                           + f"  {smiles_canon}")
            else:
                n_invalid += 1
                tqdm.write(f"[{status}] {f.name}  raw={raw_smiles}")

            if visualize:
                # 函数内延迟导入，配合 viz_utils.redirect_viz_root() 生效：
                # 每次调用重读模块属性，任务级重定向才能落到自己的目录。
                from utils.base.viz_utils import VIZ_MOLSCRIBE
                viz_path = VIZ_MOLSCRIBE / f"{f.stem}_overlay.png"
                _save_overlay(img_rgb, pred, smiles_canon, status, viz_path)

    csv_f.close()

    print("\n" + "=" * 50)
    print(f"输入图片:        {n_in}")
    print(f"成功（OK）:       {n_ok}")
    print(f"无效或待复核:    {n_invalid}")
    print(f"读图失败:        {n_read_fail}")
    print(f"\nCSV:     {csv_path}")
    print(f"molfile: {output_dir}/*.mol")
    if visualize:
        from utils.base.viz_utils import VIZ_MOLSCRIBE
        print(f"可视化: {VIZ_MOLSCRIBE}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="用 MolScribe 把 crop_dataset_processed/ 里的单分子图转成 "
                    "SMILES + molfile，汇总到 crop_dataset_smiles/results.csv。"
    )
    parser.add_argument("--input", default=str(INPUT_DIR),
                        help=f"输入目录（默认 {INPUT_DIR.name}/，即 arrow_sr 输出）")
    parser.add_argument("--output", default=str(OUTPUT_DIR),
                        help=f"输出目录（默认 {OUTPUT_DIR.name}/）")
    parser.add_argument("--device", default="cuda",
                        help="cuda 或 cpu（CUDA 不可用时自动回退 cpu）")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"MolScribe 内部 batch 大小（默认 {BATCH_SIZE}）")
    parser.add_argument("--visualize", action="store_true",
                        help="生成原子/键叠加图到 vis/molscribe/")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                        help="关闭可视化")
    parser.set_defaults(visualize=True)
    args = parser.parse_args()
    run(
        input_dir=Path(args.input),
        output_dir=Path(args.output),
        device=args.device,
        batch_size=args.batch_size,
        visualize=args.visualize,
    )
