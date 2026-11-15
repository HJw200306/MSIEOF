# 虚线箭头合成数据生成资产

本目录保存 `arrow_eraser_old_backup` 中现存的合成数据绘图依赖：

- `draw/`：分子结构、化学键掩膜、Markush 元素、虚线连接线和其他绘图模块。
- `draw/markush/bond_segmentation.py`：从 bond-only SVG 生成化学键语义掩膜。
- `draw/line_draw.py`：虚线等连接线绘制工具。

## 完整性说明

历史文档说明合成数据由 `scripts/synth_arrows.py` 生成，再由 `scripts/build_multiclass_dataset.py` 合并为三分类数据集。但这两个入口脚本已经不在 `arrow_eraser_old_backup` 中，在当前 `MSIEOF` 项目其他位置也没有找到源代码或 `.pyc` 副本。

因此，这里完整保留了旧备份中**仍然存在的生成依赖代码**，但不能声称已经恢复完整的一键合成流程。现成的合成样本已经包含在：

```text
../dataset/DashedArrow-Seg/train
../dataset/DashedArrow-Seg/val_synthetic
```

若以后找回 `synth_arrows.py` 和 `build_multiclass_dataset.py`，应放在本目录，并把输出路径改为 `../dataset/DashedArrow-Seg` 或单独的临时生成目录。
