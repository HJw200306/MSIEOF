# 共享推理代码

`runtime/arrow_sr.py` 是 `utils/pipeline/arrow_sr.py` 的兼容转发入口。MSIEOF 当前串联两个步骤：按需超分辨率、统一四分类箭头去除。

主运行入口使用 `model/unified_arrow_4class.pth`，类别为背景、实线/弯曲箭头、虚线箭头、化学键。加载时会检查 checkpoint 元数据、类别顺序和分割头输出通道。

统一模型在同一张未擦除图像上一次预测，实线与虚线类别合并为箭头，化学键类别优先保护。项目根目录的 `arrow_sr.py` 仅保留为兼容入口，实际实现统一由 `utils/pipeline/arrow_sr.py` 维护。
