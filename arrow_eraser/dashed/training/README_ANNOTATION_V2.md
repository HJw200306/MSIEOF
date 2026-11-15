# 虚线箭头 + 化学键三分类数据工作区

> 整理说明（2026-10-27）：标注器已经归档到 `../annotation/seg_annotator/`，可通过 `../../启动虚线箭头标注器.bat` 打开，历史标注工作区位于 `../annotation/annotation_v2/`。本文提到的 `scripts/prepare_annotation_v2.py`、`scripts/audit_annotations_v2.py`、`scripts/synth_arrows.py` 和 `scripts/build_multiclass_dataset.py` 在旧备份中已经缺失，相关命令目前不能执行；完整边界见 `../../docs/old_backup_coverage.md`。

当前只保留三分类训练与标注流程。旧虚线二分类权重、训练入口和验证入口已经移除，
避免与背景/虚线箭头/化学键三分类 checkpoint 混用。

## 类别约定

| 像素值 | 含义 | 是否为模型输出类 |
|---:|---|---|
| 0 | background | 是 |
| 1 | dashed_arrow | 是 |
| 2 | bond | 是 |
| 255 | ignore | 否，只在损失和指标中跳过 |

## 1. 生成真实图预标注

使用项目解释器运行：

```powershell
D:\miniconda3\envs\msirost\python.exe scripts\prepare_annotation_v2.py
```

脚本执行以下操作：

1. 将 `True_dataset/labels/seed/*.json` 原样复制到
   `annotation_v2/original_arrow_annotations/`；
2. 按历史数据构建方法，从箭头 polygon 内提取实际墨迹，写为类别 1；
3. 使用 `../model/misp_weights_no_freeze.pth` 的类别 2 预标注化学键；
4. 化学键在重叠位置优先写为类别 2，以学习推理阶段的键保护规则；
5. 重叠数量仅写入元数据作质量审计，不生成额外待处理区域；
6. `255` 不由预标注程序产生，只能由人工用于确实无法判定的区域；
7. 生成 `manifest.jsonl`、预览图、哈希和逐图元数据。

重新运行默认会复用已经生成的预标注，也绝不会删除
`corrected_masks/`。需要重跑模型预标注时加 `--force`。

## 2. 人工纠错

双击项目根目录的 `启动标注器.bat`，或运行：

```powershell
D:\miniconda3\envs\msirost\python.exe seg_annotator\run.py
```

浏览器会打开 `http://127.0.0.1:5011`。常用操作：

- `1/2/3/4`：背景、虚线箭头、化学键、ignore；
- `B`：画笔，`E`：橡皮擦，`P`：多边形；
- 橡皮擦只把类别 1（虚线箭头）的误标像素恢复为背景，不会擦掉类别 2（化学键）或 255；
- 鼠标滚轮缩放，中键或按住空格拖动；
- `Ctrl+Z/Ctrl+Y`：撤销/重做；
- `Ctrl+S`：保存草稿；
- `Ctrl+Enter`：标记已审核并进入下一张；
- `A/D`：上一张/下一张；`F`：适应窗口。

预标注不会自动生成 `255`。只有人工确认确实无法判断的区域才标记为 `255`。

预标注位于 `annotation_v2/preannotations/`，始终只读。人工结果写入
`annotation_v2/corrected_masks/`，恢复预标注时旧人工结果会进入
`annotation_v2/backups/`。

## 3. 审核人工结果

```powershell
D:\miniconda3\envs\msirost\python.exe scripts\audit_annotations_v2.py --require-reviewed
```

报告位于 `annotation_v2/audit/annotation_audit.csv` 和
`annotation_v2/audit/annotation_audit.json`。检查内容包括配对、尺寸、非法像素、
空箭头/空化学键、未审核样本以及相对预标注的改动量。

## 4. 合成数据和训练集

`scripts/synth_arrows.py` 现在在生成图片的同时输出精确语义掩膜：虚线箭头由
相同绘图几何同步生成，化学键来自 bond-only SVG。二者重叠时写为类别 2；合成数据
不会自动生成 `255`。

所有真实图审核完成后构建数据集：

```powershell
D:\miniconda3\envs\msirost\python.exe scripts\build_multiclass_dataset.py
```

默认生成 250 张 journal 风格和 250 张 patent 风格合成图。数据按来源组划分，
同一论文/页面的真实裁剪和同一 SMILES 的合成图不会跨集合：

- `dataset_v2/train`：训练数据；
- `dataset_v2/val`：只含真实图，用于选模和早停；
- `dataset_v2/val_synthetic`：独立合成验证集，只作辅助监控。

构建脚本默认拒绝未审核的真实图，`--allow-preannotations` 只用于管线测试。

## 5. 训练三分类模型

```powershell
D:\miniconda3\envs\msirost\python.exe train_arrow_eraser_v2.py
```

也可以双击项目根目录的 `启动三分类训练.bat`。

输出位于 `runs/arrow_eraser_v2/`。选模指标是虚线箭头和化学键的前景平均 IoU；
该指标只取真实图验证集。合成验证指标同时写入 `history.csv`，但不参与选模。
交叉熵和 Dice 都排除像素值 255。CUDA 训练默认启用混合精度；仅在排错时使用
`--no-amp` 关闭。

