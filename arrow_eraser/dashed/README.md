# 虚线箭头资产

## 数据集

`dataset/DashedArrow-Seg` 是经过整理的三分类数据集：

- 0：background
- 1：dashed_arrow
- 2：bond
- 255：ignore index

数据量为 529 个训练样本、20 个真实验证样本、50 个合成验证样本；训练集包含 79 个真实样本和 450 个合成样本。

`dataset/DashedArrow-Seg-Augmented` 是在上述数据上生成的可追溯增强版：

- 训练集：2116张，即529张原图，以及环境变化、箭头风格变化、低分辨率退化各529张；
- 真实验证集：20张，按字节原样复制，未参与增强；
- 合成验证集：50张，按字节原样复制，未参与增强；
- `manifest.csv` 记录父样本、增强类别、随机种子、参数与文件哈希；
- `quality_report.json` 记录标签、掩膜、分组泄漏、精确图像泄漏及低采样可见性检查，当前结果为通过。

增强集可用下面的命令重新生成或只做复核：

```powershell
python dashed/data_generation/build_augmented_dataset.py
python dashed/data_generation/build_augmented_dataset.py --validate-only
```

## 模型与训练

- 架构：U-Net++
- 编码器：ResNet-101
- 当前权重：`../models/dashed_arrow_model.pth`
- 训练入口：`training/train_arrow_eraser_v2.py`
- 默认数据路径：`dataset/DashedArrow-Seg`
- 默认输出路径：`runs/unetpp_resnet101`

在 `arrow_eraser` 目录执行：

```powershell
python dashed/training/train_arrow_eraser_v2.py
```

使用增强数据训练时执行：

```powershell
python dashed/training/train_arrow_eraser_v2.py --data dashed/dataset/DashedArrow-Seg-Augmented
```

训练脚本现在按真实/合成各50%选择来源，再按父样本均衡抽样；裁剪按50%箭头、25%箭头—化学键邻近区、25%随机区域混合进行。模型选择规则为：先满足真实验证集化学键误删率不高于0.3%，再最大化箭头 IoU；合成验证集只作为分布监控。

2026-10-28 的小型增强实验没有得到可替换当前权重的模型：当前模型在20张真实虚线验证图上的后处理箭头 IoU 为0.783367，训练流程修正版为0.713934，加入文献版式候选数据后为0.739009。当前生产权重未替换。完整设置、限制和下一步见 `runs/DASHED_ENHANCEMENT_EXPERIMENT.md`。

## 数据生成与人工标注

- `data_generation/`：从旧备份归档的分子、化学键掩膜和虚线绘图依赖。历史合成入口 `synth_arrows.py` 与数据集构建入口 `build_multiclass_dataset.py` 在旧备份中已经缺失，详见该目录的 README。
- `annotation/seg_annotator/`：虚线箭头/化学键三分类人工标注器。
- `annotation/annotation_v2/`：可直接继续编辑的历史标注工作区。
- `../启动虚线箭头标注器.bat`：一键启动入口。

旧备份功能覆盖情况见 `../docs/old_backup_coverage.md`。
