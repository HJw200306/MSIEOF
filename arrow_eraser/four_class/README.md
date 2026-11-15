# 统一四分类训练包

架构：ResNet-101 U-Net++。本目录提供已导出的四分类数据、可直接运行的训练/评估入口，以及共享全部输入与配置的三分类对照。

正式四分类训练现已完成：`runs/four_seed42/best.pth` 为第79轮最佳检查点。2026-11-03 起，剥离训练状态后的部署副本 `model/unified_arrow_4class.pth` 已替代生产双模型融合流程；迁移细节见 `../MODEL_MIGRATION.md`。

## 数据

`dataset/UnifiedArrow-Type-Seg-v1/` 包含4043张图片：

| 划分 | 实线来源 | 虚线来源 | 合计 |
|---|---:|---:|---:|
| train | 1467 | 2116 | 3583 |
| val | 193 | 20 | 213 |
| test | 197 | 0 | 197 |
| val_synthetic | 0 | 50 | 50 |

类别：0背景，1实线/连续弯曲箭头，2虚线箭头（含弯曲虚线），3化学键，255忽略。虚线箭头的连续箭头头部也属于类别2。虚线化学键、楔形键仍属于化学键。箭头与化学键重叠位置保留化学键。

实线来源从人工工作区的类型编码恢复：1→1，4→2，2/3/6→3。虚线来源按其已有标签语义映射1→2、2→3；未根据文件名前缀猜测类型。重复图的标签沿用现有像素投票规则；测试图solid_1744的类型冲突通过原图目视检查确定为连续弯曲箭头，详见`type_resolutions.json`。原标注工作区未修改。

全部四分类标签合并为三分类后与历史统一标签逐像素一致。`masks_three_class/`另存三分类派生掩膜；训练代码直接读取四分类母标签并按`--classes`动态映射。`manifest.csv`记录图像、标签、来源、父样本、哈希；其中历史source_root是来源记录，训练不访问该绝对路径。复制整个four_class目录到另一台电脑即可使用。

本次沿用已有人工审核结果，未重新完成所有图片的人工类型复核。精确重复/父样本/来源组划分由审计检查；历史来源信息不足时，无法据此保证无近重复泄漏。现有20张虚线真实验证图参与过历史选模；test并不包含独立的虚线来源测试集。120张文档候选图未加入。历史增强图与掩膜（含lowres标签）原样保留，便于B/C配对对照。

## 训练

在本目录打开终端，激活安装好PyTorch及依赖的环境：

```powershell
python -m pip install -r requirements.txt
python verify_dataset.py
python train.py --classes 4 --output runs/four_seed42 --seed 42
```

当前电脑可使用 `D:\miniconda3\envs\msirost\python.exe`。首次正式训练使用ImageNet预训练编码器，需要本机缓存或联网下载。`--encoder-weights none`仅适用于明确要从随机初始化开始的实验，B/C必须一致。

也可双击`train_four_class.bat`；默认调用PATH中的python，可设置`ARROW_PYTHON`指定解释器。三分类对照使用：

```powershell
python train.py --classes 3 --output runs/three_seed42 --seed 42
```

默认参数：80轮、每轮2934次抽样、batch size=1、输入512×512、AdamW、学习率8e-5、weight decay=1e-4、余弦衰减至1e-6，CUDA自动混合精度。批大小可按训练机器显存调大，但B/C必须完全一致。这里是从ImageNet编码器初始化的新训练，不是从现有三分类权重微调。

每轮两个来源各50%，来源内均匀选父样本，再随机选原图/增强视图。每个索引的随机数由seed/epoch/index固定，模型通道数与worker数量不会改变输入；父样本有放回抽样，不保证每轮恰好访问一次。裁剪50%箭头、25%箭头与键邻近、25%随机；RGB白色边界、mask=255，掩膜旋转最近邻。归一化使用ImageNet均值/标准差。

损失为0.5加权CE＋0.5前景Dice：四分类CE权重[.2,1,1,1.35]、Dice[0,.25,.25,.5]；三分类CE[.2,1,1.35]、Dice[0,.5,.5]。所有255像素不参与损失和指标。

固定预算完成80轮，按真实验证集化学键误擦率≤0.3%约束，在实线/虚线条件化箭头IoU等权均值最高时保存best.pth。0.3%是本实验起始阈值，可用`--max-bond-erasure`修改，B/C保持一致且训练前确定。如果所有轮都不满足约束，只产生last.pth，不会把不合格权重称为best。

输出包含config.json、history.jsonl、best.pth和last.pth。支持中断续训：

```powershell
python train.py --classes 4 --resume runs/four_seed42/last.pth --seed 42
```

续训需提供与原训练一致的epochs、size、batch-size、epoch-samples、lr等参数，脚本会校验。每轮完成后保存状态；中途打断的一轮从头重跑。

## 评价

```powershell
python train.py --classes 4 --evaluate runs/four_seed42/best.pth --split val
python train.py --classes 4 --evaluate runs/four_seed42/best.pth --split test
python train.py --classes 3 --evaluate runs/three_seed42/best.pth --split val
```

输出metrics.json、per_image.json及权重/数据哈希。四分类先argmax再将1/2合并为箭头、3映射为键，按相同擦除规则评价；不先把概率相加再argmax。报告总体IoU、Dice、precision、recall、残留率、键误删率、背景误擦率和无箭头图误擦率。没有对应样本时指标为null，而不是0。

类型条件化指标：在包含该类型的图上，以合并后的箭头预测计算，另一箭头类型真值像素忽略；这是去除质量指标，不是四分类类型识别IoU。同时保留4×4混淆矩阵分析类型错误。评估不包含OCSR运行或速度基准，这两项需另接相同后端验证。

## 检查与重建

```powershell
python test_core.py
python train.py --classes 4 --smoke-test --cpu
python train.py --classes 4 --smoke-test --smoke-size 512
```

smoke-test运行两个训练样本的前向、反向与评估前向，不保存正式权重、不下载ImageNet权重。数据重建仅在原项目内使用`python build_dataset.py --output 新目录`，需要相邻solid及unified原始数据。已有导出拒绝覆盖；`--refresh-export`只用于有意重新导出已识别的本包数据，原始标注不修改。
