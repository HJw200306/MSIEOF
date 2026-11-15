# Arrow Eraser 箭头分割资产

> 生产模型已于 2026-11-03 更新为统一四分类 ResNet-101 U-Net++：`model/unified_arrow_4class.pth`。旧实线/虚线双模型已归档到 `model/legacy_arrow_models/`，不再由主流程加载。更新前后对照见 [MODEL_MIGRATION.md](MODEL_MIGRATION.md)。下文保留的独立模型说明用于历史实验复现。

本目录现已统一保存 MSIEOF 反应箭头去除模块的训练数据、训练代码、模型权重、推理代码和资产核查记录。新增结构如下，原有 `models/`、`runs/` 和 `scripts/` 兼容路径保持不变：

```text
arrow_eraser/
├── solid/                      # 实线及弯曲箭头
│   ├── dataset/                # 已核验的历史 VOC 数据集
│   ├── training_legacy_vgg_unet/ # VGG16 U-Net 训练工程
│   ├── runtime/                # 当前项目使用的实线模型加载代码
│   └── annotation/             # 实线箭头/化学键预标注与人工复核
├── dashed/                     # 虚线箭头
│   ├── dataset/DashedArrow-Seg # 三分类数据集
│   ├── training/               # U-Net++ ResNet-101 训练代码
│   ├── annotation/             # 三分类标注器与历史标注工作区
│   └── data_generation/        # 旧备份中现存的数据生成依赖
├── common/runtime/             # 串联实线和虚线模型的推理流程
└── docs/                       # 资产清单、数据核查和模型方案
```

虚线模型训练入口为 `dashed/training/train_arrow_eraser_v2.py`，实线历史工程入口为 `solid/training_legacy_vgg_unet/train.py`。实线历史数据只含背景与箭头两类，不能直接复现当前三分类实线权重；详情见 `solid/README.md` 和 `docs/asset_inventory.csv`。

虚线三分类标注器可双击根目录的 `启动虚线箭头标注器.bat` 启动。旧备份功能覆盖和缺失源文件说明见 `docs/old_backup_coverage.md`。

实线三分类标注器可双击根目录的 `启动实线箭头化学键标注器.bat` 启动。2011份标注已全部人工审核，并经重复图片整合形成1857张唯一图片的正式 `solid/dataset/SolidArrow-Seg`：训练1467、验证193、测试197，跨划分精确重复为0。

不建议未经统一数据重训就用任一现有模型替代另一模型。当前统一三分类模型（背景、全部箭头、化学键）已经完成训练和对比，双模型仍保留为部署基线；完整结果见 `unified/README.md` 和 `unified/comparisons/`。

## 旧版说明

本目录包含用于化学结构图像中箭头去除的模型和脚本。

## 📁 目录结构

```
arrow_eraser/
├── models/                    # 模型文件
│   ├── dashed_arrow_model.pth    # 虚线箭头模型（UNet++ ResNet101, 260MB）
│   └── solid_arrow_model.pth     # 实线箭头模型（VGG16-UNet, 95MB）
├── scripts/                   # 处理脚本
│   ├── dashed_arrow_basic.py           # 虚线箭头基础处理
│   ├── dashed_arrow_postprocess.py     # 虚线箭头后处理实验（9种策略）
│   ├── solid_arrow_test.py             # 实线箭头单图测试
│   ├── solid_arrow_batch.py            # 实线箭头批量测试
│   └── （模型对比统一使用 unified/compare_models.py）
└── README.md                  # 本文件

## 🎯 模型信息

### 1. 虚线箭头模型（Dashed Arrow Model）

- **文件**: `models/dashed_arrow_model.pth`
- **原始位置**: `C:\Users\Admin\PycharmProjects\msirost\model\bond_arrow_best.pth`
- **架构**: UNet++ with ResNet101
- **类别**: 3类
  - `background` (背景)
  - `dashed_arrow` (虚线箭头)
  - `bond` (化学键)
- **大小**: 260MB
- **输入**: RGB图像（自动padding到32倍数）
- **输出**: 分割掩膜（红色=虚线箭头，绿色=化学键）

### 2. 实线箭头模型（Solid Arrow Model）

- **文件**: `models/solid_arrow_model.pth`
- **原始位置**: `C:\Users\Admin\PycharmProjects\msirost\model\misp_weights_no_freeze.pth`
- **架构**: VGG16-UNet
- **类别**: 3类
  - `_background_` (背景)
  - `arrow` (箭头)
  - `bond` (化学键)
- **大小**: 95MB
- **输入**: RGB图像（512x512）
- **输出**: 分割掩膜（红色=箭头，绿色=化学键）

## 📜 脚本说明

### 虚线箭头处理

#### 1. `dashed_arrow_basic.py` - 基础处理
- **功能**: 使用虚线箭头模型进行基本的箭头去除
- **后处理**: v5策略（键保护混合方法）
- **输出**: 去箭头图片、掩膜可视化、对比图

#### 2. `dashed_arrow_postprocess.py` - 后处理实验 ⭐ 重要
- **功能**: 测试9种不同的后处理策略
- **策略列表**:
  1. `baseline` - 基线（无后处理）
  2. `v1_dilate3` - 膨胀3x3
  3. `v1_dilate5` - 膨胀5x5
  4. `v2_morph` - 形态学(close5+dilate3)
  5. `v3_conf30` - 置信度阈值0.30
  6. **`v3_conf25`** - 置信度阈值0.25 ⭐ 推荐
  7. **`v4_hybrid25`** - 混合方法(conf0.25+dilate3) ⭐ 推荐
  8. `v4_hybrid30` - 混合方法(conf0.30+dilate3)
  9. `v5_bond_aware` - 键保护混合

- **推荐策略**: v3_conf25 或 v4_hybrid25（视觉效果最佳）
- **输出**: 每种策略的处理结果、掩膜、SMILES对比

**用法示例**:
```bash
python dashed_arrow_postprocess.py
```

输出到: `C:\Users\Admin\PycharmProjects\msirost\outputs\improved\`

### 实线箭头处理

#### 3. `solid_arrow_test.py` - 单图测试
- **功能**: 测试实线箭头模型对单张图片的效果
- **输出**: 原图、去箭头图、掩膜可视化

#### 4. `solid_arrow_batch.py` - 批量测试 ⭐
- **功能**: 批量测试多张图片，包含SMILES识别和对比
- **输出**: 每张图片的完整可视化和SMILES对比结果

**用法示例**:
```bash
python solid_arrow_batch.py
```

输出到: `C:\Users\Admin\PycharmProjects\msirost\outputs\solid_arrow_vgg16\`

#### 5. `compare_models.py` - 模型对比
- **功能**: 对比虚线箭头模型和所有箭头模型的效果
- **输出**: 对比可视化和SMILES相似度分析

## 🔧 依赖要求

```python
torch >= 2.0
torchvision
segmentation_models_pytorch
opencv-python
pillow
numpy
pandas
molscribe
rdkit
```

## 📊 使用示例

### 示例1: 虚线箭头后处理实验

```python
cd C:/Users/Admin/PycharmProjects/msirost/arrow_eraser/scripts
python dashed_arrow_postprocess.py
```

查看结果:
- 输出目录: `C:\Users\Admin\PycharmProjects\msirost\outputs\improved\`
- 查看 `postprocess_comparison.csv` 了解各策略效果
- 比较 `v3_conf25.png` 和 `v4_hybrid25.png` 的视觉效果

### 示例2: 实线箭头批量测试

```python
cd C:/Users/Admin/PycharmProjects/msirost/arrow_eraser/scripts
python solid_arrow_batch.py
```

查看结果:
- 输出目录: `C:\Users\Admin\PycharmProjects\msirost\outputs\solid_arrow_vgg16\`
- 每张图片的 `4_comparison.png` 显示（原图|去箭头|掩膜）对比
- 查看 `results_summary.csv` 了解SMILES匹配效果

## 📈 性能对比

### 检测能力对比（以JNP_2024_04_1059-1066_004.png为例）

| 模型 | 检测像素数 | 检测覆盖率 |
|------|-----------|-----------|
| 虚线箭头模型（基线） | 2,369 | 0.122% |
| 虚线箭头模型（v3_conf25） | 3,770 | 0.194% |
| 虚线箭头模型（v4_hybrid25） | 8,902 | 0.457% |
| 实线箭头模型 | 4,488 | 0.231% |

### 推荐策略

1. **虚线箭头去除**: 使用 `v3_conf25` 或 `v4_hybrid25`
   - v3更保守，适合要求精确的场景
   - v4更激进，去除更彻底

2. **实线箭头去除**: 使用VGG16-UNet基线模型
   - 检测能力强
   - 能有效区分箭头和化学键

## ⚠️ 注意事项

1. **只使用三分类虚线模型**：
   - 当前权重为 `models/dashed_arrow_model.pth`
   - 旧虚线二分类权重、训练/验证入口及相关说明已移除

2. **模型路径更新**: 
   - 所有新脚本已更新为使用 `models/` 目录下的模型
   - 如需使用其他位置的模型，修改脚本中的 `MODEL_PATH` 变量

3. **输出目录**: 
   - 默认输出到 `C:\Users\Admin\PycharmProjects\msirost\outputs\`
   - 可以在脚本中修改 `OUTPUT_DIR` 变量

## 📝 更新日志

### 2024-10-19
- ✅ 清理旧文件并整理三分类资产
- ✅ 重新组织目录结构
- ✅ 统一模型命名（dashed_arrow_model.pth, solid_arrow_model.pth）
- ✅ 整理所有处理脚本到 `scripts/` 目录
- ✅ 创建完整的文档说明

## 🔗 相关资源

- MolScribe模型: `D:\workspace\Model\swin_base_char_aux_1m680k.pth`
- 输出结果目录: `C:\Users\Admin\PycharmProjects\msirost\outputs\`
- 测试图片目录: `E:\MSIREOT_DATA\journal\images\`

## 👥 维护信息

如需更新模型或脚本，请保持以下结构：
- 模型文件放在 `models/` 目录
- 处理脚本放在 `scripts/` 目录
- 示例结果放在 `examples/` 目录
- 更新 README.md 文档
