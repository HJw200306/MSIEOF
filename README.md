# MSIEOF

Molecular Structure Image Extraction and Optimization Framework 面向化学文献页面，依次执行结构提取、邻近残片清理、候选筛选、选择性超分辨率、统一四分类箭头去除、MolScribe 识别与 RDKit 规范化。

## 本次发布范围

- 页面提取、优化、结构识别和评测源代码；箭头模型训练与标注工具。
- UnifiedArrow-Type-Seg-v1：4043 条图像记录及对应四分类和三分类掩膜，提供 SHA-256 清单。
- 数据压缩包使用 Git LFS。PJChemPageDS 完整页面、RealMol-SMILES 和 Page-to-SMILES 真值目前未包含在本仓库中；不应把箭头分割数据等同于论文全部数据。
- 不包含模型权重、训练缓存、原文 PDF 或未发表稿件。下载本仓库不能直接重现全部论文结果。

## 获取数据

```powershell
git lfs install
git clone https://github.com/HJw200306/MSIEOF.git
cd MSIEOF
git lfs pull
python scripts/unpack_dataset.py
```

数据位于 `arrow_eraser/four_class/dataset/UnifiedArrow-Type-Seg-v1`。类别为 0 背景、1 实线及连续弯曲箭头、2 虚线箭头、3 化学键、255 忽略。划分为 train 3583、val 213、test 197、val_synthetic 50。独立测试集未覆盖虚线来源，不应据此报告完整虚线泛化性能。

## 环境与验证

Python 3.10，建议为主流程与外部 DECIMER/YoDe 对照分别建立环境。`requirements_actual.txt` 是原机器历史环境快照，含 CUDA 专用版本，不是跨平台一键安装清单。箭头训练的依赖见 `arrow_eraser/four_class/requirements.txt`；标注器依赖见 `markush_annotator/requirements-annotator.txt`。本次核查的运行环境与结果记录在 `docs/VALIDATION.md`。

```powershell
python -m pip install -r arrow_eraser/four_class/requirements.txt
python arrow_eraser/four_class/verify_dataset.py
python arrow_eraser/four_class/test_core.py
```

## 主流程

准备权重后，在项目根目录按顺序执行下列模块。权重文件及用途见 [模型清单](docs/MODELS.md)。主流程的模型由工程相对路径读取；部分历史训练和外部基准脚本仍含原机器绝对路径，使用前需按自己的数据位置配置，不能视为可直接复现的正式入口。

```powershell
python -m utils.pipeline.extract path/to/page.png
python -m utils.pipeline.fragment_cleanup --help
python -m utils.pipeline.post_segment --help
python -m utils.pipeline.arrow_sr --help
python -m utils.pipeline.img2smiles --help
```

完整 Page-to-SMILES 入口和基准说明见 [utils/evaluation/README.md](utils/evaluation/README.md)。自备页面和真值后使用：

```powershell
python -m utils.evaluation.evaluate_page2smiles_full --test-set-root path/to/pages --test-set-jsonl path/to/ground_truth.jsonl --max-pages 1 --output-dir outputs/smoke
```

## 文档与许可

- [数据说明](data/README.md)
- [验证记录](docs/VALIDATION.md)
- [OCSR 文献阅读与开题材料](docs/OCSR_READING.md)
- [Molecules 格式核对](docs/MOLECULES_CHECKLIST.md)

第三方工具、权重和文献页面遵循各自许可；本次未替作者指定新的项目开源许可证，也未把第三方资产重新授权。代码已公开可见，使用、修改和再分发授权需由项目作者另行明确。
