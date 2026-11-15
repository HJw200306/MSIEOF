# 统一模型与双模型对比

统一模型在另一台电脑完成训练，随后已运行完整模型比较。入口为 `compare_models.py` / `compare.bat`，原始配置、汇总、逐图指标、预测掩膜、擦除结果和可视化均保存在 `comparisons/`。

## 默认比较对象

- dual_shared：model/misp_weights_no_freeze.pth（VGG16-U-Net，三类）＋arrow_eraser/models/dashed_arrow_model.pth（ResNet101-U-Net++，三类，第47轮）。
- unified_shared：runs/unetpp_r101_20261027_235511/best.pth（统一三类，第45轮）。

两个独立模型都在同一张未擦除图像上预测，将箭头取并集、化学键取并集，化学键优先。统一模型也使用argmax箭头且化学键优先，擦除区域统一置白。结果提供原始分割指标及最终擦除指标。未加入测试集调阈值。

实线预处理复现 split_tools/unet.py 的512方形灰边等比例缩放、/255归一化、softmax后去灰边并线性恢复原尺寸；新版虚线和统一模型用ImageNet归一化、白边补至32倍数、原尺寸全图预测。两者FP32运行，不加载ImageNet初始权重。

## 新版三分类比较流程

本比较程序只使用三分类模型，已经移除旧二分类选项及其保护流程依赖。虚线标签为0背景、1虚线箭头、2化学键；实线和统一模型为0背景、1箭头、2化学键。权重输出通道必须为3，有类别名称元数据时还会核对类别顺序。

虚线权重选择顺序：显式--dashed参数；否则查找arrow_eraser/dashed/runs下唯一的best.pth；若没有训练输出则使用arrow_eraser/models/dashed_arrow_model.pth。这份三分类权重是dashed/README.md指向的归档模型。若发现多个训练输出则要求明确指定，不根据测试成绩或修改时间自动选择。

双模型规则：箭头=(实线预测1 OR 虚线预测1)，化学键=(实线预测2 OR 虚线预测2)，擦除=箭头 AND NOT 化学键。因此虚线模型也参与化学键保护。两个模型均读取原图，原图只在最终融合后擦除。统一模型按同样类别和化学键优先规则处理。

主系统当时的实现（现归档演进为 `utils/pipeline/arrow_sr.py`）已于 2026-10-28 同步为相同的三分类双模型融合规则；比较目录仍是独立实验实现，不依赖主运行入口。

比较范围为完整的箭头预测、掩膜融合与擦除阶段。超分辨率、去噪等共用上游处理不重复运行；输入必须是两套方案共同使用的箭头阶段图像。若采用SR后的数据，GT也必须与该尺寸对齐。

## 使用命令

在 unified 目录执行（或双击compare.bat使用默认test）：

```powershell
& 'D:\miniconda3\envs\msirost\python.exe' compare_models.py --dry-run
& 'D:\miniconda3\envs\msirost\python.exe' compare_models.py --unified runs/unetpp_r101_20261027_235511/best.pth --split test
```

dry-run只核对路径、清单，不加载权重、不推理。runs只有一个best.pth时可自动选中；多个时必须显式指定，避免用测试结果选权重。

可选命令：

```powershell
# 小规模诊断，不能当作完整实验结果
python compare_models.py --split val --limit 4
# 全部验证样本：分别报告实线来源193张、虚线来源20张
python compare_models.py --split val
# 显式使用当前归档的新版虚线三分类模型
python compare_models.py --dashed ../models/dashed_arrow_model.pth
```

--data可指定其他符合相同manifest格式的数据集；--solid、--dashed可换用重训后的独立模型。--device cpu使用CPU，--no-visuals关闭拼图但仍保存去除图片和掩膜。默认两次首图预热不计时。

## 输出

comparisons/时间戳/下保存：

- config.json：权重路径/哈希、数据清单哈希、环境、三分类映射、擦除规则版本、相关代码哈希。
- summary.json：总体、按数据来源汇总的像素指标、参数量和CUDA峰值已分配显存。
- paired.csv：同一图片在各方案下的指标及统一减双模型的差值。
- report.md：简要结果表，不自动宣布胜负。
- 每个方法的cleaned/、predictions/、erase_masks/、per_image.jsonl：清理结果、0/1/2预测、最终擦除二值掩膜、原始计数。
- visuals/：原图、GT彩色标签、各方案去除结果横向拼图。红箭头、绿化学键、灰忽略区域。

计时包含预处理、前向、掩膜融合及擦除，不包含磁盘读写、载入模型、指标计算和拼图；CUDA同步计时。各方案分别加载到GPU并释放，峰值显存不会把两套方案的模型累计在一起。不是纯网络前向耗时，也不是整个MSIEOF的端到端耗时。

## 指标定义与结论范围

分割IoU/Dice按累积混淆矩阵计算。箭头残留率=GT箭头像素中未被擦除的比例；化学键误删率=GT化学键像素中被擦除的比例；背景擦除率=GT背景像素中被擦除的比例。255不参与；分母为0返回null/NA，不伪装为完美成绩。以上是标签区域擦除率，不是OCSR识别准确率，也不是视觉上残余墨迹的独立测量。

test现有197张来自实线数据；来源标签不能视为纯箭头类型，因为该来源也有虚线和历史风格变化图。val包含两种来源但已用于选模，只能作诊断。旧独立模型历史训练集与现有test的交叉情况尚未得到证明。正式宣称能够替代双模型，需要所有模型均未见过的独立真实实线、虚线、混合及无箭头样本。

脚本不调用OCSR；cleaned目录可输入相同识别器进行后续配对评估。本次比较新统一模型与已有独立模型属于系统替代实验，不能单独证明收益来自统一架构：训练数据与增强使用不同。要隔离训练方案影响，应重训独立模型并固定数据、预算、预处理及后处理。

全图预测不会自动缩小或裁剪。显存不足时程序报错，不静默跳过困难图片；输出目录可能留下未完成的逐图文件，只有出现summary.json和report.md才代表全流程结束。
