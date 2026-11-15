# Page-to-SMILES 评测入口

## 正式入口

- `evaluate_page2smiles.py`：YOLO 检测、裁剪和 MolScribe 识别的 baseline。
- `evaluate_page2smiles_full.py`：包含候选筛选、残片清理、质量分流、超分和统一四分类箭头去除的完整流程。
- `build_pagesmiles_jsonl.py`：从 approved 标注构建评测 JSONL 和标签表。
- `run_quick_test.py`：只跑前 10 页的快速环境检查。
- `benchmarks/`：DECIMER、YODE 和论文表 B 的外部系统对照。

## 运行

在项目根目录执行：

```powershell
python -m utils.evaluation.run_quick_test
python -m utils.evaluation.evaluate_page2smiles --max-pages 10
python -m utils.evaluation.evaluate_page2smiles_full --max-pages 10
python -m utils.evaluation.benchmarks.table_b_detection
python -m utils.evaluation.benchmarks.table_b_end2end
```

正式评测前应固定数据清单、权重哈希、IoU 阈值和随机种子。快速测试只验证链路可运行，不应作为论文结果。
