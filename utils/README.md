# `utils` 目录说明

工具代码按职责分为七个子包。新代码应使用完整路径导入；旧的平铺导入由顶层 `utils/__init__.py` 惰性兼容层继续支持。

| 目录 | 职责 |
|---|---|
| `pipeline/` | 正式图像提取、清理、筛选、超分、箭头去除及结构识别链路 |
| `base/` | 图像、化学图、SMILES、可视化和项目路径等共享底层库 |
| `evaluation/` | 正式端到端评测与外部系统基准 |
| `experiments/` | 消融实验和非生产后处理方案 |
| `report/` | PNG、HTML 和论文表格输出 |
| `data_tools/` | 数据下载、筛选和准备 |
| `maintenance/` | 兼容修复与迁移核验 |

推荐以模块形式运行独立入口：

```powershell
python -m utils.pipeline.extract
python -m utils.pipeline.arrow_sr
python -m utils.evaluation.evaluate_page2smiles_full
python -m utils.report.report_interactive
```

规范导入示例：

```python
from utils.pipeline.arrow_sr import ArrowSRProcessor
from utils.base.chemical_graph import convert_graph
from utils.pipeline.resolution_classifier import ResolutionClassifier
```

迁移期内，`from utils.arrow_sr import ArrowSRProcessor` 和拼写有误的历史路径 `utils.resolution_classfier` 仍然可用。新增代码不要继续使用旧路径。

测试统一保存在项目根目录的 `utils_test/`：

```powershell
python -m unittest discover -s utils_test -t . -p "test*.py"
```

VGG16 预训练权重只保留在项目根目录的 `model_data/`。原 `utils/model_data/` 是哈希相同的重复副本，已在本次整理中删除。
