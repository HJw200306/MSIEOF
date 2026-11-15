# PyCharm 运行指南 - 分子图结构标注

本指南说明如何在PyCharm中一步步完成分子图结构标注任务。

---

## 📋 前提条件

1. 已有语义分割标注数据：
   - `E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\json\PageToSMILES\*.json`
   - `E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal\images\PageToSMILES\*.png`

2. 已安装依赖：
   ```bash
   pip install -r markush_annotator/requirements-annotator.txt
   ```

---

## 🚀 三步走流程

### **步骤1：预处理 - 生成标注器数据**

#### 操作步骤：
1. 在PyCharm中打开 `markush_annotator/prepare_seg_annotation.py`
2. 检查脚本顶部的配置（第33-45行）：
   ```python
   # ============ 配置区域 - 可以根据需要修改 ============
   SEG_ROOT = Path(r"E:\MSIREOT_DATA\DSeg\Semantic_Segmentation\patentAndjournal")
   DEVICE = "auto"  # 优先使用 CUDA，不可用时自动回退 CPU
   ```
3. 右键点击编辑器 → **Run 'prepare_seg_annotation'**
4. 观察控制台输出，等待处理完成

#### 输出内容：
- `data/seg_page2smiles/images/` - 裁剪后的分子图像
- `data/seg_page2smiles/annotations/` - MolScribe预标注
- `data/seg_page2smiles/manifest.jsonl` - 索引文件
- `data/seg_page2smiles/mapping.json` - 映射关系

#### 预期日志：
```
======================================================================
步骤1: 从分割标注生成标注器数据
======================================================================

检查输入目录...
  JSON目录: E:\MSIREOT_DATA\DSeg\...\json\PageToSMILES
  图像目录: E:\MSIREOT_DATA\DSeg\...\images\PageToSMILES

创建输出目录...
  输出目录: <项目目录>\markush_annotator\data\seg_page2smiles

初始化 MolScribe (设备: cuda)...
  ✓ MolScribe 初始化成功

找到 XX 个标注文件

处理标注: 100%|██████████| XX/XX [XX:XX<00:00, X.XXit/s]

======================================================================
✓ 步骤1完成！
======================================================================
  输入文件:     XX 个
  分子实例:     XXX 个
  输出目录:     ...\data\seg_page2smiles
======================================================================
```

#### 故障排除：
- **GPU内存不足**：修改 `DEVICE = "cpu"`
- **找不到文件**：检查 `SEG_ROOT` 路径是否正确

---

### **步骤2：人工标注 - 精修图结构**

#### 操作步骤：
1. 在PyCharm中打开 `markush_annotator/run.py`
2. 默认会自动使用项目内的 `markush_annotator/data/seg_page2smiles`；无需修改电脑用户名或绝对路径。
3. 右键点击编辑器 → **Run 'run'**
4. 浏览器会自动打开标注界面（http://127.0.0.1:8765）

#### 标注界面操作：
1. **查看预标注**：
   - 左侧列表选择一个分子实例
   - 右侧显示MolScribe预标注的图结构

2. **精修原子**：
   - 点击原子可以修改类型（C/N/O/R基团等）
   - 拖动原子调整位置
   - 右键删除错误的原子
   - 双击空白处添加新原子

3. **精修化学键**：
   - 点击化学键修改类型（单键/双键/三键/芳香键）
   - 右键删除错误的键
   - 点击两个原子连接新键

4. **保存标注**：
   - 点击界面上的 "Save" 按钮
   - 标注保存到 `data/seg_page2smiles/corrected_annotations/`

5. **继续下一个**：
   - 左侧列表选择下一个实例
   - 重复上述步骤

#### 完成标注：
- 标注完所有实例后，关闭浏览器
- 在PyCharm中停止 `run.py`（点击红色方块停止按钮）

#### 预期日志：
```
======================================================================
步骤2: Markush Visual Annotator - 图结构标注
======================================================================
  服务器地址: http://127.0.0.1:8765
  数据目录: <项目目录>\markush_annotator\data\seg_page2smiles
======================================================================

标注说明:
  1. 在浏览器中打开上述URL
  2. 查看MolScribe预标注结果
  3. 精修原子位置和化学键
  4. 保存标注
  5. 完成所有标注后，关闭服务器
  6. 运行 export_seg_results.py 导出结果
======================================================================

INFO:     Opening Markush Visual Annotator: http://127.0.0.1:8765
INFO:     Started server process [XXXX]
INFO:     Uvicorn running on http://127.0.0.1:8765
```

---

### **步骤3：导出结果 - 回写到原始格式**

#### 操作步骤：
1. 确保已完成步骤2的标注并保存
2. 在PyCharm中打开 `markush_annotator/export_seg_results.py`
3. 右键点击编辑器 → **Run 'export_seg_results'**
4. 观察控制台输出，等待导出完成

#### 输出内容：
- `E:\MSIREOT_DATA\DSeg\...\json\PageToSMILES_approved\*.json` - 扩展后的标注文件

#### 输出格式：
每个分子的shape新增以下字段：
```json
{
  "label": "mol",
  "points": [[x, y], ...],
  "atoms": [
    {"x": 0.5, "y": 0.3, "symbol": "C"},
    {"x": 0.6, "y": 0.4, "symbol": "N"}
  ],
  "bonds": [
    [0, 1, 1],  // [起始原子, 结束原子, 键类型]
    [1, 2, 2]   // 1=单键, 2=双键, 3=三键, 4=芳香键
  ],
  "smiles": "c1ccccc1N"  // 从图结构生成
}
```

#### 预期日志：
```
======================================================================
步骤3: 图结构标注导出工具
======================================================================

检查输入目录...
  精修标注: ...\data\seg_page2smiles\corrected_annotations
  映射文件: ...\data\seg_page2smiles\mapping.json
  原始标注: E:\MSIREOT_DATA\...\json\PageToSMILES
  找到 XXX 个精修标注文件

加载映射文件: ...\mapping.json
  找到 XXX 个实例映射

收集精修后的标注...
  涉及 XX 个原始文件

回写标注到原始格式...
  CN102863967B_038.json: 5 updated, 0 skipped
  CN102947262B_118.json: 8 updated, 0 skipped
  ...

======================================================================
✓ 步骤3完成！
======================================================================
  输入目录:     ...\corrected_annotations
  输出目录:     E:\MSIREOT_DATA\...\json\PageToSMILES_approved
  已更新:       XXX 个分子
  已跳过:       0 个分子
======================================================================

输出格式说明：
  每个shape新增字段：
    - atoms:  精修后的原子列表
    - bonds:  精修后的化学键列表
    - smiles: 从图结构生成的SMILES (使用RDKit)

数据标注流程全部完成！
======================================================================
```

---

## 📊 数据流图示

```
步骤1: prepare_seg_annotation.py
┌─────────────────────────────────────┐
│ 分割标注 (LabelMe)                      │
│ - json/PageToSMILES/*.json          │
│ - images/PageToSMILES/*.png         │
└──────────────┬──────────────────────┘
               │ 裁剪 + MolScribe预标注
               ▼
┌─────────────────────────────────────┐
│ 标注器数据                               │
│ - data/seg_page2smiles/images/      │
│ - data/seg_page2smiles/annotations/ │
│ - data/seg_page2smiles/mapping.json │
└──────────────┬──────────────────────┘
               │
步骤2: run.py  │ 人工精修
               ▼
┌─────────────────────────────────────┐
│ 精修后标注                               │
│ - corrected_annotations/*.json      │
└──────────────┬──────────────────────┘
               │
步骤3: export_seg_results.py
               ▼
┌─────────────────────────────────────┐
│ 最终输出 (扩展LabelMe)                    │
│ - json/PageToSMILES_approved/*.json │
│   (包含 atoms + bonds + smiles)       │
└─────────────────────────────────────┘
```

---

## ⚠️ 注意事项

1. **备份数据**：
   - 在运行步骤3前，备份原始 `json/PageToSMILES/` 目录
   - 导出结果保存在独立的 `PageToSMILES_approved/` 目录

2. **GPU显存**：
   - 如果步骤1报CUDA错误，修改 `DEVICE = "cpu"`
   - CPU模式会慢一些，但更稳定

3. **标注质量**：
   - MolScribe预标注可能有错误
   - 仔细检查每个分子的原子和化学键
   - 尤其注意R基团、超原子、立体键

4. **映射关系**：
   - `mapping.json` 保存了实例ID到原始标注的映射
   - 不要手动修改或删除此文件

---

## 🔧 故障排除

### 问题1：步骤1找不到JSON文件
**现象**：`[WARN] 没有找到任何JSON文件！`

**解决**：
- 检查 `SEG_ROOT` 路径是否正确
- 确认 `json/PageToSMILES/` 目录下有 `.json` 文件

### 问题2：步骤2浏览器没有自动打开
**现象**：服务器启动但浏览器未打开

**解决**：
- 手动在浏览器中访问 `http://127.0.0.1:8765`
- 或修改脚本运行配置添加 `--no-browser` 参数

### 问题3：步骤3没有精修标注
**现象**：`[WARN] 没有找到任何精修标注文件！`

**解决**：
- 确保步骤2中点击了"Save"按钮保存标注
- 检查 `data/seg_page2smiles/corrected_annotations/` 目录是否有文件

### 问题4：MolScribe初始化失败
**现象**：`✗ MolScribe 初始化失败`

**解决**：
- 检查是否安装了 `molscribe` 包
- 尝试 `DEVICE = "cpu"` 模式
- 确认模型权重文件存在：`model/swin_base_char_aux_1m680k.pth`

---

## 📝 快速检查清单

**开始前：**
- [ ] 分割标注数据已准备好
- [ ] 依赖包已安装
- [ ] GPU可用（可选）

**步骤1后：**
- [ ] `data/seg_page2smiles/images/` 有裁剪图像
- [ ] `data/seg_page2smiles/annotations/` 有预标注
- [ ] `data/seg_page2smiles/mapping.json` 已生成

**步骤2后：**
- [ ] 所有分子已精修完成
- [ ] `corrected_annotations/` 有保存的标注
- [ ] 服务器已关闭

**步骤3后：**
- [ ] `json/PageToSMILES_approved/` 有输出文件
- [ ] 输出JSON包含 atoms、bonds、smiles 字段
- [ ] 统计数量与步骤1一致

---

## 🎯 完成！

全部步骤完成后，你将得到：
- 原始分割标注 + 图结构标注（atoms + bonds）
- 每个分子都有对应的SMILES表示
- 数据格式为扩展的LabelMe格式，兼容原有工具

可以用于：
- 训练分子识别模型
- 评估MolScribe性能
- 构建化学结构数据集
