# 模型资产与路径

权重未包含在此次 Git 上传中，不能据此宣称完整推理复现已验证。文件应置于工程根目录 `model/`。

|文件|用途|
|---|---|
|best.pt|YOLO 页面实例分割|
|post_seg_unetpp.pth|候选分子图像筛选|
|classifier_fixed.pkl|13 维质量特征分类器|
|HAT_S_ftv5.pth|选择性超分辨率|
|unified_arrow_4class.pth|统一四分类箭头分割|
|swin_base_char_aux_1m680k.pth|MolScribe 识别|

MolScribe 上游：https://github.com/thomas0809/MolScribe 。下载与使用遵循上游许可证。自训练权重的固定下载地址、版本与归档 DOI 尚待项目作者发布。
