# 数据发布说明

`UnifiedArrow-Type-Seg-v1.zip` 为本次可定位并核验的数据包，通过 Git LFS 存储；解压脚本会先验证清单中的 SHA-256。压缩包保留相对路径和来源信息，未改变图像、掩膜、样本编号及划分。

每条记录含原图、四分类掩膜和三分类派生掩膜。`release_manifest.csv` 列出压缩包解压后各文件的大小与哈希；`manifest.csv` 列出样本来源、父样本、划分及图像/掩膜路径。原机器来源路径已转换为工程相对路径，并在质量记录中保留原清单哈希。

PJChemPageDS、PatCID-RePage 页面、RealMol-SMILES 及 Page-to-SMILES 真值未在本次可访问目录中找到完整文件，本次未发布。这一状态不等于论文中相关实验未进行。PatCID 原始数据应通过原作者资源获取：https://github.com/DS4SD/PatCID 和 https://zenodo.org/records/10978812 。

本数据包沿用已有工程中的图像和标注，未新授予覆盖所有源图的许可。尤其真实文献图片的后续再分发需遵循其来源条款；未附带第三方原文 PDF。
