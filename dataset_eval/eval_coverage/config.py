"""与桌面 Coverage 基准一致的表4参数。"""

EXTRACT_CONF = 0.25
POST_SEG_THRESHOLD = 0.01
# 论文第 4.1.3 节规定的完整提取阈值。
RECALL_THRESHOLD = 0.90
COLOR_AWARE_V_BG_THR = 221
COLOR_AWARE_S_BG_THR = 40
SUBSETS = ("benchmark_D2C-RND", "benchmark_D2C-UNI")
OFFICES = ("cn", "eu", "jp", "kr", "us")
