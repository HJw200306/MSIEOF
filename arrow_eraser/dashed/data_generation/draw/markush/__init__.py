"""Markush 结构变异扩展。

子模块:
    pipeline         —— 通用原语 (找原子/键、索引化键 OBB、去背景、按锚点旋转粘贴)。
    sub_variation    —— 取代基异变 (任意原子按概率换成 R/X/Y 等符号)。
    ap_variation     —— 取代基连接点变异 (端点碳 + 单键 → 波浪/折线/虚线/星号 marker)。
    repeat_variation —— 链长 / 重复次数变异 ( (...)_n 形式的左括号 + 右括号 + 下标,
                       三个独立组件; chain / whole / ring1 三种放置模式)。
    pos_variation    —— 取代基位置变异 (环上"任意位置可取代": 在环中心画 'c' 标签,
                       沿某环键外法线方向延伸一根单键 overlay)。
    mol_markush_gen  —— 在 mol_obb_gen 基础上集成上述变异 + YOLO-OBB 数据集 CLI。

所有 *_variation 子模块共用同一套原语 (pipeline):
    find_*_candidates       —— 选目标原子/键
    erase_polygons          —— 去掉被替换的原画素
    paste_with_anchor       —— 旋转 + 平移 + 粘贴 overlay
    compute_pasted_obb      —— 从粘贴结果反推 YOLO-OBB 多边形
    shift_* (atom_coords / atom_boxes / polygons)  —— 画布扩边时同步几何
"""
