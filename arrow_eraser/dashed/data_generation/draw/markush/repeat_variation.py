"""重复次数变异 (Repeat Count Variation)。

绘制 (...)_n 形式的括号 + 下标 overlay, 表达分子链段的可变重复次数。
左右括号与下标被当作三个独立但相互关联的组件 (class: left_bracket / right_bracket
/ bracket_subscript), 各自有自己的 OBB, 但几何位置由同一组锚点计算得出, 保证:
    - 左括号在左, 右括号在右
    - 两个括号平行
    - 平行方向 ⟂ 两括号中心连线 (即不会出现"扇形张开"的怪异平面)
    - 下标默认在右下, 少量情况在其它三个角

候选选择 (find_chain_repeat_candidates):
    chain 模式 —— 链上 *两根* acyclic 单键, 左/右括号分别贴在两键中点, 中间夹一段
                 acyclic 原子, 视为可重复单元。两个键的方向不必平行 (用户分子的链
                 通常是 zig-zag 折线), 我们靠 "把括号统一旋到 ⟂ 中心连线" 来保证
                 两个括号本身平行。

whole 模式 (find_whole_molecule_repeat_candidate):
    括号紧贴整个分子的最左 / 最右两侧, 不依附具体键; 通常配方括号; 适用于做
    "整张分子可重复 N 次" 的标注样式。

实现思路与 ap_variation 一致:
    1. 用 draw_bracket_pair / draw_bracket_subscript 渲染独立 PIL 组件;
    2. 计算每个组件应放置的"世界锚点 (画布像素坐标)" + "目标方向";
    3. rotate_pil_around_anchor + paste_with_anchor 完成对齐 + 复合;
    4. 用 compute_pasted_obb 得到每个组件的紧致 OBB, 加入 placed_overlays;
    5. 画布扩边时同步平移所有外部几何记录, 与 ap_variation 复用同一套 shift_*.

类别名 (写入 placed_overlays 第二项):
    left_bracket
    right_bracket
    bracket_subscript
"""

import math
import random

import numpy as np
from rdkit import Chem

from PIL import Image as PILImage

from draw.bracket_draw import (
    draw_bracket_pair,
    draw_bracket_subscript,
    generate_subscript_content,
)
from draw.markush.pipeline import (
    paste_with_anchor,
    rotate_pil_around_anchor,
    compute_pasted_obb,
    dist,
    shift_atom_pixel_coords,
    shift_axis_boxes,
    shift_polygons_dict,
    shift_polygon_list,
    check_candidate_overlap,
    estimate_overlay_obb,
    _build_existing_polys,
)

# 括号安全渲染粗细钳位 (与 ap_variation._RENDER_THICKNESS_SAFE 一致)
_BRACKET_THICKNESS_SAFE_MAX = 8.0
_BRACKET_THICKNESS_SAFE_MIN = 1.5

# 可变高度: 仅对水平括号生效 (角度容差)
_HORIZONTAL_ANGLE_TOL_DEG = 15.0
# 二分搜索步数上限
_HEIGHT_BINARY_SEARCH_STEPS = 6


# ============================================================
# 候选搜索
# ============================================================
def _is_bracket_eligible_bond(bond):
    """该键可作为左 / 右括号的"贴键中点"对象吗?

    要求:
        - 单键
        - 不在环里 (避免括号穿环)
        - 当前没有立体方向 (BondDir == NONE)
        - 两端原子均非芳香
        - 两端原子均非 stereo center (CHI_UNSPECIFIED) —— 与 AP 同理:
          挡住"find 时 BondDir 还是 NONE, set_liti 后会被自动绘成楔形键"的情况.
        - 两端原子的 degree 都 >= 2 (避免落到链端 — 那里是 AP 的领地, 括号容易撞 AP)
    """
    if bond.GetBondType() != Chem.BondType.SINGLE:
        return False
    if bond.IsInRing():
        return False
    if bond.GetBondDir() != Chem.BondDir.NONE:
        return False
    a, b = bond.GetBeginAtom(), bond.GetEndAtom()
    if a.GetIsAromatic() or b.GetIsAromatic():
        return False
    if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return False
    if b.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
        return False
    if a.GetDegree() < 2 or b.GetDegree() < 2:
        return False
    return True


def _find_simple_path_between_bonds(mol, b_left, b_right, max_inner_atoms):
    """返回原子路径 [l_outer, l_inner, ...inner..., r_inner, r_outer] 或 None.

    inner 原子必须非芳香, 且不在环内 —— 否则括号会"跨越环结构", 不符合
    "重复某段链" 的语义。允许 inner 为空 (两根键共用一个原子)。
    """
    l_atoms = (b_left.GetBeginAtomIdx(), b_left.GetEndAtomIdx())
    r_atoms = (b_right.GetBeginAtomIdx(), b_right.GetEndAtomIdx())

    # 邻接键: 共享一个原子 -> 直接构成路径, 重复单元就是那个共享原子
    shared = set(l_atoms) & set(r_atoms)
    if shared:
        s = next(iter(shared))
        l_outer = l_atoms[0] if l_atoms[1] == s else l_atoms[1]
        r_outer = r_atoms[0] if r_atoms[1] == s else r_atoms[1]
        return [l_outer, s, r_outer]

    best = None
    for l_outer, l_inner in ((l_atoms[0], l_atoms[1]), (l_atoms[1], l_atoms[0])):
        for r_inner, r_outer in ((r_atoms[0], r_atoms[1]), (r_atoms[1], r_atoms[0])):
            # BFS 从 l_inner 到 r_inner, 禁止回到 l_outer (不能从 b_left 的另一端绕回去)
            # 也禁止穿过 r_outer (不能借道 b_right 的另一端绕路)
            forbidden = {l_outer, r_outer}
            visited = {l_inner}
            queue = [(l_inner, [l_outer, l_inner])]
            while queue:
                node, path = queue.pop(0)
                inner_count = len(path) - 2  # path 含 l_outer + l_inner 已 = 2
                if inner_count > max_inner_atoms:
                    continue
                if node == r_inner:
                    full = path + [r_outer]
                    if best is None or len(full) < len(best):
                        best = full
                    break
                atom = mol.GetAtomWithIdx(node)
                for nb in atom.GetNeighbors():
                    nbi = nb.GetIdx()
                    if nbi in visited or nbi in forbidden:
                        continue
                    # inner 中间结点必须非芳香、非环; 但 r_inner 自身可以放宽
                    # (它会被 _is_bracket_eligible_bond 的两端非芳香约束兜住)
                    if nbi != r_inner:
                        if nb.GetIsAromatic() or nb.IsInRing():
                            continue
                    visited.add(nbi)
                    queue.append((nbi, path + [nbi]))
    return best


def find_chain_repeat_candidates(mol, *, max_inner_atoms=3, max_candidates=12):
    """返回 [{left_bond_idx, right_bond_idx, path_atoms, inner_atoms}].

    path_atoms = [l_outer, l_inner, ...inner_chain..., r_inner, r_outer]
    inner_atoms = path_atoms[1:-1]  (左右括号之间的原子, 即可重复单元)
    """
    elig = [b for b in mol.GetBonds() if _is_bracket_eligible_bond(b)]
    if len(elig) < 1:
        return []

    cands = []
    for i, b_left in enumerate(elig):
        # 自配对: 共用同一根键 (即左右括号都在同一根键上) 没有意义, 跳过
        for b_right in elig[i + 1:]:
            path = _find_simple_path_between_bonds(mol, b_left, b_right,
                                                    max_inner_atoms)
            if path is None:
                continue
            cands.append({
                'left_bond_idx': b_left.GetIdx(),
                'right_bond_idx': b_right.GetIdx(),
                'path_atoms': path,
                'inner_atoms': path[1:-1],
            })
            if len(cands) >= max_candidates:
                return cands
    return cands


def find_whole_molecule_repeat_candidate(mol):
    """整体重复 (整张分子被一对方括号包住). 只生一个候选, 由调用方决定要不要落."""
    n_atoms = mol.GetNumAtoms()
    if n_atoms < 2:
        return None
    return {
        'mode': 'whole',
        'atoms': list(range(n_atoms)),
    }


def find_ring_single_atom_repeat_candidates(mol, *, min_ring_size=5, max_ring_size=7):
    """返回 [{'atom_idx', 'on_bond_idx', 'on_bond_other', 'ring_atoms'}].

    "环单原子" 模式: 选环上某个原子 A, 它的两个环邻居 B/C 都没有外部取代基 (此时
    上面 A 的环结构干净, 适合套括号). 左括号会贴在 bond(A, B) 的中点, 右括号
    "悬空" 在 A 的另一侧 (对称位置), 用于保证两括号平行 —— 即便 A 在环上几何
    并不"位于 B 与 C 中间".

    要求:
        - A 有且仅有 2 个环邻居 B/C (即 A 不是稠环交叉点)
        - A 没有非环邻居 (否则取代基会撞括号)
        - B/C 没有非环邻居 (清出"夹层"空间)
        - 环大小 ∈ [min_ring_size, max_ring_size]
    """
    ri = mol.GetRingInfo()
    if not ri.NumRings():
        return []
    atom_rings = ri.AtomRings()
    out = []
    for ring_atoms in atom_rings:
        if not (min_ring_size <= len(ring_atoms) <= max_ring_size):
            continue
        ring_set = set(ring_atoms)
        for ai in ring_atoms:
            atom = mol.GetAtomWithIdx(ai)
            if atom.GetAtomicNum() == 0:
                continue
            ring_nbs = [nb.GetIdx() for nb in atom.GetNeighbors()
                        if nb.GetIdx() in ring_set]
            non_ring_nbs = [nb.GetIdx() for nb in atom.GetNeighbors()
                            if nb.GetIdx() not in ring_set]
            if len(ring_nbs) != 2 or non_ring_nbs:
                continue
            # 检查两侧邻居本身也没有外部取代基
            b1 = mol.GetAtomWithIdx(ring_nbs[0])
            b2 = mol.GetAtomWithIdx(ring_nbs[1])
            if _has_outer_ring_neighbor(b1, ring_set) or \
               _has_outer_ring_neighbor(b2, ring_set):
                continue
            # 随机挑一根 ring bond 当 "左括号定位键"
            chosen_nb = random.choice(ring_nbs)
            bond = mol.GetBondBetweenAtoms(ai, chosen_nb)
            if bond is None:
                continue
            out.append({
                'mode': 'ring1',
                'atom_idx': ai,
                'on_bond_idx': bond.GetIdx(),
                'on_bond_other': chosen_nb,
                'ring_atoms': list(ring_atoms),
            })
    return out


def _has_outer_ring_neighbor(atom, ring_set):
    for nb in atom.GetNeighbors():
        if nb.GetIdx() not in ring_set:
            return True
    return False


# ============================================================
# 几何工具
# ============================================================
def _bond_midpoint(mol, bond_idx, atom_coords):
    b = mol.GetBondWithIdx(bond_idx)
    a = atom_coords.get(b.GetBeginAtomIdx())
    c = atom_coords.get(b.GetEndAtomIdx())
    if a is None or c is None:
        return None
    return ((a[0] + c[0]) / 2.0, (a[1] + c[1]) / 2.0)


def _avg_bond_pixel_length(mol, bond_idxs, atom_coords, default=20.0):
    """指定一组键, 求其平均像素长度 (RDKit 渲染后的真实长度)."""
    total = 0.0
    cnt = 0
    for bi in bond_idxs:
        b = mol.GetBondWithIdx(bi)
        pa = atom_coords.get(b.GetBeginAtomIdx())
        pb = atom_coords.get(b.GetEndAtomIdx())
        if pa is None or pb is None:
            continue
        total += math.hypot(pa[0] - pb[0], pa[1] - pb[1])
        cnt += 1
    if cnt == 0:
        return default
    return total / cnt


def _perp_distance(pt, line_a, line_b):
    """点 pt 到直线 line_a-line_b 的垂直距离."""
    ax, ay = line_a
    bx, by = line_b
    dx, dy = bx - ax, by - ay
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return math.hypot(pt[0] - ax, pt[1] - ay)
    # cross product / L
    return abs((pt[0] - ax) * dy - (pt[1] - ay) * dx) / L


def _estimate_chain_bracket_height(inner_atoms, atom_coords, lmid, rmid,
                                    local_bond_len):
    """根据 inner_atoms 偏离 left-right 中心连线的最大垂直距离, 估括号需要的高度.

    local_bond_len: 这一对括号附近"真实 RDKit 键长" (像素), 作为度量基准.
                    所有高度上下限都用它来表达, 避免用 font_size 这类与键长无关
                    的指标导致单原子括号过高 (实测会出现 height = 3× bond_len 的
                    情况, 视觉上像 "整张图被一根长棍夹住")。

    高度规则:
        - 括号内没有 inner 原子 (两根键共用一原子) -> height = bond_len (刚好套住那个原子)
        - 只有 1 个 inner 原子 -> height ≤ bond_len * 1.2 (用户要求: 单原子时不超过键长)
        - 多个 inner 原子 -> height ∈ [bond_len * 0.6, bond_len * 2.2],
                            基础值 = 2 * max_perp + 0.3 * bond_len (留出 padding)
    """
    bl = max(local_bond_len, 4.0)
    if not inner_atoms:
        return bl
    max_perp = 0.0
    for a_idx in inner_atoms:
        p = atom_coords.get(a_idx)
        if p is None:
            continue
        d = _perp_distance(p, lmid, rmid)
        if d > max_perp:
            max_perp = d
    if len(inner_atoms) <= 1:
        # 单原子: 严格上限 = bond_len * 1.2 (用户明确要求)
        base = max_perp * 2.0 + 0.3 * bl
        return max(min(base, bl * 1.2), bl * 0.6)
    base = 2.0 * max_perp + 0.3 * bl
    return max(min(base, bl * 2.2), bl * 0.6)


def _screen_angle_rad(p_from, p_to):
    return math.atan2(p_to[1] - p_from[1], p_to[0] - p_from[0])


def _rotate_offset(dx, dy, angle_rad):
    """把 (dx, dy) 在屏幕坐标系下绕原点逆时针旋转 angle_rad 角度.
    屏幕坐标 Y 向下 -> 数学上等价于绕原点顺时针旋转 |angle|.
    我们用标准旋转矩阵, 由调用方按"屏幕坐标 0° 朝右, +y 朝下, 角度的正方向 = 屏幕
    顺时针" 来理解传入的 angle_rad."""
    cs = math.cos(angle_rad)
    sn = math.sin(angle_rad)
    return (dx * cs - dy * sn, dx * sn + dy * cs)


# ============================================================
# 主入口
# ============================================================
def draw_repeat_brackets(cv_img, mol, *,
                          chain_candidates,
                          whole_candidate,
                          ring1_candidates=None,
                          atom_coords, atom_boxes, bond_obbs_indexed,
                          placed_overlays,
                          bond_line_width, svg_scale, font_size_px,
                          chain_prob=0.7, whole_prob=0.20, ring1_prob=0.20,
                          max_count=3,
                          subscript_prob=0.85,
                          guaranteed_first=False):
    """主入口. 与 ap_variation.draw_attachment_points 同款签名风格.

    Args:
        cv_img: BGR np.ndarray (会被原地修改 / 扩边).
        mol: RDKit Mol.
        chain_candidates: find_chain_repeat_candidates 返回的列表.
        whole_candidate: find_whole_molecule_repeat_candidate 返回的 dict 或 None.
        atom_coords, atom_boxes, bond_obbs_indexed, placed_overlays: 与 ap_variation 共享,
            画布扩张时会被同步 shift; 新加的左/右括号 + 下标 OBB 会 append 进
            placed_overlays.
        bond_line_width, svg_scale, font_size_px: 用于估计括号粗细 / 高度.
        chain_prob: 每个 chain 候选独立选中概率.
        whole_prob: 整体方括号的概率 (与 chain 是 OR 关系; 整体若画就只画一次).
        max_count: 全图最多放几对括号 (一对 = 左 + 右, 加可选下标).
        subscript_prob: 出现下标的概率.
        guaranteed_first: 若为 True, 首个候选以 100% 概率尝试, 成功后后续候选
            概率衰减为 base_prob * 0.55.

    Returns:
        (new_cv_img, reserved_atoms set[int], placed_count int).
        reserved_atoms = 落在某个括号"夹层" 里的原子, 让上层 sub_variation 知道
            不要再把这些原子换成 R (容易跟下标 / 括号本身视觉冲突)。
    """
    reserved_atoms = set()
    placed_count = 0

    # 跟踪已占用的键, 防止不同 bracket pair 贴到同一根键上.
    used_bond_ids = set()

    # 构建候选池: chain + ring1 混合打乱, whole 放最后 (整分子括号视觉冲突高)
    pool = []
    if chain_candidates:
        for c in chain_candidates:
            pool.append(('chain', c))
    if ring1_candidates:
        for c in ring1_candidates:
            pool.append(('ring1', c))
    random.shuffle(pool)
    if whole_candidate is not None:
        pool.insert(random.randint(0, len(pool)), ('whole', whole_candidate))

    def _effective_prob(base_prob):
        if not guaranteed_first:
            return base_prob
        if placed_count == 0:
            return 1.0
        return base_prob * 0.55

    for kind, payload in pool:
        if placed_count >= max_count:
            break

        # 键冲突检查
        if kind == 'chain':
            if payload['left_bond_idx'] in used_bond_ids or \
               payload['right_bond_idx'] in used_bond_ids:
                continue
            prob = _effective_prob(chain_prob)
        elif kind == 'ring1':
            if payload['on_bond_idx'] in used_bond_ids:
                continue
            prob = _effective_prob(ring1_prob)
        else:  # whole
            prob = _effective_prob(whole_prob)

        if random.random() >= prob:
            continue

        # 尝试放置
        if kind == 'chain':
            res = _place_chain_brackets(
                cv_img, mol, payload,
                atom_coords=atom_coords, atom_boxes=atom_boxes,
                bond_obbs_indexed=bond_obbs_indexed,
                placed_overlays=placed_overlays,
                bond_line_width=bond_line_width, svg_scale=svg_scale,
                font_size_px=font_size_px,
                subscript_prob=subscript_prob)
            if res is not None:
                cv_img, used_atoms = res
                reserved_atoms |= used_atoms
                placed_count += 1
                used_bond_ids.add(payload['left_bond_idx'])
                used_bond_ids.add(payload['right_bond_idx'])
        elif kind == 'ring1':
            res = _place_ring1_brackets(
                cv_img, mol, payload,
                atom_coords=atom_coords, atom_boxes=atom_boxes,
                bond_obbs_indexed=bond_obbs_indexed,
                placed_overlays=placed_overlays,
                bond_line_width=bond_line_width, svg_scale=svg_scale,
                font_size_px=font_size_px,
                subscript_prob=subscript_prob)
            if res is not None:
                cv_img, used_atoms = res
                reserved_atoms |= used_atoms
                placed_count += 1
                used_bond_ids.add(payload['on_bond_idx'])
        else:  # whole
            res = _place_whole_brackets(
                cv_img, mol, payload,
                atom_coords=atom_coords, atom_boxes=atom_boxes,
                bond_obbs_indexed=bond_obbs_indexed,
                placed_overlays=placed_overlays,
                bond_line_width=bond_line_width, svg_scale=svg_scale,
                font_size_px=font_size_px,
                subscript_prob=subscript_prob)
            if res is not None:
                cv_img, used_atoms = res
                reserved_atoms |= used_atoms
                placed_count += 1

    return cv_img, reserved_atoms, placed_count


# ============================================================
# chain 模式实现
# ============================================================
def _place_chain_brackets(cv_img, mol, cand, *,
                           atom_coords, atom_boxes, bond_obbs_indexed,
                           placed_overlays,
                           bond_line_width, svg_scale, font_size_px,
                           subscript_prob):
    lmid = _bond_midpoint(mol, cand['left_bond_idx'], atom_coords)
    rmid = _bond_midpoint(mol, cand['right_bond_idx'], atom_coords)
    if lmid is None or rmid is None:
        return None

    # 确保左括号在视觉左侧 (按屏幕 +x 排序)，防止旋转导致括号反向。
    if lmid[0] > rmid[0] or (abs(lmid[0] - rmid[0]) < 1.0 and lmid[1] > rmid[1]):
        lmid, rmid = rmid, lmid

    bracket_axis_len = dist(lmid, rmid)
    if bracket_axis_len < 4.0:
        return None

    bond_pixel_width = max(0.8, bond_line_width * svg_scale)
    # 用两根被括号选中的真实键长平均值, 作为度量基准
    local_bond_len = _avg_bond_pixel_length(
        mol, [cand['left_bond_idx'], cand['right_bond_idx']], atom_coords)
    height = _estimate_chain_bracket_height(
        cand['inner_atoms'], atom_coords, lmid, rmid, local_bond_len)

    return _render_and_paste_bracket_pair(
        cv_img, lmid, rmid, height,
        atom_coords=atom_coords, atom_boxes=atom_boxes,
        bond_obbs_indexed=bond_obbs_indexed,
        placed_overlays=placed_overlays,
        bond_pixel_width=bond_pixel_width, svg_scale=svg_scale,
        font_size_px=font_size_px,
        subscript_prob=subscript_prob,
        bracket_type=random.choice(['square', 'paren']),
        reserved_atom_pool=cand['inner_atoms'],
    )


# ============================================================
# whole 模式实现
# ============================================================
def _place_whole_brackets(cv_img, mol, cand, *,
                           atom_coords, atom_boxes, bond_obbs_indexed,
                           placed_overlays,
                           bond_line_width, svg_scale, font_size_px,
                           subscript_prob):
    atoms = cand['atoms']
    pts = [atom_coords[i] for i in atoms if i in atom_coords]
    if len(pts) < 2:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    mol_w, mol_h = x_max - x_min, y_max - y_min
    margin_x = max(font_size_px * 0.8, mol_w * 0.08, 6.0)

    # 括号始终左右放置, 不再支持上下放置 (上下放置会与左右括号判断冲突)
    lmid = (x_min - margin_x, (y_min + y_max) / 2)
    rmid = (x_max + margin_x, (y_min + y_max) / 2)
    height = max(max(mol_w, mol_h) * 1.15, font_size_px * 2.0)

    bond_pixel_width = max(0.8, bond_line_width * svg_scale)
    return _render_and_paste_bracket_pair(
        cv_img, lmid, rmid, height,
        atom_coords=atom_coords, atom_boxes=atom_boxes,
        bond_obbs_indexed=bond_obbs_indexed,
        placed_overlays=placed_overlays,
        bond_pixel_width=bond_pixel_width, svg_scale=svg_scale,
        font_size_px=font_size_px,
        subscript_prob=subscript_prob,
        bracket_type='square',  # 全分子括号
        reserved_atom_pool=atoms,
    )


# ============================================================
# ring1 模式实现 (单环原子, 一括号贴键, 另一括号悬空)
# ============================================================
def _place_ring1_brackets(cv_img, mol, cand, *,
                           atom_coords, atom_boxes, bond_obbs_indexed,
                           placed_overlays,
                           bond_line_width, svg_scale, font_size_px,
                           subscript_prob):
    """左括号: 贴在 bond(atom_idx, on_bond_other) 中点;
    右括号: 悬空, 位于 atom_idx 沿 (atom_idx − left_mid) 方向继续走相同距离的对侧.
    两个括号天然平行, 因为它们的本地坐标系都用同一个 angle_rad (lmid→rmid 方向).
    """
    a_idx = cand['atom_idx']
    other_idx = cand['on_bond_other']
    if a_idx not in atom_coords or other_idx not in atom_coords:
        return None

    a_xy = atom_coords[a_idx]
    other_xy = atom_coords[other_idx]
    # 左括号: bond 中点
    lmid = ((a_xy[0] + other_xy[0]) / 2.0, (a_xy[1] + other_xy[1]) / 2.0)
    # 右括号: 关于 a_xy 对称
    rmid = (2 * a_xy[0] - lmid[0], 2 * a_xy[1] - lmid[1])

    # 确保左括号在视觉左侧 (按屏幕 +x 排序)
    if lmid[0] > rmid[0] or (abs(lmid[0] - rmid[0]) < 1.0 and lmid[1] > rmid[1]):
        lmid, rmid = rmid, lmid

    if dist(lmid, rmid) < 4.0:
        return None

    bond_pixel_width = max(0.8, bond_line_width * svg_scale)
    # 单原子环 1 模式: 括号内只夹一个原子 -> 高度严格 ≤ bond_len * 1.2
    local_bond_len = dist(a_xy, other_xy)
    height = _estimate_chain_bracket_height(
        [a_idx], atom_coords, lmid, rmid, local_bond_len)

    return _render_and_paste_bracket_pair(
        cv_img, lmid, rmid, height,
        atom_coords=atom_coords, atom_boxes=atom_boxes,
        bond_obbs_indexed=bond_obbs_indexed,
        placed_overlays=placed_overlays,
        bond_pixel_width=bond_pixel_width, svg_scale=svg_scale,
        font_size_px=font_size_px,
        subscript_prob=subscript_prob,
        bracket_type=random.choice(['square', 'paren']),  # 环单原子括号
        reserved_atom_pool=[a_idx],
    )


# ============================================================
# 括号高度上限: 分子前景像素高度
# ============================================================
def _molecule_fg_height(atom_coords):
    """返回分子前景像素 box 的高度 (用于水平括号的最高可能高度)."""
    pts = list(atom_coords.values())
    if len(pts) < 2:
        return 100.0
    ys = [p[1] for p in pts]
    return max(ys) - min(ys) + 12  # 加小边距


# ============================================================
# 通用渲染 + 粘贴
# ============================================================
def _render_and_paste_bracket_pair(cv_img, lmid, rmid, height, *,
                                    atom_coords, atom_boxes, bond_obbs_indexed,
                                    placed_overlays,
                                    bond_pixel_width, svg_scale, font_size_px,
                                    subscript_prob,
                                    bracket_type,
                                    reserved_atom_pool):
    """渲染左 + 右括号 (可选下标), 粘贴到 cv_img.

    几何约定:
        - 两个括号都"竖立" 在其本地坐标系: 高度方向 = 上下, 开口方向 = 左/右.
        - 用 angle_rad = atan2(rmid - lmid) 把 horizontal 旋到屏幕方向 angle_rad.
        - 这意味着括号的高度轴 (本地 Y) 旋转到 angle_rad + 90° 方向, 即 ⟂ 两中点连线.
        - 自动保证两个括号平行.
    """
    # 括号粗细采用与 AP 相同的安全渲染→缩放流程:
    #   1. 以安全的渲染粗细生成括号图像
    #   2. 等比缩放至目标粗细, 使括号线条视觉粗细精准匹配分子键

    # ---- 屏幕坐标系下的旋转角度 (只算一次) ----
    angle_rad = _screen_angle_rad(lmid, rmid)
    if abs(angle_rad) > math.pi / 2:
        lmid, rmid = rmid, lmid
        angle_rad = _screen_angle_rad(lmid, rmid)
    angle_deg = math.degrees(angle_rad)

    # ---- 可变高度: 仅对水平括号 (|angle| < tol 或 |angle-180| < tol) ----
    min_height = height
    is_horizontal = (abs(angle_deg) < _HORIZONTAL_ANGLE_TOL_DEG or
                     abs(abs(angle_deg) - 180) < _HORIZONTAL_ANGLE_TOL_DEG)

    if is_horizontal:
        mol_h = _molecule_fg_height(atom_coords)
        max_height = max(min_height * 1.05, mol_h)
    else:
        max_height = min_height

    existing_polys = _build_existing_polys(atom_boxes, bond_obbs_indexed, placed_overlays)

    # 二分搜索最佳高度: 先试 max, 失败则二分搜索
    bracket_tag = None
    left_br = right_br = None
    best_left = best_right = None
    best_tag = None
    chosen_height = min_height

    # 候选高度序列: max → 二分中点 → ... → min
    heights_to_probe = []
    if is_horizontal and max_height > min_height * 1.05:
        # 首试 max
        heights_to_probe.append(max_height)
        # 预计算二分中点备选
        lo, hi = min_height, max_height
        for _ in range(_HEIGHT_BINARY_SEARCH_STEPS):
            mid = (lo + hi) / 2.0
            if mid > min_height * 1.03 and mid < max_height * 0.97:
                heights_to_probe.append(mid)
            hi = mid
        heights_to_probe.append(min_height)
    else:
        heights_to_probe = [min_height]

    def _render_at_height(h):
        """在目标像素高度 h 渲染括号对, 预补偿 scale 缩放."""
        nonlocal bracket_tag
        bracket_h_px = max(int(round(h)), int(round(svg_scale * 5)))
        render_h = max(4, int(round(bracket_h_px / scale)))
        use_line_art = (bracket_type == 'paren' and h > min_height * 1.15)
        if use_line_art:
            from draw.bracket_draw import draw_bracket_pair_line
            try:
                l, r, t = draw_bracket_pair_line('paren', render_h, render_thickness)
            except Exception:
                return None, None, None
        else:
            try:
                l, r, t = draw_bracket_pair(bracket_type, render_h, render_thickness)
            except Exception:
                return None, None, None
        # 等比缩放使线粗匹配目标 thickness
        if abs(scale - 1.0) > 0.03:
            l = l.resize((max(1, int(round(l.size[0] * scale))),
                          max(1, int(round(l.size[1] * scale)))),
                         PILImage.Resampling.LANCZOS)
            r = r.resize((max(1, int(round(r.size[0] * scale))),
                          max(1, int(round(r.size[1] * scale)))),
                         PILImage.Resampling.LANCZOS)
        return l, r, t

    target_thickness = max(1.0, bond_pixel_width * random.uniform(0.85, 1.15))
    render_thickness = min(_BRACKET_THICKNESS_SAFE_MAX,
                           max(_BRACKET_THICKNESS_SAFE_MIN, target_thickness))
    scale = target_thickness / render_thickness

    for try_height in heights_to_probe:
        l, r, t = _render_at_height(try_height)
        if l is None:
            continue

        # 重叠预检查
        lw, lh = l.size
        la = (lw / 2.0, lh / 2.0)
        est_l = estimate_overlay_obb((lw, lh), lmid, la, angle_deg)
        rw, rh = r.size
        ra = (rw / 2.0, rh / 2.0)
        est_r = estimate_overlay_obb((rw, rh), rmid, ra, angle_deg)
        if check_candidate_overlap(est_l, existing_polys, cv_img.shape[:2]) or \
           check_candidate_overlap(est_r, existing_polys, cv_img.shape[:2]):
            continue  # 太高, 试下一个

        # 通过
        chosen_height = try_height
        left_br, right_br, bracket_tag = l, r, t
        left_anchor = (left_br.size[0] / 2.0, left_br.size[1] / 2.0)
        right_anchor = (right_br.size[0] / 2.0, right_br.size[1] / 2.0)
        break

    if left_br is None or right_br is None:
        return None

    # 左括号 anchor = 图像中心 → 对齐到 lmid
    rotated_l, new_l_anchor = rotate_pil_around_anchor(left_br, angle_deg, left_anchor)

    new_bg, off = paste_with_anchor(cv_img, rotated_l, lmid, new_l_anchor)
    if off != (0, 0):
        shift_atom_pixel_coords(atom_coords, *off)
        shift_axis_boxes(atom_boxes, *off)
        shift_polygons_dict(bond_obbs_indexed, *off)
        shift_polygon_list(placed_overlays, *off)
        lmid = (lmid[0] + off[0], lmid[1] + off[1])
        rmid = (rmid[0] + off[0], rmid[1] + off[1])
    cv_img = new_bg

    l_obb = compute_pasted_obb(rotated_l, lmid, new_l_anchor, (0, 0))
    if l_obb is None:
        return None
    placed_overlays.append((l_obb.astype(np.float32), f'left_bracket|{bracket_tag}'))

    # 右括号
    rw, rh = right_br.size
    right_anchor = (rw / 2.0, rh / 2.0)
    rotated_r, new_r_anchor = rotate_pil_around_anchor(right_br, angle_deg, right_anchor)

    new_bg, off = paste_with_anchor(cv_img, rotated_r, rmid, new_r_anchor)
    if off != (0, 0):
        shift_atom_pixel_coords(atom_coords, *off)
        shift_axis_boxes(atom_boxes, *off)
        shift_polygons_dict(bond_obbs_indexed, *off)
        shift_polygon_list(placed_overlays, *off)
        lmid = (lmid[0] + off[0], lmid[1] + off[1])
        rmid = (rmid[0] + off[0], rmid[1] + off[1])
    cv_img = new_bg

    r_obb = compute_pasted_obb(rotated_r, rmid, new_r_anchor, (0, 0))
    if r_obb is None:
        return cv_img, set(reserved_atom_pool)
    placed_overlays.append((r_obb.astype(np.float32), f'right_bracket|{bracket_tag}'))

    # ---- 下标 ----
    if random.random() < subscript_prob:
        sub_text = generate_subscript_content()
        if sub_text:
            try:
                sub_img = draw_bracket_subscript(sub_text,
                                                  font_size=max(8, int(font_size_px * 0.9)))
            except Exception:
                sub_img = None
            if sub_img is not None:
                cv_img = _paste_subscript(
                    cv_img, sub_img, lmid, rmid,
                    bracket_size=(rw, rh), angle_rad=angle_rad,
                    atom_coords=atom_coords, atom_boxes=atom_boxes,
                    bond_obbs_indexed=bond_obbs_indexed,
                    placed_overlays=placed_overlays)

    return cv_img, set(reserved_atom_pool)


def _paste_subscript(cv_img, sub_img, lmid, rmid, *,
                      bracket_size, angle_rad,
                      atom_coords, atom_boxes, bond_obbs_indexed,
                      placed_overlays):
    """下标优先右下角; 若与分子键/已有 overlay 重叠则依次尝试其它三个角。

    bracket_size: (rw, rh) 右括号未旋转时的画布宽高;
                   旋转 angle_rad 后, 宽 (rw) 沿 (cos a, sin a), 高 (rh) 沿 (-sin a, cos a).
    """
    rw, rh = bracket_size
    sw, sh = sub_img.size
    H, W = cv_img.shape[:2]

    # 已有多边形 (用于重叠预检查)
    existing_polys = _build_existing_polys(atom_boxes, bond_obbs_indexed, placed_overlays)

    # 按优先级依次尝试四个角
    corners = ['right_bottom', 'right_top', 'left_bottom', 'left_top']

    def _compute_world_anchor(corner):
        sign_x = 1 if 'right' in corner else -1
        sign_y = 1 if 'bottom' in corner else -1
        anchor_world = rmid if 'right' in corner else lmid
        pad = max(2, int(round(rw * 0.6)))
        dx_local = sign_x * (rw * 0.5 + sw * 0.5 + pad)
        dy_local = sign_y * (rh * 0.5 - sh * 0.3)
        cs, sn = math.cos(angle_rad), math.sin(angle_rad)
        world_dx = dx_local * cs - dy_local * sn
        world_dy = dx_local * sn + dy_local * cs
        return (anchor_world[0] + world_dx, anchor_world[1] + world_dy)

    best_corner = None
    best_world_anchor = None
    for corner in corners:
        wa = _compute_world_anchor(corner)
        # 下标水平放置, OBB 即为轴对齐矩形
        est_obb = np.array([
            [wa[0] - sw / 2, wa[1] - sh / 2],
            [wa[0] + sw / 2, wa[1] - sh / 2],
            [wa[0] + sw / 2, wa[1] + sh / 2],
            [wa[0] - sw / 2, wa[1] + sh / 2],
        ], dtype=np.float32)
        if not check_candidate_overlap(est_obb, existing_polys, (H, W),
                                        strict_threshold=0.20, lenient_threshold=0.40):
            best_corner = corner
            best_world_anchor = wa
            break

    if best_corner is None:
        # 全部重叠, 退回右下角
        best_world_anchor = _compute_world_anchor('right_bottom')

    sub_world_anchor = best_world_anchor
    # 下标本身不旋转 (即使括号倾斜, 下标常规上仍水平摆放) —— 这与化学绘图惯例一致。
    sub_anchor_local = (sw / 2.0, sh / 2.0)

    new_bg, off = paste_with_anchor(cv_img, sub_img, sub_world_anchor, sub_anchor_local)
    if off != (0, 0):
        shift_atom_pixel_coords(atom_coords, *off)
        shift_axis_boxes(atom_boxes, *off)
        shift_polygons_dict(bond_obbs_indexed, *off)
        shift_polygon_list(placed_overlays, *off)
        sub_world_anchor = (sub_world_anchor[0] + off[0], sub_world_anchor[1] + off[1])
    cv_img = new_bg

    obb = compute_pasted_obb(sub_img, sub_world_anchor, sub_anchor_local, (0, 0))
    if obb is not None:
        placed_overlays.append((obb.astype(np.float32), 'bracket_subscript'))
    return cv_img
