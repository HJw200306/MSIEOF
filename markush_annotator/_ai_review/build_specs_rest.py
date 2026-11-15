"""生成 121-200 的 spec: 以 auto_spec.detect 为基, 叠加人工核对的例外。

spec schema (新): {"<idx>": {"waves":[{"fake":[...],"attach":k}...],
                             "bond_wave":[[u,v]...],  # 把已存在的某根键改成 type-7 (中间横跨的波浪键)
                             "relabels":{...}, "note":"...", "skip":true}}
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import auto_spec as A  # noqa: E402

LO, HI = 121, 200
OUT = Path(__file__).parent / "specs_rest.json"

# 损坏/过于复杂无法自动重建 -> 不写 corrected, 保持原预标注
SKIP = {129, 130, 131, 132, 133, 135, 136}

# 人工核对覆盖: idx -> [ [fake_list, attach], ... ]
WAVES_OVERRIDE = {
    134: [[[0], 1], [[11], 10]],                       # 聚合物两端各一条
    139: [[[0, 1], 5], [[2, 3], 5], [[4], 5], [[7], 5]],  # N 周围四条 + 甲基
    140: [[[2, 4], 3], [[0, 1], 3]],                   # N+ 左右各一条
    146: [[[0], 1]],                                   # 波浪线在 P 上方
    147: [[[0, 10], 7]],                               # 波浪线在顶端 P 上方
    148: [[[4], 1]],                                   # 波浪线在 Si 上方
    150: [[[0], 1]],                                   # 左上方羰基
    193: [[[0], 1]],                                   # 最左端假碳 (启发式误判到 12)
    199: [[[0], 1], [[11], 10]],                       # 右端补一条
    200: [[[0], 1], [[14], 13]],                       # 右端补一条
}

# 图中"横跨/中途"的波浪键 (两端都是真实原子, 直接把该键改成 wave)
BOND_WAVE = {
    174: [5, 6],   # CH–NH2
    175: [4, 5],   # 环–COOH
    177: [6, 7],   # 环–OH
    180: [1, 5],   # CH–丁基链
    191: [5, 9],   # CH–C(=O)
    192: [5, 6],   # CH–NH
    193: [5, 6],   # CH–NH
    194: [5, 6],   # CH–N(CH3)2
}


def main():
    out = {}
    for i in range(LO, HI + 1):
        if i in SKIP:
            out[str(i)] = {"skip": True, "note": "坐标/结构损坏或复杂聚合物, 无法自动重建"}
            continue
        if i in WAVES_OVERRIDE:
            waves = [{"fake": list(f), "attach": a} for f, a in WAVES_OVERRIDE[i]]
        else:
            d = A.detect(i)
            waves = d["waves"] if d else []
        spec = {"waves": waves}
        if i in BOND_WAVE:
            spec["bond_wave"] = [list(BOND_WAVE[i])]
        out[str(i)] = spec
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {len(out)} specs -> {OUT}")
    n_wave = sum(len(v.get("waves", [])) for v in out.values())
    print(f"total waves={n_wave}  bond_wave={len(BOND_WAVE)}  skip={len(SKIP)}")


if __name__ == "__main__":
    main()
