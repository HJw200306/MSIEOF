"""把人工/自动得出的 spec 转成 apply_diffs.py 能吃的 diff 文件。

spec (JSON): { "<idx>": {"waves":[{"fake":[...],"attach":k,"tip":[nx,ny]?}, ...],
                         "relabels":{"<idx>":"文本"}, "note":"...", "skip":true} }

规则: 波浪线 = 一根 type-7 键; 删掉 MolScribe 用来描波浪线的假碳,
在波浪线外端补一个 '*' 连接点原子, 用 order 7 连到片段里第一个真实原子。
"""
import json
import sys
from pathlib import Path

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")


def raw_norm(idx):
    ann = json.load(open(D / "annotations" / f"{idx:06d}.json", encoding="utf-8"))
    return [(a["x"], a["y"]) for a in ann["atoms"]]


def build(idx, spec):
    waves = spec.get("waves", [])
    allfake = sorted({k for w in waves for k in w.get("fake", [])})
    keep = [k for k in range(10000) if k not in set(allfake)]
    remap = {old: new for new, old in enumerate(keep)}

    edits = []
    for k, lab in (spec.get("relabels") or {}).items():
        edits.append({"relabel": int(k), "label": lab})
    for u, v in (spec.get("bond_wave") or []):
        edits.append({"bond": [int(u), int(v), 7]})
    if allfake:
        edits.append({"delete": allfake})

    pts = raw_norm(idx)
    for w in waves:
        attach = w["attach"]
        if attach in allfake:
            continue
        if w.get("tip") is not None:
            tx, ty = w["tip"]
        else:
            ax, ay = pts[attach]
            fk = max(w["fake"], key=lambda k: (pts[k][0] - ax) ** 2 + (pts[k][1] - ay) ** 2)
            tx, ty = pts[fk]
        edits.append({"add": {"symbol": "*", "label": w.get("tip_label"),
                              "x0": tx, "y0": ty,
                              "connect": remap[attach], "order": 7}})

    e = {"index": idx, "edits": edits, "notes": spec.get("note", ""),
         "confidence": spec.get("confidence", 0.85),
         "verdict": spec.get("verdict", "editable")}
    if spec.get("skip"):
        e["skip_corrected"] = True
    return e


def main(spec_path, out_path):
    specs = json.load(open(spec_path, encoding="utf-8"))
    diffs = [build(int(k), v) for k, v in sorted(specs.items(), key=lambda kv: int(kv[0]))]
    Path(out_path).write_text(json.dumps(diffs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(diffs)} diffs -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
