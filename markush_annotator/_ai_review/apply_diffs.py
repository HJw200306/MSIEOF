"""把 VLM 产出的结构化 diff 套用到 raw 标注上，落库为 corrected_annotations/{i}.json。

diff 条目 (每条为一个 dict):
  {"relabel": idx, "label": "Me"}                 # symbol 置 '*', label 置文本 (superatom/R基)
  {"symbol": idx, "value": "S"}                   # 仅改元素符号
  {"delete": [i, j, ...]}                          # 删除节点 (自动重映射键索引)
  {"add": {"label": "C5H11"/"*", "x0","y0", "connect": idx, "order": 1}}  # 加一个节点并连到 connect

坐标: raw 为归一化, 乘 PNG 尺寸得像素 (与用户 corrected 文件约定一致)。
状态: 落库后 review_status=pending (待审核)。
"""
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator")
sys.path.insert(0, str(ROOT))
from backend.data_manager import graph_to_smiles  # noqa: E402

D = ROOT / "data" / "seg_page2smiles"
BOND_MAP = {"single": 1, "double": 2, "triple": 3, "aromatic": 4, "wedge": 5, "dash": 6}


def load_raw_pixels(i):
    ann = json.load(open(D / "annotations" / f"{i:06d}.json", encoding="utf-8"))
    W, H = Image.open(D / "images" / f"{i:06d}.png").size
    atoms = [{
        "x": round(a["x"] * W, 2), "y": round(a["y"] * H, 2),
        "confidence": a.get("confidence", 1.0),
        "symbol": a.get("atom_symbol", "C"), "label": None,
    } for a in ann["atoms"]]
    bonds = [[b["endpoint_atoms"][0], b["endpoint_atoms"][1],
              BOND_MAP.get(b.get("bond_type", "single"), 1)] for b in ann["bonds"]]
    return atoms, bonds, (W, H)


def apply_diff(i, diff):
    atoms, bonds, (W, H) = load_raw_pixels(i)
    for e in diff.get("edits", []):
        if "relabel" in e:
            atoms[e["relabel"]]["symbol"] = e.get("symbol", "*")
            atoms[e["relabel"]]["label"] = e["label"]
        elif "symbol" in e:
            atoms[e["symbol"]]["symbol"] = e["value"]
            if e.get("label") is not None:
                atoms[e["symbol"]]["label"] = e["label"]
        elif "bond" in e:
            u, v, order = e["bond"]
            for b in bonds:
                if (b[0] == u and b[1] == v) or (b[0] == v and b[1] == u):
                    b[2] = order

    deletes = set()
    for e in diff.get("edits", []):
        if "delete" in e:
            deletes.update(e["delete"])

    if deletes:
        keep = [k for k in range(len(atoms)) if k not in deletes]
        remap = {old: new for new, old in enumerate(keep)}
        atoms = [atoms[k] for k in keep]
        bonds = [[remap[u], remap[v], o] for u, v, o in bonds
                 if u in remap and v in remap and u != v]

    for e in diff.get("edits", []):
        if "add" in e:
            a = e["add"]
            new_idx = len(atoms)
            atoms.append({
                "x": round(a["x0"] * W, 2), "y": round(a["y0"] * H, 2),
                "confidence": 1.0, "symbol": a.get("symbol", "*"),
                "label": a.get("label"),
            })
            if a.get("connect") is not None and a["connect"] in range(len(atoms)):
                bonds.append([new_idx, a["connect"], a.get("order", 1)])

    esmi = graph_to_smiles(atoms, bonds, [])
    payload = {
        "index": i, "esmiles": esmi, "image_size": [W, H], "coverage": 1.0,
        "atoms": atoms, "bonds": bonds, "sgroups": [], "brackets": [],
        "ocr_atoms": [], "superatom_merges": [],
        "ai_review": {"verdict": diff.get("verdict", "editable"),
                      "confidence": diff.get("confidence"),
                      "notes": diff.get("notes", "")},
    }
    return payload, esmi


def run(diffs_path, write=False, status="pending"):
    diffs = json.load(open(diffs_path, encoding="utf-8"))
    status_lines = []
    for diff in diffs:
        i = diff["index"]
        payload, esmi = apply_diff(i, diff)
        print(f"--- idx {i}  conf={diff.get('confidence')}  verdict={diff.get('verdict')}")
        print(f"    notes : {diff.get('notes','')}")
        print(f"    esmiles: {esmi[:100]}")
        if diff.get("skip_corrected"):
            print("    [SKIP] 不写 corrected 文件 (损坏样本, 只登记状态供人工处理)")
            status_lines.append({"index": i, "review_status": "pending",
                                 "review_notes": f"AI预纠错: {diff.get('notes','')}"})
            continue
        if write:
            (D / "corrected_annotations" / f"{i:06d}.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            status_lines.append({"index": i, "review_status": status,
                                 "review_notes": f"AI预纠错: {diff.get('notes','')}"})
    if write and status_lines:
        sp = D / "review_status.jsonl"
        existing = {}
        order = []
        if sp.exists():
            for line in sp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    existing[r["index"]] = r
                    order.append(r["index"])
        for r in status_lines:
            if r["index"] not in existing:
                order.append(r["index"])
            existing[r["index"]] = r
        with open(sp, "w", encoding="utf-8") as f:
            for idx in order:
                f.write(json.dumps(existing[idx], ensure_ascii=False) + "\n")
        print(f"[STATUS] wrote {len(status_lines)} rows, total {len(order)}")


if __name__ == "__main__":
    diffs_path = sys.argv[1]
    write = "--write" in sys.argv
    run(diffs_path, write=write)
