"""渲染 202-400 的「原图 | 拓扑+编号」叠加图到 ov_batch2/，并可拼接触表。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_overlay as R  # noqa: E402

D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
OUT = D / "_ai_review" / "ov_batch2"

if __name__ == "__main__":
    idxs = list(range(202, 401))
    OUT.mkdir(parents=True, exist_ok=True)
    for i in idxs:
        R.render(i, out_dir=OUT, src="annotations")
    print(f"rendered {len(idxs)} overlays -> {OUT}")
