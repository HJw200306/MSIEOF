"""对齐后几何检测: corrected 图结构缩放到原图墨迹包围盒, 再查
  (漏键) 两原子距离≈键长、中间有墨迹、图中无键
  (多键) 图中两原子有键、但中间几乎无墨迹
坐标先按墨迹包围盒做仿射对齐, 消除 MolScribe 布局的整体缩放偏移。"""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")

def load(i):
    d = json.load(open(D/"corrected_annotations"/f"{i:06d}.json", encoding="utf-8"))
    img = Image.open(D/"images"/f"{i:06d}.png").convert("L")
    return d, np.array(img)

def align_atoms(atoms, ink):
    ys, xs = np.where(ink < 200)
    ix0, ix1, iy0, iy1 = xs.min(), xs.max(), ys.min(), ys.max()
    gx = [a["x"] for a in atoms]; gy = [a["y"] for a in atoms]
    gx0, gx1, gy0, gy1 = min(gx), max(gx), min(gy), max(gy)
    P = []
    for a in atoms:
        x = ix0 + (a["x"]-gx0)/(gx1-gx0+1e-9)*(ix1-ix0)
        y = iy0 + (a["y"]-gy0)/(gy1-gy0+1e-9)*(iy1-iy0)
        P.append([x, y])
    return np.array(P)

def run(i, ink_thr=140):
    d, img = load(i)
    H, W = img.shape
    ink = img < ink_thr
    at = d["atoms"]; bonds = d["bonds"]
    P = align_atoms(at, img)
    bonded = set()
    for b in bonds:
        u, v = b[0], b[1]
        if u != v: bonded.add((min(u,v), max(u,v)))
    blen = [np.linalg.norm(P[u]-P[v]) for (u,v) in bonded] or [10]
    med = float(np.median(blen))
    missing = []
    for a in range(len(at)):
        for c in range(a+1, len(at)):
            if (a,c) in bonded: continue
            dist = float(np.linalg.norm(P[a]-P[c]))
            if not (0.7*med < dist < 1.35*med): continue
            ts = np.linspace(0.28, 0.72, 16)
            hits = 0
            for t in ts:
                x, y = P[a]*(1-t) + P[c]*t
                xi, yi = int(round(x)), int(round(y))
                if 0 <= xi < W and 0 <= yi < H and ink[max(0,yi-1):yi+2, max(0,xi-1):xi+2].any():
                    hits += 1
            if hits/len(ts) > 0.72:
                missing.append((a, c, round(dist/med, 2), round(hits/len(ts), 2)))
    extra = []
    for (u,v) in bonded:
        ts = np.linspace(0.2, 0.8, 16); hits = 0
        for t in ts:
            x,y = P[u]*(1-t)+P[v]*t
            xi,yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H and ink[max(0,yi-1):yi+2, max(0,xi-1):xi+2].any():
                hits += 1
        if hits/len(ts) < 0.30:
            extra.append((u, v, round(hits/len(ts),2)))
    return missing, extra

if __name__ == "__main__":
    for i in range(833, 923):
        try:
            m, e = run(i)
        except Exception as ex:
            print(i, "ERR", ex); continue
        if m or e:
            print(f"{i}  missing={m}  extra={[x[:2] for x in e]}")
