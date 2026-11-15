"""几何检测: 两原子距离≈键长、中间有墨迹、raw无键 -> 疑似漏键。"""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
D=Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
def ink_mask(p):
    im=Image.open(p).convert("L"); a=np.array(im)
    return (a<140)
def run(i):
    raw=json.load(open(D/"annotations"/f"{i:06d}.json",encoding="utf-8"))
    W,H=Image.open(D/"images"/f"{i:06d}.png").size
    ink=ink_mask(D/"images"/f"{i:06d}.png")
    at=raw["atoms"]; P=np.array([[a["x"]*W,a["y"]*H] for a in at])
    bonded=set()
    blen=[]
    for b in raw["bonds"]:
        u,v=b["endpoint_atoms"][:2]; bonded.add((min(u,v),max(u,v)))
        blen.append(np.linalg.norm(P[u]-P[v]))
    if not blen: return []
    med=np.median(blen)
    out=[]
    n=len(at)
    for a in range(n):
        for c in range(a+1,n):
            if (a,c) in bonded: continue
            d=np.linalg.norm(P[a]-P[c])
            if not (0.75*med < d < 1.25*med): continue
            # sample ink along segment, exclude 25% near each atom
            ts=np.linspace(0.30,0.70,15)
            hits=0; tot=0
            for t in ts:
                x,y=P[a]*(1-t)+P[c]*t
                xi,yi=int(round(x)),int(round(y))
                tot+=1
                if 0<=xi<W and 0<=yi<H:
                    win=ink[max(0,yi-1):yi+2,max(0,xi-1):xi+2]
                    if win.any(): hits+=1
            cov=hits/tot
            if cov>0.75:
                out.append((a,c,round(d/med,2),round(cov,2)))
    return out
if __name__=="__main__":
    for i in range(833,923):
        r=run(i)
        if r: print(i, r)
