"""把 corrected 图结构仿射缩放到原图墨迹包围盒后叠加, 判断是纯缩放问题还是结构错位。"""
import sys, json, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
BC = {1:(16,185,129),2:(2,132,199),3:(234,88,12),4:(124,58,237),5:(79,70,229),6:(225,29,72),7:(217,70,239)}

def run(i, S=8):
    d = json.load(open(D/"corrected_annotations"/f"{i:06d}.json", encoding="utf-8"))
    src = Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    W,H = src.size
    a = np.asarray(src.convert("L"))
    ink = a < 200
    ys,xs = np.where(ink)
    ix0,ix1,iy0,iy1 = xs.min(),xs.max(),ys.min(),ys.max()
    at = d["atoms"]
    gx=[p["x"] for p in at]; gy=[p["y"] for p in at]
    gx0,gx1,gy0,gy1 = min(gx),max(gx),min(gy),max(gy)
    def mapx(x): return ix0 + (x-gx0)/(gx1-gx0+1e-9)*(ix1-ix0)
    def mapy(y): return iy0 + (y-gy0)/(gy1-gy0+1e-9)*(iy1-iy0)
    P=[(mapx(p["x"])*S, mapy(p["y"])*S) for p in at]
    big = src.resize((W*S,H*S), Image.LANCZOS)
    dr = ImageDraw.Draw(big)
    w = max(2, round(2.2*S/8))
    for b in d["bonds"]:
        u,v,t=b[0],b[1],int(b[2]); x1,y1=P[u]; x2,y2=P[v]
        dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy)
        if L==0: continue
        col=BC.get(t,(16,185,129))
        if t==7:
            nx,ny=-dy/L,dx/L; amp=max(2.0*S/8,min(L*0.06,9*S/8)); cyc=max(2,min(6,round(L/(18*S/8)))); segs=cyc*10
            pts=[]
            for s in range(segs+1):
                tt=s/segs; off=math.sin(tt*math.pi*2*cyc)*amp
                pts.append((x1+dx*tt+nx*off,y1+dy*tt+ny*off))
            dr.line(pts, fill=col, width=w, joint="curve")
        else:
            dr.line([(x1,y1),(x2,y2)], fill=col, width=w)
    r=3*S/8
    for (x,y) in P: dr.ellipse([x-r,y-r,x+r,y+r], fill=(255,0,0))
    return big

if __name__=="__main__":
    for a in sys.argv[1:]:
        i=int(a); im=run(i); p=TMP/f"align_{i:06d}.png"; im.save(p); print(p, im.size)
