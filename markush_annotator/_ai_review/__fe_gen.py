"""用 build_corr 生成的 payload (含新对齐坐标) 仿前端渲染, 验证叠加是否与原图重合。"""
import sys, math
from pathlib import Path
from PIL import Image, ImageDraw
import build_corr as BC
D = BC.D
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
BC_col = {1:(16,185,129),2:(2,132,199),3:(234,88,12),4:(124,58,237),5:(79,70,229),6:(225,29,72),7:(217,70,239)}

def render(i, S=8):
    p, _, _, _ = BC.build(i)
    ws, hs = p["image_size"]
    src = Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    big = src.resize((ws*S, hs*S), Image.LANCZOS)
    dr = ImageDraw.Draw(big)
    at = p["atoms"]
    P = [(a["x"]*S, a["y"]*S) for a in at]
    for b in p["bonds"]:
        u,v,t = b[0],b[1],int(b[2]); x1,y1=P[u]; x2,y2=P[v]
        dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy)
        if L==0: continue
        nx,ny=-dy/L,dx/L; col=BC_col.get(t,(16,185,129)); w=max(2,round(2.2*S/8))
        if t==1: dr.line([(x1,y1),(x2,y2)],fill=col,width=w)
        elif t==2:
            bd=3.2*S/8
            dr.line([(x1+nx*bd,y1+ny*bd),(x2+nx*bd,y2+ny*bd)],fill=col,width=w)
            dr.line([(x1-nx*bd,y1-ny*bd),(x2-nx*bd,y2-ny*bd)],fill=col,width=w)
        elif t==7:
            amp=max(2.0*S/8,min(L*0.06,9*S/8)); cyc=max(2,min(6,round(L/(18*S/8)))); segs=cyc*10
            pts=[]
            for s in range(segs+1):
                tt=s/segs; off=math.sin(tt*math.pi*2*cyc)*amp
                pts.append((x1+dx*tt+nx*off,y1+dy*tt+ny*off))
            dr.line(pts,fill=col,width=w,joint="curve")
        else: dr.line([(x1,y1),(x2,y2)],fill=col,width=w)
    r=5.5*S/8
    for (x,y) in P: dr.ellipse([x-r,y-r,x+r,y+r],fill=(15,23,42),outline=(255,255,255),width=max(1,round(1.5*S/8)))
    return big

if __name__=="__main__":
    for a in sys.argv[1:]:
        i=int(a); im=render(i); q=TMP/f"feg_{i:06d}.png"; im.save(q); print(q, im.size)
