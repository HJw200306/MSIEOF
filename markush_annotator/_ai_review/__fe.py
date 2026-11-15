"""仿前端 mol_editor 渲染: 底图 + 绿/蓝/品红键 + 深色原子点 + 虚线边界。用于比对用户截图。"""
import sys, json, math
from pathlib import Path
from PIL import Image, ImageDraw
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
BC = {1:(16,185,129),2:(2,132,199),3:(234,88,12),4:(124,58,237),5:(79,70,229),6:(225,29,72),7:(217,70,239)}

def render(i, S=8):
    d = json.load(open(D/"corrected_annotations"/f"{i:06d}.json", encoding="utf-8"))
    ws, hs = d["image_size"]
    src = Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    big = src.resize((ws*S, hs*S), Image.LANCZOS)
    dr = ImageDraw.Draw(big)
    # 虚线边界
    for x in range(0, ws*S, 8):
        dr.line([(x,0),(x+4,0)], fill=(148,163,184)); dr.line([(x,hs*S-1),(x+4,hs*S-1)], fill=(148,163,184))
    for y in range(0, hs*S, 8):
        dr.line([(0,y),(0,y+4)], fill=(148,163,184)); dr.line([(ws*S-1,y),(ws*S-1,y+4)], fill=(148,163,184))
    at = d["atoms"]
    P = [(a["x"]*S, a["y"]*S) for a in at]
    for b in d["bonds"]:
        u,v,t = b[0],b[1],int(b[2])
        x1,y1 = P[u]; x2,y2 = P[v]
        dx,dy = x2-x1, y2-y1; L = math.hypot(dx,dy)
        if L==0: continue
        nx,ny = -dy/L, dx/L
        col = BC.get(t,(16,185,129)); w = max(2, round(2.2*S/8))
        if t==1:
            dr.line([(x1,y1),(x2,y2)], fill=col, width=w)
        elif t==2:
            bd = 3.2*S/8
            dr.line([(x1+nx*bd,y1+ny*bd),(x2+nx*bd,y2+ny*bd)], fill=col, width=w)
            dr.line([(x1-nx*bd,y1-ny*bd),(x2-nx*bd,y2-ny*bd)], fill=col, width=w)
        elif t==7:
            amp = max(2.0*S/8, min(L*0.06, 9*S/8)); cyc = max(2,min(6,round(L/(18*S/8)))); segs=cyc*10
            pts=[]
            for s in range(segs+1):
                tt=s/segs; off=math.sin(tt*math.pi*2*cyc)*amp
                pts.append((x1+dx*tt+nx*off, y1+dy*tt+ny*off))
            dr.line(pts, fill=col, width=w, joint="curve")
    r = 5.5*S/8
    for (x,y) in P:
        dr.ellipse([x-r,y-r,x+r,y+r], fill=(15,23,42), outline=(255,255,255), width=max(1,round(1.5*S/8)))
    return big

if __name__=="__main__":
    for a in sys.argv[1:]:
        i=int(a); im=render(i); p=TMP/f"fe_{i:06d}.png"; im.save(p); print(p, im.size)
