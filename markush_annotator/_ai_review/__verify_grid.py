"""从磁盘 corrected 直接用其坐标(不做任何 remap)仿前端叠加原图, 拼网格核对是否已重合。"""
import json, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
BC = {1:(16,185,129),2:(2,132,199),3:(234,88,12),4:(124,58,237),5:(79,70,229),6:(225,29,72),7:(217,70,239)}
TILE_W = 380
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
def tile(i):
    d = json.load(open(D/"corrected_annotations"/f"{i:06d}.json", encoding="utf-8"))
    src = Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    W,H = src.size
    S = TILE_W/W
    big = src.resize((int(W*S),int(H*S)), Image.LANCZOS)
    dr = ImageDraw.Draw(big)
    at = d["atoms"]; P=[(a["x"]*S,a["y"]*S) for a in at]
    for b in d["bonds"]:
        u,v,t=b[0],b[1],int(b[2]); x1,y1=P[u]; x2,y2=P[v]
        dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy)
        if L==0: continue
        col=BC.get(t,(16,185,129)); w=2
        if t==7:
            nx,ny=-dy/L,dx/L; amp=max(2.0,min(L*0.06,9)); cyc=max(2,min(6,round(L/18))); segs=cyc*10
            pts=[]
            for s in range(segs+1):
                tt=s/segs; off=math.sin(tt*math.pi*2*cyc)*amp
                pts.append((x1+dx*tt+nx*off,y1+dy*tt+ny*off))
            dr.line(pts,fill=col,width=w,joint="curve")
        elif t==2:
            nx,ny=-dy/L,dx/L; bd=3.2
            dr.line([(x1+nx*bd,y1+ny*bd),(x2+nx*bd,y2+ny*bd)],fill=col,width=w)
            dr.line([(x1-nx*bd,y1-ny*bd),(x2-nx*bd,y2-ny*bd)],fill=col,width=w)
        else: dr.line([(x1,y1),(x2,y2)],fill=col,width=w)
    for at_ in at:
        x,y=at_["x"]*S,at_["y"]*S
        dr.ellipse([x-2.5,y-2.5,x+2.5,y+2.5],fill=(15,23,42),outline=(255,255,255))
    return big
idxs=list(range(833,923)); cols=10
tiles=[(i,tile(i)) for i in idxs]
maxh=max(t.height for _,t in tiles); rows=(len(tiles)+cols-1)//cols
grid=Image.new('RGB',(cols*TILE_W, rows*(maxh+20)),(255,255,255)); dr=ImageDraw.Draw(grid); f=font(15)
for k,(i,t) in enumerate(tiles):
    r,c=divmod(k,cols); x=c*TILE_W; y=r*(maxh+20)
    grid.paste(t,(x,y+18)); dr.text((x+3,y+1),f'{i}',fill=(200,0,0),font=f)
grid.save(TMP/'verify_grid_833_922.png'); print('size',grid.size,'rows',rows)
