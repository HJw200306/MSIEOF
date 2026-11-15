"""把 833-922 corrected 图结构缩放到原图墨迹包围盒后叠加, 拼成大图逐张核对结构是否真的错。"""
import json, math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
BC = {1:(16,185,129),2:(2,132,199),3:(234,88,12),4:(124,58,237),5:(79,70,229),6:(225,29,72),7:(217,70,239)}
TILE_W = 400

def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()

def tile(i):
    d = json.load(open(D/"corrected_annotations"/f"{i:06d}.json", encoding="utf-8"))
    src = Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    W,H = src.size
    a = np.asarray(src.convert("L")); ink = a < 200
    ys,xs = np.where(ink)
    ix0,ix1,iy0,iy1 = xs.min(),xs.max(),ys.min(),ys.max()
    at = d["atoms"]; gx=[p["x"] for p in at]; gy=[p["y"] for p in at]
    gx0,gx1,gy0,gy1 = min(gx),max(gx),min(gy),max(gy)
    # 原图先裁到墨迹包围盒, 再整体缩放到 tile 宽度; 图结构 bbox 映射到同一区域
    ink = src.crop((ix0,iy0,ix1+1,iy1+1))
    Sx = TILE_W/ink.width
    VH = int(ink.height*Sx)
    def mapx(x): return (x-gx0)/(gx1-gx0+1e-9)*TILE_W
    def mapy(y): return (y-gy0)/(gy1-gy0+1e-9)*VH
    big = ink.resize((TILE_W,VH), Image.LANCZOS)
    dr = ImageDraw.Draw(big); w = 2
    for b in d["bonds"]:
        u,v,t=b[0],b[1],int(b[2]); x1,y1=mapx(at[u]['x']),mapy(at[u]['y']); x2,y2=mapx(at[v]['x']),mapy(at[v]['y'])
        dx,dy=x2-x1,y2-y1; L=math.hypot(dx,dy)
        if L==0: continue
        col=BC.get(t,(16,185,129))
        if t==7:
            nx,ny=-dy/L,dx/L; amp=max(2.0,min(L*0.06,9)); cyc=max(2,min(6,round(L/18))); segs=cyc*10
            pts=[]
            for s in range(segs+1):
                tt=s/segs; off=math.sin(tt*math.pi*2*cyc)*amp
                pts.append((x1+dx*tt+nx*off,y1+dy*tt+ny*off))
            dr.line(pts, fill=col, width=w, joint="curve")
        else:
            dr.line([(x1,y1),(x2,y2)], fill=col, width=w)
    for at_ in at:
        x,y=mapx(at_['x']),mapy(at_['y'])
        dr.ellipse([x-2.5,y-2.5,x+2.5,y+2.5], fill=(255,0,0))
    return big.crop((0,0,big.width,min(big.height, VH+20)))

idxs = list(range(833,923))
cols = 10
tiles = []
for i in idxs:
    try: tiles.append((i, tile(i)))
    except Exception as e: print('ERR',i,e)
maxh = max(t.height for _,t in tiles)
rows = (len(tiles)+cols-1)//cols
grid = Image.new('RGB',(cols*TILE_W, rows*(maxh+22)),(255,255,255))
dr = ImageDraw.Draw(grid)
f = font(16)
for k,(i,t) in enumerate(tiles):
    r,c = divmod(k,cols)
    x = c*TILE_W; y = r*(maxh+22)
    grid.paste(t,(x,y+20))
    dr.text((x+3,y+1), f'{i}', fill=(200,0,0), font=f)
grid.save(TMP/'align_grid_833_922.png')
print('rows',rows,'maxh',maxh,'->', grid.size)
