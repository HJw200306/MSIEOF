"""全对比度叠加: 原图 + raw图键(彩线) + 原子点。原图有笔画无彩线=缺键。"""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
BC={1:(0,170,0),2:(0,90,255),3:(160,0,190),4:(0,150,150),5:(230,0,90),6:(230,0,90),7:(255,0,0)}
def render(i, sc=None):
    ann=json.load(open(D/"annotations"/f"{i:06d}.json",encoding="utf-8"))
    img=Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    W,H=img.size
    if sc is None: sc=max(6,min(12,round(760/W)))
    big=img.resize((W*sc,H*sc),Image.LANCZOS)
    dr=ImageDraw.Draw(big)
    at=ann["atoms"]
    P=[(round(a["x"]*W*sc),round(a["y"]*H*sc)) for a in at]
    for b in ann["bonds"]:
        u,v=b["endpoint_atoms"][0],b["endpoint_atoms"][1]
        t=__import__("build_base").BOND_MAP.get(str(b["bond_type"]).lower(),1)
        dr.line([P[u],P[v]],fill=BC.get(t,(0,170,0)),width=max(2,sc//3))
    for k,(x,y) in enumerate(P):
        r=max(3,sc-2)
        dr.ellipse([x-r,y-r,x+r,y+r],outline=(255,140,0),width=2)
        dr.text((x+r,y-r-14),str(k),fill=(200,0,0),font=font(max(12,sc*3)))
    return big
if __name__=="__main__":
    for a in sys.argv[1:]:
        i=int(a); im=render(i)
        p=TMP/f"ov2_{i:06d}.png"; im.save(p); print(p, im.size)
