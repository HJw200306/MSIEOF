"""sq_sheet.py i1 i2 ... -> 原图接触表, 每格标注 当前C~C数."""
import sys, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(s):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,s)
        except Exception: pass
    return ImageFont.load_default()
idx=[int(x) for x in sys.argv[1:]]
cols=3; cell=460; pad,lab=6,22
rows=(len(idx)+cols-1)//cols
W=cols*(cell+pad)+pad; H=rows*(int(cell*0.6)+pad+lab)+pad
cv=Image.new("RGB",(W,H),(255,255,255)); dr=ImageDraw.Draw(cv); f=font(18)
for t,i in enumerate(idx):
    a=json.load(open(D/"corrected_annotations"/f"{i:06d}.json",encoding="utf-8"))
    cc=sum(1 for b in a["bonds"] if b[2]==7)
    im=Image.open(D/"images"/f"{i:06d}.png").convert("RGB")
    sc=cell/im.width; im=im.resize((cell,max(1,int(im.height*sc))),Image.LANCZOS)
    r,c=divmod(t,cols); x=pad+c*(cell+pad); y=pad+r*(int(cell*0.6)+pad+lab)
    cv.paste(im,(x,y+lab))
    dr.rectangle([x,y,x+cell,y+lab],outline=(0,0,0))
    dr.text((x+2,y+2),f"idx {i}   graph C~C={cc}",fill=(200,0,0),font=f)
out=TMP/"sqsheet.png"; cv.save(out); print(out,cv.size)
