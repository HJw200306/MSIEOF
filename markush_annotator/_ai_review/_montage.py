import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
D = Path(r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\_ai_review\ov_raw_832_922")
TMP = Path(r"C:\Users\Admin\AppData\Local\Temp")
def font(sz):
    for n in ("arial.ttf","Arial.ttf","DejaVuSans.ttf"):
        try: return ImageFont.truetype(n,sz)
        except Exception: pass
    return ImageFont.load_default()
lo,hi=map(int,sys.argv[1:3])
cols=int(sys.argv[3]) if len(sys.argv)>3 else 3
cell=int(sys.argv[4]) if len(sys.argv)>4 else 470
full = "--full" in sys.argv
items=[i for i in range(lo,hi+1) if (D/f"{i:06d}.png").exists()]
rows=(len(items)+cols-1)//cols
pad,lab=8,26
canvas=Image.new("RGB",(cols*(cell+pad)+pad, rows*(int(cell*0.72)+pad+lab)+pad),(240,240,240))
dr=ImageDraw.Draw(canvas); f=font(20)
rowh=int(cell*0.72)
for t,i in enumerate(items):
    im=Image.open(D/f"{i:06d}.png").convert("RGB")
    if not full:
        W,left=im.size
        # render layout: left panel width = (W-gap)/2 . crop right panel
        # detect by known gap 24, top 34
        top=34; gap=24
        pw=(W-gap)//2
        im=im.crop((pw+gap, top, W, im.height))
    sc=cell/im.width
    im=im.resize((cell,max(1,int(im.height*sc))))
    r,c=divmod(t,cols)
    x=pad+c*(cell+pad); y=pad+r*(rowh+pad+lab)
    canvas.paste(im,(x,y+lab))
    dr.text((x,y+2),f"idx {i}",fill=(200,0,0),font=f)
out=TMP/f"mont_{lo}_{hi}.png"; canvas.save(out); print(out,canvas.size)
