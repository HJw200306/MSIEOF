import sys
from pathlib import Path
from PIL import Image, ImageDraw
TMP=Path(r"C:\Users\Admin\AppData\Local\Temp")
lo,hi=map(int,sys.argv[1:3]); cols=2; cell=770; pad=8
items=list(range(lo,hi+1)); rows=(len(items)+cols-1)//cols
s=Image.open(TMP/f"fin_{lo:06d}.png"); rowh=s.height*cell//s.width
canvas=Image.new("RGB",(cols*(cell+pad)+pad, rows*(rowh+pad)+pad),(235,235,235))
for t,i in enumerate(items):
    im=Image.open(TMP/f"fin_{i:06d}.png").convert("RGB")
    im=im.resize((cell,int(im.height*cell/im.width)))
    r,c=divmod(t,cols); canvas.paste(im,(pad+c*(cell+pad),pad+r*(rowh+pad)))
out=TMP/f"finsheet_{lo}_{hi}.png"; canvas.save(out); print(out,canvas.size)
