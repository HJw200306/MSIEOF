import sys
from PIL import Image
B = r"C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles"
for i in [int(x) for x in sys.argv[1:]]:
    im = Image.open(rf"{B}\images\{i:06d}.png").convert("RGB")
    im = im.resize((im.size[0]*6, im.size[1]*6), Image.LANCZOS)
    im.save(rf"{B}\_ai_review\zoom\_s{i:06d}.png")
    print(i, "saved")
