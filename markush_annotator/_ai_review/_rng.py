import sys
sys.path.insert(0, r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\_ai_review')
import os, ovp2
os.makedirs(r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles\_ai_review\zoom', exist_ok=True)
lo, hi = int(sys.argv[1]), int(sys.argv[2])
for i in range(lo, hi+1):
    try:
        out, d = ovp2.render(i)
    except Exception as e:
        print(i, "ERR", e)
print("done")
