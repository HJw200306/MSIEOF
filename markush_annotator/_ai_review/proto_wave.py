import sys, json, re, ast, io
sys.path.insert(0,'.')
import build_base as B
from backend.data_manager import graph_to_smiles
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
D=r'C:\Users\Admin\PycharmProjects\msirost\markush_annotator\data\seg_page2smiles'

src=open('build_corr.py',encoding='utf-8').read()
m=re.search(r'OVERRIDE\s*=\s*\{',src); i=m.end()-1; depth=0;j=i
while j<len(src):
    if src[j]=='{':depth+=1
    elif src[j]=='}':
        depth-=1
        if depth==0:break
    j+=1
OV=ast.literal_eval(src[i:j+1])

def wave_connect(i):
    atoms,bonds,(W,H),flags,ann=B.base(i)
    ov=OV.get(i,{})
    D=set(ov.get('delete',[]))
    # connected components
    import collections
    g=collections.defaultdict(set)
    for u,v,o in bonds: g[u].add(v); g[v].add(u)
    # main core = component containing the most non-deleted heavy atoms? use largest comp
    seen=set(); comps=[]
    for s in range(len(atoms)):
        if s in seen: continue
        stack=[s]; comp=set()
        while stack:
            x=stack.pop()
            if x in comp: continue
            comp.add(x); seen.add(x)
            for y in g[x]:
                if y not in comp: stack.append(y)
        comps.append(comp)
    main=max(comps,key=len)
    D |= set(range(len(atoms)))-main   # drop stray disconnected fragments (e.g. 837's free CC)
    core=[k for k in range(len(atoms)) if k not in D]
    kept=set(); out=[]
    for u,v,o in bonds:
        du,dv=u in D, v in D
        if du^dv:
            fk = u if du else v
            kept.add(fk); out.append([u,v,7])
        elif du and dv:
            out.append([u,v,o])
        else:
            out.append([u,v,o])
    drops=D-kept
    keep=[k for k in range(len(atoms)) if k not in drops]
    remap={o:n for n,o in enumerate(keep)}
    A=[atoms[k] for k in keep]
    Bo=[[remap[u],remap[v],o] for u,v,o in out if u in remap and v in remap and u!=v]
    return A,Bo,(W,H)

def rend(smi,W,H):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    d=rdMolDraw2D.MolDraw2DCairo(W,H); rdMolDraw2D.PrepareAndDrawMolecule(d,m); d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText())).convert('RGB')

if __name__=='__main__':
    for i in [851,858,833,862,837,914,882,906]:
        A,Bo,(W,H)=wave_connect(i)
        s=graph_to_smiles(A,Bo,[])
        print(i,s)
