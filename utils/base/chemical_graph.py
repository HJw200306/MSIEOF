"""Versioned chemical graph contract shared by annotation, export and evaluation.

Real atoms are parsed as atoms, aliases remain labelled dummy atoms. Strict
sanitization never falls back to accepting invalid valence. Image coordinates
have their Y axis reversed before interpreting directed wedge/dash bonds.
"""
from __future__ import annotations

import copy
import json
import re
from typing import Any

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

SCHEMA_VERSION = 'chemical-graph-v2.0'
SUPERATOMS = set(('Ac Ts Ms Tf Ph Bn Bz Tol Mes Naph Boc Fmoc Cbz Piv TBDMS TMS '
    'TIPS THP Tr PMB Me Et Pr iPr nPr cPr Bu iBu nBu tBu cHex OMe OEt OPh SMe '
    'NH2 NMe2 NO2 SO2 SO3H COOH CO2H COOMe COOEt CONH2 CHO CN CF3 OCF3 SF5 '
    'CH3 H3C CH2 C2H5 CH2CH3 HO3S SO2NH2 O2S SOOO R X Y Z Ar Het Hal _AP *').split())
BOND_ORDER = {'single':1, 'double':2, 'triple':3, 'aromatic':4,
              'wedge':5, 'dash':6, 'solid wedge':5, 'dashed wedge':6,
              'wave':7, 'wavy':7, 'unspecified':7}


def _is_alias(token: str) -> bool:
    return token in SUPERATOMS or bool(re.fullmatch(r'R(?:\d+[a-z]?|[a-z])?', token))


def normalize_atom(a: dict) -> dict:
    """In-place, idempotent migration of legacy label-wrapped atom tokens."""
    token = str(a.get('label') or a.get('atom_label') or a.get('symbol') or a.get('atom_symbol') or 'C').strip()
    inner = token[1:-1] if token.startswith('[') and token.endswith(']') else token
    if token in ('*','[*]','_AP'):
        label = '_AP'
        parsed = None
    elif _is_alias(inner):
        label = inner
        parsed = None
    else:
        try:
            parsed = Chem.AtomFromSmiles(token)
        except Exception:
            parsed = None
        if parsed is None and re.fullmatch(r'[A-Z][a-z]?', token):
            try:
                parsed = Chem.Atom(token)
            except Exception:
                pass
        label = token
    if parsed is None:
        a.update(symbol='*', label=label, kind='alias')
        for k in ('formal_charge','isotope','explicit_hs','no_implicit','aromatic','source_stereo_token'):
            a.pop(k, None)
    else:
        # Existing normalized attributes override defaults from a plain symbol.
        fields = {'formal_charge':parsed.GetFormalCharge(), 'isotope':parsed.GetIsotope(),
                  'explicit_hs':parsed.GetNumExplicitHs(), 'no_implicit':parsed.GetNoImplicit(),
                  'aromatic':parsed.GetIsAromatic()}
        if a.get('kind') == 'atom' and a.get('normalized_symbol') == token:
            fields.update({k:a[k] for k in fields if k in a})
        if '@' in token:
            a['source_stereo_token'] = token
        a.update(symbol=parsed.GetSymbol().lower() if fields['aromatic'] else parsed.GetSymbol(),
                 label=None, kind='atom', **fields)
        a['normalized_symbol'] = a['symbol']
    a.pop('atom_symbol', None)
    a.pop('atom_label', None)
    return a


def normalize_bond(b: Any) -> tuple[int,int,int]:
    if isinstance(b, dict):
        u,v = b.get('endpoint_atoms', [b.get('u',b.get('begin_atom',0)), b.get('v',b.get('end_atom',1))])
        t = b.get('bond_type',1)
    else:
        u,v = b[:2]
        t = b[2] if len(b)>2 else 1
    if isinstance(t,str):
        t = BOND_ORDER[t.lower()] if not t.isdigit() else int(t)
    return int(u),int(v),int(t)


def _canonical_labelled(mol, isomeric=True):
    """Canonical CXSMILES without coordinates; labels participate in ranking.

    AtomMapNum is temporary and is cleared before writing. A separate mapping
    token in the returned key makes it collision-free across label vocabularies.
    Real chemical isotopes are never overwritten.
    """
    m=Chem.Mol(mol)
    labels=sorted({a.GetProp('atomLabel') for a in m.GetAtoms() if a.HasProp('atomLabel')})
    for a in m.GetAtoms():
        if a.HasProp('atomLabel'):
            a.SetAtomMapNum(labels.index(a.GetProp('atomLabel'))+1)
    key=Chem.MolToSmiles(m,canonical=True,isomericSmiles=isomeric)
    return json.dumps(labels,ensure_ascii=False,separators=(',',':'))+'|'+key


def _graph_equivalence_key_mol(mol):
    """Canonical graph key used by GEM.

    GEM evaluates the chemical graph after normalizing presentation details:
    alias text (R1/R2/X/Me/...) is collapsed to a generic detected placeholder,
    explicit H is converted to implicit H where possible, and stereochemical
    drawing direction is ignored. Real atoms, formal charge, isotope, aromaticity,
    connectivity and bond order still have to match exactly.
    """
    m=Chem.Mol(mol)
    for atom in m.GetAtoms():
        atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        atom.SetAtomMapNum(0)
        if atom.GetAtomicNum()==0:
            atom.SetIsotope(0)
            for prop in ('atomLabel','molFileAlias','_displayLabel','_displayLabelW'):
                if atom.HasProp(prop):atom.ClearProp(prop)
    for bond in m.GetBonds():
        bond.SetStereo(Chem.BondStereo.STEREONONE)
        bond.SetBondDir(Chem.BondDir.NONE)
    try:
        m=Chem.RemoveHs(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=False)


def convert_graph(atoms, bonds, brackets=None, image_size=None, coordinates_normalized=False) -> dict:
    aa=[normalize_atom(copy.deepcopy(a)) for a in atoms]
    out={'schema':SCHEMA_VERSION,'atoms':aa,'bonds':[], 'valid':False,
         'smiles':'','canonical':None,'graph_key':None,'errors':[],'warnings':[]}
    if not aa:
        out['errors'].append({'message':'empty graph'})
        return out
    rw=Chem.RWMol()
    for a in aa:
        if a['kind']=='alias':
            atom=Chem.Atom(0);atom.SetProp('atomLabel',a['label'])
        else:
            atom=Chem.Atom(a['symbol'].capitalize())
            atom.SetFormalCharge(int(a['formal_charge']));atom.SetIsotope(int(a['isotope']))
            atom.SetNumExplicitHs(int(a['explicit_hs']));atom.SetNoImplicit(bool(a['no_implicit']))
            atom.SetIsAromatic(bool(a['aromatic']))
        rw.AddAtom(atom)
    types={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,
           4:Chem.BondType.AROMATIC,5:Chem.BondType.SINGLE,6:Chem.BondType.SINGLE,7:Chem.BondType.UNSPECIFIED}
    try:
        for b in bonds:
            u,v,t=normalize_bond(b)
            if not(0<=u<len(aa) and 0<=v<len(aa)) or u==v or t not in types:
                raise ValueError(f'invalid bond {u},{v},{t}')
            if rw.GetBondBetweenAtoms(u,v):
                raise ValueError(f'duplicate bond {u},{v}')
            rw.AddBond(u,v,types[t]);out['bonds'].append([u,v,t])
            if t==4:
                rw.GetAtomWithIdx(u).SetIsAromatic(True);rw.GetAtomWithIdx(v).SetIsAromatic(True)
            if t in (5,6):
                rw.GetBondBetweenAtoms(u,v).SetBondDir(Chem.BondDir.BEGINWEDGE if t==5 else Chem.BondDir.BEGINDASH)
    except Exception as e:
        out['errors'].append({'message':str(e)})
        return out
    m=rw.GetMol()
    try:
        # RDKit 2022's chemistry diagnostics may query explicit valence before
        # initializing the property cache (notably aromatic dummy neighbours).
        # This initialization is permissive; the following full diagnostics
        # and SanitizeMol still enforce chemical validity.
        m.UpdatePropertyCache(strict=False)
        problems=Chem.DetectChemistryProblems(m)
        for p in problems:
            rec={'message':p.Message(),'type':p.GetType()}
            if hasattr(p,'GetAtomIdx'):rec['atom_index']=p.GetAtomIdx()
            if hasattr(p,'GetAtomIndices'):rec['atom_indices']=list(p.GetAtomIndices())
            out['errors'].append(rec)
        if out['errors']:return out
        Chem.SanitizeMol(m)
        have_xy=all('x' in a and 'y' in a for a in aa)
        if have_xy:
            w,h=image_size or (1,1)
            conf=Chem.Conformer(len(aa));conf.Set3D(False)
            for i,a in enumerate(aa):
                x,y=float(a['x']),float(a['y'])
                if coordinates_normalized:x,y=x*w,y*h
                conf.SetAtomPosition(i,(x,-y,0))
            m.AddConformer(conf)
            Chem.AssignChiralTypesFromBondDirs(m,replaceExistingTags=True)
            Chem.DetectBondStereochemistry(m)
            Chem.AssignStereochemistry(m,cleanIt=True,force=True)
        elif any(t in (5,6) for _,_,t in out['bonds']):
            out['warnings'].append('stereo_coordinates_missing')
        wedge_nodes={i for u,v,t in out['bonds'] if t in (5,6) for i in (u,v)}
        unresolved=[i for i,a in enumerate(aa) if a.get('source_stereo_token') and i not in wedge_nodes]
        if unresolved:
            out['warnings'].append('stereo_token_without_direction:'+','.join(map(str,unresolved)))
        if brackets:
            out['warnings'].append('brackets_require_graph_review')
        out['smiles']=Chem.MolToCXSmiles(m,Chem.SmilesWriteParams(),Chem.CXSmilesFields.CX_ATOM_LABELS)
        out['canonical']=_canonical_labelled(m,True)
        out['graph_key']=_graph_equivalence_key_mol(m)
        out['valid']=True
    except Exception as e:
        out['errors'].append({'message':str(e),'type':type(e).__name__})
    return out


def canonicalize_strict(smiles: str, isomeric=True):
    if not smiles:return None
    try:
        m=Chem.MolFromSmiles(smiles)
        return _canonical_labelled(m,isomeric) if m is not None else None
    except Exception:
        return None


def graph_equivalence_key(smiles: str):
    """Return the relaxed graph key used by GEM, or ``None`` if invalid."""
    if not smiles:return None
    try:
        m=Chem.MolFromSmiles(smiles)
        return _graph_equivalence_key_mol(m) if m is not None else None
    except Exception:
        return None


def graph_equivalent(s1: str,s2: str) -> bool:
    """Graph equivalence ignoring alias text, stereo drawing and explicit H."""
    a,b=graph_equivalence_key(s1),graph_equivalence_key(s2)
    return a is not None and b is not None and a==b


def tanimoto_strict(s1: str,s2: str):
    """Morgan skeleton similarity; alias text does not participate."""
    try:
        a,b=Chem.MolFromSmiles(s1),Chem.MolFromSmiles(s2)
        if a is None or b is None:return None
        return float(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(a,2,nBits=2048),AllChem.GetMorganFingerprintAsBitVect(b,2,nBits=2048)))
    except Exception:return None
