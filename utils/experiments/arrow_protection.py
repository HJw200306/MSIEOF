"""Conservative arrow erasure using image evidence, without GT or OCSR results.

Parameters are fixed before the v2 ablation; no test-score-driven tuning.
Connected neutral ink touching a chemical-bond prediction is protected as a
whole. Nearby small components protect letters and isolated wedge strokes.
"""
from __future__ import annotations
import cv2
import numpy as np

PROTECTION_VERSION='conservative-ink-v1.0'
PROTECTION_PARAMETERS={'dashed_probability':0.90,'ink_threshold':200,
    'neutral_range':60,'bond_dilation_linewidth':1.5,
    'near_text_linewidth':6,'solid_min_aspect':2.5,'dashed_min_aspect':2.5}


def protect_and_erase(rgb, solid_mask, bond_mask, dashed_probability):
    rgb=np.asarray(rgb,dtype=np.uint8)
    h,w=rgb.shape[:2]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    ink=gray<PROTECTION_PARAMETERS['ink_threshold']
    neutral=ink & (np.ptp(rgb,axis=2)<PROTECTION_PARAMETERS['neutral_range'])
    dist=cv2.distanceTransform(neutral.astype(np.uint8),cv2.DIST_L2,3)
    linewidth=float(np.clip(2*np.median(dist[neutral]) if neutral.any() else 1,1,6))
    radius=max(1,int(round(1.5*linewidth)))
    core=cv2.dilate(bond_mask.astype(np.uint8),np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
    protected=core.copy()
    n,cc,stats,_=cv2.connectedComponentsWithStats(neutral.astype(np.uint8),connectivity=8)
    small_limit=max(12*linewidth,0.10*min(h,w))
    # First extend protection along complete connected chemical strokes.
    for k in range(1,n):
        x,y,cw,ch,area=stats[k]
        component=(cc[y:y+ch,x:x+cw]==k)
        overlap=bool(np.any(core[y:y+ch,x:x+cw][component]))
        if overlap:protected[y:y+ch,x:x+cw][component]=True
    # Then locate letters relative to the complete protected strokes, including
    # endpoints the segmentation only recognized in part.
    if protected.any():
        distance_to_bond=cv2.distanceTransform((~protected).astype(np.uint8),cv2.DIST_L2,3)
    else:
        distance_to_bond=np.full((h,w),np.inf)
    for k in range(1,n):
        x,y,cw,ch,area=stats[k]
        component=(cc[y:y+ch,x:x+cw]==k)
        near=float(distance_to_bond[y:y+ch,x:x+cw][component].min())
        small=(max(cw,ch)<=small_limit and area>=2)
        if small and near<=max(6*linewidth,0.6*max(cw,ch)):
            protected[y:y+ch,x:x+cw][component]=True
    # Only modestly expand accepted protection; avoid a fixed pixel radius
    # across original/SR resolutions.
    protected=cv2.dilate(protected.astype(np.uint8),np.ones((3,3),np.uint8)).astype(bool)
    dashed_mask=dashed_probability>=PROTECTION_PARAMETERS['dashed_probability']
    accepted={}
    region_log=[]
    for name,mask in [('solid',solid_mask),('dashed',dashed_mask)]:
        gap=max(1,int(round(2*linewidth)))
        group=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,np.ones((2*gap+1,2*gap+1),np.uint8))
        count,groups,gstats,_=cv2.connectedComponentsWithStats(group,connectivity=8)
        allowed=np.zeros((h,w),bool)
        for k in range(1,count):
            x,y,cw,ch,_=gstats[k]
            region=(groups[y:y+ch,x:x+cw]==k) & mask[y:y+ch,x:x+cw]
            ink_region=region & ink[y:y+ch,x:x+cw]
            ys,xs=np.where(ink_region)
            if len(xs)<3:continue
            colored=float(np.mean(np.ptp(rgb[y:y+ch,x:x+cw],axis=2)[ink_region]>=60))
            eig=np.linalg.eigvalsh(np.cov(np.stack([xs,ys]))) if len(xs)>3 else np.array([1,1])
            aspect=float(np.sqrt(max(eig[-1],1)/max(eig[0],1)))
            span=max(cw,ch)
            if name=='solid':
                confirmed=(span>=6*linewidth and colored>=0.35) or (span>=12*linewidth and aspect>=2.5)
            else:
                pieces=cv2.connectedComponents(ink_region.astype(np.uint8),connectivity=8)[0]-1
                confirmed=(span>=8*linewidth and colored>=0.35) or (span>=16*linewidth and aspect>=2.5 and pieces>=3)
            if confirmed:allowed[y:y+ch,x:x+cw] |= region
            region_log.append({'branch':name,'bbox':[int(x),int(y),int(cw),int(ch)],
                'ink_pixels':int(len(xs)),'colored_fraction':colored,'aspect':aspect,
                'accepted':bool(confirmed)})
        accepted[name]=allowed & ~protected
    erase=accepted['solid'] | accepted['dashed']
    out=rgb.copy();out[erase]=255
    return out,{'solid_proposed':solid_mask,'bond':bond_mask,
        'dashed_proposed':dashed_probability>=0.5,'protected':protected,
        'solid_erased':accepted['solid'],'dashed_erased':accepted['dashed'],
        'erased':erase,'linewidth':linewidth,'regions':region_log,
        'version':PROTECTION_VERSION}
