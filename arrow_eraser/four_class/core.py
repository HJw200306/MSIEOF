"""Shared data, losses and comparable metrics for three/four-class training."""
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

CLASS_NAMES = {3: ['background', 'arrow', 'bond'],
               4: ['background', 'solid_arrow', 'dashed_arrow', 'bond']}


def collapse(a):
    out = a.copy()
    out[(a == 1) | (a == 2)] = 1
    out[a == 3] = 2
    return out


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(root):
    with (root/'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def augment(image, mask, size, rng):
    # Identical transforms for both class definitions, using the four-class GT.
    if rng.random() < .5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < .5:
        image, mask = image[::-1], mask[::-1]
    k = rng.randrange(4)
    image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
    h, w = mask.shape
    if rng.random() < .45:
        matrix = cv2.getRotationMatrix2D((w/2, h/2), rng.uniform(-6, 6), rng.uniform(.92, 1.08))
        matrix[:, 2] += [rng.uniform(-.04, .04)*w, rng.uniform(-.04, .04)*h]
        image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
        mask = cv2.warpAffine(mask, matrix, (w, h), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    ph, pw = max(0, size-h), max(0, size-w)
    image = cv2.copyMakeBorder(image, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(255,255,255))
    mask = cv2.copyMakeBorder(mask, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=255)
    h, w = mask.shape
    arrow = (mask == 1) | (mask == 2)
    draw, points = rng.random(), None
    if draw < .5:
        points = np.argwhere(arrow)
    elif draw < .75:
        kernel = np.ones((21,21), np.uint8)
        near = (cv2.dilate(arrow.astype(np.uint8), kernel) > 0) & (cv2.dilate((mask == 3).astype(np.uint8), kernel) > 0)
        points = np.argwhere(near & (arrow | (mask == 3)))
        if len(points) == 0:
            points = np.argwhere(arrow)
    if points is not None and len(points):
        cy, cx = points[rng.randrange(len(points))]
        y = rng.randint(max(0, int(cy)-size+1), min(int(cy), h-size))
        x = rng.randint(max(0, int(cx)-size+1), min(int(cx), w-size))
    else:
        y, x = rng.randint(0, h-size), rng.randint(0, w-size)
    return image[y:y+size, x:x+size].copy(), mask[y:y+size, x:x+size].copy()


class ArrowDataset(Dataset):
    def __init__(self, root, split, classes=4, size=512, seed=42, epoch_samples=0):
        self.root, self.split, self.classes = Path(root), split, classes
        self.size, self.seed, self.epoch = size, seed, 0
        self.rows = [r for r in read_rows(self.root) if r['split'] == split]
        if not self.rows:
            raise ValueError(f'No {split} samples')
        groups = {'solid': defaultdict(list), 'dashed': defaultdict(list)}
        for r in self.rows:
            groups[r['domain']][r['parent_id']].append(r)
        self.parents = {d: [g[k] for k in sorted(g)] for d,g in groups.items()}
        self.length = len(self.rows)
        if split == 'train':
            if not all(self.parents.values()):
                raise ValueError('Both source domains required')
            self.length = epoch_samples or 2*max(map(len, self.parents.values()))
            if self.length % 2:
                raise ValueError('epoch_samples must be even')

    def __len__(self):
        return self.length

    def choice(self, index):
        # Local RNG independent of model creation, worker count and class count.
        rng = random.Random(f'{self.seed}:{self.epoch}:{index}')
        if self.split == 'train':
            row = rng.choice(rng.choice(self.parents['solid' if index % 2 == 0 else 'dashed']))
        else:
            row = self.rows[index]
        return row, rng

    def __getitem__(self, index):
        row, rng = self.choice(index)
        with Image.open(self.root/row['image']) as im:
            image = np.array(im.convert('RGB'))
        with Image.open(self.root/row['mask']) as im:
            mask = np.array(im.convert('L'))
        if self.split == 'train':
            image, mask = augment(image, mask, self.size, rng)
        else:
            h,w = mask.shape
            image = cv2.copyMakeBorder(image,0,(-h)%32,0,(-w)%32,cv2.BORDER_CONSTANT,value=(255,255,255))
            mask = cv2.copyMakeBorder(mask,0,(-h)%32,0,(-w)%32,cv2.BORDER_CONSTANT,value=255)
        target = collapse(mask) if self.classes == 3 else mask
        image = (image.astype(np.float32)/255 - np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)
        return (torch.from_numpy(image.transpose(2,0,1).copy()),
                torch.from_numpy(target.astype(np.int64)), torch.from_numpy(mask.astype(np.int64)), row['id'])


def loss_fn(logits, target):
    n = logits.shape[1]
    valid = target != 255
    if not valid.any():
        return logits.sum()*0
    weights = logits.new_tensor([.2,1.,1.35] if n == 3 else [.2,1.,1.,1.35])
    ce = F.cross_entropy(logits, target, weight=weights, ignore_index=255)
    safe = target.masked_fill(~valid, 0)
    truth = F.one_hot(safe,n).permute(0,3,1,2).float()*valid[:,None]
    prob = logits.float().softmax(1)*valid[:,None]
    dice = (2*(prob*truth).sum((0,2,3))+1)/((prob+truth).sum((0,2,3))+1)
    dw = logits.new_tensor([0,.5,.5] if n == 3 else [0,.25,.25,.5])
    return .5*ce + .5*((1-dice)*dw).sum()


def ratio(a,b):
    return float(a/b) if b else None


class Metrics:
    def __init__(self):
        self.cm = np.zeros((3,3),np.int64)
        self.typed = np.zeros((4,4),np.int64)
        self.types = {1: np.zeros(3,np.int64), 2: np.zeros(3,np.int64)}
        self.neg = np.zeros(4,np.int64)
        self.images = 0

    def update(self, pred, truth, classes):
        merged = collapse(pred) if classes == 4 else pred
        gt = collapse(truth)
        valid = truth != 255
        self.cm += np.bincount(gt[valid]*3+merged[valid], minlength=9).reshape(3,3)
        if classes == 4:
            self.typed += np.bincount(truth[valid]*4+pred[valid], minlength=16).reshape(4,4)
        erased = merged == 1
        for t in (1,2):
            if (truth == t).any():
                # Type-conditioned removal metric, not type-classification IoU.
                use = valid & (truth != 3-t)
                tp = (erased & (truth == t)).sum()
                fp = (erased & use & (truth != t)).sum()
                fn = (~erased & (truth == t)).sum()
                self.types[t] += [tp,fp,fn]
        if not ((truth == 1)|(truth == 2)).any():
            n = (erased & valid).sum()
            self.neg += [1,int(n>0),int(valid.sum()),int(n)]
        self.images += 1

    def result(self):
        c = self.cm
        tp,fp,fn = c[1,1],c[:,1].sum()-c[1,1],c[1].sum()-c[1,1]
        out = {'images':self.images,'confusion':c.tolist(),'arrow_iou':ratio(tp,tp+fp+fn),
               'arrow_dice':ratio(2*tp,2*tp+fp+fn),'arrow_precision':ratio(tp,tp+fp),
               'arrow_recall':ratio(tp,tp+fn),'arrow_residual_rate':ratio(fn,tp+fn),
               'bond_false_erasure_rate':ratio(c[2,1],c[2].sum()),
               'background_erasure_rate':ratio(c[0,1],c[0].sum()),
               'negative_images':int(self.neg[0]),'negative_image_false_positive_rate':ratio(self.neg[1],self.neg[0]),
               'negative_pixel_erasure_rate':ratio(self.neg[3],self.neg[2]),
               'four_class_confusion':self.typed.tolist()}
        for t,name in [(1,'solid'),(2,'dashed')]:
            tp,fp,fn = self.types[t]
            out[name] = {'iou':ratio(tp,tp+fp+fn),'recall':ratio(tp,tp+fn),
                         'residual_rate':ratio(fn,tp+fn),'counts':[int(tp),int(fp),int(fn)]}
        return out
