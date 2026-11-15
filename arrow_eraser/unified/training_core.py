#!/usr/bin/env python3
"""Train 3-class semantic segmentation: background / arrow / bond."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from datetime import datetime
from pathlib import Path

import albumentations as albu
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


ROOT = Path(__file__).resolve().parent
MODULE_ROOT = ROOT.parent
CLASS_NAMES = ["background", "arrow", "bond"]
IGNORE_INDEX = 255


class ArrowBondUNetPlusPlus(nn.Module):
    def __init__(self, encoder: str = "resnet101", encoder_weights="imagenet"):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, size: int, training: bool):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = sorted(
            path for path in images_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and (masks_dir / f"{path.stem}.png").is_file()
        )
        if not self.files:
            raise RuntimeError(f"没有图像/掩膜配对: {images_dir}")
        self.training = training
        self.size = size
        if training:
            transforms = [
                albu.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0),
                albu.CropNonEmptyMaskIfExists(height=size, width=size, p=1.0),
                albu.HorizontalFlip(p=.5),
                albu.VerticalFlip(p=.5),
                albu.RandomRotate90(p=.5),
                albu.ShiftScaleRotate(
                    shift_limit=.04, scale_limit=.08, rotate_limit=6, p=.45,
                    border_mode=cv2.BORDER_CONSTANT, value=255, mask_value=0,
                ),
            ]
            self.augmentation = albu.Compose(transforms)
        else:
            # Validation uses the complete image.  Cropping a wide chemistry
            # figure to its centre would silently omit arrows near both ends.
            self.augmentation = None
        self.normalize = albu.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        image_path = self.files[index]
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None or image.shape[:2] != mask.shape:
            raise ValueError(f"样本损坏或尺寸不一致: {image_path}")
        values = set(int(v) for v in np.unique(mask))
        if not values <= {0, 1, 2, IGNORE_INDEX}:
            raise ValueError(f"掩膜含非法值 {values}: {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augmentation is not None:
            augmented = self.augmentation(image=image, mask=mask)
            image, mask = augmented["image"], augmented["mask"]
        else:
            height, width = mask.shape
            pad_h = (32 - height % 32) % 32
            pad_w = (32 - width % 32) % 32
            if pad_h or pad_w:
                image = cv2.copyMakeBorder(
                    image, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255),
                )
                mask = cv2.copyMakeBorder(
                    mask, 0, pad_h, 0, pad_w,
                    cv2.BORDER_CONSTANT, value=IGNORE_INDEX,
                )
        image = self.normalize(image=image)["image"].transpose(2, 0, 1).astype(np.float32)
        return torch.from_numpy(image), torch.from_numpy(mask.astype(np.int64))


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    one_hot = torch.nn.functional.one_hot(safe_target, 3).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = torch.softmax(logits, dim=1) * valid_f
    one_hot *= valid_f
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = (probs + one_hot).sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    return 1.0 - dice.mean()


def update_confusion(confusion: np.ndarray, pred: np.ndarray, target: np.ndarray) -> None:
    valid = target != IGNORE_INDEX
    encoded = target[valid] * 3 + pred[valid]
    confusion += np.bincount(encoded, minlength=9).reshape(3, 3)


def metrics(confusion: np.ndarray) -> dict:
    tp = np.diag(confusion).astype(float)
    fp = confusion.sum(axis=0) - tp
    fn = confusion.sum(axis=1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    dice = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    result = {}
    for index, name in enumerate(CLASS_NAMES):
        result[f"iou_{name}"] = float(iou[index])
        result[f"dice_{name}"] = float(dice[index])
    result["foreground_miou"] = float(iou[1:].mean())
    result["foreground_mdice"] = float(dice[1:].mean())
    return result


def run_epoch(model, loader, device, optimizer, ce_loss, dice_weight: float, *, scaler=None, use_amp: bool = False):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((3, 3), dtype=np.int64)
    total_loss = 0.0
    iterator = tqdm(loader, desc="train" if training else "val", leave=False)
    for images, targets in iterator:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss_ce = ce_loss(logits, targets)
                loss_dice = masked_dice_loss(logits, targets)
                loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
            if training and scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * images.size(0)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        truth = targets.detach().cpu().numpy()
        for p, t in zip(pred, truth):
            update_confusion(confusion, p, t)
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader.dataset), metrics(confusion)
