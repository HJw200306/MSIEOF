#!/usr/bin/env python3
"""Train 3-class semantic segmentation: background / dashed_arrow / bond."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from collections import defaultdict
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
CLASS_NAMES = ["background", "dashed_arrow", "bond"]
IGNORE_INDEX = 255


class ArrowBondUNetPlusPlus(nn.Module):
    def __init__(self, encoder: str = "resnet101"):
        super().__init__()
        self.model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
            activation=None,
        )

    def forward(self, x):
        return self.model(x)


class SegmentationDataset(Dataset):
    def __init__(self, images_dir: Path, masks_dir: Path, size: int, training: bool,
                 extra_train_data: Path | None = None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.files = sorted(
            path for path in images_dir.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
            and (masks_dir / f"{path.stem}.png").is_file()
        )
        self.samples = [(path.stem, path, masks_dir / f"{path.stem}.png") for path in self.files]
        if training and extra_train_data is not None:
            extra_images, extra_masks = extra_train_data / "images", extra_train_data / "masks"
            for path in sorted(extra_images.glob("*.png")):
                mask_path = extra_masks / path.name
                if not mask_path.is_file():
                    raise FileNotFoundError(f"额外数据缺少掩膜: {mask_path}")
                self.samples.append((f"document_{path.stem}", path, mask_path))
        self.sample_ids = [sample_id for sample_id, _, _ in self.samples]
        if not self.files:
            raise RuntimeError(f"没有图像/掩膜配对: {images_dir}")
        self.training = training
        self.size = size
        if training:
            self.augmentation = albu.Compose([
                albu.HorizontalFlip(p=.5),
                albu.VerticalFlip(p=.5),
                albu.RandomRotate90(p=.5),
                albu.ShiftScaleRotate(
                    shift_limit=.04, scale_limit=.08, rotate_limit=6, p=.45,
                    interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                    value=(255, 255, 255), mask_value=IGNORE_INDEX,
                ),
            ])
        else:
            # Validation uses the complete image.  Cropping a wide chemistry
            # figure to its centre would silently omit arrows near both ends.
            self.augmentation = None
        self.normalize = albu.Normalize(mean=(.485, .456, .406), std=(.229, .224, .225))

    def __len__(self):
        return len(self.samples)

    def _mixed_crop(self, image: np.ndarray, mask: np.ndarray):
        """50% arrow-centred, 25% arrow/bond-neighbourhood, 25% random."""
        h, w = mask.shape
        pad_h, pad_w = max(0, self.size - h), max(0, self.size - w)
        if pad_h or pad_w:
            image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_CONSTANT, value=(255, 255, 255))
            mask = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_CONSTANT, value=IGNORE_INDEX)
            h, w = mask.shape
        draw = random.random()
        candidates = None
        if draw < .50:
            candidates = np.argwhere(mask == 1)
        elif draw < .75:
            arrow = (mask == 1).astype(np.uint8)
            bond = (mask == 2).astype(np.uint8)
            kernel = np.ones((21, 21), np.uint8)
            contact = (cv2.dilate(arrow, kernel) > 0) & (cv2.dilate(bond, kernel) > 0)
            candidates = np.argwhere(contact & ((mask == 1) | (mask == 2)))
            if not len(candidates):
                candidates = np.argwhere(mask == 1)
        if candidates is not None and len(candidates):
            cy, cx = candidates[random.randrange(len(candidates))]
            y_min, y_max = max(0, int(cy) - self.size + 1), min(int(cy), h - self.size)
            x_min, x_max = max(0, int(cx) - self.size + 1), min(int(cx), w - self.size)
            y0 = random.randint(y_min, max(y_min, y_max))
            x0 = random.randint(x_min, max(x_min, x_max))
        else:
            y0 = random.randint(0, h - self.size)
            x0 = random.randint(0, w - self.size)
        return image[y0:y0+self.size, x0:x0+self.size], mask[y0:y0+self.size, x0:x0+self.size]

    def __getitem__(self, index: int):
        _, image_path, mask_path = self.samples[index]
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
            image, mask = self._mixed_crop(image, mask)
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


def masked_dice_loss(logits: torch.Tensor, target: torch.Tensor,
                     mode: str = "all") -> torch.Tensor:
    valid = target != IGNORE_INDEX
    safe_target = target.masked_fill(~valid, 0)
    one_hot = torch.nn.functional.one_hot(safe_target, 3).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = torch.softmax(logits, dim=1) * valid_f
    one_hot *= valid_f
    intersection = (probs * one_hot).sum(dim=(0, 2, 3))
    denominator = (probs + one_hot).sum(dim=(0, 2, 3))
    dice = (2 * intersection + 1.0) / (denominator + 1.0)
    selected = dice if mode == "all" else dice[1:]
    return 1.0 - selected.mean()


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
        result[f"precision_{name}"] = float(tp[index] / max(tp[index] + fp[index], 1))
        result[f"recall_{name}"] = float(tp[index] / max(tp[index] + fn[index], 1))
    result["foreground_miou"] = float(iou[1:].mean())
    result["foreground_mdice"] = float(dice[1:].mean())
    result["arrow_residual_rate"] = float(fn[1] / max(tp[1] + fn[1], 1))
    result["bond_false_erasure_rate"] = float(confusion[2, 1] / max(confusion[2].sum(), 1))
    result["background_erasure_rate"] = float(confusion[0, 1] / max(confusion[0].sum(), 1))
    return result


def run_epoch(model, loader, device, optimizer, ce_loss, dice_weight: float, *,
              dice_mode: str = "all", scaler=None, use_amp: bool = False):
    training = optimizer is not None
    model.train(training)
    confusion = np.zeros((3, 3), dtype=np.int64)
    total_loss = 0.0
    negative_images = negative_images_erased = 0
    negative_pixels = negative_pixels_erased = 0
    iterator = tqdm(loader, desc="train" if training else "val", leave=False)
    for images, targets in iterator:
        images, targets = images.to(device), targets.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss_ce = ce_loss(logits, targets)
                loss_dice = masked_dice_loss(logits, targets, dice_mode)
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
            valid = t != IGNORE_INDEX
            if not np.any(t == 1):
                negative_images += 1
                erased = valid & (p == 1)
                negative_images_erased += int(erased.any())
                negative_pixels += int(valid.sum())
                negative_pixels_erased += int(erased.sum())
        iterator.set_postfix(loss=f"{loss.item():.4f}")
    values = metrics(confusion)
    values.update(
        no_arrow_images=negative_images,
        no_arrow_image_false_positive_rate=float(negative_images_erased / max(negative_images, 1)),
        no_arrow_pixel_erasure_rate=float(negative_pixels_erased / max(negative_pixels, 1)),
    )
    return total_loss / len(loader.dataset), values


class BalancedParents(Dataset):
    """50/50 real/synthetic; then uniform parent and uniform view."""
    def __init__(self, base: SegmentationDataset, rows: list[dict], length: int):
        indices = {sample_id: i for i, sample_id in enumerate(base.sample_ids)}
        groups = {"real": defaultdict(list), "synthetic": defaultdict(list)}
        for row in rows:
            if row["split"] != "train":
                continue
            domain = "real" if row["kind"] == "real" else "synthetic"
            groups[domain][row["parent_id"]].append(indices[row["id"]])
        for sample_id, index in indices.items():
            if sample_id.startswith("document_"):
                groups["synthetic"][sample_id].append(index)
        self.base = base
        self.parents = {domain: list(parent_views.values())
                        for domain, parent_views in groups.items()}
        self.length = length
        if not all(self.parents.values()):
            raise ValueError("真实/合成父样本均不能为空")

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        domain = "real" if index % 2 == 0 else "synthetic"
        return self.base[random.choice(random.choice(self.parents[domain]))]


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def main() -> int:
    parser = argparse.ArgumentParser(description="训练三分类虚线箭头语义分割模型")
    parser.add_argument("--data", type=Path, default=MODULE_ROOT / "dataset" / "DashedArrow-Seg")
    parser.add_argument("--output", type=Path, default=MODULE_ROOT / "runs" / "unetpp_resnet101")
    parser.add_argument("--encoder", default="resnet101")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--dice-weight", type=float, default=.5)
    parser.add_argument("--dice-mode", choices=("all", "foreground"), default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch-samples", type=int, default=0,
                        help="每轮抽样数；0=较大来源父样本数的2倍")
    parser.add_argument("--extra-train-data", type=Path,
                        help="额外候选数据目录，须含images/和masks/；仅加入训练")
    parser.add_argument("--init-checkpoint", type=Path,
                        help="从三分类checkpoint初始化，用于固定起点的小型微调实验")
    parser.add_argument("--max-bond-false-erasure", type=float, default=.003,
                        help="选模时允许的真实验证化学键误删率上限")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--no-amp", action="store_true", help="禁用 CUDA 混合精度")
    args = parser.parse_args()
    if args.epoch_samples < 0 or args.max_bond_false_erasure < 0:
        parser.error("epoch-samples和max-bond-false-erasure必须非负")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    manifest_path = data / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"父样本均衡采样需要manifest.csv: {manifest_path}")
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    extra_data = args.extra_train_data.expanduser().resolve() if args.extra_train_data else None
    train_base = SegmentationDataset(
        data / "train" / "images", data / "train" / "masks", args.size, True,
        extra_train_data=extra_data,
    )
    parent_counts = defaultdict(set)
    for row in manifest_rows:
        if row["split"] == "train":
            domain = "real" if row["kind"] == "real" else "synthetic"
            parent_counts[domain].add(row["parent_id"])
    extra_count = 0 if extra_data is None else len(list((extra_data / "images").glob("*.png")))
    sample_count = args.epoch_samples or 2 * max(
        len(parent_counts["real"]), len(parent_counts["synthetic"]) + extra_count
    )
    if sample_count % 2:
        sample_count += 1
    train_set = BalancedParents(train_base, manifest_rows, sample_count)
    val_set = SegmentationDataset(data / "val" / "images", data / "val" / "masks", args.size, False)
    synthetic_val_images = data / "val_synthetic" / "images"
    synthetic_val_masks = data / "val_synthetic" / "masks"
    synthetic_val_set = (
        SegmentationDataset(synthetic_val_images, synthetic_val_masks, args.size, False)
        if synthetic_val_images.is_dir() and synthetic_val_masks.is_dir()
        else None
    )
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_set, args.batch_size, shuffle=False, num_workers=args.workers,
        worker_init_fn=seed_worker, generator=generator,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, 1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
    synthetic_val_loader = (
        DataLoader(synthetic_val_set, 1, shuffle=False, num_workers=args.workers, pin_memory=device.type == "cuda")
        if synthetic_val_set is not None else None
    )
    model = ArrowBondUNetPlusPlus(args.encoder).to(device)
    if args.init_checkpoint:
        checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
        if checkpoint.get("class_names") not in (None, CLASS_NAMES):
            raise ValueError("初始化checkpoint类别不匹配")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=.5, patience=4)
    class_weights = torch.tensor([.20, 1.0, 1.35], device=device)
    ce_loss = nn.CrossEntropyLoss(weight=class_weights, ignore_index=IGNORE_INDEX)
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best = -1.0
    stale = 0
    history_path = output / "history.csv"
    config = vars(args).copy(); config["data"] = str(data); config["output"] = str(output); config["device"] = str(device); config["amp"] = use_amp
    config.update(
        epoch_samples_actual=sample_count,
        parent_counts={key: len(value) for key, value in parent_counts.items()},
        extra_train_samples=extra_count,
        sampling="50% real / 50% synthetic; uniform parent then uniform view",
        crop_sampling="50% arrow / 25% arrow-bond neighbourhood / 25% random",
        padding="RGB white; new mask pixels ignore=255",
        selection=("maximize real-val arrow IoU subject to bond false erasure "
                   f"<= {args.max_bond_false_erasure}"),
        manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )
    (output / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    metric_names = list(metrics(np.zeros((3, 3), dtype=int)).keys())
    metric_names += ["no_arrow_images", "no_arrow_image_false_positive_rate",
                     "no_arrow_pixel_erasure_rate"]
    fields = [
        "epoch", "train_loss", "val_loss", "synthetic_val_loss", "lr",
        "selection_eligible", "selection_score",
    ]
    fields += [f"val_{key}" for key in metric_names]
    fields += [f"synthetic_val_{key}" for key in metric_names]
    with open(history_path, "w", newline="", encoding="utf-8-sig") as history:
        writer = csv.DictWriter(history, fieldnames=fields); writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, _ = run_epoch(
                model, train_loader, device, optimizer, ce_loss, args.dice_weight,
                dice_mode=args.dice_mode, scaler=scaler, use_amp=use_amp,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            val_loss, val_metrics = run_epoch(
                model, val_loader, device, None, ce_loss, args.dice_weight,
                dice_mode=args.dice_mode, use_amp=use_amp,
            )
            if synthetic_val_loader is not None:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                synthetic_val_loss, synthetic_val_metrics = run_epoch(
                    model, synthetic_val_loader, device, None, ce_loss, args.dice_weight,
                    dice_mode=args.dice_mode, use_amp=use_amp,
                )
            else:
                synthetic_val_loss = float("nan")
                synthetic_val_metrics = {key: float("nan") for key in metric_names}
            eligible = val_metrics["bond_false_erasure_rate"] <= args.max_bond_false_erasure
            score = val_metrics["iou_dashed_arrow"] if eligible else -1.0
            scheduler.step(score)
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "synthetic_val_loss": synthetic_val_loss,
                "lr": optimizer.param_groups[0]["lr"],
                "selection_eligible": eligible,
                "selection_score": score,
            }
            row.update({f"val_{key}": value for key, value in val_metrics.items()})
            row.update({f"synthetic_val_{key}": value for key, value in synthetic_val_metrics.items()})
            writer.writerow(row); history.flush()
            print(
                f"epoch {epoch:03d}: train={train_loss:.4f} real_val={val_loss:.4f} "
                f"selection={score:.4f} arrow={val_metrics['iou_dashed_arrow']:.4f} "
                f"bond={val_metrics['iou_bond']:.4f} "
                f"synthetic_fg_mIoU={synthetic_val_metrics['foreground_miou']:.4f}"
            )
            checkpoint = {
                "schema_version": 1,
                "model_state_dict": model.state_dict(),
                "encoder": args.encoder,
                "num_classes": 3,
                "class_names": CLASS_NAMES,
                "ignore_index": IGNORE_INDEX,
                "epoch": epoch,
                "metrics": val_metrics,
                "synthetic_metrics": synthetic_val_metrics,
                "selection_split": "val_real",
                "selection_rule": config["selection"],
                "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            }
            torch.save(checkpoint, output / "last.pth")
            if eligible and score > best:
                best = score; stale = 0
                torch.save(checkpoint, output / "best.pth")
            else:
                stale += 1
                if stale >= args.patience:
                    print(f"early stop: {stale} epochs without improvement")
                    break
    print(f"best eligible arrow IoU={best:.4f}; model={output / 'best.pth'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

