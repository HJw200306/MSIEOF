"""
Test script to compare arrow erasing with and without bond protection.
Generates side-by-side comparison showing the improvement.
"""

import os
import sys
import cv2
import torch
import numpy as np
import albumentations as albu
from pathlib import Path
from PIL import Image

# Add project paths
_AE_ROOT = Path(__file__).parent
_PROJECT_ROOT = _AE_ROOT.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from arrow_earser_val import (
    ArrowUNetPlusPlus, PreprocessorWrapper,
    refine_prediction, erase_arrows, detect_bonds,
    get_full_image_aug, CONFIG
)
from split_tools.unet import Unet


def main():
    test_image = "journal_0041.png"

    images_dir = Path(CONFIG["BASE_PATH"]) / "val" / "images"
    img_path = images_dir / test_image

    if not img_path.exists():
        print(f"❌ Test image not found: {img_path}")
        print("Available images:")
        for f in sorted(images_dir.glob("*.png"))[:5]:
            print(f"  - {f.name}")
        return

    print(f"Testing bond protection on: {test_image}\n")

    # Load models
    print("Loading dashed arrow model...")
    dashed_model = ArrowUNetPlusPlus(num_classes=CONFIG["CLASSES"]).to(CONFIG["DEVICE"])
    state = torch.load(CONFIG["MODEL_PATH"], map_location=CONFIG["DEVICE"], weights_only=True)
    dashed_model.load_state_dict(state)
    dashed_model.eval()

    print("Loading bond detection model...")
    cuda = torch.cuda.is_available()
    bond_model = Unet(
        str(CONFIG["BOND_MODEL_PATH"]),
        num_classes=3,
        input_shape=[512, 512],
        cuda=cuda
    )
    print("✓ Models loaded\n")

    # Load and process image
    image = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = image_rgb.shape[:2]

    # Preprocessing
    aug = get_full_image_aug()
    preprocessor = PreprocessorWrapper()
    normalize = albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    # Augment
    augmented = aug(image=image_rgb, mask=np.zeros((h_orig, w_orig), dtype=np.uint8))
    img_aug = augmented["image"]

    # Normalize and convert to tensor
    normalized = normalize(image=img_aug)
    img_tensor = torch.from_numpy(
        normalized["image"].transpose(2, 0, 1).astype("float32")
    ).unsqueeze(0).to(CONFIG["DEVICE"])

    # Inference - dashed arrow detection
    print("Detecting dashed arrows...")
    with torch.no_grad():
        output = dashed_model(img_tensor)
        pred_raw = torch.argmax(output, dim=1).cpu().numpy().squeeze().astype(np.uint8)

    pred_raw = pred_raw[:h_orig, :w_orig]
    pred_refined = refine_prediction(pred_raw)

    # Detect bonds
    print("Detecting chemical bonds...")
    bond_mask = detect_bonds(bond_model, image_rgb)

    # Count overlaps
    overlap_pixels = np.sum(pred_refined & bond_mask)
    arrow_pixels = np.sum(pred_refined)
    bond_pixels = np.sum(bond_mask)

    print(f"\nDetection results:")
    print(f"  Dashed arrow pixels: {arrow_pixels}")
    print(f"  Bond pixels: {bond_pixels}")
    print(f"  Overlap pixels: {overlap_pixels} ({100*overlap_pixels/arrow_pixels:.1f}% of arrow)")

    # Erase without bond protection
    print("\nErasing without bond protection...")
    erased_no_protection = erase_arrows(image_rgb, pred_refined, bond_mask=None)

    # Erase with bond protection
    print("Erasing with bond protection...")
    erased_with_protection = erase_arrows(image_rgb, pred_refined, bond_mask=bond_mask)

    # Create comparison visualization
    output_dir = _AE_ROOT / "comparison"
    output_dir.mkdir(exist_ok=True)

    original_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    # Create bond overlay for visualization
    bond_overlay = original_bgr.copy()
    bond_overlay[bond_mask] = [0, 0, 255]  # Red for bonds
    bond_overlay = cv2.addWeighted(original_bgr, 0.7, bond_overlay, 0.3, 0)

    # Create arrow overlay
    arrow_overlay = original_bgr.copy()
    arrow_overlay[pred_refined > 0] = [0, 255, 255]  # Yellow for arrows
    arrow_overlay = cv2.addWeighted(original_bgr, 0.7, arrow_overlay, 0.3, 0)

    # Create 4-panel comparison
    h = original_bgr.shape[0]
    divider = np.full((h, 3, 3), 80, dtype=np.uint8)

    comparison = np.hstack([
        original_bgr, divider,
        erased_no_protection, divider,
        erased_with_protection, divider,
        bond_overlay
    ])

    # Add labels
    label_height = 30
    labels = np.full((label_height, comparison.shape[1], 3), 255, dtype=np.uint8)

    panel_width = original_bgr.shape[1]
    label_texts = [
        "Original",
        "Without Protection",
        "With Bond Protection",
        "Bond Detection"
    ]

    for i, text in enumerate(label_texts):
        x_pos = i * (panel_width + 3) + 10
        cv2.putText(labels, text, (x_pos, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    final_comparison = np.vstack([labels, comparison])

    # Save results
    comparison_path = output_dir / f"{test_image.replace('.png', '_bond_protection_comparison.png')}"
    cv2.imwrite(str(comparison_path), final_comparison)

    # Save individual masks
    cv2.imwrite(str(output_dir / f"{test_image.replace('.png', '_arrow_mask.png')}"),
                pred_refined * 255)
    cv2.imwrite(str(output_dir / f"{test_image.replace('.png', '_bond_mask.png')}"),
                bond_mask.astype(np.uint8) * 255)

    print(f"\n✓ Comparison saved to: {comparison_path}")
    print(f"  Layout: [Original | No Protection | With Protection | Bond Overlay]")
    print(f"\nBond protection prevents erasing {overlap_pixels} pixels that overlap with molecular bonds!")


if __name__ == "__main__":
    main()
