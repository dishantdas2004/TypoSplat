import sys
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.data.mask_generator import get_letter_mask

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_letter_mask.py /path/to/output/<sample_id>")
        sys.exit(1)

    sample_dir = sys.argv[1]
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    meta_path = os.path.join(sample_dir, "metadata.json")
    mesh_path = os.path.join(sample_dir, "mesh.ply")

    if not view_matches or not os.path.exists(meta_path) or not os.path.exists(mesh_path):
        raise SystemExit(f"Missing view.png, mesh.ply, or metadata.json in {sample_dir}")

    # Load RGB
    img = Image.open(view_matches[0]).convert("RGB")
    img_np = np.array(img)
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    # Generate the mask at 148x148
    mask_148_tensor = get_letter_mask(mesh_path, meta, device=device)
    mask_148 = mask_148_tensor[0, 0].cpu().numpy()
    
    # Upsample it purely for visual overlay with the 518x518 RGB image
    mask_148_upscaled = torch.nn.functional.interpolate(
        mask_148_tensor.unsqueeze(0), size=(518, 518), mode='nearest'
    )[0, 0, 0].cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img_np)
    axes[0].set_title("Original RGB")
    axes[0].axis('off')

    axes[1].imshow(mask_148, cmap='gray')
    axes[1].set_title("Downsampled Letter Mask (148x148)")
    axes[1].axis('off')

    # Create a red overlay
    overlay = np.zeros_like(img_np)
    overlay[:, :, 0] = 255 # Red channel
    
    axes[2].imshow(img_np)
    axes[2].imshow(overlay, alpha=mask_148_upscaled * 0.5)
    axes[2].set_title("Mask Overlay (Should perfectly hug text)")
    axes[2].axis('off')

    plt.tight_layout()
    out_path = os.path.abspath(os.path.join(sample_dir, "mask_overlay_check.png"))
    plt.savefig(out_path, dpi=150)
    print(f"\n[SUCCESS] Saved Letter Mask check to: \n -> {out_path}")

if __name__ == "__main__":
    main()