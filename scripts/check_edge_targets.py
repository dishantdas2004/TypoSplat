import sys
import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

# Setup paths for VS Code and terminal
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.edge_predictor import get_gt_depth_edges

try:
    import OpenEXR
    import Imath
except ImportError:
    raise SystemExit("Install OpenEXR: pip install OpenEXR")

def load_exr_depth(path):
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = list(header['channels'].keys())
    target = next((c for c in ['V', 'Z', 'R'] if c in channels), channels[0])
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr_file.channel(target, pt)
    depth = np.frombuffer(raw, dtype=np.float32).reshape(height, width).copy()
    return depth

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_edge_targets.py /path/to/output/<sample_id>")
        sys.exit(1)

    sample_dir = sys.argv[1]
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    depth_matches = glob.glob(os.path.join(sample_dir, "*depth*.exr"))

    if not view_matches or not depth_matches:
        raise SystemExit(f"Missing view.png or depth.exr in {sample_dir}")

    img = np.array(Image.open(view_matches[0]).convert("RGB"))
    depth_np = load_exr_depth(depth_matches[0])
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()

    kernel_sizes = [5, 9, 15]
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    
    axes[0].imshow(img)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')

    axes[1].imshow(depth_np, cmap='inferno', vmax=10.0)
    axes[1].set_title("Depth (vmax=10)")
    axes[1].axis('off')

    print(f"\n=== LCN Empirical Test for {os.path.basename(sample_dir)} ===")

    for i, k in enumerate(kernel_sizes):
        # Run LCN target generator
        targets, bg_mask = get_gt_depth_edges(depth_tensor, pool_kernel_size=k)
        
        targets_1d = targets[0, 0]
        bg_mask_1d = bg_mask[0, 0]
        
        # Calculate P99 ONLY on the valid pixels (ignore the zeroed background)
        valid_targets = targets_1d[~bg_mask_1d]
        if valid_targets.numel() > 0:
            p99 = torch.quantile(valid_targets, 0.99).item()
        else:
            p99 = 1.0
            
        print(f"Kernel Size {k:2d} -> P99: {p99:.3f} | Max: {valid_targets.max().item():.3f}")

        # For visualization, we cap the display at the P99 to see relative contrast
        axes[i+2].imshow(targets_1d.numpy(), cmap='hot', vmin=0, vmax=p99)
        axes[i+2].set_title(f"LCN (k={k})\nDisplay vmax={p99:.2f}")
        axes[i+2].axis('off')

    plt.tight_layout()
    out_path = os.path.abspath(os.path.join(sample_dir, "lcn_target_check.png"))
    plt.savefig(out_path, dpi=150)
    print(f"\n[SUCCESS] Saved LCN comparison to: \n -> {out_path}")

if __name__ == "__main__":
    main()