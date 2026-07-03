import sys
import os
import glob
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image
import json

# Route Python to the TypoSplat root directory
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

# Single Source of Truth
from src.models.edge_predictor import get_gt_normal_edges

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
        print("Usage: python scripts/test_normal_edges.py /path/to/output/<sample_id>")
        sys.exit(1)

    sample_dir = sys.argv[1]
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    depth_matches = glob.glob(os.path.join(sample_dir, "*depth*.exr"))
    meta_path = os.path.join(sample_dir, "metadata.json")

    if not view_matches or not depth_matches or not os.path.exists(meta_path):
        raise SystemExit(f"Missing view.png, depth.exr, or metadata.json in {sample_dir}")

    img = np.array(Image.open(view_matches[0]).convert("RGB"))
    depth_np = load_exr_depth(depth_matches[0])
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Use the consolidated model function but ask for raw values to calculate P99
    edge_map = get_gt_normal_edges(depth_tensor, meta, normalize=False)
    edge_np = edge_map[0, 0].numpy()
    
    # --- FIX: Downsample the boolean mask to 148x148 to match the edge map ---
    valid_mask_518 = (depth_tensor < 50.0).float()
    valid_mask_148 = F.interpolate(valid_mask_518, size=(148, 148), mode='nearest')
    valid_mask_np = valid_mask_148[0, 0].numpy().astype(bool)
    
    # Calculate P99 using the correctly sized mask
    valid_edges = edge_np[valid_mask_np]
    p99 = np.percentile(valid_edges, 99) if len(valid_edges) > 0 else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img)
    axes[0].set_title("RGB Image")
    axes[0].axis('off')

    axes[1].imshow(depth_np, cmap='inferno', vmax=10.0)
    axes[1].set_title("Depth Map")
    axes[1].axis('off')

    axes[2].imshow(edge_np, cmap='hot', vmin=0, vmax=p99)
    axes[2].set_title(f"Normal-Based Edges (Imported)\nMax: {edge_np.max():.2f} | P99: {p99:.2f}")
    axes[2].axis('off')

    plt.tight_layout()
    out_path = os.path.abspath(os.path.join(sample_dir, "normal_target_check.png"))
    plt.savefig(out_path, dpi=150)
    print(f"\n[SUCCESS] Saved Normal Edge check to: \n -> {out_path}")

if __name__ == "__main__":
    main()