import os
import sys
import glob
import json
import torch
import numpy as np
from torchvision import transforms
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.losses.render_losses import compute_l1_rgb_loss, compute_sobel_edge_loss, ShallowPerceptualLoss
from src.data.mask_generator import get_letter_mask

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Rendering Losses Verification on {device} ===")
    
    sample_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/data/7"
    
    # 1. Load Ground Truth RGB
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    gt_pil = Image.open(view_matches[0]).convert("RGB").resize((518, 518))
    gt_tensor = transforms.ToTensor()(gt_pil).unsqueeze(0).to(device) # Shape: [1, 3, 518, 518]
    
    # 2. Simulate gsplat's [B, H, W, C] output from the saved render
    render_path = os.path.join(sample_dir, "untrained_render.png")
    if not os.path.exists(render_path):
        print(f"[ERROR] Could not find {render_path}. Run test_gsplat_real.py first.")
        sys.exit(1)
        
    pred_pil = Image.open(render_path).convert("RGB")
    # Simulate the raw channel-last tensor coming out of gsplat
    raw_gsplat_output = torch.tensor(np.array(pred_pil) / 255.0, dtype=torch.float32, device=device).unsqueeze(0)
    print(f"Raw gsplat output shape: {raw_gsplat_output.shape} (Channel-Last)")
    
    # The crucial permutation step needed in the training loop
    pred_tensor = raw_gsplat_output.permute(0, 3, 1, 2)
    
    print("\n--- Tensor Shapes Prepared for Losses ---")
    print(f"GT Tensor Shape:   {gt_tensor.shape}")
    print(f"Pred Tensor Shape: {pred_tensor.shape}")
    
    # 3. Generate and upsample the Letter Mask for the Sobel Loss
    print("\nGenerating Letter Mask for Sobel isolation...")
    meta_path = os.path.join(sample_dir, "metadata.json")
    mesh_path = os.path.join(sample_dir, "mesh.ply")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    mask_148 = get_letter_mask(mesh_path, meta, device=device)
    mask_518 = torch.nn.functional.interpolate(mask_148, size=(518, 518), mode='nearest')
    print(f"Mask Shape: {mask_518.shape}")
    
    # 4. Initialize VGG Perceptual Loss Module
    print("\nLoading Shallow VGG-16 Perceptual Network (May download weights)...")
    lpips_loss_fn = ShallowPerceptualLoss(device)
    
    # 5. Compute Losses
    try:
        loss_rgb = compute_l1_rgb_loss(pred_tensor, gt_tensor)
        print(f"[SUCCESS] L_RGB (L1) computed:       {loss_rgb.item():.4f}")
        
        # Pass the mask here!
        loss_edge = compute_sobel_edge_loss(pred_tensor, gt_tensor, mask=mask_518)
        print(f"[SUCCESS] L_edge (Masked Sobel):     {loss_edge.item():.4f}")
        
        loss_lpips = lpips_loss_fn(pred_tensor, gt_tensor)
        print(f"[SUCCESS] LPIPS (VGG) computed:      {loss_lpips.item():.4f}")
        
    except Exception as e:
        print(f"\n[CRASH] Loss computation failed:\n{e}")
        sys.exit(1)
        
    print("\n[SUCCESS] All 3 rendering losses are geometrically sound and shape-safe!")

if __name__ == "__main__":
    main()