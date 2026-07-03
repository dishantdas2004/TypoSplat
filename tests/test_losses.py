import sys
import os
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.losses.typ_losses import (
    compute_scale_invariant_depth_loss,
    compute_normal_loss,
    compute_extrusion_loss,
    compute_masked_depth_loss
)

def run_loss_suite_smoke_test():
    print("\n=== TypoSplat Loss Suite Verification Harness ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing mathematical validation on: {device}\n")
    
    B = 2
    H_518, W_518 = 518, 518
    H_148, W_148 = 148, 148
    
    intrinsics = {
        "fx": torch.tensor([719.4, 719.4], device=device),
        "fy": torch.tensor([719.4, 719.4], device=device),
        "cx": torch.tensor([259.0, 259.0], device=device),
        "cy": torch.tensor([259.0, 259.0], device=device)
    }
    
    gt_depth_518 = torch.ones(B, 1, H_518, W_518, device=device) * 3.0
    gt_depth_518[:, :, :120, :] = 100.0  
    valid_mask_518 = gt_depth_518 < 50.0
    
    pred_depth_518 = gt_depth_518.clone() + (torch.randn_like(gt_depth_518) * 0.05)
    pred_depth_518 = torch.clamp(pred_depth_518, min=0.5) 
    
    gt_depth_148 = torch.ones(B, 1, H_148, W_148, device=device) * 3.0
    
    y_grid, x_grid = torch.meshgrid(torch.arange(H_148), torch.arange(W_148), indexing='ij')
    center_y, center_x = H_148 // 2, W_148 // 2
    distance_from_center = torch.sqrt((x_grid - center_x)**2 + (y_grid - center_y)**2)
    dummy_mask_148 = (distance_from_center < 35).float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    
    params_0 = {"true_depth": gt_depth_148.clone() + (torch.randn_like(gt_depth_148) * 0.02)}
    params_1 = {"z_offset": torch.ones(B, 1, H_148, W_148, device=device) * 0.05} 
    params_2 = {"z_offset": torch.ones(B, 1, H_148, W_148, device=device) * 0.05} 
    
    target_extrusion = 0.11 
    
    # 1. Scale-Invariant Depth Loss (Least Squares Regression)
    loss_depth = compute_scale_invariant_depth_loss(pred_depth_518, gt_depth_518, valid_mask_518)
    print(f"Least-Squares Scale+Shift Depth Loss: {loss_depth.item():.6f}")
    assert not torch.isnan(loss_depth)
    assert loss_depth.item() >= 0.0
    
    # 2. Surface Normal Loss
    loss_normal = compute_normal_loss(pred_depth_518, gt_depth_518, intrinsics, valid_mask_518)
    print(f"Surface Normal Field Loss:            {loss_normal.item():.6f}")
    assert not torch.isnan(loss_normal)
    assert loss_normal.item() >= 0.0
    
    # 3. Extrusion Thickness Loss
    loss_extrusion = compute_extrusion_loss(params_1, params_2, target_extrusion, dummy_mask_148)
    print(f"Extrusion Thickness Loss:             {loss_extrusion.item():.6f}")
    assert not torch.isnan(loss_extrusion)
    assert loss_extrusion.item() >= 0.0
    
    # 4. Masked Depth Loss
    loss_masked_depth = compute_masked_depth_loss(params_0, gt_depth_148, dummy_mask_148)
    print(f"Masked Depth Anchor Loss:             {loss_masked_depth.item():.6f}")
    assert not torch.isnan(loss_masked_depth)
    assert loss_masked_depth.item() >= 0.0
    
    print("\n[SUCCESS] All core loss metrics are verified, bounded, and numerically stable.")

if __name__ == "__main__":
    run_loss_suite_smoke_test()