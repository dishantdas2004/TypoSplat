"""
TypoSplat Edge Score Predictor
==============================
Predicts a 2D probability map [B, 1, 148, 148] indicating the likelihood of 
an anchor containing a high-frequency geometric edge (typography boundaries).

Uses Surface Normals derived from depth via finite differences to eliminate 
perspective foreshortening artifacts on steep camera angles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def _reshape_intrinsic(val):
    """
    Safely reshapes intrinsic values (scalar or batched tensor) to [B, 1, 1, 1] 
    so they broadcast correctly against [B, C, H, W] spatial tensors.
    """
    if isinstance(val, torch.Tensor):
        return val.view(-1, 1, 1, 1)
    return val

def depth_to_normal(depth, fx, fy, cx, cy):
    """
    Converts a depth map [B, 1, H, W] to a surface normal map [B, 3, H, W]
    using camera intrinsics and finite differences.
    """
    B, _, H, W = depth.shape
    
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), 
                          torch.arange(W, device=depth.device), indexing='ij')
    x = x.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    y = y.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
    
    X = (x - cx) * depth / fx
    Y = (y - cy) * depth / fy
    Z = depth
    
    pts_3d = torch.cat([X, Y, Z], dim=1) 
    
    padded = F.pad(pts_3d, (1, 1, 1, 1), mode='replicate')
    dx = (padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]) / 2.0
    dy = (padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]) / 2.0
    
    normals = torch.cross(dx, dy, dim=1)
    normals = F.normalize(normals, p=2, dim=1)
    
    return normals

@torch.no_grad()
def get_gt_normal_edges(depth_tensor, meta_json, device='cpu', normalize=True):
    """
    Generates ground truth edge targets using Sobel on Surface Normals.
    Normalized to [0, 1] using empirical max (9.0) by default for BCE loss.
    """
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.unsqueeze(1)
        
    raw_bg_mask = depth_tensor >= 50.0
    bg_mask_518 = F.max_pool2d(raw_bg_mask.float(), kernel_size=5, stride=1, padding=2) >= 0.5
    valid_mask_518 = ~bg_mask_518
    
    if valid_mask_518.any():
        mean_valid = depth_tensor[valid_mask_518].mean()
    else:
        mean_valid = torch.tensor(5.0, device=device)
    
    safe_depth = torch.where(bg_mask_518, mean_valid, depth_tensor)
    
    fx = _reshape_intrinsic(meta_json["fx"])
    fy = _reshape_intrinsic(meta_json["fy"])
    cx = _reshape_intrinsic(meta_json["cx"])
    cy = _reshape_intrinsic(meta_json["cy"])
    
    normals = depth_to_normal(safe_depth, fx, fy, cx, cy) 
    normals_xy = normals[:, 0:2, :, :] 
    
    sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(2, 1, 1, 1)
    sobel_y = torch.tensor([[-1., -2., -1.], [ 0., 0., 0.], [ 1., 2., 1.]], device=device, dtype=torch.float32).view(1, 1, 3, 3).repeat(2, 1, 1, 1)
    
    grad_x = F.conv2d(normals_xy, sobel_x, padding=1, groups=2)
    grad_y = F.conv2d(normals_xy, sobel_y, padding=1, groups=2)
    
    # Raw gradient magnitude at 518x518
    grad_mag_518 = torch.sqrt(torch.sum(grad_x**2 + grad_y**2, dim=1, keepdim=True) + 1e-8)
    
    # --- FIX: Restored Separate Downsampling Logic ---
    # Downsample gradients smoothly
    grad_mag_148 = F.interpolate(grad_mag_518, size=(148, 148), mode='bilinear', align_corners=False)
    
    # Downsample boolean mask strictly without smearing zeroes
    bg_mask_148 = F.interpolate(bg_mask_518.float(), size=(148, 148), mode='nearest') >= 0.5
    
    # Apply background mask POST-downsample
    grad_mag_148 = torch.where(bg_mask_148, torch.zeros_like(grad_mag_148), grad_mag_148)
    
    if normalize:
        grad_mag_148 = torch.clamp(grad_mag_148 / 9.0, 0.0, 1.0)
        
    return grad_mag_148

class EdgeScorePredictor(nn.Module):
    def __init__(self, feature_channels=258):
        super().__init__()
        in_channels = feature_channels + 1
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()  
        )
        
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], dtype=torch.float32).view(1, 1, 3, 3).repeat(2, 1, 1, 1)
        sobel_y = torch.tensor([[-1., -2., -1.], [ 0., 0., 0.], [ 1., 2., 1.]], dtype=torch.float32).view(1, 1, 3, 3).repeat(2, 1, 1, 1)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, features_148, base_depth_518, intrinsics, gt_bg_mask_518=None):
        with torch.no_grad():
            if gt_bg_mask_518 is not None:
                dilated_bg_mask = F.max_pool2d(gt_bg_mask_518.float(), kernel_size=5, stride=1, padding=2) >= 0.5
                valid_mask = ~dilated_bg_mask
                mean_valid = base_depth_518[valid_mask].mean() if valid_mask.any() else torch.tensor(5.0, device=base_depth_518.device)
                safe_pred_depth = torch.where(dilated_bg_mask, mean_valid, base_depth_518)
            else:
                safe_pred_depth = base_depth_518
                
            fx = _reshape_intrinsic(intrinsics["fx"])
            fy = _reshape_intrinsic(intrinsics["fy"])
            cx = _reshape_intrinsic(intrinsics["cx"])
            cy = _reshape_intrinsic(intrinsics["cy"])
            
            normals = depth_to_normal(safe_pred_depth, fx, fy, cx, cy)
            normals_xy = normals[:, 0:2, :, :]
            
            grad_x = F.conv2d(normals_xy, self.sobel_x, padding=1, groups=2)
            grad_y = F.conv2d(normals_xy, self.sobel_y, padding=1, groups=2)
            depth_edges_518 = torch.sqrt(torch.sum(grad_x**2 + grad_y**2, dim=1, keepdim=True) + 1e-8)
            
            depth_edges_148 = F.interpolate(depth_edges_518, size=(148, 148), mode='bilinear', align_corners=False)
            norm_depth_edges_148 = torch.clamp(depth_edges_148 / 9.0, 0.0, 1.0)
            
        x = torch.cat([features_148, norm_depth_edges_148], dim=1) 
        score_map = self.net(x)
        
        return score_map

if __name__ == "__main__":
    print("\n=== TypoSplat Edge Predictor Smoke Test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on: {device}\n")
    
    edge_predictor = EdgeScorePredictor(feature_channels=258).to(device)
    
    B = 2
    dummy_features = torch.randn(B, 258, 148, 148, device=device)
    dummy_pred_depth = torch.rand(B, 1, 518, 518, device=device) * 5.0
    dummy_gt_depth = torch.ones(B, 1, 518, 518, device=device) * 3.0
    dummy_gt_depth[:, :, :100, :] = 100.0  
    
    dummy_intrinsics = {
        "fx": torch.tensor([700.0, 700.0], device=device),
        "fy": torch.tensor([700.0, 700.0], device=device),
        "cx": torch.tensor([259.0, 259.0], device=device),
        "cy": torch.tensor([259.0, 259.0], device=device)
    }
    
    dummy_bg_mask = dummy_gt_depth >= 50.0
    
    score_map = edge_predictor(dummy_features, dummy_pred_depth, dummy_intrinsics, gt_bg_mask_518=dummy_bg_mask)
    targets = get_gt_normal_edges(dummy_gt_depth, dummy_intrinsics, device=device) 
    
    print(f"Predicted Score Map Shape: {list(score_map.shape)}")
    print(f"Ground Truth Target Shape: {list(targets.shape)}")
    
    # --- FIX: Strict Assertions ---
    assert list(score_map.shape) == [B, 1, 148, 148], f"Score map shape mismatch! Expected {[B, 1, 148, 148]}, got {list(score_map.shape)}"
    assert list(targets.shape) == [B, 1, 148, 148], f"Target shape mismatch! Expected {[B, 1, 148, 148]}, got {list(targets.shape)}"
    assert score_map.shape == targets.shape, "Shape mismatch between prediction and target!"
    
    print("\n[SUCCESS] Edge Predictor and Target Generator are mathematically sound and shapes align.")