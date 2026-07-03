"""
TypoSplat Loss Suite
====================
Implements the core geometric, structural, and scale-invariant loss functions
for typography-focused 3D Gaussian Splatting reconstruction.
"""

import torch
import torch.nn.functional as F
# Import both the normal converter and the safety helper
from src.models.edge_predictor import depth_to_normal, _reshape_intrinsic

def compute_scale_invariant_depth_loss(pred_depth, gt_depth, valid_mask):
    """
    Computes Scale and Shift Invariant Depth Loss via Least-Squares regression.
    Aligns prediction to ground truth (s * pred + t = gt) before computing 
    the residual L2 error. Only computes over valid_mask pixels.
    """
    batch_size = pred_depth.shape[0]
    loss_sum = 0.0
    
    for i in range(batch_size):
        mask_i = valid_mask[i, 0]
        n = mask_i.float().sum()
        
        if n < 2: # Need at least 2 points to compute variance/covariance
            continue
            
        # Isolate valid pixels
        P = pred_depth[i, 0][mask_i]
        G = gt_depth[i, 0][mask_i]
        
        # Calculate sums for covariance
        sum_P = P.sum()
        sum_G = G.sum()
        
        # Covariance and Variance components
        var_P = (P * P).sum() - (sum_P * sum_P) / n
        cov_PG = (P * G).sum() - (sum_P * sum_G) / n
        
        # Optimal Scale (s) and Shift (t) via Least Squares
        s = cov_PG / (var_P + 1e-6)
        t = (sum_G - s * sum_P) / n
        
        # Align the prediction
        aligned_pred = s * P + t
        
        # Compute residual MSE
        loss_i = torch.sum((aligned_pred - G) ** 2) / n
        loss_sum += loss_i
        
    return loss_sum / batch_size

def compute_normal_loss(pred_depth, gt_depth, intrinsics, valid_mask):
    """
    Derives surface normal maps for both predicted and ground truth depths via
    finite differences, evaluating the cosine distance between orientation vectors.
    """
    # FIX: Safely reshape batched intrinsics to avoid broadcasting crashes
    fx = _reshape_intrinsic(intrinsics["fx"])
    fy = _reshape_intrinsic(intrinsics["fy"])
    cx = _reshape_intrinsic(intrinsics["cx"])
    cy = _reshape_intrinsic(intrinsics["cy"])
    
    pred_normals = depth_to_normal(pred_depth, fx, fy, cx, cy) 
    gt_normals = depth_to_normal(gt_depth, fx, fy, cx, cy)     
    
    cosine_dist = 1.0 - torch.sum(pred_normals * gt_normals, dim=1, keepdim=True)
    
    # FIX: Guard against silent NaN poisoning on empty masks
    n_pixels = valid_mask.float().sum()
    if n_pixels == 0:
        return torch.tensor(0.0, device=pred_depth.device)
        
    loss = torch.sum(cosine_dist * valid_mask.float()) / n_pixels
    return loss

def compute_extrusion_loss(params_1, params_2, target_extrusion, letter_mask_148):
    """
    Enforces structural depth consistency across layers by comparing cumulative 
    Z-offsets against the ground truth scalar extrusion value.
    """
    pred_extrusion = params_1["z_offset"] + params_2["z_offset"]
    target_tensor = torch.full_like(pred_extrusion, target_extrusion)
    
    sq_err = (pred_extrusion - target_tensor) ** 2
    
    n_pixels = letter_mask_148.float().sum()
    if n_pixels == 0:
        return torch.tensor(0.0, device=pred_extrusion.device)
        
    loss = torch.sum(sq_err * letter_mask_148.float()) / n_pixels
    return loss

def compute_masked_depth_loss(params_0, gt_depth_148, letter_mask_148):
    """
    Binds the front-face Gaussian centers (Layer 0 true depth) directly to the 
    downsampled ground truth depth map layout.
    """
    pred_front_depth = params_0["true_depth"]
    
    abs_err = torch.abs(pred_front_depth - gt_depth_148)
    
    n_pixels = letter_mask_148.float().sum()
    if n_pixels == 0:
        return torch.tensor(0.0, device=pred_front_depth.device)
        
    loss = torch.sum(abs_err * letter_mask_148.float()) / n_pixels
    return loss