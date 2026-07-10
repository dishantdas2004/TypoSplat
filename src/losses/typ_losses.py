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
from gsplat import rasterization
from src.losses.render_losses import compute_l1_rgb_loss

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

def compute_anisotropy_loss(scales, r_bound=10.0):
    """
    Penalizes Gaussians whose aspect ratio (max_scale / min_scale) exceeds r_bound.
    """
    scale_max = scales.max(dim=-1).values
    scale_min = scales.min(dim=-1).values
    aspect_ratio = scale_max / (scale_min + 1e-8)
    penalty = torch.clamp(aspect_ratio - r_bound, min=0.0)
    return penalty.mean()

def _get_relative_viewmat(c2w_A_list, c2w_B_list, device):
    """Helper to calculate OpenCV viewmat from Camera A to Camera B."""
    c2w_A_blender = torch.tensor(c2w_A_list, dtype=torch.float32, device=device)
    c2w_B_blender = torch.tensor(c2w_B_list, dtype=torch.float32, device=device)
    
    S = torch.tensor([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=torch.float32, device=device)
    
    c2w_A_cv = c2w_A_blender @ S
    c2w_B_cv = c2w_B_blender @ S
    
    w2c_B_cv = torch.linalg.inv(c2w_B_cv)
    return (w2c_B_cv @ c2w_A_cv).unsqueeze(0)

def compute_novel_view_loss(means, quats, scales, opacities, colors, viewmats_B, Ks_B, gt_rgb_B, mask_518_B, lpips_fn):
    from gsplat import rasterization
    from src.losses.render_losses import compute_l1_rgb_loss, compute_sobel_edge_loss
    
    render_colors_B, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmats_B, Ks=Ks_B, width=518, height=518,
    )
    
    pred_rgb_B_raw = render_colors_B.permute(0, 3, 1, 2)
    
    # 1. Hard-gate the render for L1 and Sobel to prevent blocky 148-grid bleeding
    pred_rgb_B_masked = pred_rgb_B_raw * mask_518_B
    
    loss_rgb_B = compute_l1_rgb_loss(pred_rgb_B_masked, gt_rgb_B, mask=mask_518_B)
    loss_edge_B = compute_sobel_edge_loss(pred_rgb_B_masked, gt_rgb_B, mask=mask_518_B)
    
    # 2. Pass RAW unmasked render to LPIPS (VGG) to prevent artificial cutout edges
    loss_lpips_B = lpips_fn(pred_rgb_B_raw, gt_rgb_B, mask=mask_518_B)
    
    return loss_rgb_B, loss_edge_B, loss_lpips_B, render_colors_B

def compute_centroid_loss(means, viewmats_B, K_B, mask_518_B, device, sigma=200.0):
    """
    Bypasses gsplat's rasterizer entirely (plain matrix projection), so it
    can provide gradient even when Gaussians are fully off-screen and
    therefore invisible/zero-gradient to the photometric novel-view loss.
    Uses a robust, bounded loss (not plain L1) so the gradient decays to
    zero for hopelessly-far targets, instead of pushing the calibrator
    toward infinity forever.
    """
    means_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)
    cam_coords = (viewmats_B[0] @ means_h.T).T[:, :3]
    
    valid = cam_coords[:, 2] > 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
        
    z = cam_coords[valid, 2:3]
    xy = cam_coords[valid, :2]
    
    fx, fy = K_B[0, 0, 0], K_B[0, 1, 1]
    cx, cy = K_B[0, 0, 2], K_B[0, 1, 2]
    
    u = (fx * xy[:, 0:1] / z) + cx
    v = (fy * xy[:, 1:2] / z) + cy
    
    pred_centroid = torch.cat([u, v], dim=1).mean(dim=0)
    
    target_px = torch.nonzero(mask_518_B.squeeze(), as_tuple=False).float()
    target_centroid = target_px.mean(dim=0).flip(0)
    
    dist_sq = torch.sum((pred_centroid - target_centroid) ** 2)
    loss = 1.0 - torch.exp(-dist_sq / (2.0 * sigma ** 2))
    
    return loss

def compute_zoffset_regularization(params_1, params_2, target_per_layer=0.0562):
    """
    compute_extrusion_loss only constrains z_offset_1 + z_offset_2 (the sum).
    Nothing stops the network satisfying that sum while putting all the
    correction into one offset and collapsing/inverting the other, breaking
    the physical Layer1/Layer2 separation. This is a gentle prior (not a
    hard constraint) pulling each offset individually toward its expected
    share of the real mean extrusion depth (0.1124 / 2 = 0.0562).
    """
    reg1 = ((params_1["z_offset"] - target_per_layer) ** 2).mean()
    reg2 = ((params_2["z_offset"] - target_per_layer) ** 2).mean()
    return reg1 + reg2

def compute_opacity_sparsity_loss(opacities):
    """
    Pushes opacities toward 0.0 or 1.0, penalizing grayish/middle values.
    Maximum penalty is applied when opacity is exactly 0.5.
    """
    return (opacities * (1.0 - opacities)).mean()