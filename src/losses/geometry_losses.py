"""
TypoSplat Geometry Losses
=========================
Calculates physical structural losses using the 2D projected letter mask.
"""

import torch
import torch.nn.functional as F

def compute_extrusion_loss(params_1, params_2, target_extrusion, letter_mask_148):
    """
    Enforces the total predicted extrusion thickness against the metadata ground truth.
    Only computes loss for anchors inside the letter mask.
    """
    # Total physical thickness = offset from Layer 1 + offset from Layer 2
    pred_extrusion = params_1["z_offset"] + params_2["z_offset"]
    
    # Broadcast the scalar target to match the shape
    target_tensor = torch.full_like(pred_extrusion, target_extrusion)
    
    # Masked MSE Loss
    loss = F.mse_loss(
        pred_extrusion * letter_mask_148, 
        target_tensor * letter_mask_148,
        reduction='mean'
    )
    
    return loss

def compute_masked_depth_loss(params_0, gt_depth_148, letter_mask_148):
    """
    Aligns the front face (Layer 0) of the Gaussians strictly to the ground truth depth.
    Acts as a highly efficient proxy for Chamfer Distance.
    """
    pred_front_depth = params_0["true_depth"]
    
    # Masked L1 Loss for robust depth alignment
    loss = F.l1_loss(
        pred_front_depth * letter_mask_148,
        gt_depth_148 * letter_mask_148,
        reduction='mean'
    )
    
    return loss