"""
TypoSplat Stage 0: Single-Sample Overfit Training Loop
======================================================
Proves that the end-to-end architecture (VGGT -> Decoder -> gsplat) 
can successfully receive gradients and minimize the rendering and geometric losses.
"""

import os
import sys
import glob
import json
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from src.losses.render_losses import compute_l1_rgb_loss, compute_sobel_edge_loss, ShallowPerceptualLoss
from src.losses.typ_losses import compute_masked_depth_loss, compute_extrusion_loss
from src.data.mask_generator import get_letter_mask
from gsplat import rasterization

def load_exr_depth(filepath, device):
    """Loads a 1-channel EXR depth map and formats it to [1, 1, H, W]"""
    depth_np = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if depth_np is None:
        raise ValueError(f"Failed to load EXR from {filepath}")
    if len(depth_np.shape) == 3:
        depth_np = depth_np[:, :, 0]
        
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(torch.float32).to(device)
    return depth_tensor

def flatten_decoder_outputs_camera_space(params_0, params_1, params_2, intrinsics, device, H_out=518, H_in=148):
    """Unprojects pixels into 3D Camera Space and applies final color activations."""
    fx, fy, cx, cy = intrinsics
    scale_factor = float(H_out) / float(H_in) 
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H_in, device=device, dtype=torch.float32), 
        torch.arange(H_in, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    
    all_means, all_quats, all_scales, all_opacities, all_colors = [], [], [], [], []
    
    for params in [params_0, params_1, params_2]:
        u_148 = x_grid + params["xy_offset"][0, 0] + 0.5
        v_148 = y_grid + params["xy_offset"][0, 1] + 0.5
        
        u_518 = u_148 * scale_factor
        v_518 = v_148 * scale_factor
        
        Z = params["true_depth"][0, 0]
        X = (u_518 - cx) * Z / fx
        Y = (v_518 - cy) * Z / fy
        
        means = torch.stack([X, Y, Z], dim=-1).view(-1, 3) 
        quats = params["rot"][0].permute(1, 2, 0).view(-1, 4)         
        scales = params["scale"][0].permute(1, 2, 0).view(-1, 3)      
        opacities = params["opacity"][0].view(-1)
        colors = torch.sigmoid(params["sh_dc"][0].permute(1, 2, 0).view(-1, 3))      
        
        all_means.append(means)
        all_quats.append(quats)
        all_scales.append(scales)
        all_opacities.append(opacities)
        all_colors.append(colors)
        
    return (
        torch.cat(all_means, dim=0),
        torch.cat(all_quats, dim=0),
        torch.cat(all_scales, dim=0),
        torch.cat(all_opacities, dim=0),
        torch.cat(all_colors, dim=0)
    )

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TypoSplat: Stage 0 Training (Single Sample Overfit) ===")
    
    # 1. Setup & Data Loading
    sample_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/data/7"
    meta_path = os.path.join(sample_dir, "metadata.json")
    mesh_path = os.path.join(sample_dir, "mesh.ply")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    depth_matches = glob.glob(os.path.join(sample_dir, "*depth*.exr"))
    
    gt_pil = Image.open(view_matches[0]).convert("RGB").resize((518, 518))
    gt_rgb = transforms.ToTensor()(gt_pil).unsqueeze(0).to(device) # [1, 3, 518, 518]
    
    # FIX: Load True Ground Truth Depth and downsample to 148
    gt_depth_518 = load_exr_depth(depth_matches[0], device)
    gt_depth_148 = torch.nn.functional.interpolate(gt_depth_518, size=(148, 148), mode='nearest')
    
    print("Pre-computing letter masks...")
    mask_148 = get_letter_mask(mesh_path, meta, device=device) # [1, 1, 148, 148]
    mask_518 = torch.nn.functional.interpolate(mask_148, size=(518, 518), mode='nearest')
    
    # 2. Model Initialization
    vggt = VGGTWrapper().to(device) 
    
    for param in vggt.parameters():
        param.requires_grad = False
        
    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    lpips_fn = ShallowPerceptualLoss(device)
    
    # 3. Optimizer setup
    trainable_params = list(upsampler.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    
    intrinsics_tuple = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    Ks = torch.tensor([[
        [meta["fx"],  0, meta["cx"]],
        [ 0, meta["fy"], meta["cy"]],
        [ 0,  0,  1]
    ]], dtype=torch.float32, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)

    # 4. The Training Loop
    iterations = 250 # Increased to 250 to give gradients time to settle
    loss_history = []
    
    print("\nStarting Training Loop...")
    upsampler.train()
    decoder.train()
    
    # Compute base VGGT features ONCE outside the loop since they are frozen
    with torch.no_grad():
        vggt_out = vggt.forward_with_features(gt_rgb) 
        base_patch_tokens = vggt_out["patch_tokens"]
        base_depth_518 = vggt_out["depth"] # FIX: Use real VGGT depth for the decoder
    
    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        
        # --- Forward Pass ---
        upsampled_features = upsampler(base_patch_tokens)
        params_0, params_1, params_2 = decoder(upsampled_features, base_depth_518)
        
        # --- Flatten to 3D ---
        means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
            params_0, params_1, params_2, intrinsics_tuple, device
        )
        
        # --- Rasterization ---
        render_colors, render_alphas, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=518, height=518,
        )
        
        pred_rgb = render_colors.permute(0, 3, 1, 2)
        
        # --- Compute Losses ---
        loss_rgb = compute_l1_rgb_loss(pred_rgb, gt_rgb)
        loss_edge = compute_sobel_edge_loss(pred_rgb, gt_rgb, mask=mask_518)
        loss_lpips = lpips_fn(pred_rgb, gt_rgb)
        
        loss_depth = compute_masked_depth_loss(params_0, gt_depth_148, mask_148)
        loss_extrusion = compute_extrusion_loss(params_1, params_2, target_extrusion=meta["extrusion_depth"], letter_mask_148=mask_148)
        
        # Total Unweighted Loss
        total_loss = loss_rgb + loss_edge + loss_lpips + loss_depth + loss_extrusion
        
        # --- Backpropagation ---
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (i+1) % 50 == 0:
            print(f"Iter {i+1:03d} | Loss: {total_loss.item():.4f} (RGB: {loss_rgb.item():.4f}, Edge: {loss_edge.item():.4f}, LPIPS: {loss_lpips.item():.4f}, Depth: {loss_depth.item():.4f})")
            
    # 5. Final Output & Validation
    print("\n[SUCCESS] Stage 0 Overfit Complete!")
    print(f"Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")
    
    out_img = render_colors[0].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(gt_rgb[0].permute(1, 2, 0).cpu().numpy())
    axes[0].set_title("Ground Truth")
    axes[0].axis('off')
    
    axes[1].imshow(out_img)
    axes[1].set_title("Final Trained Render")
    axes[1].axis('off')
    
    out_path = os.path.join(sample_dir, "overfit_result.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved POC render to -> {out_path}")

if __name__ == "__main__":
    main()