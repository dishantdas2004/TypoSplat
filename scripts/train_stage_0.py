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
import OpenEXR
import Imath

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from src.losses.render_losses import compute_l1_rgb_loss, compute_sobel_edge_loss, ShallowPerceptualLoss
from src.losses.typ_losses import compute_scale_invariant_depth_loss, compute_extrusion_loss
from src.data.mask_generator import get_letter_mask
from gsplat import rasterization

def load_exr_depth(filepath, device):
    """Loads a 1-channel EXR depth map using official OpenEXR bindings"""
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    channels = list(header['channels'].keys())
    channel_name = None
    for candidate in ('Z', 'R', 'V'):
        if candidate in channels:
            channel_name = candidate
            break
    if not channel_name:
        channel_name = channels[0] 
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr_file.channel(channel_name, pt)
    
    depth_np = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    depth_tensor = torch.from_numpy(depth_np.copy()).unsqueeze(0).unsqueeze(0).to(device)
    
    return depth_tensor

def flatten_decoder_outputs_camera_space(params_0, params_1, params_2, intrinsics, device, mask_148=None, H_out=518, H_in=148):
    """Unprojects pixels into 3D Camera Space and applies final color activations.
       Suppresses background Gaussian opacity if a mask is provided."""
    fx, fy, cx, cy = intrinsics
    scale_factor = float(H_out) / float(H_in) 
    
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H_in, device=device, dtype=torch.float32), 
        torch.arange(H_in, device=device, dtype=torch.float32), 
        indexing='ij'
    )
    
    all_means, all_quats, all_scales, all_opacities, all_colors = [], [], [], [], []
    
    # Flatten the mask once to align with the 1D Gaussian arrays
    if mask_148 is not None:
        flat_mask = mask_148[0, 0].float().view(-1)
    
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
        colors = torch.sigmoid(params["sh_dc"][0].permute(1, 2, 0).view(-1, 3))      
        
        # Hard Opacity Gate: Multiply by the float mask
        opacities = params["opacity"][0].view(-1)
        if mask_148 is not None:
            opacities = opacities * flat_mask
            
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
    
    sample_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/data/7"
    meta_path = os.path.join(sample_dir, "metadata.json")
    mesh_path = os.path.join(sample_dir, "mesh.ply")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    depth_matches = glob.glob(os.path.join(sample_dir, "*depth*.exr"))
    
    gt_pil = Image.open(view_matches[0]).convert("RGB").resize((518, 518))
    gt_rgb = transforms.ToTensor()(gt_pil).unsqueeze(0).to(device)
    
    gt_depth_518 = load_exr_depth(depth_matches[0], device)
    gt_depth_148 = torch.nn.functional.interpolate(gt_depth_518, size=(148, 148), mode='nearest')
    
    print("Pre-computing letter masks...")
    mask_148 = get_letter_mask(mesh_path, meta, device=device)
    mask_518 = torch.nn.functional.interpolate(mask_148, size=(518, 518), mode='nearest')
    
    vggt = VGGTWrapper().to(device) 
    
    for param in vggt.parameters():
        param.requires_grad = False
        
    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    lpips_fn = ShallowPerceptualLoss(device)
    
    trainable_params = list(upsampler.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    
    intrinsics_tuple = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    Ks = torch.tensor([[
        [meta["fx"],  0, meta["cx"]],
        [ 0, meta["fy"], meta["cy"]],
        [ 0,  0,  1]
    ]], dtype=torch.float32, device=device)
    viewmats = torch.eye(4, device=device).unsqueeze(0)

    iterations = 250 
    loss_history = []
    
    print("\nStarting Training Loop...")
    upsampler.train()
    decoder.train()
    
    with torch.no_grad():
        vggt_out = vggt.forward_with_features(gt_rgb) 
        base_patch_tokens = vggt_out["patch_tokens"]
        base_depth_518 = vggt_out["depth"]
    
    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        
        # --- Forward Pass ---
        upsampled_features = upsampler(base_patch_tokens)
        params_0, params_1, params_2 = decoder(upsampled_features, base_depth_518)
        
        # --- Rasterization ---
        # Pass mask_148 to suppress background opacity
        means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
            params_0, params_1, params_2, intrinsics_tuple, device, mask_148=mask_148
        )
        
        render_colors, render_alphas, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats, Ks=Ks, width=518, height=518,
        )
        
        pred_rgb = render_colors.permute(0, 3, 1, 2)
        
        # --- Compute Losses (Strictly Aligned Scope) ---
        # Image-space losses now receive mask_518
        loss_rgb = compute_l1_rgb_loss(pred_rgb, gt_rgb, mask=mask_518)
        loss_edge = compute_sobel_edge_loss(pred_rgb, gt_rgb, mask=mask_518)
        loss_lpips = lpips_fn(pred_rgb, gt_rgb, mask=mask_518)
        
        # Depth is now Scale-Invariant, evaluated on Layer 1 (Front of the extrusion)
        layer_1_depth = params_0["true_depth"] + params_1["z_offset"]
        loss_depth = compute_scale_invariant_depth_loss(layer_1_depth, gt_depth_148, mask_148.bool())
        
        # Extrusion handles relative thickness of the layers
        loss_extrusion = compute_extrusion_loss(params_1, params_2, target_extrusion=meta["extrusion_depth"], letter_mask_148=mask_148)
        
        # Total Unweighted Loss
        total_loss = loss_rgb + loss_edge + loss_lpips + loss_depth + loss_extrusion
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        
        if (i+1) % 50 == 0:
            print(f"Iter {i+1:03d} | Loss: {total_loss.item():.4f} (RGB: {loss_rgb.item():.4f}, Edge: {loss_edge.item():.4f}, LPIPS: {loss_lpips.item():.4f}, Depth: {loss_depth.item():.4f}, Extrusion: {loss_extrusion.item():.8f})")
            
    print("\n[SUCCESS] Stage 0 Overfit Complete!")
    print(f"Initial Loss: {loss_history[0]:.4f} -> Final Loss: {loss_history[-1]:.4f}")
    
    out_img = render_colors[0].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # We multiply GT by the mask for the visualization so they match nicely
    mask_vis = mask_518[0].permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(gt_rgb[0].permute(1, 2, 0).cpu().numpy() * mask_vis)
    axes[0].set_title("Ground Truth (Masked)")
    axes[0].axis('off')
    
    axes[1].imshow(out_img)
    axes[1].set_title("Final Trained Render")
    axes[1].axis('off')
    
    out_path = os.path.join(sample_dir, "overfit_result.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved POC render to -> {out_path}")

if __name__ == "__main__":
    main()