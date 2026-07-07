"""
TypoSplat Stage 0: Single-Sample Overfit Training Loop
Incorporates Novel-View supervision to prevent degenerate flat/needle Gaussians.
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
from src.losses.typ_losses import (
    compute_scale_invariant_depth_loss, 
    compute_extrusion_loss, 
    compute_normal_loss, 
    compute_anisotropy_loss,
    compute_novel_view_loss
)
from src.data.mask_generator import get_letter_mask
from gsplat import rasterization

def load_exr_depth(filepath, device):
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    channels = list(header['channels'].keys())
    channel_name = next((c for c in ('Z', 'R', 'V') if c in channels), channels[0])
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    raw = exr_file.channel(channel_name, pt)
    depth_np = np.frombuffer(raw, dtype=np.float32).reshape(height, width)
    return torch.from_numpy(depth_np.copy()).unsqueeze(0).unsqueeze(0).to(device)

def flatten_decoder_outputs_camera_space(params_0, params_1, params_2, intrinsics, device, mask_148=None, H_out=518, H_in=148):
    fx, fy, cx, cy = intrinsics
    scale_factor = float(H_out) / float(H_in) 
    y_grid, x_grid = torch.meshgrid(torch.arange(H_in, device=device, dtype=torch.float32), torch.arange(H_in, device=device, dtype=torch.float32), indexing='ij')
    all_means, all_quats, all_scales, all_opacities, all_colors = [], [], [], [], []
    
    flat_mask = mask_148[0, 0].float().view(-1) if mask_148 is not None else None
    
    for params in [params_0, params_1, params_2]:
        u_148 = x_grid + params["xy_offset"][0, 0] + 0.5
        v_148 = y_grid + params["xy_offset"][0, 1] + 0.5
        u_518, v_518 = u_148 * scale_factor, v_148 * scale_factor
        
        Z = params["true_depth"][0, 0]
        X = (u_518 - cx) * Z / fx
        Y = (v_518 - cy) * Z / fy
        
        means = torch.stack([X, Y, Z], dim=-1).view(-1, 3) 
        quats = params["rot"][0].permute(1, 2, 0).view(-1, 4)         
        scales = params["scale"][0].permute(1, 2, 0).view(-1, 3)      
        colors = torch.sigmoid(params["sh_dc"][0].permute(1, 2, 0).view(-1, 3))      
        
        opacities = params["opacity"][0].view(-1)
        if flat_mask is not None:
            opacities = opacities * flat_mask
            
        all_means.append(means)
        all_quats.append(quats)
        all_scales.append(scales)
        all_opacities.append(opacities)
        all_colors.append(colors)
        
    return (torch.cat(all_means, dim=0), torch.cat(all_quats, dim=0), torch.cat(all_scales, dim=0), torch.cat(all_opacities, dim=0), torch.cat(all_colors, dim=0))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TypoSplat: Stage 0 Training (Dual-View Overfit) ===")
    
    sample_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/data/19"
    meta_path = os.path.join(sample_dir, "metadata.json")
    mesh_path = os.path.join(sample_dir, "mesh.ply")
    
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    view_A_paths = glob.glob(os.path.join(sample_dir, "*view_A*.png"))
    view_B_paths = glob.glob(os.path.join(sample_dir, "*view_B*.png"))
    depth_A_paths = glob.glob(os.path.join(sample_dir, "*depth_A*.exr"))
    
    # Load Camera A
    gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
    gt_depth_518_A = load_exr_depth(depth_A_paths[0], device)
    gt_depth_148_A = torch.nn.functional.interpolate(gt_depth_518_A, size=(148, 148), mode='nearest')
    
    # Load Camera B
    gt_rgb_B = transforms.ToTensor()(Image.open(view_B_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
    
    print("Pre-computing letter masks for both views...")
    mask_148_A = get_letter_mask(mesh_path, meta, device=device)
    mask_518_A = torch.nn.functional.interpolate(mask_148_A, size=(518, 518), mode='nearest')
    
    mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device=device)
    mask_518_B = torch.nn.functional.interpolate(mask_148_B, size=(518, 518), mode='nearest')
    
    vggt = VGGTWrapper().to(device) 
    for param in vggt.parameters():
        param.requires_grad = False
        
    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    lpips_fn = ShallowPerceptualLoss(device)
    
    trainable_params = list(upsampler.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(trainable_params, lr=1e-4)
    
    # Camera A Setup
    intrinsics_tuple_A = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    scale_148 = 148.0 / 518.0
    intrinsics_dict_148_A = {
        "fx": meta["fx"] * scale_148, "fy": meta["fy"] * scale_148, 
        "cx": meta["cx"] * scale_148, "cy": meta["cy"] * scale_148
    }
    Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
    viewmats_A = torch.eye(4, device=device).unsqueeze(0)

    iterations = 5000 
    loss_history = []
    
    print("\nStarting Training Loop...")
    upsampler.train()
    decoder.train()
    
    with torch.no_grad():
        vggt_out = vggt.forward_with_features(gt_rgb_A) 
        base_patch_tokens = vggt_out["patch_tokens"]
        base_depth_518 = vggt_out["depth"]
    
    for i in tqdm(range(iterations)):
        optimizer.zero_grad()
        
        upsampled_features = upsampler(base_patch_tokens)
        params_0, params_1, params_2 = decoder(upsampled_features, base_depth_518)
        
        means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
            params_0, params_1, params_2, intrinsics_tuple_A, device, mask_148=mask_148_A
        )
        
        # Camera A Render
        render_colors_A, _, _ = rasterization(
            means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
            viewmats=viewmats_A, Ks=Ks_A, width=518, height=518,
        )
        pred_rgb_A = render_colors_A.permute(0, 3, 1, 2)
        
        # --- Losses ---
        loss_rgb = compute_l1_rgb_loss(pred_rgb_A, gt_rgb_A, mask=mask_518_A)
        loss_edge = compute_sobel_edge_loss(pred_rgb_A, gt_rgb_A, mask=mask_518_A)
        loss_lpips = lpips_fn(pred_rgb_A, gt_rgb_A, mask=mask_518_A)
        
        layer_1_depth = params_0["true_depth"] + params_1["z_offset"]
        loss_depth = compute_scale_invariant_depth_loss(layer_1_depth, gt_depth_148_A, mask_148_A.bool())
        loss_extrusion = compute_extrusion_loss(params_1, params_2, meta["extrusion_depth"], mask_148_A)
        loss_aniso = compute_anisotropy_loss(scales, r_bound=10.0)
        loss_normal = compute_normal_loss(layer_1_depth, gt_depth_148_A, intrinsics_dict_148_A, mask_148_A)
        
        # Novel View Loss (Encapsulates Camera B Render & L1 Calculation)
        loss_novel_view, render_colors_B = compute_novel_view_loss(
            means, quats, scales, opacities, colors, 
            meta, meta["camera_B"], gt_rgb_B, mask_518_B, device
        )
        
        if i == 0:
            print(f"\n--- RAW NOVEL VIEW MAGNITUDE (ITER 0) ---")
            print(f"Raw Novel View: {loss_novel_view.item():.4f}")
            sys.exit(0) # STOP EXECUTION TO TUNE LAMBDA
        
        total_loss = (
            1.0 * loss_rgb + 
            1.0 * loss_edge + 
            0.002 * loss_lpips + 
            50.0 * loss_depth + 
            1000.0 * loss_extrusion +
            1.0 * loss_aniso + 
            1.0 * loss_normal +
            1.0 * loss_novel_view # Adjust this after checking Iter 0
        )
        
        total_loss.backward()
        optimizer.step()
        loss_history.append(total_loss.item())
        
        if (i+1) % 500 == 0:
            print(f"Iter {i+1:04d} | Total: {total_loss.item():.4f} | Aniso: {loss_aniso.item():.4f} | Novel View: {loss_novel_view.item():.4f}")
            
    # Save a comparison render showing both views
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0,0].imshow(gt_rgb_A[0].permute(1, 2, 0).cpu().numpy() * mask_518_A[0].permute(1, 2, 0).cpu().numpy())
    axes[0,0].set_title("GT Camera A")
    axes[0,1].imshow(render_colors_A[0].detach().cpu().numpy())
    axes[0,1].set_title("Render Camera A")
    
    axes[1,0].imshow(gt_rgb_B[0].permute(1, 2, 0).cpu().numpy() * mask_518_B[0].permute(1, 2, 0).cpu().numpy())
    axes[1,0].set_title("GT Camera B")
    axes[1,1].imshow(render_colors_B[0].detach().cpu().numpy())
    axes[1,1].set_title("Render Camera B")
    
    out_path = os.path.join(sample_dir, "overfit_result_dual.png")
    plt.savefig(out_path, dpi=150)

    # --- DIAGNOSTICS ---
    scale_max = scales.max(dim=-1).values
    scale_min = scales.min(dim=-1).values
    aspect_ratios = scale_max / (scale_min + 1e-8)
    
    print(f"\n--- ANISOTROPY (ASPECT RATIO) ---")
    print(f"Mean Ratio: {aspect_ratios.mean().item():.2f}")
    print(f"Max Ratio:  {aspect_ratios.max().item():.2f}")

if __name__ == "__main__":
    main()