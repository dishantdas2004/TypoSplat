"""
Real gsplat Forward Pass Verification
=====================================
Wires the real VGGT + TypoSplatUpsampler + TypoSplatDecoder outputs into gsplat.
Renders directly in Camera Space (Identity Viewmat) for training loop efficiency.
"""

import os
import sys
import glob
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from gsplat import rasterization

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
        
        # Means are now in Camera Space
        means = torch.stack([X, Y, Z], dim=-1).view(-1, 3) 
        
        quats = params["rot"][0].permute(1, 2, 0).view(-1, 4)         
        scales = params["scale"][0].permute(1, 2, 0).view(-1, 3)      
        opacities = params["opacity"][0].view(-1)
        
        # Sigmoid Activation for Colors [0, 1]
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
    print(f"=== TypoSplat: Real Rendering Test on {device} ===")
    
    sample_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/data/7"
    meta_path = os.path.join(sample_dir, "metadata.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    vggt = VGGTWrapper().to(device) 
    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    
    # Load Real Image for accurate VGGT Depth Prediction
    view_matches = glob.glob(os.path.join(sample_dir, "*view*.png"))
    img_pil = Image.open(view_matches[0]).convert("RGB").resize((518, 518))
    real_img = transforms.ToTensor()(img_pil).unsqueeze(0).to(device)
    dummy_base_depth = torch.ones(1, 1, 518, 518, device=device) * 5.0
    
    vggt_out = vggt.forward_with_features(real_img) 
    upsampled_features = upsampler(vggt_out["patch_tokens"])
    params_list = decoder(upsampled_features, dummy_base_depth)
    
    intrinsics_tuple = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
    Ks = torch.tensor([[
        [meta["fx"],  0, meta["cx"]],
        [ 0, meta["fy"], meta["cy"]],
        [ 0,  0,  1]
    ]], dtype=torch.float32, device=device)

    means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
        params_list[0], params_list[1], params_list[2], intrinsics_tuple, device
    )
    
    # Train-View Camera is Identity because means are natively in Camera Space
    viewmats = torch.eye(4, device=device).unsqueeze(0)

    print("\nExecuting gsplat.rasterization()...")
    render_colors, render_alphas, info = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmats, Ks=Ks, width=518, height=518,
    )
    
    print("\n[SUCCESS] Real sample rendered without crashing!")
    print(f"Alpha min/max/mean: {render_alphas.min().item():.4f}, {render_alphas.max().item():.4f}, {render_alphas.mean().item():.4f}")
    print(f"Color min/max/mean: {render_colors.min().item():.4f}, {render_colors.max().item():.4f}, {render_colors.mean().item():.4f}")
    
    out_img = render_colors[0].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))
    axes.imshow(out_img)
    axes.set_title("Untrained Gaussian Render (Raw)")
    axes.axis('off')
    
    out_path = os.path.join(sample_dir, "untrained_render.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved raw render to -> {out_path}")

if __name__ == "__main__":
    main()