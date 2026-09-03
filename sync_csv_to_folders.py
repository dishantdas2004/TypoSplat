import os
import sys
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sync diagnostic CSV with existing local folders and append to a master CSV.")
    parser.add_argument("--input_csv", type=str, default="diagnostics_train_id_0-5303.csv", help="Path to the new/raw diagnostic CSV")
    parser.add_argument("--data_dir", type=str, default="data", help="Local directory containing the filtered sample folders")
    parser.add_argument("--output_csv", type=str, default="master_diagnostics.csv", help="The master CSV to create or update")
    args = parser.parse_args()

    # 1. Get the list of all currently existing sample folders
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' not found.")
        sys.exit(1)
        
    existing_folders = [f for f in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, f))]
    print(f"Found {len(existing_folders)} sample folders in '{args.data_dir}'.")

    # 2. Load the input diagnostic CSV
    try:
        input_df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"ERROR: Could not find input CSV '{args.input_csv}'.")
        sys.exit(1)
        
    if 'Sample' not in input_df.columns:
        print("ERROR: Could not find a 'Sample' column in the input CSV.")
        sys.exit(1)

    # 3. Filter the input CSV to ONLY include rows where the folder exists
    # Convert 'Sample' to string to match folder names reliably
    input_df['Sample'] = input_df['Sample'].astype(str)
    filtered_df = input_df[input_df['Sample'].isin(existing_folders)]
    
    print(f"Extracted {len(filtered_df)} matching rows from '{args.input_csv}'.")

    # 4. Create or Update the Master CSV
    if os.path.exists(args.output_csv):
        print(f"Master CSV '{args.output_csv}' found. Merging new data...")
        master_df = pd.read_csv(args.output_csv)
        master_df['Sample'] = master_df['Sample'].astype(str)
        
        # Combine the old master and the new filtered data
        combined_df = pd.concat([master_df, filtered_df], ignore_index=True)
        
        # Drop duplicates based on 'Sample' ID, keeping the most recent entry
        combined_df.drop_duplicates(subset=['Sample'], keep='last', inplace=True)
        
        # Save back to disk
        combined_df.to_csv(args.output_csv, index=False)
        print(f"Successfully updated '{args.output_csv}'. It now contains {len(combined_df)} total samples.")
        
    else:
        print(f"Master CSV '{args.output_csv}' not found. Creating a new one...")
        filtered_df.to_csv(args.output_csv, index=False)
        print(f"Successfully created '{args.output_csv}' with {len(filtered_df)} samples.")

if __name__ == "__main__":
    main()























"""
Overfit Diagnostic Check (Camera A L1 Optimization + Camera B SSIM/Variance Audit)
==================================================================================
Trains on 4 fixed samples for 3000 steps using Camera A L1 loss.
Evaluates Camera A (PSNR) and Camera B (PSNR, overall SSIM) every 300 steps 
to monitor photometric reconstruction quality and structural drift.
"""

import os
import sys
import glob
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Hardcoded to TypoSplat project root
sys.path.append("/home/isiuts/TypoSplat")

from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from src.data.mask_generator import get_letter_mask
from src.losses.render_losses import compute_l1_rgb_loss
from src.losses.typ_losses import _get_relative_viewmat
from gsplat import rasterization

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
        
        # Leaky clamp activation for sh_dc (slope 0.05) preserves gradient for out-of-bounds values
        x_raw = params["sh_dc"][0].permute(1, 2, 0).view(-1, 3)
        colors = torch.where(
            x_raw < 0.0,
            0.05 * x_raw,
            torch.where(
                x_raw > 1.0,
                1.0 + 0.05 * (x_raw - 1.0),
                x_raw
            ))

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
    device = torch.device("cuda")
    
    print("Loading models...")
    vggt = VGGTWrapper().to(device).eval()
    for p in vggt.parameters(): p.requires_grad = False

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    
    ckpt_path = "/home/isiuts/stage1/checkpoint_epoch_22.pt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    upsampler.load_state_dict(ckpt['upsampler'])
    decoder.load_state_dict(ckpt['decoder'])

    for param in decoder.calibrator.parameters():
        param.requires_grad = False

    upsampler.train()
    decoder.train()

    optimizer = optim.Adam(list(upsampler.parameters()) + list(decoder.parameters()), lr=1e-4)

    # Load 4 fixed samples
    data_dir = "/home/isiuts/validation-set"
    sample_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:4]
    
    print(f"Pre-processing {len(sample_dirs)} samples...")
    samples = []
    
    for sid in sample_dirs:
        d = os.path.join(data_dir, sid)
        meta = json.load(open(os.path.join(d, "metadata.json")))
        
        # Camera A Setup
        view_A_path = glob.glob(os.path.join(d, "*view_A*.png"))[0]
        gt_A = transforms.ToTensor()(Image.open(view_A_path).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
        mask148_A = get_letter_mask(os.path.join(d, "mesh.ply"), meta, device=device)
        mask518_A = F.interpolate(mask148_A.float(), size=(518, 518), mode='nearest').bool()
        Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmats_A = torch.eye(4, device=device).unsqueeze(0)
        
        # Camera B Setup
        meta_B = meta["camera_B"]
        view_B_path = glob.glob(os.path.join(d, "*view_B*.png"))[0]
        gt_B = transforms.ToTensor()(Image.open(view_B_path).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
        mask148_B = get_letter_mask(os.path.join(d, "mesh.ply"), meta_B, device=device)
        mask518_B = F.interpolate(mask148_B.float(), size=(518, 518), mode='nearest').bool()
        Ks_B = torch.tensor([[[meta_B["fx"], 0, meta_B["cx"]], [0, meta_B["fy"], meta_B["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], device)
        
        # Precompute VGGT features once to save compute in the loop
        with torch.no_grad():
            vggt_out = vggt.forward_with_features(gt_A)
            patch_tokens = vggt_out["patch_tokens"].detach()
            base_depth = vggt_out["depth"].detach()

        samples.append({
            "sid": sid,
            "meta": meta,
            "gt_A": gt_A,
            "mask148_A": mask148_A,
            "mask518_A": mask518_A,
            "Ks_A": Ks_A,
            "viewmats_A": viewmats_A,
            "gt_B": gt_B,
            "mask148_B": mask148_B,
            "mask518_B": mask518_B,
            "Ks_B": Ks_B,
            "viewmats_B": viewmats_B,
            "patch_tokens": patch_tokens,
            "base_depth": base_depth
        })

    # --- Initial Ground-Truth Camera-B Variance Check ---
    print("\n--- Initial Ground-Truth Camera-B Variance Check ---")
    C2_ref = 9e-4
    for i, s in enumerate(samples):
        # Convert to grayscale (Y = 0.2989R + 0.5870G + 0.1140B)
        gt_b_gray = 0.2989 * s["gt_B"][0, 0] + 0.5870 * s["gt_B"][0, 1] + 0.1140 * s["gt_B"][0, 2]
        mask_b = s["mask518_B"][0, 0]
        masked_pixels = gt_b_gray[mask_b]
        
        if len(masked_pixels) > 0:
            var = masked_pixels.var().item()
            status = "WITHIN 10x range" if (9e-5 <= var <= 9e-3) else "OUTSIDE 10x range"
            print(f"Sample {i} | Grayscale Variance: {var:.6e} | {status} (vs C2={C2_ref})")
        else:
            print(f"Sample {i} | Mask empty, cannot compute variance.")
    print("----------------------------------------------------\n")

    print(f"Starting Overfit Loop (3000 steps, lr=1e-4)...")
    
    for step in range(3001):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        
        for i, s in enumerate(samples):
            # Forward pass
            up_feat = upsampler(s["patch_tokens"])
            
            if step == 0 and i == 0:
                print(f"[DEBUG] up_feat.requires_grad: {up_feat.requires_grad}")
            
            p_list, _, _, _, _ = decoder(up_feat, s["base_depth"], s["patch_tokens"])
            
            m, q, scale, o, c = flatten_decoder_outputs_camera_space(
                p_list[0], p_list[1], p_list[2], 
                (s["meta"]["fx"], s["meta"]["fy"], s["meta"]["cx"], s["meta"]["cy"]), 
                device, mask_148=s["mask148_A"]
            )
            
            # Render Camera A
            render_A, _, _ = rasterization(
                means=m.float(), quats=q.float(), scales=scale.float(), opacities=o.float(), colors=c.float(),
                viewmats=s["viewmats_A"], Ks=s["Ks_A"], width=518, height=518
            )
            
            pred_rgb_A_raw = render_A.permute(0, 3, 1, 2)
            pred_rgb_A_masked = pred_rgb_A_raw * s["mask518_A"]
            
            # Loss computed ONLY on Camera A
            loss = compute_l1_rgb_loss(pred_rgb_A_masked, s["gt_A"], mask=s["mask518_A"])
            total_loss = total_loss + loss
            
            if step % 300 == 0:
                # Masked PSNR Computation (Camera A)
                pred_A_np = (pred_rgb_A_raw[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                gt_A_np = (s["gt_A"][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                mask_A_np = s["mask518_A"][0, 0].cpu().numpy().astype(bool)
                
                pred_A_masked = pred_A_np[mask_A_np]
                gt_A_masked = gt_A_np[mask_A_np]
                
                psnr_A = psnr_metric(gt_A_masked, pred_A_masked, data_range=255) if len(gt_A_masked) > 0 else 0.0
                
                # --- Read-Only Evaluation on Camera B ---
                with torch.no_grad():
                    render_B, _, _ = rasterization(
                        means=m.float(), quats=q.float(), scales=scale.float(), opacities=o.float(), colors=c.float(),
                        viewmats=s["viewmats_B"], Ks=s["Ks_B"], width=518, height=518
                    )
                    pred_rgb_B_raw = render_B.permute(0, 3, 1, 2)
                
                pred_B_np = (pred_rgb_B_raw[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                gt_B_np = (s["gt_B"][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                mask_B_np = s["mask518_B"][0, 0].cpu().numpy().astype(bool)
                
                # PSNR B
                pred_B_masked = pred_B_np[mask_B_np]
                gt_B_masked = gt_B_np[mask_B_np]
                psnr_B = psnr_metric(gt_B_masked, pred_B_masked, data_range=255) if len(gt_B_masked) > 0 else 0.0

                # Full SSIM (Bounding Box Crop) B
                ys, xs = np.where(mask_B_np)
                ssim_B = 0.0
                
                if len(ys) > 0 and len(xs) > 0 and (ys.max() - ys.min() + 1) >= 7 and (xs.max() - xs.min() + 1) >= 7:
                    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                    pred_B_crop = pred_B_np[y0:y1, x0:x1]
                    gt_B_crop = gt_B_np[y0:y1, x0:x1]
                    
                    ssim_B = ssim_metric(gt_B_crop, pred_B_crop, channel_axis=-1, data_range=255)
                
                print(f"Step {step:4d} | S{i} | Cam A [L1: {loss.item():.4f}, PSNR: {psnr_A:5.2f}] || Cam B [PSNR: {psnr_B:5.2f}, SSIM: {ssim_B:.4f}]")

        # Average and Backward
        avg_loss = total_loss / len(samples)
        avg_loss.backward()
        
        # Critical Debugging Check at Step 0
        if step == 0:
            total_grad_norm = sum(p.grad.norm().item() for p in list(upsampler.parameters()) + list(decoder.parameters()) if p.grad is not None)
            none_grad_count = sum(1 for p in list(upsampler.parameters()) + list(decoder.parameters()) if p.grad is None)
            
            print(f"[DEBUG] Step 0 grad norm: {total_grad_norm:.6f}, params with None grad: {none_grad_count}")
            
            if total_grad_norm < 1e-5 or none_grad_count > 10:
                print("========================================================================")
                print("[!] CRITICAL WARNING: Gradients are dead or missing on Step 0!")
                print("    Optimization is not happening. Check computational graph.")
                print("========================================================================")
        
        optimizer.step()

if __name__ == "__main__":
    main()




























"""
Overfit Diagnostic Check (Camera A L1 + Camera B Depth + Orientation Loss)
========================================================================
Trains on 4 fixed samples for 3000 steps using:
1. Camera A L1 RGB loss
2. Novel view Camera B Depth Loss (W_DEPTH_B = 10.0)
3. Minimum-scale Orientation Loss (W_ORIENT = 2.0)

Camera B RGB remains strictly read-only for held-out evaluation.
"""

import os
import sys
import glob
import json
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import OpenEXR
import Imath
from PIL import Image
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

sys.path.append("/home/isiuts/TypoSplat")

from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from src.data.mask_generator import get_letter_mask
from src.losses.render_losses import compute_l1_rgb_loss
from src.losses.typ_losses import _get_relative_viewmat
from gsplat import rasterization

def load_exr_depth_cpu(filepath):
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
    return torch.from_numpy(depth_np.copy()).unsqueeze(0).unsqueeze(0)

def quat_to_rotmat(q):
    """Converts quaternions (w, x, y, z) to 3x3 rotation matrices."""
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    R = torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z),     2 * (x * z + w * y),
        2 * (x * y + w * z),     1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y),     2 * (y * z + w * x),     1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)
    return R

def compute_novel_view_depth_loss(depth_pred, depth_gt, mask):
    m = mask.view_as(depth_pred).bool()
    if m.sum() == 0:
        return depth_pred.new_zeros(())
    return (depth_pred[m] - depth_gt[m]).abs().mean()

def compute_orientation_loss(quats, scales, normals_gt, valid_mask=None):
    R = quat_to_rotmat(quats)
    min_idx = scales.argmin(dim=1)
    n_pred = R[torch.arange(R.shape[0], device=R.device), :, min_idx]
    n_pred = F.normalize(n_pred, dim=-1)
    n_gt = F.normalize(normals_gt, dim=-1)
    loss = 1.0 - (n_pred * n_gt).sum(-1).abs()
    if valid_mask is not None:
        loss = loss[valid_mask]
    return loss.mean() if loss.numel() > 0 else loss.sum()

def flatten_decoder_outputs_camera_space(params_0, params_1, params_2, intrinsics, device, 
                                          mask_148=None, normal_maps=None, H_out=518, H_in=148):
    fx, fy, cx, cy = intrinsics
    scale_factor = float(H_out) / float(H_in) 
    y_grid, x_grid = torch.meshgrid(torch.arange(H_in, device=device, dtype=torch.float32), torch.arange(H_in, device=device, dtype=torch.float32), indexing='ij')
    all_means, all_quats, all_scales, all_opacities, all_colors, all_normals = [], [], [], [], [], []
    flat_mask = mask_148[0, 0].float().view(-1) if mask_148 is not None else None

    for layer_idx, params in enumerate([params_0, params_1, params_2]):
        u_148 = x_grid + params["xy_offset"][0, 0] + 0.5
        v_148 = y_grid + params["xy_offset"][0, 1] + 0.5
        u_518, v_518 = u_148 * scale_factor, v_148 * scale_factor

        Z = params["true_depth"][0, 0]
        X = (u_518 - cx) * Z / fx
        Y = (v_518 - cy) * Z / fy

        means = torch.stack([X, Y, Z], dim=-1).view(-1, 3) 
        quats = params["rot"][0].permute(1, 2, 0).view(-1, 4)         
        scales = params["scale"][0].permute(1, 2, 0).view(-1, 3)      
        
        x_raw = params["sh_dc"][0].permute(1, 2, 0).view(-1, 3)
        colors = torch.where(
            x_raw < 0.0,
            0.05 * x_raw,
            torch.where(
                x_raw > 1.0,
                1.0 + 0.05 * (x_raw - 1.0),
                x_raw
            ))

        opacities = params["opacity"][0].view(-1)
        if flat_mask is not None:
            opacities = opacities * flat_mask

        if normal_maps is not None:
            n_flat = torch.from_numpy(normal_maps[layer_idx]).to(device).view(-1, 3)
            all_normals.append(n_flat)

        all_means.append(means)
        all_quats.append(quats)
        all_scales.append(scales)
        all_opacities.append(opacities)
        all_colors.append(colors)

    normals_out = torch.cat(all_normals, dim=0) if normal_maps is not None else None
    return (torch.cat(all_means, dim=0), torch.cat(all_quats, dim=0), torch.cat(all_scales, dim=0), 
            torch.cat(all_opacities, dim=0), torch.cat(all_colors, dim=0), normals_out)

def main():
    device = torch.device("cuda")
    
    print("Loading models...")
    vggt = VGGTWrapper().to(device).eval()
    for p in vggt.parameters(): p.requires_grad = False

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    
    # [!] UPDATE THIS PATH to where your checkpoint actually lives on this machine
    ckpt_path = "/home/isiuts/stage1/checkpoint_epoch_22.pt" 
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    upsampler.load_state_dict(ckpt['upsampler'])
    decoder.load_state_dict(ckpt['decoder'])

    for param in decoder.calibrator.parameters():
        param.requires_grad = False

    upsampler.train()
    decoder.train()

    optimizer = optim.Adam(list(upsampler.parameters()) + list(decoder.parameters()), lr=1e-4)

    data_dir = "/home/isiuts/validation-set"
    sample_dirs = sorted([os.path.join(data_dir, d) for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])[:4]
    
    print(f"Pre-processing {len(sample_dirs)} samples...")
    samples = []
    
    for sid in sample_dirs:
        meta = json.load(open(os.path.join(sid, "metadata.json")))
        
        # Camera A Setup
        view_A_path = glob.glob(os.path.join(sid, "*view_A*.png"))[0]
        gt_A = transforms.ToTensor()(Image.open(view_A_path).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
        mask148_A = get_letter_mask(os.path.join(sid, "mesh.ply"), meta, device=device)
        mask518_A = F.interpolate(mask148_A.float(), size=(518, 518), mode='nearest').bool()
        Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmats_A = torch.eye(4, device=device).unsqueeze(0)
        
        # Camera B Setup
        meta_B = meta["camera_B"]
        view_B_path = glob.glob(os.path.join(sid, "*view_B*.png"))[0]
        gt_B = transforms.ToTensor()(Image.open(view_B_path).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
        mask148_B = get_letter_mask(os.path.join(sid, "mesh.ply"), meta_B, device=device)
        mask518_B = F.interpolate(mask148_B.float(), size=(518, 518), mode='nearest').bool()
        Ks_B = torch.tensor([[[meta_B["fx"], 0, meta_B["cx"]], [0, meta_B["fy"], meta_B["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], device)
        
        # Load offline generated GT Depth B and GT Normals A
        gt_depth_B = load_exr_depth_cpu(os.path.join(sid, "depth_B.exr")).to(device)
        gt_normal_A = [
            np.load(os.path.join(sid, "normal_A_layer01.npy")),
            np.load(os.path.join(sid, "normal_A_layer01.npy")),
            np.load(os.path.join(sid, "normal_A_layer2.npy")),
        ]

        with torch.no_grad():
            vggt_out = vggt.forward_with_features(gt_A)
            patch_tokens = vggt_out["patch_tokens"].detach()
            base_depth = vggt_out["depth"].detach()

        samples.append({
            "sid": sid,
            "meta": meta,
            "gt_A": gt_A,
            "mask148_A": mask148_A,
            "mask518_A": mask518_A,
            "Ks_A": Ks_A,
            "viewmats_A": viewmats_A,
            "gt_B": gt_B,
            "mask148_B": mask148_B,
            "mask518_B": mask518_B,
            "Ks_B": Ks_B,
            "viewmats_B": viewmats_B,
            "gt_depth_B": gt_depth_B,
            "gt_normal_A": gt_normal_A,
            "patch_tokens": patch_tokens,
            "base_depth": base_depth
        })

    # --- Initial Ground-Truth Camera-B Variance Check ---
    print("\n--- Initial Ground-Truth Camera-B Variance Check ---")
    C2_ref = 9e-4
    for i, s in enumerate(samples):
        gt_b_gray = 0.2989 * s["gt_B"][0, 0] + 0.5870 * s["gt_B"][0, 1] + 0.1140 * s["gt_B"][0, 2]
        mask_b = s["mask518_B"][0, 0]
        masked_pixels = gt_b_gray[mask_b]
        
        if len(masked_pixels) > 0:
            var = masked_pixels.var().item()
            status = "WITHIN 10x range" if (9e-5 <= var <= 9e-3) else "OUTSIDE 10x range"
            print(f"Sample {i} | Grayscale Variance: {var:.6e} | {status} (vs C2={C2_ref})")
        else:
            print(f"Sample {i} | Mask empty, cannot compute variance.")
    print("----------------------------------------------------\n")

    print(f"Starting Overfit Loop (3000 steps, lr=1e-4)...")
    
    W_DEPTH_B = 10.0
    W_ORIENT = 2.0

    for step in range(3001):
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        
        for i, s in enumerate(samples):
            up_feat = upsampler(s["patch_tokens"])
            
            p_list, _, _, _, _ = decoder(up_feat, s["base_depth"], s["patch_tokens"])
            
            m, q, scale, o, c, n = flatten_decoder_outputs_camera_space(
                p_list[0], p_list[1], p_list[2], 
                (s["meta"]["fx"], s["meta"]["fy"], s["meta"]["cx"], s["meta"]["cy"]), 
                device, mask_148=s["mask148_A"], normal_maps=s["gt_normal_A"]
            )
            
            # 1. Camera A Render + L1 Loss
            render_A, _, _ = rasterization(
                means=m.float(), quats=q.float(), scales=scale.float(), opacities=o.float(), colors=c.float(),
                viewmats=s["viewmats_A"], Ks=s["Ks_A"], width=518, height=518
            )
            pred_rgb_A_raw = render_A.permute(0, 3, 1, 2)
            pred_rgb_A_masked = pred_rgb_A_raw * s["mask518_A"]
            loss_l1_A = compute_l1_rgb_loss(pred_rgb_A_masked, s["gt_A"], mask=s["mask518_A"])
            
            # 2. Camera B Depth Loss (with gradient, RGB+ED mode)
            render_B_train, _, _ = rasterization(
                means=m.float(), quats=q.float(), scales=scale.float(), opacities=o.float(), colors=c.float(),
                viewmats=s["viewmats_B"], Ks=s["Ks_B"], width=518, height=518, render_mode="RGB+ED"
            )
            depth_B_pred = render_B_train[..., 3]
            loss_depth_B = compute_novel_view_depth_loss(depth_B_pred, s["gt_depth_B"], s["mask518_B"])

            # 3. Orientation Loss
            flat_mask_A = s["mask148_A"][0, 0].bool().view(-1).repeat(3)
            loss_orient = compute_orientation_loss(q, scale, n, valid_mask=flat_mask_A)

            # Combined training loss
            loss = loss_l1_A + W_DEPTH_B * loss_depth_B + W_ORIENT * loss_orient
            total_loss = total_loss + loss
            
            if step % 300 == 0:
                # Camera A PSNR
                pred_A_np = (pred_rgb_A_raw[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                gt_A_np = (s["gt_A"][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                mask_A_np = s["mask518_A"][0, 0].cpu().numpy().astype(bool)
                
                pred_A_masked = pred_A_np[mask_A_np]
                gt_A_masked = gt_A_np[mask_A_np]
                psnr_A = psnr_metric(gt_A_masked, pred_A_masked, data_range=255) if len(gt_A_masked) > 0 else 0.0
                
                # Camera B Read-Only RGB Evaluation (NO gradient)
                with torch.no_grad():
                    render_B_eval, _, _ = rasterization(
                        means=m.float(), quats=q.float(), scales=scale.float(), opacities=o.float(), colors=c.float(),
                        viewmats=s["viewmats_B"], Ks=s["Ks_B"], width=518, height=518
                    )
                    pred_rgb_B_raw = render_B_eval.permute(0, 3, 1, 2)
                
                pred_B_np = (pred_rgb_B_raw[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                gt_B_np = (s["gt_B"][0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8).transpose(1, 2, 0)
                mask_B_np = s["mask518_B"][0, 0].cpu().numpy().astype(bool)
                
                pred_B_masked = pred_B_np[mask_B_np]
                gt_B_masked = gt_B_np[mask_B_np]
                psnr_B = psnr_metric(gt_B_masked, pred_B_masked, data_range=255) if len(gt_B_masked) > 0 else 0.0

                ys, xs = np.where(mask_B_np)
                ssim_B = 0.0
                if len(ys) > 0 and len(xs) > 0 and (ys.max() - ys.min() + 1) >= 7 and (xs.max() - xs.min() + 1) >= 7:
                    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                    ssim_B = ssim_metric(gt_B_np[y0:y1, x0:x1], pred_B_np[y0:y1, x0:x1], channel_axis=-1, data_range=255)
                
                print(f"Step {step:4d} | S{i} | Losses [L1_A: {loss_l1_A.item():.4f}, Depth_B: {loss_depth_B.item():.4f}, Orient: {loss_orient.item():.4f}] || Cam A PSNR: {psnr_A:5.2f} || Cam B [PSNR: {psnr_B:5.2f}, SSIM: {ssim_B:.4f}]")

        avg_loss = total_loss / len(samples)
        avg_loss.backward()
        
        if step == 0:
            total_grad_norm = sum(p.grad.norm().item() for p in list(upsampler.parameters()) + list(decoder.parameters()) if p.grad is not None)
            none_grad_count = sum(1 for p in list(upsampler.parameters()) + list(decoder.parameters()) if p.grad is None)
            print(f"[DEBUG] Step 0 grad norm: {total_grad_norm:.6f}, params with None grad: {none_grad_count}")

        optimizer.step()

if __name__ == "__main__":
    main()