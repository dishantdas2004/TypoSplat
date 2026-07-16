"""
TypoSplat Stage 0: Mini-Batch Overfit (Depth Calibrator + Centroid Bootstrap)
Accepts multiple sample directories to provide diverse gradient signal to the Calibrator.
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
    compute_novel_view_loss,
    compute_centroid_loss,
    compute_zoffset_regularization,
    compute_opacity_sparsity_loss,
    _get_relative_viewmat
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
    print(f"=== TypoSplat: Stage 0 Training (Mini-Batch Calibrator Test) ===")

    # Accept multiple directories from CLI
    sample_dirs = sys.argv[1:] if len(sys.argv) > 1 else ["/content/data/19"]
    print(f"Loading {len(sample_dirs)} samples for batched training...")

    vggt = VGGTWrapper().to(device) 
    for param in vggt.parameters():
        param.requires_grad = False

    dataset = []

    # Pre-cache all GT data and VGGT features to save memory and time
    for sample_dir in sample_dirs:
        meta_path = os.path.join(sample_dir, "metadata.json")
        mesh_path = os.path.join(sample_dir, "mesh.ply")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        view_A_paths = glob.glob(os.path.join(sample_dir, "*view_A*.png"))
        view_B_paths = glob.glob(os.path.join(sample_dir, "*view_B*.png"))
        depth_A_paths = glob.glob(os.path.join(sample_dir, "*depth_A*.exr"))

        gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
        gt_depth_518_A = load_exr_depth(depth_A_paths[0], device)
        gt_depth_148_A = torch.nn.functional.interpolate(gt_depth_518_A, size=(148, 148), mode='nearest')
        gt_rgb_B = transforms.ToTensor()(Image.open(view_B_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)

        mask_148_A = get_letter_mask(mesh_path, meta, device=device)
        mask_518_A = torch.nn.functional.interpolate(mask_148_A, size=(518, 518), mode='nearest')
        mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device=device)
        mask_518_B = torch.nn.functional.interpolate(mask_148_B, size=(518, 518), mode='nearest')

        intrinsics_tuple_A = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        scale_148 = 148.0 / 518.0
        intrinsics_dict_148_A = {
            "fx": meta["fx"] * scale_148, "fy": meta["fy"] * scale_148, 
            "cx": meta["cx"] * scale_148, "cy": meta["cy"] * scale_148
        }
        Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmats_A = torch.eye(4, device=device).unsqueeze(0)

        meta_B = meta["camera_B"]
        Ks_B = torch.tensor([[[meta_B["fx"], 0, meta_B["cx"]], [0, meta_B["fy"], meta_B["cy"]], [0, 0, 1]]], dtype=torch.float32, device=device)
        viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], device)

        with torch.no_grad():
            vggt_out = vggt.forward_with_features(gt_rgb_A) 

        dataset.append({
            "dir": sample_dir,
            "meta": meta,
            "gt_rgb_A": gt_rgb_A,
            "gt_depth_148_A": gt_depth_148_A,
            "gt_rgb_B": gt_rgb_B,
            "mask_148_A": mask_148_A,
            "mask_518_A": mask_518_A,
            "mask_518_B": mask_518_B,
            "intrinsics_tuple_A": intrinsics_tuple_A,
            "intrinsics_dict_148_A": intrinsics_dict_148_A,
            "Ks_A": Ks_A,
            "viewmats_A": viewmats_A,
            "Ks_B": Ks_B,
            "viewmats_B": viewmats_B,
            "patch_tokens": vggt_out["patch_tokens"],
            "base_depth": vggt_out["depth"]
        })

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    lpips_fn = ShallowPerceptualLoss(device)

    # --- Split Optimizer Groups (Calibrator gets 10x slower LR) ---
    calibrator_params = list(decoder.calibrator.parameters())
    base_params = list(upsampler.parameters()) + [p for n, p in decoder.named_parameters() if 'calibrator' not in n]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': calibrator_params, 'lr': 1e-5}
    ])

    iterations = 2000 
    batch_size = len(dataset)

    print("\nStarting Training Loop...")
    upsampler.train()
    decoder.train()

    for i in tqdm(range(iterations)):
        # --- FIXED: Calibrator LR Warmup ---
        if i < 500:
            warmup_lr = 1e-7 + (i / 500.0) * (1e-5 - 1e-7)
            optimizer.param_groups[1]['lr'] = warmup_lr
        else:
            optimizer.param_groups[1]['lr'] = 1e-5

        optimizer.zero_grad()

        batch_total_loss = 0.0
        log_metrics = {} 

        for data in dataset:
            upsampled_features = upsampler(data["patch_tokens"])

            # --- Unpack raw_calib_out for L2 Regularization ---
            params_list, calib_scale, calib_shift, raw_calib_out = decoder(upsampled_features, data["base_depth"], data["patch_tokens"])
            params_0, params_1, params_2 = params_list

            means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
                params_0, params_1, params_2, data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
            )

            render_colors_A, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518,
            )
            pred_rgb_A_raw = render_colors_A.permute(0, 3, 1, 2)

            # --- Mask-Resolution Fix (Branching) ---
            pred_rgb_A_masked = pred_rgb_A_raw * data["mask_518_A"]

            loss_rgb = compute_l1_rgb_loss(pred_rgb_A_masked, data["gt_rgb_A"], mask=data["mask_518_A"])
            loss_edge = compute_sobel_edge_loss(pred_rgb_A_masked, data["gt_rgb_A"], mask=data["mask_518_A"])
            loss_lpips = lpips_fn(pred_rgb_A_raw, data["gt_rgb_A"], mask=data["mask_518_A"])

            layer_1_depth = params_0["true_depth"] + params_1["z_offset"]
            loss_depth = compute_scale_invariant_depth_loss(layer_1_depth, data["gt_depth_148_A"], data["mask_148_A"].bool())
            loss_extrusion = compute_extrusion_loss(params_1, params_2, data["meta"]["extrusion_depth"], data["mask_148_A"])
            loss_aniso = compute_anisotropy_loss(scales, r_bound=10.0)
            loss_normal = compute_normal_loss(layer_1_depth, data["gt_depth_148_A"], data["intrinsics_dict_148_A"], data["mask_148_A"])

            # --- Symmetric Novel View Photometric Suite ---
            loss_rgb_B, loss_edge_B, loss_lpips_B, render_colors_B = compute_novel_view_loss(
                means, quats, scales, opacities, colors, data["viewmats_B"], data["Ks_B"], data["gt_rgb_B"], data["mask_518_B"], lpips_fn
            )

            loss_novel_view = loss_rgb_B + loss_edge_B + (0.002 * loss_lpips_B)

            loss_centroid = compute_centroid_loss(means, data["viewmats_B"], data["Ks_B"], data["mask_518_B"], device)
            loss_zreg = compute_zoffset_regularization(params_1, params_2)
            loss_opacity_sparsity = compute_opacity_sparsity_loss(opacities)

            # --- Soft L2 Regularization on raw calibrator output ---
            loss_calib_reg = (raw_calib_out ** 2).mean()

            centroid_weight = max(0.0, 1.0 - i / 1500.0)

            # --- Print raw magnitudes on Iter 0 for ALL samples ---
            if i == 0:
                sample_name = os.path.basename(data["dir"])
                print(f"\n--- CAMERA B RAW MAGNITUDES: Sample {sample_name} (ITER 0) ---")
                print(f"Raw RGB B:   {loss_rgb_B.item():.4f}")
                print(f"Raw Edge B:  {loss_edge_B.item():.4f}")
                print(f"Raw LPIPS B: {loss_lpips_B.item():.4f}")
                print("-" * 50)

            # Extract NV Gradient for diagnostic before the graph is consumed
            nv_grad_mag = 0.0
            if (i+1) % 500 == 0:
                nv_grad = torch.autograd.grad(loss_novel_view, means, retain_graph=True, allow_unused=True)[0]
                nv_grad_mag = nv_grad.abs().mean().item() if nv_grad is not None else 0.0

            sample_loss = (
                1.0 * loss_rgb + 
                1.0 * loss_edge + 
                0.002 * loss_lpips + 
                50.0 * loss_depth + 
                1000.0 * loss_extrusion +
                1.0 * loss_aniso + 
                1.0 * loss_normal +
                0.5 * loss_novel_view +
                centroid_weight * 0.05 * loss_centroid +
                0.05 * loss_zreg +
                1.0 * loss_opacity_sparsity +
                0.01 * loss_calib_reg  # Added Calibrator L2 Penalty
            )

            # Accumulate gradient safely
            (sample_loss / batch_size).backward()
            batch_total_loss += sample_loss.item() / batch_size

            # Store Calibrator metrics for 500s AND early checks
            if (i+1) % 500 == 0 or (i+1) in [100, 200, 300, 400]:
                sample_name = os.path.basename(data["dir"])
                if "calib_tracking" not in log_metrics:
                    log_metrics["calib_tracking"] = {}

                log_metrics["calib_tracking"][sample_name] = {
                    "scale": calib_scale[0,0].item(),
                    "shift": calib_shift[0,0].item()
                }

            # Keep the other heavy metrics only on 500s
            if (i+1) % 500 == 0:
                log_metrics["rgb"] = loss_rgb.item()
                log_metrics["nv"] = loss_novel_view.item()
                log_metrics["cent"] = loss_centroid.item()
                log_metrics["nv_grad"] = nv_grad_mag

                with torch.no_grad():
                    means_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)
                    points_camB = (data["viewmats_B"][0] @ means_h.T).T
                    Ks_B = data["Ks_B"]

                    # --- FIXED: Robust Z bounds for internal off-screen logging ---
                    Z_MIN = 1.0
                    valid_z = points_camB[:, 2] > Z_MIN
                    Z_safe = points_camB[:, 2].clone()
                    Z_safe[~valid_z] = 1.0

                    x_proj = (points_camB[:, 0] / Z_safe) * Ks_B[0,0,0] + Ks_B[0,0,2]
                    y_proj = (points_camB[:, 1] / Z_safe) * Ks_B[0,1,1] + Ks_B[0,1,2]

                    in_frame = valid_z & (x_proj >= 0) & (x_proj <= 518) & (y_proj >= 0) & (y_proj <= 518)
                    out_of_bounds = ~in_frame

                    log_metrics["frac_off"] = out_of_bounds.float().mean().item()

        optimizer.step()

        # --- Early Blow-up Catch (Iterates over ALL samples) ---
        if (i+1) in [100, 200, 300, 400]:
            print(f"\n[Early Check Iter {i+1:04d}] Calibrator Tracking:")
            for s_name, vals in log_metrics["calib_tracking"].items():
                print(f"     > Sample {s_name}: Scale = {vals['scale']:.4f}, Shift = {vals['shift']:.4f}")

        if (i+1) % 500 == 0:
            print(f"\nIter {i+1:04d} | Batch Avg Total: {batch_total_loss:.4f}")
            print(f"   [Calibrator Tracking]")
            for s_name, vals in log_metrics["calib_tracking"].items():
                print(f"     > Sample {s_name}: Scale = {vals['scale']:.4f}, Shift = {vals['shift']:.4f}")

            print(f"   [General Metrics (Sample {os.path.basename(dataset[-1]['dir'])})]")
            print(f"     > RGB: {log_metrics['rgb']:.4f} | Novel View: {log_metrics['nv']:.4f} | Centroid: {log_metrics['cent']:.4f}")
            print(f"     > NV Grad Mag: {log_metrics['nv_grad']:.10f} | Off-screen B: {log_metrics['frac_off']:.2%}")
            print("-" * 50)

        # --- Checkpointing every 1000 iterations ---
        if (i+1) % 1000 == 0:
            checkpoint_path = f"checkpoint_iter{i+1}.pt"
            torch.save({
                'upsampler': upsampler.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': i+1
            }, checkpoint_path)
            print(f"   [!] Saved checkpoint to {checkpoint_path}")

    print("\n[SUCCESS] Stage 0 Mini-Batch Overfit Complete!")

    last_data = dataset[-1]
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes[0,0].imshow(last_data["gt_rgb_A"][0].permute(1, 2, 0).cpu().numpy() * last_data["mask_518_A"][0].permute(1, 2, 0).cpu().numpy())
    axes[0,0].set_title("GT Camera A")

    axes[0,1].imshow(pred_rgb_A_masked[0].permute(1,2,0).detach().cpu().numpy())
    axes[0,1].set_title("Render Camera A (518x518 Gated)")

    axes[1,0].imshow(last_data["gt_rgb_B"][0].permute(1, 2, 0).cpu().numpy() * last_data["mask_518_B"][0].permute(1, 2, 0).cpu().numpy())
    axes[1,0].set_title("GT Camera B")

    vis_render_B = render_colors_B.permute(0,3,1,2) * last_data["mask_518_B"]
    axes[1,1].imshow(vis_render_B[0].permute(1,2,0).detach().cpu().numpy())
    axes[1,1].set_title("Render Camera B (518x518 Gated)")

    out_path = os.path.join(last_data["dir"], "batch_result_dual.png")
    plt.savefig(out_path, dpi=150)

if __name__ == "__main__":
    main()