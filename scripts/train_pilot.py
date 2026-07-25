"""
TypoSplat Stage 1 Pilot Training
Epoch-based mini-batch training for scale-agnostic dataset pools.
Includes held-out evaluation logging and loss magnitude audits.
"""

import os
import sys
import glob
import json
import random
import argparse
import torch
import torch.optim as optim
import pandas as pd  
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
    compute_calibrator_regression_loss, 
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

# --- Evaluation Helper with Per-Sample Tracking ---
def evaluate(eval_set, upsampler, decoder, device):
    upsampler.eval()
    decoder.eval()
    scale_errors, shift_errors = [], []
    per_sample_results = []
    with torch.no_grad():
        for data in eval_set:
            upsampled_features = upsampler(data["patch_tokens"])
            _, calib_scale, calib_shift, _, attn_weights = decoder(upsampled_features, data["base_depth"], data["patch_tokens"])
            
            pred_scale = calib_scale.item()
            pred_shift = calib_shift.item()
            
            scale_errors.append(abs(pred_scale - data["target_opt_scale"]))
            shift_errors.append(abs(pred_shift - data["target_opt_shift"]))
            
            per_sample_results.append({
                "sample": os.path.basename(data["dir"]),
                "pred_scale": pred_scale,
                "target_scale": data["target_opt_scale"],
                "pred_shift": pred_shift,
                "target_shift": data["target_opt_shift"],
                "attn_max": attn_weights.max().item(),
                "attn_std": attn_weights.std().item(),
            })
            
    upsampler.train()
    decoder.train()
    
    mean_scale_err = sum(scale_errors) / len(scale_errors)
    mean_shift_err = sum(shift_errors) / len(shift_errors)
    return mean_scale_err, mean_shift_err, per_sample_results


def main():
    parser = argparse.ArgumentParser(description="TypoSplat Pilot Epoch Training")
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Parent directory containing sample folders")
    parser.add_argument("--diag_csv", type=str, default="/content/diagnostic_results.csv", help="Path to diagnostic CSV")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=30, help="Total number of epochs")
    parser.add_argument("--anneal_epochs", type=float, default=4.0, help="Epochs over which to anneal calibrator regression weight")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== TypoSplat: Stage 1 Pilot Training ===")

    try:
        diagnostic_df = pd.read_csv(args.diag_csv)
        diagnostic_df = diagnostic_df.set_index("Sample")
    except FileNotFoundError:
        print(f"WARNING: {args.diag_csv} not found! Training will fail if not provided.")
        sys.exit(1)

    vggt = VGGTWrapper().to(device) 
    for param in vggt.parameters():
        param.requires_grad = False

    sample_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                   if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    print(f"Discovered {len(sample_dirs)} valid samples in {args.data_dir}. Loading into memory...")

    dataset = []

    for sample_dir in tqdm(sample_dirs, desc="Pre-caching dataset"):
        sample_id = os.path.basename(sample_dir)
        try:
            target_opt_scale = float(diagnostic_df.loc[int(sample_id), "Opt_Scale"])
            target_opt_shift = float(diagnostic_df.loc[int(sample_id), "Opt_Shift"])
        except KeyError:
            continue

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
            "base_depth": vggt_out["depth"],
            "target_opt_scale": target_opt_scale,
            "target_opt_shift": target_opt_shift
        })

    print(f"Successfully loaded {len(dataset)} samples. Ready for training.")

    # --- Held-out Split Logic ---
    eval_size = 64 if len(dataset) > 64 else max(1, len(dataset) // 5)
    eval_set = dataset[-eval_size:]
    train_set = dataset[:-eval_size]
    print(f"Dataset split: {len(train_set)} for training, {len(eval_set)} for evaluation.")

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    lpips_fn = ShallowPerceptualLoss(device)

    calibrator_params = list(decoder.calibrator.parameters())
    base_params = list(upsampler.parameters()) + [p for n, p in decoder.named_parameters() if 'calibrator' not in n]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': calibrator_params, 'lr': 1e-5}
    ])

    batch_size = args.batch_size
    iters_per_epoch = len(train_set) // batch_size
    
    if iters_per_epoch == 0:
        print("ERROR: Training set size is smaller than batch size. Reduce batch size or add samples.")
        sys.exit(1)

    anneal_iters = int(iters_per_epoch * args.anneal_epochs)
    log_freq = max(1, iters_per_epoch // 4)

    print("\n=======================================================")
    print(f"Train Samples:   {len(train_set)}")
    print(f"Eval Samples:    {len(eval_set)}")
    print(f"Batch Size:      {batch_size}")
    print(f"Iters per Epoch: {iters_per_epoch}")
    print(f"Total Epochs:    {args.num_epochs}")
    print(f"Anneal Window:   {args.anneal_epochs} epochs ({anneal_iters} iters)")
    print(f"Logging Freq:    Every {log_freq} iters")
    print("=======================================================\n")

    upsampler.train()
    decoder.train()

    global_iter = 0

    for epoch in range(1, args.num_epochs + 1):
        random.shuffle(train_set)
        
        for batch_start in range(0, len(train_set) - batch_size + 1, batch_size):
            batch = train_set[batch_start : batch_start + batch_size]
            global_iter = (epoch - 1) * iters_per_epoch + (batch_start // batch_size)

            if global_iter < 1000:
                calib_lr = 1e-7 + (global_iter / 1000.0) * (1e-5 - 1e-7)
                optimizer.param_groups[1]['lr'] = calib_lr
            else:
                optimizer.param_groups[1]['lr'] = 1e-5

            optimizer.zero_grad()
            batch_total_loss = 0.0
            
            is_log_step = (global_iter % log_freq == 0)
            log_metrics = {} 

            for sample_idx, data in enumerate(batch):
                upsampled_features = upsampler(data["patch_tokens"])

                params_list, calib_scale, calib_shift, _, attn_weights = decoder(upsampled_features, data["base_depth"], data["patch_tokens"])
                params_0, params_1, params_2 = params_list

                means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
                    params_0, params_1, params_2, data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                )

                render_colors_A, _, _ = rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                    viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518,
                )
                pred_rgb_A_raw = render_colors_A.permute(0, 3, 1, 2)
                pred_rgb_A_masked = pred_rgb_A_raw * data["mask_518_A"]

                loss_rgb = compute_l1_rgb_loss(pred_rgb_A_masked, data["gt_rgb_A"], mask=data["mask_518_A"])
                loss_edge = compute_sobel_edge_loss(pred_rgb_A_masked, data["gt_rgb_A"], mask=data["mask_518_A"])
                loss_lpips = lpips_fn(pred_rgb_A_raw, data["gt_rgb_A"], mask=data["mask_518_A"])

                layer_1_depth = params_0["true_depth"] + params_1["z_offset"]
                loss_depth = compute_scale_invariant_depth_loss(layer_1_depth, data["gt_depth_148_A"], data["mask_148_A"].bool())
                loss_extrusion = compute_extrusion_loss(params_1, params_2, data["meta"]["extrusion_depth"], data["mask_148_A"])
                loss_aniso = compute_anisotropy_loss(scales, r_bound=10.0)
                loss_normal = compute_normal_loss(layer_1_depth, data["gt_depth_148_A"], data["intrinsics_dict_148_A"], data["mask_148_A"])

                loss_rgb_B, loss_edge_B, loss_lpips_B, render_colors_B = compute_novel_view_loss(
                    means, quats, scales, opacities, colors, data["viewmats_B"], data["Ks_B"], data["gt_rgb_B"], data["mask_518_B"], lpips_fn, iteration=global_iter
                )

                loss_novel_view = loss_rgb_B + loss_edge_B + (0.002 * loss_lpips_B)
                loss_centroid = compute_centroid_loss(means, data["viewmats_B"], data["Ks_B"], data["mask_518_B"], device)
                loss_zreg = compute_zoffset_regularization(params_1, params_2)
                loss_opacity_sparsity = compute_opacity_sparsity_loss(opacities)

                centroid_weight = max(0.0, 1.0 - global_iter / 1500.0)
                calib_reg_weight = max(0.2, 1.0 - global_iter / anneal_iters)

                loss_calib_target = compute_calibrator_regression_loss(
                    calib_scale, calib_shift, 
                    torch.tensor(data["target_opt_scale"], device=device), 
                    torch.tensor(data["target_opt_shift"], device=device)
                )

                # --- Loss Weight Magnitude Audit (Only First Sample of First Batch) ---
                if global_iter == 0 and sample_idx == 0:
                    print("\n=== LOSS WEIGHT AUDIT (iter 0, first sample in first batch) ===")
                    terms = {
                        "rgb": (1.0, loss_rgb.item()),
                        "edge": (1.0, loss_edge.item()),
                        "lpips": (0.002, loss_lpips.item()),
                        "depth": (50.0, loss_depth.item()),
                        "extrusion": (1000.0, loss_extrusion.item()),
                        "aniso": (1.0, loss_aniso.item()),
                        "normal": (1.0, loss_normal.item()),
                        "novel_view": (0.5, loss_novel_view.item()),
                        "centroid": (centroid_weight * 0.05, loss_centroid.item()),
                        "zreg": (0.05, loss_zreg.item()),
                        "opacity_sparsity": (1.0, loss_opacity_sparsity.item()),
                        "calib_reg": (calib_reg_weight * 2.0, loss_calib_target.item()),
                    }
                    total = sum(w * v for w, v in terms.values())
                    for name, (w, v) in sorted(terms.items(), key=lambda x: -x[1][0]*x[1][1]):
                        contribution = w * v
                        print(f"  {name:20s}: raw={v:10.4f}  weight={w:8.4f}  contribution={contribution:10.4f}  ({contribution/total:.1%} of total)")
                    print(f"  {'TOTAL':20s}: {total:.4f}")
                    print("=" * 60 + "\n")

                if is_log_step:
                    nv_grad = torch.autograd.grad(loss_novel_view, means, retain_graph=True, allow_unused=True)[0]
                    nv_grad_mag = nv_grad.abs().mean().item() if nv_grad is not None else 0.0

                    with torch.no_grad():
                        non_zero_pixels = torch.nonzero(render_colors_B[0].sum(-1) > 0.01, as_tuple=False)
                        mask_pixels = torch.nonzero(data["mask_518_B"].squeeze() > 0.5, as_tuple=False)
                        sample_name = os.path.basename(data["dir"])
                        if len(non_zero_pixels) > 0:
                            print(f"[BBOX CHECK - Epoch {epoch} | Sample {sample_name}] Render bbox: Y[{non_zero_pixels[:,0].min().item()}-{non_zero_pixels[:,0].max().item()}] X[{non_zero_pixels[:,1].min().item()}-{non_zero_pixels[:,1].max().item()}]")
                        else:
                            print(f"[BBOX CHECK - Epoch {epoch} | Sample {sample_name}] Render bbox: EMPTY (no pixels above threshold)")
                        print(f"[BBOX CHECK - Epoch {epoch} | Sample {sample_name}] Mask bbox:   Y[{mask_pixels[:,0].min().item()}-{mask_pixels[:,0].max().item()}] X[{mask_pixels[:,1].min().item()}-{mask_pixels[:,1].max().item()}]")

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
                    (calib_reg_weight * 2.0 * loss_calib_target) 
                )
                
                (sample_loss / batch_size).backward()
                batch_total_loss += sample_loss.item() / batch_size

                if is_log_step:
                    sample_name = os.path.basename(data["dir"])
                    if "calib_tracking" not in log_metrics:
                        log_metrics["calib_tracking"] = {}

                    log_metrics["calib_tracking"][sample_name] = {
                        "scale": calib_scale[0,0].item(),
                        "shift": calib_shift[0,0].item(),
                        "target_scale": data["target_opt_scale"],
                        "target_shift": data["target_opt_shift"],
                        "reg_loss": loss_calib_target.item()
                    }

                    log_metrics["rgb"] = loss_rgb.item()
                    log_metrics["nv"] = loss_novel_view.item()
                    log_metrics["cent"] = loss_centroid.item()
                    log_metrics["nv_grad"] = nv_grad_mag
                    log_metrics["calib_reg_weight"] = calib_reg_weight * 2.0

                    with torch.no_grad():
                        means_h = torch.cat([means, torch.ones_like(means[:, :1])], dim=1)
                        points_camB = (data["viewmats_B"][0] @ means_h.T).T
                        Ks_B = data["Ks_B"]

                        Z_MIN = 1.0
                        valid_z = points_camB[:, 2] > Z_MIN
                        Z_safe = points_camB[:, 2].clone()
                        Z_safe[~valid_z] = 1.0

                        x_proj = (points_camB[:, 0] / Z_safe) * Ks_B[0,0,0] + Ks_B[0,0,2]
                        y_proj = (points_camB[:, 1] / Z_safe) * Ks_B[0,1,1] + Ks_B[0,1,2]

                        in_frame = valid_z & (x_proj >= 0) & (x_proj <= 518) & (y_proj >= 0) & (y_proj <= 518)
                        out_of_bounds = ~in_frame

                        log_metrics["frac_off"] = out_of_bounds.float().mean().item()

                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    axes[0,0].imshow(data["gt_rgb_A"][0].permute(1, 2, 0).cpu().numpy() * data["mask_518_A"][0].permute(1, 2, 0).cpu().numpy())
                    axes[0,0].set_title("GT Camera A")
                    axes[0,1].imshow(pred_rgb_A_masked[0].permute(1,2,0).detach().cpu().numpy())
                    axes[0,1].set_title(f"Render A (Epoch {epoch} Iter {global_iter})")
                    axes[1,0].imshow(data["gt_rgb_B"][0].permute(1, 2, 0).cpu().numpy() * data["mask_518_B"][0].permute(1, 2, 0).cpu().numpy())
                    axes[1,0].set_title("GT Camera B")
                    vis_render_B = render_colors_B.permute(0,3,1,2) * data["mask_518_B"]
                    axes[1,1].imshow(vis_render_B[0].permute(1,2,0).detach().cpu().numpy())
                    axes[1,1].set_title(f"Render B (Epoch {epoch} Iter {global_iter})")
                    
                    out_path = os.path.join(data["dir"], f"render_epoch_{epoch}_iter_{global_iter}.png")
                    plt.savefig(out_path, dpi=150)
                    plt.close(fig)

                    fig_attn, ax_attn = plt.subplots(figsize=(5, 5))
                    attn_map = attn_weights[0, 0].view(37, 37).cpu().detach().numpy()
                    im = ax_attn.imshow(attn_map, cmap='viridis')
                    ax_attn.set_title(f"Attention Map (Epoch {epoch} Iter {global_iter})")
                    fig_attn.colorbar(im, ax=ax_attn)
                    attn_out_path = os.path.join(data["dir"], f"attn_map_epoch_{epoch}_iter_{global_iter}.png")
                    plt.savefig(attn_out_path, dpi=150)
                    plt.close(fig_attn)

            optimizer.step()

            if is_log_step:
                print(f"\n[Epoch {epoch}/{args.num_epochs} | Global Iter {global_iter}] Batch Avg Total Loss: {batch_total_loss:.4f}")
                print(f"   [Calibrator Tracking]")
                for s_name, vals in log_metrics["calib_tracking"].items():
                    print(f"     > Sample {s_name}: Scale = {vals['scale']:.4f} (Target: {vals['target_scale']:.4f}) | Shift = {vals['shift']:.4f} (Target: {vals['target_shift']:.4f})")
                    print(f"       Reg Loss: {vals['reg_loss']:.4f} (Weight: {log_metrics['calib_reg_weight']:.2f})")

                print(f"   [General Metrics (Sample {os.path.basename(batch[-1]['dir'])})]")
                print(f"     > RGB: {log_metrics['rgb']:.4f} | Novel View: {log_metrics['nv']:.4f} | Centroid: {log_metrics['cent']:.4f}")
                print(f"     > NV Grad Mag: {log_metrics['nv_grad']:.10f} | Off-screen B: {log_metrics['frac_off']:.2%}")
                print("-" * 50)
                
                # --- NEW: Training Log CSV Appending ---
                train_log_path = "training_log.csv"
                write_header = not os.path.exists(train_log_path)
                with open(train_log_path, "a") as f:
                    if write_header:
                        f.write("epoch,global_iter,batch_total_loss,nv_grad_mag,off_screen_frac\n")
                    f.write(f"{epoch},{global_iter},{batch_total_loss:.6f},{log_metrics['nv_grad']:.10f},{log_metrics['frac_off']:.6f}\n")

        # --- Epoch End Processing ---
        checkpoint_path = f"checkpoint_epoch_{epoch}.pt"
        torch.save({
            'upsampler': upsampler.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_iter': global_iter
        }, checkpoint_path)
        
        # --- Held-out Evaluation Pass ---
        mean_scale_err, mean_shift_err, per_sample_results = evaluate(eval_set, upsampler, decoder, device)
        print(f"\n[EPOCH {epoch} EVAL] Mean |scale_err| = {mean_scale_err:.4f} | Mean |shift_err| = {mean_shift_err:.4f}")
        
        # Log to Detailed Per-Sample CSV (Tracking first 10 for detailed traces)
        per_sample_csv_path = "eval_per_sample_history.csv"
        write_header_per_sample = not os.path.exists(per_sample_csv_path)
        with open(per_sample_csv_path, "a") as f:
            if write_header_per_sample:
                f.write("epoch,sample,pred_scale,target_scale,pred_shift,target_shift,attn_max,attn_std\n")
            for r in per_sample_results[:10]:
                f.write(f"{epoch},{r['sample']},{r['pred_scale']:.6f},{r['target_scale']:.6f},{r['pred_shift']:.6f},{r['target_shift']:.6f},{r['attn_max']:.6f},{r['attn_std']:.6f}\n")

        # Log Aggregate Metrics to Eval History
        mean_attn_max = sum(r["attn_max"] for r in per_sample_results) / len(per_sample_results)
        eval_csv_path = "eval_history.csv"
        write_header_eval = not os.path.exists(eval_csv_path)
        with open(eval_csv_path, "a") as f:
            if write_header_eval:
                f.write("epoch,mean_scale_err,mean_shift_err,mean_attn_max\n")
            f.write(f"{epoch},{mean_scale_err:.6f},{mean_shift_err:.6f},{mean_attn_max:.6f}\n")

        print(f"[!] End of Epoch {epoch} -> Saved checkpoint to {checkpoint_path}\n")

    print(f"\n[SUCCESS] Stage 1 Pilot Complete ({args.num_epochs} Epochs)!")

if __name__ == "__main__":
    main()