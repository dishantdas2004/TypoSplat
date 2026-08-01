"""
TypoSplat Stage 1 Full Training
===============================
Epoch-based mini-batch training with Lazy Loading Two-Tier Dataset (Disk/RAM).
Supports separate evaluation datasets with automatic overlap checking.
Includes Google Drive checkpoint and cache persistence, train-vs-val tracking,
and per-epoch loss component breakdowns.
"""

import os
import sys
import glob
import json
import shutil
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
from torch.utils.data import Dataset, DataLoader
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
from src.utils.cache_tier import get_cache_tier
from gsplat import rasterization

# ==========================================
# 1. Dataset & Two-Tier Cache Helpers
# ==========================================

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

def restore_disk_tier_from_drive(sample_dirs, drive_backup_root):
    if not os.path.exists(drive_backup_root):
        return
    restored = 0
    for sample_dir in sample_dirs:
        cache_path = os.path.join(sample_dir, "cached_features.pt")
        if os.path.exists(cache_path):
            continue
        sample_id = os.path.basename(sample_dir)
        src = os.path.join(drive_backup_root, f"{sample_id}.pt")
        if os.path.exists(src):
            shutil.copy2(src, cache_path)
            restored += 1
    if restored:
        print(f"[INFO] Restored {restored} disk-tier cache files from Drive backup.")

def build_or_load_ram_cache(all_sample_dirs, vggt, device, backup_path):
    """
    Globally loads or computes the RAM-tier cache for all samples (train + eval).
    Guarantees all RAM-tier tensors stay in System RAM (CPU) in fp16.
    """
    ram_dirs = [d for d in all_sample_dirs if get_cache_tier(int(os.path.basename(d))) == "ram"]
    ram_cache = {}
    
    if os.path.exists(backup_path):
        print(f"Loading global RAM-tier cache from Drive backup: {backup_path}...")
        ram_cache = torch.load(backup_path, map_location='cpu')
        
    missing_dirs = [d for d in ram_dirs if int(os.path.basename(d)) not in ram_cache]
    
    if missing_dirs:
        print(f"Computing {len(missing_dirs)} missing RAM-tier samples...")
        vggt.eval()
        for d in tqdm(missing_dirs, desc="RAM-tier preload"):
            sample_id = int(os.path.basename(d))
            meta_path = os.path.join(d, "metadata.json")
            mesh_path = os.path.join(d, "mesh.ply")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            view_A_paths = glob.glob(os.path.join(d, "*view_A*.png"))
            gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
            
            with torch.no_grad():
                vggt_out = vggt.forward_with_features(gt_rgb_A)
            mask_148_A = get_letter_mask(mesh_path, meta, device=device)
            mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device=device)
            
            # Explicitly force to CPU and fp16
            ram_cache[sample_id] = {
                "patch_tokens": vggt_out["patch_tokens"].cpu().half(),
                "base_depth": vggt_out["depth"].cpu().half(),
                "mask_148_A": mask_148_A.cpu().half(),
                "mask_148_B": mask_148_B.cpu().half(),
            }
        
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        torch.save(ram_cache, backup_path)
        print(f"Saved updated global RAM-tier backup to {backup_path}")
        
    return ram_cache

class TypoSplatDataset(Dataset):
    def __init__(self, sample_dirs, diagnostic_df, ram_cache, vggt, device):
        self.sample_dirs = sample_dirs
        self.diagnostic_df = diagnostic_df
        self.ram_cache = ram_cache
        self.vggt = vggt
        self.device = device

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        sample_id = int(os.path.basename(sample_dir))
        tier = get_cache_tier(sample_id)
        
        target_opt_scale = float(self.diagnostic_df.loc[sample_id, "Opt_Scale"])
        target_opt_shift = float(self.diagnostic_df.loc[sample_id, "Opt_Shift"])

        meta_path = os.path.join(sample_dir, "metadata.json")
        mesh_path = os.path.join(sample_dir, "mesh.ply")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        view_A_paths = glob.glob(os.path.join(sample_dir, "*view_A*.png"))
        view_B_paths = glob.glob(os.path.join(sample_dir, "*view_B*.png"))
        depth_A_paths = glob.glob(os.path.join(sample_dir, "*depth_A*.exr"))
        
        if tier == "ram":
            cached_data = self.ram_cache[sample_id]
        else:
            cache_path = os.path.join(sample_dir, "cached_features.pt")
            try:
                if not os.path.exists(cache_path):
                    raise FileNotFoundError("Cache missing.")
                cached_data = torch.load(cache_path, map_location='cpu')
                if not all(k in cached_data for k in ("patch_tokens", "base_depth", "mask_148_A", "mask_148_B")):
                    raise ValueError("Incomplete cache data.")
            except Exception:
                # On-the-fly self-healing
                gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    vggt_out = self.vggt.forward_with_features(gt_rgb_A)
                mask_148_A = get_letter_mask(mesh_path, meta, device=self.device)
                mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device=self.device)
                
                cached_data = {
                    "patch_tokens": vggt_out["patch_tokens"].cpu().half(),
                    "base_depth": vggt_out["depth"].cpu().half(),
                    "mask_148_A": mask_148_A.cpu().half(),
                    "mask_148_B": mask_148_B.cpu().half()
                }
                try:
                    torch.save(cached_data, cache_path)
                except Exception:
                    pass
        
        # Upcast to FP32 and transfer to target GPU device
        patch_tokens = cached_data["patch_tokens"].float().to(self.device)
        base_depth = cached_data["base_depth"].float().to(self.device)
        mask_148_A = cached_data["mask_148_A"].float().to(self.device)
        mask_148_B = cached_data["mask_148_B"].float().to(self.device)

        mask_518_A = torch.nn.functional.interpolate(mask_148_A, size=(518, 518), mode='nearest')
        mask_518_B = torch.nn.functional.interpolate(mask_148_B, size=(518, 518), mode='nearest')

        gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(self.device)
        gt_depth_518_A = load_exr_depth(depth_A_paths[0], self.device)
        gt_depth_148_A = torch.nn.functional.interpolate(gt_depth_518_A, size=(148, 148), mode='nearest')
        gt_rgb_B = transforms.ToTensor()(Image.open(view_B_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(self.device)

        intrinsics_tuple_A = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        scale_148 = 148.0 / 518.0
        intrinsics_dict_148_A = {
            "fx": meta["fx"] * scale_148, "fy": meta["fy"] * scale_148, 
            "cx": meta["cx"] * scale_148, "cy": meta["cy"] * scale_148
        }
        Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32, device=self.device)
        viewmats_A = torch.eye(4, device=self.device).unsqueeze(0)

        meta_B = meta["camera_B"]
        Ks_B = torch.tensor([[[meta_B["fx"], 0, meta_B["cx"]], [0, meta_B["fy"], meta_B["cy"]], [0, 0, 1]]], dtype=torch.float32, device=self.device)
        viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], self.device)

        return {
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
            "patch_tokens": patch_tokens,
            "base_depth": base_depth,
            "target_opt_scale": target_opt_scale,
            "target_opt_shift": target_opt_shift
        }

# ==========================================
# 2. Helpers (Flatten & Evaluate)
# ==========================================

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

def evaluate(dataloader, upsampler, decoder, device):
    upsampler.eval()
    decoder.eval()
    scale_errors, shift_errors, per_sample_results = [], [], []
    
    with torch.no_grad():
        for batch in dataloader:
            for data in batch:
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

# ==========================================
# 3. Main Training Script
# ==========================================

def main():
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="TypoSplat Stage 1 Full Training")
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Parent directory containing train sample folders")
    parser.add_argument("--diag_csv", type=str, default="/content/master_diagnostics.csv", help="Path to single master diagnostic CSV")
    parser.add_argument("--eval_data_dir", type=str, default=None, help="Directory containing separate eval sample folders (REQUIRED for true validation)")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/TypoSplat/stage1_checkpoints", help="Persistent storage directory on Google Drive")
    parser.add_argument("--disk_backup_dir", type=str, default="/content/drive/MyDrive/TypoSplat/disk_cache_backup", help="Path to disk cache backup on Drive")
    parser.add_argument("--ram_backup_path", type=str, default="/content/drive/MyDrive/TypoSplat/ram_cache_backup.pt", help="Path to single bulk RAM cache backup on Drive")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== TypoSplat: Stage 1 Training ===")
    print(f"Checkpoints and Logs routing to: {args.checkpoint_dir}\n")

    # --- Load Master Diagnostic CSV ---
    try:
        master_diag_df = pd.read_csv(args.diag_csv).set_index("Sample")
    except FileNotFoundError:
        print(f"ERROR: {args.diag_csv} not found!")
        sys.exit(1)

    vggt = VGGTWrapper().to(device)
    for param in vggt.parameters():
        param.requires_grad = False
    vggt.eval()

    # --- Discover and Filter Train Directories ---
    raw_train_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                      if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    train_dirs = [d for d in raw_train_dirs if int(os.path.basename(d)) in master_diag_df.index]

    # --- Discover and Filter Eval Directories ---
    if args.eval_data_dir and args.eval_data_dir != args.data_dir:
        raw_eval_dirs = [os.path.join(args.eval_data_dir, d) for d in os.listdir(args.eval_data_dir) 
                         if os.path.isdir(os.path.join(args.eval_data_dir, d)) and os.path.exists(os.path.join(args.eval_data_dir, d, "metadata.json"))]
        eval_dirs = [d for d in raw_eval_dirs if int(os.path.basename(d)) in master_diag_df.index]
    else:
        # Fallback to random slice if no separate eval folder provided
        random.shuffle(train_dirs)
        eval_size = 350
        eval_dirs = train_dirs[-eval_size:]
        train_dirs = train_dirs[:-eval_size]

    # --- Disjointness Safety Assertion & Sanity Check ---
    train_ids = set(int(os.path.basename(d)) for d in train_dirs)
    eval_ids = set(int(os.path.basename(d)) for d in eval_dirs)
    collision = train_ids & eval_ids
    assert not collision, f"CRITICAL ERROR: {len(collision)} sample IDs overlap between train and eval pools!"
    
    print(f"[CHECK] Train pool: {len(train_dirs)} samples | Eval pool: {len(eval_dirs)} samples (expected 350)")
    if len(eval_dirs) != 350:
        print(f"[WARNING] Eval pool size ({len(eval_dirs)}) does not match expected 350 — check --eval_data_dir.")

    all_sample_dirs = train_dirs + eval_dirs

    # --- Disk Tier Restore ---
    restore_disk_tier_from_drive(all_sample_dirs, args.disk_backup_dir)

    # --- Global CPU RAM Cache Preload ---
    global_ram_cache = build_or_load_ram_cache(all_sample_dirs, vggt, device, args.ram_backup_path)

    # --- Construct Datasets ---
    train_dataset = TypoSplatDataset(train_dirs, master_diag_df, global_ram_cache, vggt, device)
    eval_dataset = TypoSplatDataset(eval_dirs, master_diag_df, global_ram_cache, vggt, device)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)

    fixed_train_eval_dirs = train_dirs[:50]
    fixed_train_eval_dataset = TypoSplatDataset(fixed_train_eval_dirs, master_diag_df, global_ram_cache, vggt, device)
    fixed_train_eval_dataloader = DataLoader(fixed_train_eval_dataset, batch_size=16, shuffle=False, collate_fn=lambda x: x)
    
    fixed_vis_dirs = train_dirs[:2]
    fixed_vis_dataset = TypoSplatDataset(fixed_vis_dirs, master_diag_df, global_ram_cache, vggt, device)
    fixed_vis_dataloader = DataLoader(fixed_vis_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    lpips_fn = ShallowPerceptualLoss(device)

    calibrator_params = list(decoder.calibrator.parameters())
    base_params = list(upsampler.parameters()) + [p for n, p in decoder.named_parameters() if 'calibrator' not in n]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': calibrator_params, 'lr': 1e-5}
    ])

    num_epochs = 35
    iters_per_epoch = len(train_dataset) // args.batch_size
    bootstrap_iters = int(iters_per_epoch * 4.0)
    anneal_iters = int(iters_per_epoch * 4.0)

    start_epoch = 1
    global_iter = 0
    val_err_history = []
    
    existing_checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_epoch_*.pt"))
    if existing_checkpoints:
        latest_ckpt = max(existing_checkpoints, key=lambda p: int(p.split('_epoch_')[-1].split('.pt')[0]))
        print(f"\n[INFO] Found existing checkpoint: {latest_ckpt}")
        print("Restoring weights and resuming training...")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        upsampler.load_state_dict(ckpt_data['upsampler'])
        decoder.load_state_dict(ckpt_data['decoder'])
        optimizer.load_state_dict(ckpt_data['optimizer'])
        start_epoch = ckpt_data['epoch'] + 1
        global_iter = ckpt_data['global_iter']
        print(f"Resuming from Epoch {start_epoch}, Global Iteration {global_iter}\n")

    print("\n=======================================================")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Iters per Epoch: {iters_per_epoch}")
    print(f"Total Epochs:    {num_epochs} (Starting at {start_epoch})")
    print(f"Bootstrap Iters: {bootstrap_iters}")
    print(f"Anneal Iters:    {anneal_iters}")
    print("=======================================================\n")

    upsampler.train()
    decoder.train()

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_total_loss = 0.0
        
        # Initialize component loss tracking for this epoch
        epoch_component_losses = {k: 0.0 for k in [
            "rgb", "edge", "lpips", "depth", "extrusion", "aniso", 
            "normal", "novel_view", "centroid", "zreg", "opacity_sparsity", "calib_reg"
        ]}
        
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            global_iter = (epoch - 1) * iters_per_epoch + batch_idx

            if global_iter < 1000:
                calib_lr = 1e-7 + (global_iter / 1000.0) * (1e-5 - 1e-7)
            else:
                calib_lr = 1e-5
            optimizer.param_groups[1]['lr'] = calib_lr

            optimizer.zero_grad()
            batch_total_loss = 0.0

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
                    means, quats, scales, opacities, colors, data["viewmats_B"], data["Ks_B"], data["gt_rgb_B"], data["mask_518_B"], lpips_fn, iteration=global_iter, warmup_iters=bootstrap_iters
                )

                loss_novel_view = loss_rgb_B + loss_edge_B + (0.002 * loss_lpips_B)
                loss_centroid = compute_centroid_loss(means, data["viewmats_B"], data["Ks_B"], data["mask_518_B"], device)
                loss_zreg = compute_zoffset_regularization(params_1, params_2)
                loss_opacity_sparsity = compute_opacity_sparsity_loss(opacities)

                centroid_weight = max(0.0, 1.0 - global_iter / bootstrap_iters)
                calib_reg_weight = max(0.2, 1.0 - global_iter / anneal_iters)

                loss_calib_target = compute_calibrator_regression_loss(
                    calib_scale, calib_shift, 
                    torch.tensor(data["target_opt_scale"], device=device), 
                    torch.tensor(data["target_opt_shift"], device=device)
                )

                # Capture weighted loss components for tracking
                weighted_terms = {
                    "rgb": 1.0 * loss_rgb.item(),
                    "edge": 1.0 * loss_edge.item(),
                    "lpips": 0.002 * loss_lpips.item(),
                    "depth": 50.0 * loss_depth.item(),
                    "extrusion": 1000.0 * loss_extrusion.item(),
                    "aniso": 1.0 * loss_aniso.item(),
                    "normal": 1.0 * loss_normal.item(),
                    "novel_view": 0.5 * loss_novel_view.item(),
                    "centroid": centroid_weight * 0.05 * loss_centroid.item(),
                    "zreg": 0.05 * loss_zreg.item(),
                    "opacity_sparsity": 1.0 * loss_opacity_sparsity.item(),
                    "calib_reg": calib_reg_weight * 2.0 * loss_calib_target.item()
                }
                
                for k, v in weighted_terms.items():
                    epoch_component_losses[k] += v / args.batch_size

                if global_iter == 0 and sample_idx == 0:
                    print("\n=== LOSS WEIGHT AUDIT (iter 0, first sample) ===")
                    total = sum(weighted_terms.values())
                    for name, v in sorted(weighted_terms.items(), key=lambda x: -x[1]):
                        print(f"  {name:20s}: contribution={v:10.4f}  ({v/total:.1%} of total)")
                    print(f"  {'TOTAL':20s}: {total:.4f}")
                    print("=" * 60 + "\n")

                if batch_idx == len(train_dataloader) - 1 and sample_idx == 0:
                    nv_grad = torch.autograd.grad(loss_novel_view, means, retain_graph=True, allow_unused=True)[0]
                    nv_grad_mag = nv_grad.abs().mean().item() if nv_grad is not None else 0.0

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
                        frac_offscreen = (~in_frame).float().mean().item()
                    
                    last_batch_metrics = (nv_grad_mag, frac_offscreen)

                sample_loss = sum(weighted_terms.values())
                
                (sample_loss / args.batch_size).backward()
                batch_total_loss += sample_loss.item() / args.batch_size

            optimizer.step()
            epoch_total_loss += batch_total_loss

        # --- Aggregate and Log Per-Epoch Stats ---
        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        train_log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        write_header = not os.path.exists(train_log_path)
        with open(train_log_path, "a") as f:
            if write_header:
                f.write("epoch,global_iter,avg_epoch_loss,last_batch_nv_grad_mag,last_batch_off_screen_frac\n")
            f.write(f"{epoch},{global_iter},{avg_epoch_loss:.6f},{last_batch_metrics[0]:.10f},{last_batch_metrics[1]:.6f}\n")

        # Log per-component loss averages
        avg_components = {k: v / len(train_dataloader) for k, v in epoch_component_losses.items()}
        comp_log_path = os.path.join(args.checkpoint_dir, "loss_components_log.csv")
        comp_write_header = not os.path.exists(comp_log_path)
        with open(comp_log_path, "a") as f:
            headers = ["rgb", "edge", "lpips", "depth", "extrusion", "aniso", "normal", "novel_view", "centroid", "zreg", "opacity_sparsity", "calib_reg"]
            if comp_write_header:
                f.write("epoch,global_iter," + ",".join(headers) + "\n")
            f.write(f"{epoch},{global_iter}," + ",".join(f"{avg_components[k]:.6f}" for k in headers) + "\n")

        # --- Periodic Visualizations ---
        if epoch % 2 == 0:
            upsampler.eval()
            decoder.eval()
            with torch.no_grad():
                for vis_batch in fixed_vis_dataloader:
                    data = vis_batch[0]
                    sample_name = os.path.basename(data["dir"])
                    
                    up_feat = upsampler(data["patch_tokens"])
                    p_list, _, _, _, attn_w = decoder(up_feat, data["base_depth"], data["patch_tokens"])
                    m, q, s, o, c = flatten_decoder_outputs_camera_space(
                        p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                    )
                    
                    r_colors_A, _, _ = rasterization(
                        means=m, quats=q, scales=s, opacities=o, colors=c,
                        viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518,
                    )
                    
                    _, _, _, r_colors_B = compute_novel_view_loss(
                        m, q, s, o, c, data["viewmats_B"], data["Ks_B"], data["gt_rgb_B"], data["mask_518_B"], lpips_fn, iteration=10000, warmup_iters=1
                    )
                    
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    axes[0,0].imshow(data["gt_rgb_A"][0].permute(1, 2, 0).cpu().numpy() * data["mask_518_A"][0].permute(1, 2, 0).cpu().numpy())
                    axes[0,0].set_title("GT Camera A")
                    axes[0,1].imshow((r_colors_A.permute(0,3,1,2) * data["mask_518_A"])[0].permute(1,2,0).detach().cpu().numpy())
                    axes[0,1].set_title(f"Render A (Epoch {epoch})")
                    axes[1,0].imshow(data["gt_rgb_B"][0].permute(1, 2, 0).cpu().numpy() * data["mask_518_B"][0].permute(1, 2, 0).cpu().numpy())
                    axes[1,0].set_title("GT Camera B")
                    axes[1,1].imshow((r_colors_B.permute(0,3,1,2) * data["mask_518_B"])[0].permute(1,2,0).detach().cpu().numpy())
                    axes[1,1].set_title(f"Render B (Epoch {epoch})")
                    
                    plt.savefig(os.path.join(args.checkpoint_dir, f"{sample_name}_render_ep{epoch}.png"), dpi=150)
                    plt.close(fig)

                    fig_attn, ax_attn = plt.subplots(figsize=(5, 5))
                    im = ax_attn.imshow(attn_w[0, 0].view(37, 37).cpu().numpy(), cmap='viridis')
                    ax_attn.set_title(f"Attention (Epoch {epoch})")
                    fig_attn.colorbar(im, ax=ax_attn)
                    plt.savefig(os.path.join(args.checkpoint_dir, f"{sample_name}_attn_ep{epoch}.png"), dpi=150)
                    plt.close(fig_attn)
                    
            upsampler.train()
            decoder.train()

        # --- Checkpoint Saving ---
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'upsampler': upsampler.state_dict(),
            'decoder': decoder.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_iter': global_iter
        }, checkpoint_path)
        
        # --- Evaluate ---
        val_scale_err, val_shift_err, val_per_sample = evaluate(eval_dataloader, upsampler, decoder, device)
        train_scale_err, train_shift_err, _ = evaluate(fixed_train_eval_dataloader, upsampler, decoder, device)
        
        print(f"\n[EPOCH {epoch} EVAL] Val ScaleErr: {val_scale_err:.4f} | Val ShiftErr: {val_shift_err:.4f}")
        print(f"                 Train ScaleErr: {train_scale_err:.4f} | Train ShiftErr: {train_shift_err:.4f}")

        train_vs_val_csv = os.path.join(args.checkpoint_dir, "train_vs_val_history.csv")
        tv_write_header = not os.path.exists(train_vs_val_csv)
        with open(train_vs_val_csv, "a") as f:
            if tv_write_header:
                f.write("epoch,train_mean_scale_err,train_mean_shift_err,val_mean_scale_err,val_mean_shift_err\n")
            f.write(f"{epoch},{train_scale_err:.6f},{train_shift_err:.6f},{val_scale_err:.6f},{val_shift_err:.6f}\n")

        # Log ALL validation samples (no longer capping at [:10])
        per_sample_csv = os.path.join(args.checkpoint_dir, "eval_per_sample_history.csv")
        ps_write_header = not os.path.exists(per_sample_csv)
        with open(per_sample_csv, "a") as f:
            if ps_write_header:
                f.write("epoch,sample,pred_scale,target_scale,pred_shift,target_shift,attn_max,attn_std\n")
            for r in val_per_sample:
                f.write(f"{epoch},{r['sample']},{r['pred_scale']:.6f},{r['target_scale']:.6f},{r['pred_shift']:.6f},{r['target_shift']:.6f},{r['attn_max']:.6f},{r['attn_std']:.6f}\n")

        mean_attn_max = sum(r["attn_max"] for r in val_per_sample) / len(val_per_sample)
        eval_csv = os.path.join(args.checkpoint_dir, "eval_history.csv")
        eval_write_header = not os.path.exists(eval_csv)
        with open(eval_csv, "a") as f:
            if eval_write_header:
                f.write("epoch,mean_scale_err,mean_shift_err,mean_attn_max\n")
            f.write(f"{epoch},{val_scale_err:.6f},{val_shift_err:.6f},{mean_attn_max:.6f}\n")

        val_err_history.append(val_scale_err)
        if len(val_err_history) >= 5:
            recent_min = min(val_err_history[-5:-1])
            if val_scale_err >= recent_min:
                print(f"[!] Warning: Val Scale Error ({val_scale_err:.4f}) has not improved over the last 5 epochs (Best recent: {recent_min:.4f}).")

        print(f"[!] End of Epoch {epoch} -> Checkpoint saved to Drive.\n")

    print(f"\n[SUCCESS] Stage 1 Training Complete ({num_epochs} Epochs)!")

if __name__ == "__main__":
    main()