"""
TypoSplat Stage 1.5: Color Fine-Tuning (Combined Fixes)
=======================================================
Short diagnostic run (5 epochs) incorporating all validated fixes:
1. Leaky clamp activation for sh_dc (color) to prevent dead-gradient zones.
2. Moderated photometric weights (RGB=4.0, NovelView=3.0) to balance color vs geometry.
3. Minimum scale hinge penalty to cap geometric shrinkage.
Calibrator remains frozen, bootstrap geometric losses dropped.
"""

import os
import sys
import glob
import json
import shutil
import random
import argparse
import uuid
import concurrent.futures
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

from skimage.metrics import peak_signal_noise_ratio as psnr_metric, structural_similarity as ssim_metric

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
    compute_novel_view_loss,
    compute_min_scale_loss,
    _get_relative_viewmat
)
from src.data.mask_generator import get_letter_mask
from src.utils.cache_tier import get_cache_tier
from gsplat import rasterization

# ==========================================
# 0. Tunable Loss Weights (Constants)
# ==========================================
WEIGHT_RGB = 4.0           
WEIGHT_EDGE = 2.0          
WEIGHT_DEPTH = 20.0        
WEIGHT_EXTRUSION = 200.0   
WEIGHT_NORMAL = 1.0        
WEIGHT_LPIPS = 0.1         
WEIGHT_NOVEL_VIEW = 3.0    
WEIGHT_MIN_SCALE = 500.0   

# Physically grounded threshold (2 * mean(depth/fx) across val set)
TAU_SIZE = 0.006915 

# ==========================================
# 1. Dataset & Two-Tier Cache Helpers
# ==========================================

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

def restore_disk_tier_from_drive(sample_dirs, drive_backup_root):
    if not os.path.exists(drive_backup_root):
        return

    def _restore_single(sample_dir):
        cache_path = os.path.join(sample_dir, "cached_features.pt")
        if os.path.exists(cache_path):
            return 0
        sample_id = os.path.basename(sample_dir)
        src = os.path.join(drive_backup_root, f"{sample_id}.pt")
        if os.path.exists(src):
            shutil.copy2(src, cache_path)
            return 1
        return 0

    restored = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_restore_single, d): d for d in sample_dirs}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sample_dirs), desc="Restoring Disk Cache"):
            try:
                restored += future.result()
            except Exception:
                pass

    if restored:
        print(f"[INFO] Restored {restored} disk-tier cache files from Drive backup.")

def build_or_load_ram_cache(all_sample_dirs, vggt, device, backup_dir, chunk_size=500):
    ram_dirs = [d for d in all_sample_dirs if get_cache_tier(int(os.path.basename(d))) == "ram"]
    ram_cache = {}

    if os.path.exists(backup_dir) and os.path.isdir(backup_dir):
        chunk_files = glob.glob(os.path.join(backup_dir, "*.pt"))
        if chunk_files:
            print(f"Loading {len(chunk_files)} RAM-tier chunks from Drive backup...")
            for cf in tqdm(chunk_files, desc="Loading RAM chunks"):
                try:
                    chunk_data = torch.load(cf, map_location='cpu')
                    ram_cache.update(chunk_data)
                except Exception as e:
                    print(f"Warning: Failed to load chunk {cf} ({e}). It will be recomputed.")

    missing_dirs = [d for d in ram_dirs if int(os.path.basename(d)) not in ram_cache]

    if missing_dirs:
        print(f"Computing {len(missing_dirs)} missing RAM-tier samples in chunks of {chunk_size}...")
        vggt.eval()
        os.makedirs(backup_dir, exist_ok=True)

        for i in range(0, len(missing_dirs), chunk_size):
            chunk_dirs = missing_dirs[i:i + chunk_size]
            chunk_dict = {}
            chunk_idx = (i // chunk_size) + 1
            total_chunks = (len(missing_dirs) + chunk_size - 1) // chunk_size

            for d in tqdm(chunk_dirs, desc=f"RAM-tier preload (Chunk {chunk_idx}/{total_chunks})"):
                sample_id = int(os.path.basename(d))
                meta_path = os.path.join(d, "metadata.json")
                mesh_path = os.path.join(d, "mesh.ply")
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                view_A_paths = glob.glob(os.path.join(d, "*view_A*.png"))
                gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)

                with torch.no_grad():
                    vggt_out = vggt.forward_with_features(gt_rgb_A)
                
                mask_148_A = get_letter_mask(mesh_path, meta, device='cpu')
                mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device='cpu')

                chunk_dict[sample_id] = {
                    "patch_tokens": vggt_out["patch_tokens"].cpu().half(),
                    "base_depth": vggt_out["depth"].cpu().half(),
                    "mask_148_A": mask_148_A.half(),
                    "mask_148_B": mask_148_B.half(),
                }

            chunk_id = uuid.uuid4().hex[:8]
            chunk_path = os.path.join(backup_dir, f"ram_chunk_{chunk_id}.pt")
            torch.save(chunk_dict, chunk_path)
            ram_cache.update(chunk_dict)

    return ram_cache

def identity_collate(batch):
    return batch

class TypoSplatDatasetCPU(Dataset):
    def __init__(self, sample_dirs, diagnostic_df, ram_cache, vggt):
        self.sample_dirs = sample_dirs
        self.diagnostic_df = diagnostic_df
        self.ram_cache = ram_cache
        self.vggt = vggt

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
            except Exception as e:
                print(f"\n[WARN] Self-heal triggered for {sample_id} in worker process (Cache missing/corrupt).")
                print(f"       WARNING: Fallback uses CPU PyTorch3D rasterization which is extremely slow.")
                print(f"       VGGT CUDA context may fail due to fork(). Error: {e}\n")
                
                gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)
                
                try:
                    with torch.no_grad():
                        vggt_out = self.vggt.forward_with_features(gt_rgb_A)
                except:
                    vggt_out = {"patch_tokens": torch.zeros((1, 2048, 37, 37)), "depth": torch.zeros((1, 1, 518, 518))}
                
                mask_148_A = get_letter_mask(mesh_path, meta, device='cpu')
                mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device='cpu')
                
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
        
        patch_tokens = cached_data["patch_tokens"].float()
        base_depth = cached_data["base_depth"].float()
        mask_148_A = cached_data["mask_148_A"].float()
        mask_148_B = cached_data["mask_148_B"].float()

        mask_518_A = torch.nn.functional.interpolate(mask_148_A, size=(518, 518), mode='nearest')
        mask_518_B = torch.nn.functional.interpolate(mask_148_B, size=(518, 518), mode='nearest')

        gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)
        gt_depth_518_A = load_exr_depth_cpu(depth_A_paths[0])
        gt_depth_148_A = torch.nn.functional.interpolate(gt_depth_518_A, size=(148, 148), mode='nearest')
        gt_rgb_B = transforms.ToTensor()(Image.open(view_B_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)

        intrinsics_tuple_A = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        scale_148 = 148.0 / 518.0
        intrinsics_dict_148_A = {
            "fx": meta["fx"] * scale_148, "fy": meta["fy"] * scale_148, 
            "cx": meta["cx"] * scale_148, "cy": meta["cy"] * scale_148
        }
        Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32)
        viewmats_A = torch.eye(4).unsqueeze(0)

        meta_B = meta["camera_B"]
        Ks_B = torch.tensor([[[meta_B["fx"], 0, meta_B["cx"]], [0, meta_B["fy"], meta_B["cy"]], [0, 0, 1]]], dtype=torch.float32)
        viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], 'cpu')

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
# 2. Helpers (Pre-Masked LPIPS & Flatten)
# ==========================================

class PreMaskedLPIPS(torch.nn.Module):
    def __init__(self, base_lpips):
        super().__init__()
        self.base_lpips = base_lpips

    def forward(self, pred, gt, mask=None):
        if mask is not None:
            pred = pred * mask
            gt = gt * mask
        return self.base_lpips(pred, gt, mask=None)

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
        
        # [FIX]: Replaced widened hard-clamp with a leaky clamp (slope 0.05).
        # Reason: Direct measurement on the Stage 1 checkpoint showed the previous clamp-and-rescale
        # activation ([-0.2, 1.2]) still left ~78-79% of raw sh_dc values outside its bounds 
        # (measured range: min -4.86, max 2.69). This leaky-clamp guarantees a nonzero gradient 
        # (slope 0.05) everywhere, so no weights are dead, while preserving full gradient inside [0,1].
        # 0.05 is chosen over 0.01 to pull extreme outliers back within a short diagnostic run.
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

def compute_psnr_ssim(dataloader, upsampler, decoder, device, n_samples):
    upsampler.eval()
    decoder.eval()
    
    psnr_list = []
    ssim_list = []
    samples_evaluated = 0
    
    with torch.no_grad():
        for batch in dataloader:
            for data_cpu in batch:
                if samples_evaluated >= n_samples:
                    break
                
                data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data_cpu.items()}
                
                with torch.autocast('cuda', dtype=torch.float16):
                    up_feat = upsampler(data["patch_tokens"])
                    p_list, _, _, _, _ = decoder(up_feat, data["base_depth"], data["patch_tokens"])
                    m, q, s, o, c = flatten_decoder_outputs_camera_space(
                        p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                    )
                    
                    r_colors_B, _, _ = rasterization(
                        means=m.float(), quats=q.float(), scales=s.float(), opacities=o.float(), colors=c.float(),
                        viewmats=data["viewmats_B"], Ks=data["Ks_B"], width=518, height=518,
                    )
                
                pred_rgb = r_colors_B.permute(0, 3, 1, 2)[0].permute(1, 2, 0)
                gt_rgb = data["gt_rgb_B"][0].permute(1, 2, 0)
                
                pred_np = (pred_rgb.detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                gt_np = (gt_rgb.detach().float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                mask_np = data["mask_518_B"][0,0].detach().cpu().numpy().astype(bool)

                pred_masked_pixels = pred_np[mask_np]
                gt_masked_pixels = gt_np[mask_np]
                psnr_val = psnr_metric(gt_masked_pixels, pred_masked_pixels, data_range=255)
                
                ys, xs = np.where(mask_np)
                if len(ys) > 0 and len(xs) > 0 and (ys.max() - ys.min() + 1) >= 7 and (xs.max() - xs.min() + 1) >= 7:
                    y0, y1, x0, x1 = ys.min(), ys.max()+1, xs.min(), xs.max()+1
                    pred_crop = pred_np[y0:y1, x0:x1]
                    gt_crop = gt_np[y0:y1, x0:x1]
                    ssim_val = ssim_metric(gt_crop, pred_crop, channel_axis=-1, data_range=255)
                else:
                    ssim_val = 0.0 
                
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)
                samples_evaluated += 1
                
            if samples_evaluated >= n_samples:
                break
                
    upsampler.train()
    decoder.train()
    
    return np.mean(psnr_list), np.mean(ssim_list)

# ==========================================
# 3. Main Fine-Tuning Script
# ==========================================

def main():
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="TypoSplat Stage 1.5 Color Fine-Tuning")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best Stage 1 checkpoint (.pt) to load")
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Train dataset")
    parser.add_argument("--diag_csv", type=str, default="/content/master_diagnostics.csv", help="Master diagnostic CSV")
    parser.add_argument("--eval_data_dir", type=str, default=None, help="Eval dataset")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/TypoSplat/stage1_5_leaky_clamp_fix", help="Output dir")
    parser.add_argument("--disk_backup_dir", type=str, default="/content/drive/MyDrive/TypoSplat/disk_cache_backup", help="Disk cache backup")
    parser.add_argument("--ram_backup_dir", type=str, default="/content/drive/MyDrive/TypoSplat/ram_cache_backup", help="Chunked RAM cache backup")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=5, help="Total training epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Multiprocessing workers for DataLoader")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    psnr_log_path = os.path.join(args.checkpoint_dir, "psnr_ssim_log.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== TypoSplat: Stage 1.5 Color Fine-Tuning ===")
    print(f"Checkpoints and Logs routing to: {args.checkpoint_dir}\n")

    try:
        master_diag_df = pd.read_csv(args.diag_csv).set_index("Sample")
    except FileNotFoundError:
        print(f"ERROR: {args.diag_csv} not found!")
        sys.exit(1)

    vggt = VGGTWrapper().to(device)
    for param in vggt.parameters():
        param.requires_grad = False
    vggt.eval()

    raw_train_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                      if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    train_dirs = [d for d in raw_train_dirs if int(os.path.basename(d)) in master_diag_df.index]

    if args.eval_data_dir and args.eval_data_dir != args.data_dir:
        raw_eval_dirs = [os.path.join(args.eval_data_dir, d) for d in os.listdir(args.eval_data_dir) 
                         if os.path.isdir(os.path.join(args.eval_data_dir, d)) and os.path.exists(os.path.join(args.eval_data_dir, d, "metadata.json"))]
        eval_dirs = [d for d in raw_eval_dirs if int(os.path.basename(d)) in master_diag_df.index]
    else:
        random.shuffle(train_dirs)
        eval_size = 350
        eval_dirs = train_dirs[-eval_size:]
        train_dirs = train_dirs[:-eval_size]

    train_ids = set(int(os.path.basename(d)) for d in train_dirs)
    eval_ids = set(int(os.path.basename(d)) for d in eval_dirs)
    collision = train_ids & eval_ids
    assert not collision, f"CRITICAL ERROR: {len(collision)} sample IDs overlap between train and eval pools!"
    
    print(f"[CHECK] Train pool: {len(train_dirs)} samples | Eval pool: {len(eval_dirs)} samples")
    all_sample_dirs = train_dirs + eval_dirs

    restore_disk_tier_from_drive(all_sample_dirs, args.disk_backup_dir)
    global_ram_cache = build_or_load_ram_cache(all_sample_dirs, vggt, device, args.ram_backup_dir)

    train_dataset = TypoSplatDatasetCPU(train_dirs, master_diag_df, global_ram_cache, vggt)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, collate_fn=identity_collate)

    eval_dataset = TypoSplatDatasetCPU(eval_dirs, master_diag_df, global_ram_cache, vggt)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, collate_fn=identity_collate)

    fixed_vis_dirs = train_dirs[:2]
    fixed_vis_dataset = TypoSplatDatasetCPU(fixed_vis_dirs, master_diag_df, global_ram_cache, vggt)
    fixed_vis_dataloader = DataLoader(fixed_vis_dataset, batch_size=1, shuffle=False, collate_fn=identity_collate)

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    
    print(f"Loading checkpoint weights from: {args.checkpoint}")
    ckpt_data = torch.load(args.checkpoint, map_location=device)
    upsampler.load_state_dict(ckpt_data['upsampler'])
    decoder.load_state_dict(ckpt_data['decoder'])

    for param in decoder.calibrator.parameters():
        param.requires_grad = False
    print("Frozen decoder.calibrator parameters to protect converged geometry scale/shift.")

    base_lpips = ShallowPerceptualLoss(device)
    lpips_fn = PreMaskedLPIPS(base_lpips)

    base_params = list(upsampler.parameters()) + [p for n, p in decoder.named_parameters() if 'calibrator' not in n]
    optimizer = optim.Adam([{'params': base_params, 'lr': 1e-4}])
    scaler = torch.cuda.amp.GradScaler()

    num_epochs = args.num_epochs
    iters_per_epoch = len(train_dataset) // args.batch_size

    print("\n=======================================================")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Iters per Epoch: {iters_per_epoch}")
    print(f"Total Epochs:    {num_epochs}")
    print(f"Workers:         {args.num_workers}")
    print("=======================================================\n")

    print("\n--- Computing Baseline PSNR/SSIM (Epoch 0, Camera B) ---")
    base_psnr, base_ssim = compute_psnr_ssim(eval_dataloader, upsampler, decoder, device, n_samples=100)
    print(f"[EPOCH 0] PSNR: {base_psnr:.2f} | SSIM: {base_ssim:.4f} (n=100)")
    
    psnr_write_header = not os.path.exists(psnr_log_path)
    with open(psnr_log_path, "a") as f:
        if psnr_write_header:
            f.write("epoch,n_samples,psnr,ssim\n")
        f.write(f"0,100,{base_psnr:.6f},{base_ssim:.6f}\n")

    upsampler.train()
    decoder.train()

    for epoch in range(1, num_epochs + 1):
        epoch_total_loss = 0.0
        
        epoch_component_losses = {k: 0.0 for k in ["rgb", "edge", "lpips", "depth", "extrusion", "normal", "novel_view", "min_scale"]}

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            global_iter = (epoch - 1) * iters_per_epoch + batch_idx
            optimizer.zero_grad()
            batch_total_loss = 0.0

            for sample_idx, data_cpu in enumerate(batch):
                data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data_cpu.items()}
                
                with torch.autocast('cuda', dtype=torch.float16):
                    upsampled_features = upsampler(data["patch_tokens"])
                    
                    params_list, calib_scale, calib_shift, _, attn_weights = decoder(upsampled_features, data["base_depth"], data["patch_tokens"])
                    params_0, params_1, params_2 = params_list

                    means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
                        params_0, params_1, params_2, data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                    )

                    loss_min_scale = compute_min_scale_loss(scales, tau_size=TAU_SIZE)

                    render_colors_A, _, _ = rasterization(
                        means=means.float(), quats=quats.float(), scales=scales.float(), opacities=opacities.float(), colors=colors.float(),
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
                    loss_normal = compute_normal_loss(layer_1_depth, data["gt_depth_148_A"], data["intrinsics_dict_148_A"], data["mask_148_A"])

                    loss_rgb_B, loss_edge_B, _, render_colors_B = compute_novel_view_loss(
                        means.float(), quats.float(), scales.float(), opacities.float(), colors.float(), data["viewmats_B"], data["Ks_B"], data["gt_rgb_B"], data["mask_518_B"], None, iteration=global_iter, warmup_iters=1
                    )
                    
                    pred_rgb_B_raw = render_colors_B.permute(0, 3, 1, 2)
                    loss_lpips_B = lpips_fn(pred_rgb_B_raw, data["gt_rgb_B"], mask=data["mask_518_B"])

                    loss_novel_view_base = loss_rgb_B + loss_edge_B 

                    sample_loss_tensor = (
                        WEIGHT_RGB * loss_rgb +
                        WEIGHT_EDGE * loss_edge +
                        WEIGHT_LPIPS * loss_lpips +
                        WEIGHT_DEPTH * loss_depth +
                        WEIGHT_EXTRUSION * loss_extrusion +
                        WEIGHT_NORMAL * loss_normal +
                        WEIGHT_NOVEL_VIEW * loss_novel_view_base +
                        WEIGHT_LPIPS * loss_lpips_B +
                        WEIGHT_MIN_SCALE * loss_min_scale
                    )
                
                scaler.scale(sample_loss_tensor / args.batch_size).backward()
                batch_total_loss += sample_loss_tensor.item() / args.batch_size

                weighted_terms = {
                    "rgb": WEIGHT_RGB * loss_rgb.item(),
                    "edge": WEIGHT_EDGE * loss_edge.item(),
                    "lpips": WEIGHT_LPIPS * (loss_lpips.item() + loss_lpips_B.item()),
                    "depth": WEIGHT_DEPTH * loss_depth.item(),
                    "extrusion": WEIGHT_EXTRUSION * loss_extrusion.item(),
                    "normal": WEIGHT_NORMAL * loss_normal.item(),
                    "novel_view": WEIGHT_NOVEL_VIEW * loss_novel_view_base.item(),
                    "min_scale": WEIGHT_MIN_SCALE * loss_min_scale.item(),
                }
                
                for k, v in weighted_terms.items():
                    epoch_component_losses[k] += v / args.batch_size

                if global_iter == 0 and sample_idx == 0:
                    print("\n=== STAGE 1.5 LOSS WEIGHT AUDIT (iter 0, first sample) ===")
                    total = sum(weighted_terms.values())
                    for name, v in sorted(weighted_terms.items(), key=lambda x: -x[1]):
                        print(f"  {name:20s}: contribution={v:10.4f}  ({v/total:.1%} of total)")
                    print(f"  {'TOTAL':20s}: {total:.4f}")
                    print("=" * 60 + "\n")

            scaler.step(optimizer)
            scaler.update()
            epoch_total_loss += batch_total_loss

        avg_epoch_loss = epoch_total_loss / len(train_dataloader)
        
        avg_components = {k: v / len(train_dataloader) for k, v in epoch_component_losses.items()}
        comp_log_path = os.path.join(args.checkpoint_dir, "loss_components_log.csv")
        comp_write_header = not os.path.exists(comp_log_path)
        with open(comp_log_path, "a") as f:
            headers = ["rgb", "edge", "lpips", "depth", "extrusion", "normal", "novel_view", "min_scale"]
            if comp_write_header:
                f.write("epoch,global_iter,avg_epoch_loss," + ",".join(headers) + "\n")
            f.write(f"{epoch},{global_iter},{avg_epoch_loss:.6f}," + ",".join(f"{avg_components[k]:.6f}" for k in headers) + "\n")
            print(f"[EPOCH {epoch}] Avg Loss: {avg_epoch_loss:.4f} | RGB: {avg_components['rgb']:.4f} | NV: {avg_components['novel_view']:.4f} | MinScale: {avg_components['min_scale']:.4f}")

        eval_n = 350 # Evaluate all 350 samples every epoch
        print(f"\n--- Computing PSNR/SSIM (Epoch {epoch}, Camera B) ---")
        ep_psnr, ep_ssim = compute_psnr_ssim(eval_dataloader, upsampler, decoder, device, n_samples=eval_n)
        print(f"[EPOCH {epoch}] PSNR: {ep_psnr:.2f} | SSIM: {ep_ssim:.4f} (n={eval_n})")
        
        with open(psnr_log_path, "a") as f:
            f.write(f"{epoch},{eval_n},{ep_psnr:.6f},{ep_ssim:.6f}\n")

        if epoch % 1 == 0:
            upsampler.eval()
            decoder.eval()
            with torch.no_grad():
                for vis_batch in fixed_vis_dataloader:
                    data_cpu = vis_batch[0]
                    data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data_cpu.items()}
                    sample_name = os.path.basename(data["dir"])
                    
                    with torch.autocast('cuda', dtype=torch.float16):
                        up_feat = upsampler(data["patch_tokens"])
                        p_list, _, _, _, _ = decoder(up_feat, data["base_depth"], data["patch_tokens"])
                        m, q, s, o, c = flatten_decoder_outputs_camera_space(
                            p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                        )
                        
                        r_colors_A, _, _ = rasterization(
                            means=m.float(), quats=q.float(), scales=s.float(), opacities=o.float(), colors=c.float(),
                            viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518,
                        )
                        
                        _, _, _, r_colors_B = compute_novel_view_loss(
                            m.float(), q.float(), s.float(), o.float(), c.float(), data["viewmats_B"], data["Ks_B"], data["gt_rgb_B"], data["mask_518_B"], None, iteration=10000, warmup_iters=1
                        )
                    
                    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
                    axes[0,0].imshow(data["gt_rgb_A"][0].permute(1, 2, 0).cpu().numpy() * data["mask_518_A"][0].permute(1, 2, 0).cpu().numpy())
                    axes[0,0].set_title("GT Camera A")
                    axes[0,1].imshow((r_colors_A.permute(0,3,1,2) * data["mask_518_A"])[0].permute(1,2,0).float().cpu().numpy())
                    axes[0,1].set_title(f"Render A (Epoch {epoch})")
                    axes[1,0].imshow(data["gt_rgb_B"][0].permute(1, 2, 0).cpu().numpy() * data["mask_518_B"][0].permute(1, 2, 0).cpu().numpy())
                    axes[1,0].set_title("GT Camera B")
                    axes[1,1].imshow((r_colors_B.permute(0,3,1,2) * data["mask_518_B"])[0].permute(1,2,0).float().cpu().numpy())
                    axes[1,1].set_title(f"Render B (Epoch {epoch})")
                    
                    plt.savefig(os.path.join(args.checkpoint_dir, f"{sample_name}_render_ep{epoch}.png"), dpi=150)
                    plt.close(fig)
                    
            upsampler.train()
            decoder.train()

        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'upsampler': upsampler.state_dict(),
            'decoder': decoder.state_dict(),
            'epoch': epoch,
            'global_iter': global_iter
        }, checkpoint_path)
        
        print(f"[!] End of Epoch {epoch} -> Checkpoint saved to Drive.\n")

    print(f"\n[SUCCESS] Stage 1.5 Color Fine-Tuning Complete ({num_epochs} Epochs)!")

if __name__ == "__main__":
    main()