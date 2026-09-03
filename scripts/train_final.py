"""
TypoSplat: 1400-Sample Capacity Diagnostic Training Script
============================================================
Single data directory (1400 samples). Eval is a fixed, NON-HELD-OUT
subset of 300 samples drawn from that same 1400 (training-set eval,
not a generalization eval) — this directly answers the underfit vs.
overfit diagnostic from the full-dataset run.

- NUM_EPOCHS = 400, single cosine decay over the full horizon
- Loss weights reverted to the originally-validated recipe:
  W_SSIM_A = 0.2, W_SSIM_B = 0.1, W_SCALE_MAG = 1.0
- Checkpointing: best_raw.pt and best_ema.pt tracked independently,
  last.pt always overwritten, full numbered checkpoints every 10 epochs
- VGGT feature cache restored from Drive backup before training starts
- No RAM/disk tiering (not needed at this sample count)
"""

import os
import sys
import glob
import json
import time
import math
import copy
import shutil
import argparse
import concurrent.futures
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from tqdm import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append("/content/TypoSplat")
from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from src.data.mask_generator import get_letter_mask
from src.losses.typ_losses import _get_relative_viewmat
from gsplat import rasterization

# ==========================================
# 0. DRIVE CACHE RESTORE (pre-training step)
# ==========================================
def restore_disk_tier_from_drive(sample_dirs, drive_backup_root):
    """
    Copies cached_features.pt files from a Drive backup folder down to each
    sample's local directory, before training starts. Skips samples that
    already have a local cache. Parallelized since this is pure I/O.
    """
    if not drive_backup_root or not os.path.exists(drive_backup_root):
        print(f"[CACHE RESTORE] Drive backup root not found or not set ({drive_backup_root}) — skipping restore.")
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
    missing = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(_restore_single, d): d for d in sample_dirs}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(sample_dirs), desc="Restoring cache from Drive"):
            try:
                result = future.result()
                if result == 1:
                    restored += 1
            except Exception:
                missing += 1

    print(f"[CACHE RESTORE] Restored {restored} cache files from Drive backup ({drive_backup_root}).")
    if missing:
        print(f"[CACHE RESTORE] WARNING: {missing} samples failed to restore — check errors above.")


# ==========================================
# 1. DATASET & DATALOADER
# ==========================================
class TypoSplatDataset(Dataset):
    def __init__(self, sample_dirs, vggt, device):
        self.sample_dirs = sample_dirs
        self.vggt = vggt
        self.device = device

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        meta = json.load(open(os.path.join(sample_dir, "metadata.json")))
        mesh_path = os.path.join(sample_dir, "mesh.ply")
        cache_path = os.path.join(sample_dir, "cached_features.pt")

        try:
            cached_data = torch.load(cache_path, map_location='cpu', weights_only=True)
            if not all(k in cached_data for k in ("patch_tokens", "base_depth", "mask_148_A", "mask_148_B")):
                raise ValueError("Incomplete cache keys")
        except Exception as e:
            raise RuntimeError(f"Cache missing/corrupt for {sample_dir} ({str(e)}) — pre-repair caches before training, "
                               f"or set num_workers=0 if on-the-fly repair during training is required.")

        patch_tokens = cached_data["patch_tokens"].float()
        base_depth = cached_data["base_depth"].float()
        mask_148_A = cached_data["mask_148_A"].float()
        mask_148_B = cached_data["mask_148_B"].float()
        mask_518_A = F.interpolate(mask_148_A, size=(518, 518), mode='nearest')
        mask_518_B = F.interpolate(mask_148_B, size=(518, 518), mode='nearest')

        view_A_path = glob.glob(os.path.join(sample_dir, "*view_A*.png"))[0]
        view_B_path = glob.glob(os.path.join(sample_dir, "*view_B*.png"))[0]
        gt_rgb_A = transforms.ToTensor()(Image.open(view_A_path).convert("RGB").resize((518, 518))).unsqueeze(0)
        gt_rgb_B = transforms.ToTensor()(Image.open(view_B_path).convert("RGB").resize((518, 518))).unsqueeze(0)

        intrinsics_tuple_A = (meta["fx"], meta["fy"], meta["cx"], meta["cy"])
        Ks_A = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32)
        viewmats_A = torch.eye(4).unsqueeze(0)

        meta_B = meta["camera_B"]
        Ks_B = torch.tensor([[[meta_B["fx"], 0, meta_B["cx"]], [0, meta_B["fy"], meta_B["cy"]], [0, 0, 1]]], dtype=torch.float32)
        viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], 'cpu')

        return {
            "dir": sample_dir, "meta": meta,
            "gt_rgb_A": gt_rgb_A, "gt_rgb_B": gt_rgb_B,
            "mask_518_A": mask_518_A, "mask_518_B": mask_518_B,
            "mask_148_A": mask_148_A,
            "intrinsics_tuple_A": intrinsics_tuple_A,
            "Ks_A": Ks_A, "viewmats_A": viewmats_A,
            "Ks_B": Ks_B, "viewmats_B": viewmats_B,
            "patch_tokens": patch_tokens, "base_depth": base_depth,
        }

def collate_fn(batch):
    return batch

# ==========================================
# 2. COLOR ACTIVATION & FLATTEN OUTPUTS
# ==========================================
GAIN, BIAS, LEAK = 1.0, 0.72, 0.2
def function_preserving_activation(raw):
    x = GAIN * raw + BIAS
    return (LEAK * x + (1.0 - LEAK) * x.clamp(-2.0, 2.0)) / 4.0 + 0.5

def flatten_outputs(params_0, params_1, params_2, intrinsics, device, mask_148=None):
    fx, fy, cx, cy = intrinsics
    scale_factor = 518.0 / 148.0
    y_grid, x_grid = torch.meshgrid(torch.arange(148, device=device, dtype=torch.float32), torch.arange(148, device=device, dtype=torch.float32), indexing='ij')
    flat_mask = mask_148[0, 0].float().view(-1) if mask_148 is not None else None

    all_means, all_quats, all_scales, all_opacities, all_colors = [], [], [], [], []
    for params in [params_0, params_1, params_2]:
        u = (x_grid + params["xy_offset"][0, 0] + 0.5) * scale_factor
        v = (y_grid + params["xy_offset"][0, 1] + 0.5) * scale_factor
        Z = params["true_depth"][0, 0]
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        means = torch.stack([X, Y, Z], dim=-1).view(-1, 3)
        quats = params["rot"][0].permute(1, 2, 0).view(-1, 4)
        raw_sh_dc = params["sh_dc"][0].permute(1, 2, 0).view(-1, 3)

        colors = function_preserving_activation(raw_sh_dc)
        scales = params["scale"][0].permute(1, 2, 0).reshape(-1, 3)
        opacities = params["opacity"][0].view(-1)

        if flat_mask is not None:
            opacities = opacities * flat_mask

        all_means.append(means); all_quats.append(quats); all_scales.append(scales)
        all_opacities.append(opacities); all_colors.append(colors)

    return (torch.cat(all_means, 0), torch.cat(all_quats, 0), torch.cat(all_scales, 0),
            torch.cat(all_opacities, 0), torch.cat(all_colors, 0))

# ==========================================
# 3. DYCHECK SSIM & REGULARIZERS
# ==========================================
def gaussian_window(window_size, sigma):
    coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    return g.view(1, -1) * g.view(-1, 1)

def dycheck_ssim(x, y, mask, window_size=7, sigma=1.5, cov_thresh=0.3):
    C = x.shape[1]
    w = gaussian_window(window_size, sigma).to(x.device).expand(C, 1, window_size, window_size)
    M = mask.float().expand(-1, C, -1, -1)

    def conv(t):
        return F.conv2d(t, w, padding=window_size // 2, groups=C)

    den = conv(M) + 1e-8
    mu_x, mu_y = conv(M * x) / den, conv(M * y) / den
    s_xx = conv(M * x * x) / den - mu_x**2
    s_yy = conv(M * y * y) / den - mu_y**2
    s_xy = conv(M * x * y) / den - mu_x * mu_y

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2*mu_x*mu_y + C1) * (2*s_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (s_xx + s_yy + C2))

    coverage = conv(M)
    valid = (M > 0) & (coverage >= cov_thresh)
    ssim_map_masked = ssim_map[valid]

    if ssim_map_masked.numel() == 0:
        return torch.tensor(0.0).to(x.device), 0.0
    return ssim_map_masked.mean(), valid.sum().float() / (M > 0).sum().float()

def compute_scale_magnitude_penalty(scale, clamp_bound=0.06, target_fraction=0.3):
    target = clamp_bound * target_fraction
    return F.relu(scale - target).mean()

def compute_anisotropy_loss(scale, r_bound=10.0):
    max_scale = scale.max(dim=-1).values
    min_scale = scale.min(dim=-1).values
    return F.relu(max_scale / (min_scale + 1e-6) - r_bound).mean()

# ==========================================
# 4. CONTROL SUITE PRE-FLIGHT
# ==========================================
def run_control_suite(device, eval_dirs):
    print("\n--- Running Control Suite ---")
    sid_dir = eval_dirs[0]
    meta = json.load(open(os.path.join(sid_dir, "metadata.json")))
    gt_B = transforms.ToTensor()(Image.open(glob.glob(os.path.join(sid_dir, "*view_B*.png"))[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
    mask148_B = get_letter_mask(os.path.join(sid_dir, "mesh.ply"), meta["camera_B"], device=device)
    mask518_B = F.interpolate(mask148_B.float(), size=(518, 518), mode='nearest')

    score1, _ = dycheck_ssim(gt_B, gt_B, mask518_B)
    print(f"T1 (GT vs GT)                 : {score1.item():.4f}  (Expected ≈ 1.0)")

    score2, _ = dycheck_ssim(gt_B, torch.zeros_like(gt_B), mask518_B)
    print(f"T2 (GT vs Zeros)              : {score2.item():.4f}  (Expected ≈ 0.0)")

    gt_offset = torch.clamp(gt_B + 0.13, 0.0, 1.0)
    score3, _ = dycheck_ssim(gt_B, gt_offset, mask518_B)
    print(f"T3 (GT vs GT+0.13)            : {score3.item():.4f}  (Expected ≈ 0.93-0.97)")

    gt_shifted = torch.roll(gt_B, shifts=(2, 2), dims=(2, 3))
    score4, _ = dycheck_ssim(gt_B, gt_shifted, mask518_B)
    print(f"T4 (GT vs Shifted 2px)        : {score4.item():.4f}  (Expected: Clear drop)")

    import torchvision.transforms.functional as TF
    gt_blurred = TF.gaussian_blur(gt_B, kernel_size=7, sigma=2.0)
    score5, _ = dycheck_ssim(gt_B, gt_blurred, mask518_B)
    print(f"T5 (GT vs Blurred sig=2)      : {score5.item():.4f}  (Expected: Clear drop)")

    mask_dilated = F.max_pool2d(mask518_B, kernel_size=41, stride=1, padding=20)
    score6a, _ = dycheck_ssim(gt_B, gt_B, mask_dilated)
    print(f"T6a (GT vs GT, 20px dilated)  : {score6a.item():.4f}  (Expected ≈ 1.0, unchanged from T1)")

    gt_bg_white = gt_B.clone()
    gt_bg_white[mask518_B.expand(-1, 3, -1, -1) == 0] = 1.0
    gt_offset_bg_black = gt_offset.clone()
    gt_offset_bg_black[mask518_B.expand(-1, 3, -1, -1) == 0] = 0.0
    score6b, _ = dycheck_ssim(gt_bg_white, gt_offset_bg_black, mask518_B)
    print(f"T6b (GT_bg vs GT+0.13_bg)     : {score6b.item():.4f}  (Expected to match T3 perfectly)")

    if not (score1.item() > 0.99
            and score2.item() < 0.1
            and score4.item() < score1.item() - 0.1
            and score5.item() < score1.item() - 0.05
            and abs(score3.item() - score6b.item()) < 1e-4
            and abs(score1.item() - score6a.item()) < 1e-4):
        print("[!] CONTROL SUITE FAILED. Aborting.")
        sys.exit(1)
    print("Control Suite Passed.\n")

# ==========================================
# 5. EMA & LR SCHEDULE
# ==========================================
def update_ema(ema_model, model, step, start_step, decay):
    if step < start_step:
        return
    d = min(decay, (1 + step - start_step) / (10 + step - start_step))
    with torch.no_grad():
        for ema_p, p in zip(ema_model.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p, alpha=1 - d)

def get_lr(step, total_steps, warmup_steps=1000, peak_lr=2e-4, min_lr=1e-6):
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))

# ==========================================
# 6. EVALUATION LOGIC
# ==========================================
def evaluate(dataloader, upsampler, decoder, vggt, device):
    upsampler.eval()
    decoder.eval()
    vggt.eval()
    decoder.calibrator.eval()

    ssims, psnrs, collapses, scales_all, alphas_all = [], [], 0, [], []

    with torch.no_grad():
        for batch in dataloader:
            for data in batch:
                data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
                up_feat = upsampler(data["patch_tokens"])
                p_list, _, _, _, _ = decoder(up_feat, data["base_depth"], data["patch_tokens"])

                m, q, s, o, c = flatten_outputs(p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data.get("mask_148_A"))
                render_B, alpha_B, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_B"], Ks=data["Ks_B"], width=518, height=518)
                pred_B = render_B[..., :3].permute(0, 3, 1, 2)

                gt_B = data["gt_rgb_B"]
                mask_B = data["mask_518_B"]
                mask_bool_B = mask_B[0, 0].bool()

                ssim_val, _ = dycheck_ssim(pred_B, gt_B, mask_B, window_size=7, cov_thresh=0.3)
                pred_np = (pred_B[0].detach().cpu().numpy() * 255).clip(0,255).astype(np.uint8).transpose(1,2,0)
                gt_np = (gt_B[0].cpu().numpy() * 255).clip(0,255).astype(np.uint8).transpose(1,2,0)
                mask_np = mask_bool_B.cpu().numpy()

                psnr_val = psnr_metric(gt_np[mask_np], pred_np[mask_np], data_range=255) if mask_np.sum() > 0 else 0.0

                ssims.append(ssim_val.item())
                psnrs.append(psnr_val)
                if (1.0 - ssim_val.item()) > 0.95:
                    collapses += 1
                scales_all.append(s.detach().cpu().numpy().flatten())
                alphas_all.append(alpha_B[0][mask_bool_B].mean().item() if mask_bool_B.sum() > 0 else 0.0)

    upsampler.train()
    decoder.train()
    vggt.eval()
    decoder.calibrator.eval()

    return {
        "mean_ssim": np.mean(ssims), "median_ssim": np.median(ssims),
        "mean_psnr": np.mean(psnrs), "median_psnr": np.median(psnrs),
        "collapse_rate": collapses / len(ssims) if ssims else 0,
        "scale_p1": np.percentile(np.concatenate(scales_all), 1) if scales_all else 0,
        "scale_p5": np.percentile(np.concatenate(scales_all), 5) if scales_all else 0,
        "scale_p50": np.percentile(np.concatenate(scales_all), 50) if scales_all else 0,
        "scale_p95": np.percentile(np.concatenate(scales_all), 95) if scales_all else 0,
        "mean_alpha": np.mean(alphas_all),
    }

def save_visualization(eval_dirs, upsampler, decoder, vggt, device, epoch, out_dir, sample_indices=(0, 100, 199)):
    upsampler.eval()
    decoder.eval()
    vis_dir = os.path.join(out_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    for idx in sample_indices:
        if idx >= len(eval_dirs):
            continue
        sample_dir = eval_dirs[idx]
        sample_id = os.path.basename(sample_dir)
        ds = TypoSplatDataset([sample_dir], vggt, device)
        data = ds[0]
        data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

        with torch.no_grad():
            up_feat = upsampler(data["patch_tokens"])
            p_list, _, _, _, _ = decoder(up_feat, data["base_depth"], data["patch_tokens"])
            m, q, s, o, c = flatten_outputs(p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"])

            render_A, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518)
            render_B, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_B"], Ks=data["Ks_B"], width=518, height=518)

        mask_A, mask_B = data["mask_518_A"], data["mask_518_B"]
        gt_A_masked = (data["gt_rgb_A"] * mask_A)[0].permute(1, 2, 0).cpu().numpy()
        render_A_masked = (render_A.permute(0, 3, 1, 2) * mask_A)[0].permute(1, 2, 0).cpu().numpy()
        gt_B_masked = (data["gt_rgb_B"] * mask_B)[0].permute(1, 2, 0).cpu().numpy()
        render_B_masked = (render_B.permute(0, 3, 1, 2) * mask_B)[0].permute(1, 2, 0).cpu().numpy()

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        axes[0, 0].imshow(gt_A_masked); axes[0, 0].set_title("GT A")
        axes[0, 1].imshow(render_A_masked.clip(0, 1)); axes[0, 1].set_title(f"Render A (Epoch {epoch})")
        axes[1, 0].imshow(gt_B_masked); axes[1, 0].set_title("GT B")
        axes[1, 1].imshow(render_B_masked.clip(0, 1)); axes[1, 1].set_title(f"Render B (Epoch {epoch})")
        for ax in axes.flat: ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(vis_dir, f"{sample_id}_epoch{epoch}.png"), dpi=120)
        plt.close(fig)

    upsampler.train()
    decoder.train()
    vggt.eval()
    decoder.calibrator.eval()

# ==========================================
# 7. CHECKPOINT SAVE HELPER
# ==========================================
def save_full_checkpoint(path, upsampler, decoder, ema_upsampler, ema_decoder, optimizer, global_step, epoch):
    torch.save({
        'upsampler': upsampler.state_dict(),
        'decoder': decoder.state_dict(),
        'ema_upsampler': ema_upsampler.state_dict(),
        'ema_decoder': ema_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': global_step,
        'epoch': epoch
    }, path)

# ==========================================
# 8. MAIN TRAINING LOOP
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_dir", type=str, required=True,
                         help="Directory containing the 1400 samples. All 1400 are used for training; "
                              "the eval subset is drawn from this SAME directory (not held out).")
    parser.add_argument("--checkpoint", type=str, default="/content/drive/MyDrive/TypoSplat/stage1/checkpoint_epoch_22.pt")
    parser.add_argument("--out_dir", type=str, default="/content/drive/MyDrive/TypoSplat/run_1400")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num_eval_samples", type=int, default=300,
                         help="Number of samples (from train_data_dir, sorted, non-held-out) used for eval.")
    parser.add_argument("--drive_backup_root", type=str, default=None,
                         help="Path to Drive-backed cache folder (e.g. a mounted disk_cache_backup_1400 dir). "
                              "If set, cached_features.pt files are restored from here before training starts.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    print("Loading dataset...")
    all_sids = sorted([d for d in os.listdir(args.train_data_dir) if os.path.isdir(os.path.join(args.train_data_dir, d))])
    all_dirs = [os.path.join(args.train_data_dir, s) for s in all_sids if os.path.exists(os.path.join(args.train_data_dir, s, "metadata.json"))]

    train_dirs = all_dirs
    eval_dirs = all_dirs[:args.num_eval_samples]

    print(f"[CHECK] Train: {len(train_dirs)} samples (from {args.train_data_dir})")
    print(f"[CHECK] Eval: {len(eval_dirs)} samples (first {args.num_eval_samples}, sorted, drawn from the SAME "
          f"directory as training — NOT held out. This is a training-set eval by design.)")

    # --- Restore VGGT feature cache from Drive, if configured ---
    if args.drive_backup_root:
        restore_disk_tier_from_drive(all_dirs, args.drive_backup_root)

    vggt = VGGTWrapper().to(device)
    for p in vggt.parameters(): p.requires_grad = False
    vggt.eval()

    train_dataset = TypoSplatDataset(train_dirs, vggt, device)
    eval_dataset = TypoSplatDataset(eval_dirs, vggt, device)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=4)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4)

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoder(in_channels=258).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    upsampler.load_state_dict(ckpt['upsampler'])
    decoder.load_state_dict(ckpt['decoder'])

    for param in decoder.calibrator.parameters():
        param.requires_grad = False

    params = [p for p in list(upsampler.parameters()) + list(decoder.parameters()) if p.requires_grad]
    print(f"[CHECK] Trainable params: {sum(p.numel() for p in params)}")

    optimizer = torch.optim.AdamW(params, lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    ema_upsampler = copy.deepcopy(upsampler)
    ema_decoder = copy.deepcopy(decoder)
    for p in list(ema_upsampler.parameters()) + list(ema_decoder.parameters()):
        p.requires_grad = False

    start_step = 0
    start_epoch = 0
    best_raw_ssim = -1.0
    best_ema_ssim = -1.0

    if args.resume and os.path.exists(args.resume):
        print(f"Resuming from {args.resume}...")
        res_ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        upsampler.load_state_dict(res_ckpt['upsampler'])
        decoder.load_state_dict(res_ckpt['decoder'])
        ema_upsampler.load_state_dict(res_ckpt['ema_upsampler'])
        ema_decoder.load_state_dict(res_ckpt['ema_decoder'])
        optimizer.load_state_dict(res_ckpt['optimizer'])
        start_step = res_ckpt['step']
        start_epoch = res_ckpt['epoch']
        best_raw_ssim = res_ckpt.get('best_raw_ssim', -1.0)
        best_ema_ssim = res_ckpt.get('best_ema_ssim', -1.0)
        print(f"[RESUME CHECK] Successfully loaded checkpoint. Resuming from Epoch {start_epoch}, Global Step {start_step}")
        print(f"[RESUME CHECK] best_raw_ssim={best_raw_ssim:.4f}, best_ema_ssim={best_ema_ssim:.4f}")

    steps_per_epoch = len(train_loader)
    NUM_EPOCHS = 400
    total_steps = NUM_EPOCHS * steps_per_epoch

    # Originally-validated loss weights (Part 3 recipe) — NOT the 0.35/0.175/0.5
    # adjusted weights from the prior 30-epoch follow-up run, whose effect was
    # never independently measured.
    W_SSIM_A_CEIL = 0.2
    W_SSIM_B_CEIL = 0.1
    W_SCALE_MAG = 1.0

    if start_step == 0:
        # --- Pre-Flight Sequence ---
        run_control_suite(device, eval_dirs)
        print(f"Config logged: LR Warmup=1000, SSIM Warmup=800, Peak LR=2e-4, Cosine Decay over {NUM_EPOCHS} epochs, "
              f"EMA Decay=0.999 @ step 1500, W_SSIM_A={W_SSIM_A_CEIL}, W_SSIM_B={W_SSIM_B_CEIL}, W_SCALE_MAG={W_SCALE_MAG}")

        print("\n--- Running Step 0 Eval (Raw Weights) ---")
        raw_metrics = evaluate(eval_loader, upsampler, decoder, vggt, device)
        print(f"Raw  ({len(eval_dirs)}) | SSIM: {raw_metrics['mean_ssim']:.4f} | PSNR: {raw_metrics['mean_psnr']:.2f} | Col: {raw_metrics['collapse_rate']:.2%}")

        print("\n--- Running Step 0 Eval (EMA Weights) ---")
        eval_metrics = evaluate(eval_loader, ema_upsampler, ema_decoder, vggt, device)
        print(f"EMA  ({len(eval_dirs)}) | SSIM: {eval_metrics['mean_ssim']:.4f} | PSNR: {eval_metrics['mean_psnr']:.2f} | Col: {eval_metrics['collapse_rate']:.2%} | Alpha: {eval_metrics['mean_alpha']:.4f}")

        save_visualization(eval_dirs, ema_upsampler, ema_decoder, vggt, device, "0_baseline", args.out_dir)
        print(f"-> Step 0 Baseline Visualizations Saved.")

        print("\n--- Smoke Test (20 real training steps) ---")
        upsampler.train()
        decoder.train()
        vggt.eval()
        decoder.calibrator.eval()

        start_t = time.time()
        smoke_iter = iter(train_loader)

        for i in range(20):
            try:
                batch = next(smoke_iter)
            except StopIteration:
                smoke_iter = iter(train_loader)
                batch = next(smoke_iter)

            for pg in optimizer.param_groups:
                pg['lr'] = get_lr(i, total_steps)

            optimizer.zero_grad()
            step_loss = 0.0

            ssim_weight_scale = min(1.0, i / 800.0)
            w_ssim_A = W_SSIM_A_CEIL * ssim_weight_scale
            w_ssim_B = W_SSIM_B_CEIL * ssim_weight_scale

            for data in batch:
                data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}
                up_feat = upsampler(data["patch_tokens"])
                p_list, _, _, _, _ = decoder(up_feat, data["base_depth"], data["patch_tokens"])
                m, q, s, o, c = flatten_outputs(p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"])

                render_A, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518)
                pred_A = render_A[..., :3].permute(0, 3, 1, 2)
                render_B, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_B"], Ks=data["Ks_B"], width=518, height=518)
                pred_B = render_B[..., :3].permute(0, 3, 1, 2)

                M_A, M_B = data["mask_518_A"], data["mask_518_B"]
                M_loss_A = F.max_pool2d(M_A.float(), kernel_size=11, stride=1, padding=5)
                M_loss_B = F.max_pool2d(M_B.float(), kernel_size=11, stride=1, padding=5)

                mask_bool_A = M_A[0, 0].bool()
                mask_bool_B = M_B[0, 0].bool()
                m3_A = mask_bool_A.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
                m3_B = mask_bool_B.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)

                L1_A = (pred_A - data["gt_rgb_A"])[m3_A].abs().mean()
                L1_B = (pred_B - data["gt_rgb_B"])[m3_B].abs().mean()

                SSIM_A, _ = dycheck_ssim(pred_A, data["gt_rgb_A"], M_loss_A, window_size=7, cov_thresh=0.5)
                SSIM_B, _ = dycheck_ssim(pred_B, data["gt_rgb_B"], M_loss_B, window_size=7, cov_thresh=0.5)

                L_A_photo = 0.8 * L1_A + w_ssim_A * (1.0 - SSIM_A)
                L_B_photo = 0.8 * L1_B + w_ssim_B * (1.0 - SSIM_B)
                L_photo = 0.6 * L_A_photo + 0.4 * L_B_photo

                flat_mask_bool = data["mask_148_A"][0, 0].bool().view(-1)
                full_mask = torch.cat([flat_mask_bool]*3, dim=0)
                valid_scales = s[full_mask]

                L_scale = compute_scale_magnitude_penalty(valid_scales, target_fraction=0.3)
                L_aniso = compute_anisotropy_loss(valid_scales, r_bound=10.0)

                sample_loss = L_photo + W_SCALE_MAG * L_scale + 0.1 * L_aniso
                step_loss += sample_loss / len(batch)

            step_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            optimizer.step()

        end_t = time.time()

        sec_per_step = (end_t - start_t) / 20.0
        est_total_hours = (total_steps * sec_per_step) / 3600.0
        print(f"Smoke test complete. 20 steps took {end_t - start_t:.2f}s.")
        print(f"Steps per epoch: {steps_per_epoch} | Total Steps: {total_steps}")
        print(f"Estimated full run time: {est_total_hours:.2f} hours")

        input("\n>>> Pre-flight complete. Press Enter to proceed to full training or Ctrl+C to abort...")

    # --- Training Loop ---
    collapse_history = []
    scale_p1_history = []
    grad_norm_history = []

    global_step = start_step

    for epoch in range(start_epoch, NUM_EPOCHS):
        upsampler.train()
        decoder.train()
        vggt.eval()
        decoder.calibrator.eval()

        print(f"\n--- EPOCH {epoch+1}/{NUM_EPOCHS} ---")
        epoch_b_offset = []

        for batch in train_loader:
            current_lr = get_lr(global_step, total_steps)
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            optimizer.zero_grad()
            step_loss = 0.0

            ssim_weight_scale = min(1.0, global_step / 800.0)
            w_ssim_A = W_SSIM_A_CEIL * ssim_weight_scale
            w_ssim_B = W_SSIM_B_CEIL * ssim_weight_scale

            for data in batch:
                data = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}

                up_feat = upsampler(data["patch_tokens"])
                p_list, _, _, _, _ = decoder(up_feat, data["base_depth"], data["patch_tokens"])

                m, q, s, o, c = flatten_outputs(p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"])

                render_A, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518)
                pred_A = render_A[..., :3].permute(0, 3, 1, 2)

                render_B, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_B"], Ks=data["Ks_B"], width=518, height=518)
                pred_B = render_B[..., :3].permute(0, 3, 1, 2)

                M_A, M_B = data["mask_518_A"], data["mask_518_B"]
                M_loss_A = F.max_pool2d(M_A.float(), kernel_size=11, stride=1, padding=5)
                M_loss_B = F.max_pool2d(M_B.float(), kernel_size=11, stride=1, padding=5)

                mask_bool_A = M_A[0, 0].bool()
                mask_bool_B = M_B[0, 0].bool()
                m3_A = mask_bool_A.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)
                m3_B = mask_bool_B.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1)

                L1_A = (pred_A - data["gt_rgb_A"])[m3_A].abs().mean()
                L1_B = (pred_B - data["gt_rgb_B"])[m3_B].abs().mean()

                SSIM_A, _ = dycheck_ssim(pred_A, data["gt_rgb_A"], M_loss_A, window_size=7, cov_thresh=0.5)
                SSIM_B, _ = dycheck_ssim(pred_B, data["gt_rgb_B"], M_loss_B, window_size=7, cov_thresh=0.5)

                L_A_photo = 0.8 * L1_A + w_ssim_A * (1.0 - SSIM_A)
                L_B_photo = 0.8 * L1_B + w_ssim_B * (1.0 - SSIM_B)
                L_photo = 0.6 * L_A_photo + 0.4 * L_B_photo

                flat_mask_bool = data["mask_148_A"][0, 0].bool().view(-1)
                full_mask = torch.cat([flat_mask_bool]*3, dim=0)
                valid_scales = s[full_mask]

                L_scale = compute_scale_magnitude_penalty(valid_scales, target_fraction=0.3)
                L_aniso = compute_anisotropy_loss(valid_scales, r_bound=10.0)

                sample_loss = L_photo + W_SCALE_MAG * L_scale + 0.1 * L_aniso
                step_loss += sample_loss / len(batch)

                b_offset = (pred_B[0, :, mask_bool_B].mean() - data["gt_rgb_B"][0, :, mask_bool_B].mean()).item()
                epoch_b_offset.append(b_offset)

            step_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=100.0).item()
            grad_norm_history.append(grad_norm)

            optimizer.step()
            update_ema(ema_upsampler, upsampler, global_step, 1500, 0.999)
            update_ema(ema_decoder, decoder, global_step, 1500, 0.999)
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}/{total_steps} | LR: {current_lr:.6f} | Loss: {step_loss.item():.4f} | Grad: {grad_norm:.2f}")

        # --- Epoch Evaluation ---
        print("\n--- Running Epoch Evaluation (Raw Weights) ---")
        raw_metrics = evaluate(eval_loader, upsampler, decoder, vggt, device)

        print("\n--- Running Epoch Evaluation (EMA Weights) ---")
        eval_metrics = evaluate(eval_loader, ema_upsampler, ema_decoder, vggt, device)

        print(f"Raw  ({len(eval_dirs)}) | SSIM: {raw_metrics['mean_ssim']:.4f} | PSNR: {raw_metrics['mean_psnr']:.2f} | Col: {raw_metrics['collapse_rate']:.2%}")
        print(f"EMA  ({len(eval_dirs)}) | SSIM: {eval_metrics['mean_ssim']:.4f} | PSNR: {eval_metrics['mean_psnr']:.2f} | Col: {eval_metrics['collapse_rate']:.2%} | Alpha: {eval_metrics['mean_alpha']:.4f}")
        print(f"Scales (1/5/50/95) | {eval_metrics['scale_p1']:.6f} / {eval_metrics['scale_p5']:.6f} / {eval_metrics['scale_p50']:.6f} / {eval_metrics['scale_p95']:.6f}")

        collapse_history.append(eval_metrics['collapse_rate'])
        scale_p1_history.append(eval_metrics['scale_p1'])

        save_visualization(eval_dirs, ema_upsampler, ema_decoder, vggt, device, epoch + 1, args.out_dir)

        # --- Checkpointing ---
        print("\n--- Saving Checkpoints ---")
        current_epoch_num = epoch + 1

        # last.pt: always overwritten, every epoch — cheap insurance, always reflects
        # the most recent state regardless of what epoch the run stops/crashes at.
        last_path = os.path.join(args.out_dir, "last.pt")
        save_full_checkpoint(last_path, upsampler, decoder, ema_upsampler, ema_decoder, optimizer, global_step, current_epoch_num)
        print(f"-> Saved last.pt (Epoch {current_epoch_num})")

        # Full numbered checkpoint every 10 epochs only — at 400 epochs, saving raw+EMA+optimizer
        # state every single epoch is a large amount of redundant disk usage.
        if current_epoch_num % 10 == 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_epoch_{current_epoch_num}.pt")
            save_full_checkpoint(ckpt_path, upsampler, decoder, ema_upsampler, ema_decoder, optimizer, global_step, current_epoch_num)
            print(f"-> Saved {ckpt_path}")

        # best_raw.pt: tracked independently from EMA — raw and EMA are not guaranteed
        # to peak at the same epoch, so neither should be assumed to imply the other.
        if raw_metrics['mean_ssim'] > best_raw_ssim:
            best_raw_ssim = raw_metrics['mean_ssim']
            best_raw_path = os.path.join(args.out_dir, "best_raw.pt")
            torch.save({
                'upsampler': upsampler.state_dict(),
                'decoder': decoder.state_dict(),
                'epoch': current_epoch_num,
                'ssim': best_raw_ssim,
            }, best_raw_path)
            print(f"-> Saved new best_raw.pt (SSIM: {best_raw_ssim:.4f}, Epoch {current_epoch_num})")

        # best_ema.pt: tracked independently from raw.
        if eval_metrics['mean_ssim'] > best_ema_ssim:
            best_ema_ssim = eval_metrics['mean_ssim']
            best_ema_path = os.path.join(args.out_dir, "best_ema.pt")
            torch.save({
                'ema_upsampler': ema_upsampler.state_dict(),
                'ema_decoder': ema_decoder.state_dict(),
                'epoch': current_epoch_num,
                'ssim': best_ema_ssim,
            }, best_ema_path)
            print(f"-> Saved new best_ema.pt (SSIM: {best_ema_ssim:.4f}, Epoch {current_epoch_num})")

if __name__ == "__main__":
    main()