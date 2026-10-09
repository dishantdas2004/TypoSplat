"""
TypoSplat Stage 2: RGB-Skip Colour Finetune
===========================================
Warm-starts from a finished Stage 1 checkpoint (--init_checkpoint) and adds a
direct pixel-colour path to the decoder (the Stage 1 decoder never saw the
input image's RGB, only VGGT tokens, so colours were not learned).

  * New zero-initialised modules: an RGB conv added into the first conv of each
    trunk, and a per-layer 3x3 RGB skip added to the raw sh_dc output.
    Input to both: z = 4*(rgb - 0.5), with rgb = input image A area-downsampled
    to 148x148. At step 0 the model is IDENTICAL to the Stage 1 checkpoint.
  * Calibrator is TRAINED by default (calibration was still improving at the end of Stage 1):
    its own LR group (--lr_calib) and the calibrator-regression loss at its Stage 1 final weight
    (0.2 x 2.0). Pass --freeze_calibrator to freeze it instead. All geometry losses unchanged.
  * Two LR groups (new weights / existing weights), linear warmup + cosine to 10%.
  * No blur curriculum / SSIM ramp / centroid term (those were Stage 1 bootstrap
    schedules, already completed).
  * Skips are fed full-resolution RGB bilinearly sampled at each layer's own anchor
    position (detached coordinates); trunks get the area-downsampled 148x148 image.
  * EMA of weights (raw and EMA are both evaluated and checkpointed), self-calibrating
    gradient clipping (3x median of the first 200 steps), calibrator kept in eval mode.
  * Eval reports masked novel-view PSNR and view-A PSNR (a gap between them means the
    model is copying pixels onto A instead of learning colour for novel views).
  * Checkpoint selection: novel-view SSIM floor (step-0 baseline - 0.005), then max PSNR.
Auto-resumes from the latest checkpoint_epoch_*.pt in --checkpoint_dir.
Use a NEW empty --checkpoint_dir for a fresh finetune.
"""

import os
import sys
import glob
import json
import shutil
import random
import argparse
import uuid
import torch
import copy
import math
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import OpenEXR
import Imath

# Prevents the known "received 0 items of ancdata" crash with multi-worker
# DataLoaders on long runs (hit previously after ~epoch 16 in an earlier run).
torch.multiprocessing.set_sharing_strategy('file_system')

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper
from src.models.upsampler import TypoSplatUpsampler
from src.models.decoder import TypoSplatDecoder
from src.losses.render_losses import compute_l1_rgb_loss
from src.losses.typ_losses import (
    compute_scale_invariant_depth_loss,
    compute_extrusion_loss,
    compute_normal_loss,
    compute_anisotropy_loss,
    compute_centroid_loss,
    compute_zoffset_regularization,
    compute_opacity_sparsity_loss,
    compute_calibrator_regression_loss,
    _get_relative_viewmat
)
from gsplat import rasterization


class TypoSplatDecoderRGB(TypoSplatDecoder):
    """
    Stage-1 decoder + zero-initialised RGB paths. With all rgb_* weights at zero
    this is exactly the Stage 1 decoder.
    """
    def __init__(self, in_channels=258, trunk_channels=256):
        super().__init__(in_channels=in_channels, trunk_channels=trunk_channels)
        self.rgb_trunk_0_1 = nn.Conv2d(3, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.rgb_trunk_2 = nn.Conv2d(3, trunk_channels, kernel_size=3, padding=1, bias=False)
        self.rgb_skip = nn.ModuleList([nn.Conv2d(3, 3, kernel_size=3, padding=1) for _ in range(3)])
        for m in [self.rgb_trunk_0_1, self.rgb_trunk_2, *self.rgb_skip]:
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _run_trunk(trunk, features, rgb_conv, z):
        # trunk = [Conv, GELU, Conv, GELU]; RGB is added to the first conv's output.
        x = trunk[0](features) + rgb_conv(z)
        x = trunk[1](x)
        x = trunk[2](x)
        return trunk[3](x)

    @staticmethod
    def _anchor_rgb(img_518, xy_offset):
        """Bilinearly sample the full-res image at each anchor's position in view A.
        Anchor (i,j) sits at (j+0.5+dx, i+0.5+dy) in 148-grid pixel units (same as the rasteriser
        path in flatten_decoder_outputs_camera_space). Coordinates are detached."""
        B, _, H, W = xy_offset.shape
        ys = torch.arange(H, device=xy_offset.device, dtype=xy_offset.dtype).view(1, H, 1)
        xs = torch.arange(W, device=xy_offset.device, dtype=xy_offset.dtype).view(1, 1, W)
        u = xs + 0.5 + xy_offset[:, 0].detach()
        v = ys + 0.5 + xy_offset[:, 1].detach()
        grid = torch.stack([2.0 * u / W - 1.0, 2.0 * v / H - 1.0], dim=-1)
        return F.grid_sample(img_518, grid, mode="bilinear", padding_mode="border", align_corners=False)

    def forward(self, features, base_depth_518, patch_tokens, rgb_148, rgb_518):
        z = 4.0 * (rgb_148 - 0.5)

        feat_0_1 = self._run_trunk(self.trunk_0_1, features, self.rgb_trunk_0_1, z)
        feat_2 = self._run_trunk(self.trunk_2, features, self.rgb_trunk_2, z)

        params_0 = self.head_0(feat_0_1)
        params_1 = self.head_1(feat_0_1)
        params_2 = self.head_2(feat_2)
        for i, prm in enumerate((params_0, params_1, params_2)):
            z_i = 4.0 * (self._anchor_rgb(rgb_518, prm["xy_offset"]) - 0.5)
            prm["sh_dc"] = prm["sh_dc"] + self.rgb_skip[i](z_i)

        base_depth_148 = F.interpolate(base_depth_518, size=(148, 148), mode='bilinear', align_corners=False)
        scale, shift, raw_calib_out, attn_weights = self.calibrator(patch_tokens)
        base_depth_148 = scale.view(-1, 1, 1, 1) * base_depth_148 + shift.view(-1, 1, 1, 1)

        params_0["true_depth"] = base_depth_148
        params_1["true_depth"] = base_depth_148 + params_1["z_offset"]
        params_2["true_depth"] = base_depth_148 + params_1["z_offset"] + params_2["z_offset"]

        return [params_0, params_1, params_2], scale, shift, raw_calib_out, attn_weights


def rgb148_of(data):
    # Input image A (full frame, as it would be given at inference), area-downsampled
    # to the 148x148 anchor grid: [1,3,518,518] -> [1,3,148,148].
    return F.interpolate(data["gt_rgb_A"], size=(148, 148), mode="area")


def set_train_mode(upsampler, decoder):
    # Keep the calibrator in eval mode (it has no BatchNorm/dropout, so this is identical to train mode
    # when it is trained, and guarantees nothing changes when it is frozen).
    upsampler.train()
    decoder.train()
    decoder.calibrator.eval()


@torch.no_grad()
def ema_update(ema_model, model, decay, copy_weights=False):
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        if not p.requires_grad:
            continue
        if copy_weights:
            pe.copy_(p)
        else:
            pe.mul_(decay).add_(p.detach(), alpha=1.0 - decay)


def append_eval_row(csv_path, epoch, tag, scale_err, shift_err, per_sample, ssim, psnr, a_psnr):
    attn_max = sum(r["attn_max"] for r in per_sample) / len(per_sample)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a") as f:
        if write_header:
            f.write("epoch,weights,mean_scale_err,mean_shift_err,mean_attn_max,mean_novel_ssim,mean_novel_psnr,mean_A_psnr\n")
        f.write(f"{epoch},{tag},{scale_err:.6f},{shift_err:.6f},{attn_max:.6f},{ssim:.6f},{psnr:.4f},{a_psnr:.4f}\n")


def run_zero_init_equivalence_test(upsampler, decoder, sample_dataset, device):
    """At a fresh warm start the RGB decoder must reproduce the Stage 1 decoder exactly."""
    ref = TypoSplatDecoder(in_channels=258).to(device)
    ref.load_state_dict({k: v for k, v in decoder.state_dict().items() if not k.startswith("rgb_")})
    ref.eval()
    decoder.eval()
    upsampler.eval()
    sample = move_to_device(sample_dataset[0], device)
    with torch.no_grad():
        up = upsampler(sample["patch_tokens"])
        new_layers, new_scale, new_shift, _, _ = decoder(up, sample["base_depth"], sample["patch_tokens"], rgb148_of(sample), sample["gt_rgb_A"])
        ref_layers, ref_scale, ref_shift, _, _ = ref(up, sample["base_depth"], sample["patch_tokens"])
    max_diff = max((new_scale - ref_scale).abs().max().item(), (new_shift - ref_shift).abs().max().item())
    for ln, lr_ in zip(new_layers, ref_layers):
        for k in ln:
            max_diff = max(max_diff, (ln[k] - lr_[k]).abs().max().item())
    ok = max_diff < 1e-5
    print(f"[SELF-TEST C] RGB decoder == Stage 1 decoder at zero-init: max|diff|={max_diff:.3e} {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise RuntimeError(f"Self-test C failed: max|diff|={max_diff}")

# ==========================================
# 1. Dataset Helpers
# ==========================================

def load_exr_depth(filepath, device='cpu'):
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

def collate_fn(batch):
    return batch

def move_to_device(data, device):
    return {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in data.items()}

class TypoSplatDataset(Dataset):
    """
    CPU-only Dataset. No CUDA calls happen in __getitem__ so this is safe
    to use with multiple DataLoader workers (num_workers > 0).
    """
    def __init__(self, sample_dirs, diagnostic_df):
        self.sample_dirs = sample_dirs
        self.diagnostic_df = diagnostic_df

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        sample_id = int(os.path.basename(sample_dir))

        target_opt_scale = float(self.diagnostic_df.loc[sample_id, "Opt_Scale"])
        target_opt_shift = float(self.diagnostic_df.loc[sample_id, "Opt_Shift"])

        meta_path = os.path.join(sample_dir, "metadata.json")
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        view_A_paths = glob.glob(os.path.join(sample_dir, "*view_A*.png"))
        view_B_paths = glob.glob(os.path.join(sample_dir, "*view_B*.png"))
        view_C_paths = glob.glob(os.path.join(sample_dir, "*view_C*.png"))
        view_D_paths = glob.glob(os.path.join(sample_dir, "*view_D*.png"))
        depth_A_paths = glob.glob(os.path.join(sample_dir, "*depth_A*.exr"))

        cache_path = os.path.join(sample_dir, "cached_features.pt")
        try:
            if not os.path.exists(cache_path):
                raise FileNotFoundError("Cache file missing.")
            cached_data = torch.load(cache_path, map_location='cpu', weights_only=True)
            expected_keys = ("patch_tokens", "base_depth", "mask_148_A", "mask_148_B", "mask_148_C", "mask_148_D")
            if not all(k in cached_data for k in expected_keys):
                raise ValueError("Incomplete cache data keys.")
        except Exception as e:
            raise RuntimeError(f"Cache missing/corrupt for {sample_dir}. Run the precompute script first. Details: {e}")

        # Kept entirely on CPU for safe multi-worker dataloading
        patch_tokens = cached_data["patch_tokens"].float()
        base_depth = cached_data["base_depth"].float()
        mask_148_A = cached_data["mask_148_A"].float()
        mask_148_B = cached_data["mask_148_B"].float()
        mask_148_C = cached_data["mask_148_C"].float()
        mask_148_D = cached_data["mask_148_D"].float()

        mask_518_A = torch.nn.functional.interpolate(mask_148_A, size=(518, 518), mode='nearest')
        mask_518_B = torch.nn.functional.interpolate(mask_148_B, size=(518, 518), mode='nearest')
        mask_518_C = torch.nn.functional.interpolate(mask_148_C, size=(518, 518), mode='nearest')
        mask_518_D = torch.nn.functional.interpolate(mask_148_D, size=(518, 518), mode='nearest')

        gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)
        gt_depth_518_A = load_exr_depth(depth_A_paths[0], 'cpu')
        gt_depth_148_A = torch.nn.functional.interpolate(gt_depth_518_A, size=(148, 148), mode='nearest')

        gt_rgb_B = transforms.ToTensor()(Image.open(view_B_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)
        gt_rgb_C = transforms.ToTensor()(Image.open(view_C_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)
        gt_rgb_D = transforms.ToTensor()(Image.open(view_D_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0)

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

        meta_C = meta["camera_C"]
        Ks_C = torch.tensor([[[meta_C["fx"], 0, meta_C["cx"]], [0, meta_C["fy"], meta_C["cy"]], [0, 0, 1]]], dtype=torch.float32)
        viewmats_C = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_C["camera_to_world_matrix"], 'cpu')

        meta_D = meta["camera_D"]
        Ks_D = torch.tensor([[[meta_D["fx"], 0, meta_D["cx"]], [0, meta_D["fy"], meta_D["cy"]], [0, 0, 1]]], dtype=torch.float32)
        viewmats_D = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_D["camera_to_world_matrix"], 'cpu')

        return {
            "dir": sample_dir,
            "meta": meta,
            "gt_rgb_A": gt_rgb_A,
            "gt_depth_148_A": gt_depth_148_A,
            "gt_rgb_B": gt_rgb_B,
            "gt_rgb_C": gt_rgb_C,
            "gt_rgb_D": gt_rgb_D,
            "mask_148_A": mask_148_A,
            "mask_518_A": mask_518_A,
            "mask_518_B": mask_518_B,
            "mask_518_C": mask_518_C,
            "mask_518_D": mask_518_D,
            "intrinsics_tuple_A": intrinsics_tuple_A,
            "intrinsics_dict_148_A": intrinsics_dict_148_A,
            "Ks_A": Ks_A,
            "viewmats_A": viewmats_A,
            "Ks_B": Ks_B,
            "viewmats_B": viewmats_B,
            "Ks_C": Ks_C,
            "viewmats_C": viewmats_C,
            "Ks_D": Ks_D,
            "viewmats_D": viewmats_D,
            "patch_tokens": patch_tokens,
            "base_depth": base_depth,
            "target_opt_scale": target_opt_scale,
            "target_opt_shift": target_opt_shift,
            "target_opt_scale_t": torch.tensor(target_opt_scale, dtype=torch.float32),
            "target_opt_shift_t": torch.tensor(target_opt_shift, dtype=torch.float32),
        }

# ==========================================
# 2. Additions (SSIM, Scale Penalty, Act, Blur)
# ==========================================

MAX_BLUR_SIGMA = 100.0
GAIN, BIAS, LEAK = 1.0, 0.72, 0.2
def function_preserving_activation(raw):
    x = GAIN * raw + BIAS
    return (LEAK * x + (1.0 - LEAK) * x.clamp(-2.0, 2.0)) / 4.0 + 0.5

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

def make_gaussian_kernel1d(kernel_size, sigma, device):
    # Identical formula to torchvision's _get_gaussian_kernel1d, built on CPU then moved.
    half = (kernel_size - 1) * 0.5
    x = torch.linspace(-half, half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()
    return kernel1d.to(device=device, dtype=torch.float32)

def separable_gaussian_blur(img, kernel1d):
    # Two 1D passes with reflect padding. Mathematically identical to
    # torchvision.transforms.functional.gaussian_blur with a square kernel,
    # at a small fraction of the cost of the 2D version.
    C = img.shape[1]
    k = kernel1d.shape[0]
    pad = k // 2
    assert pad < img.shape[-2] and pad < img.shape[-1], "blur padding must be smaller than the image"
    x = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    w_h = kernel1d.view(1, 1, 1, k).repeat(C, 1, 1, 1)
    w_v = kernel1d.view(1, 1, k, 1).repeat(C, 1, 1, 1)
    x = F.conv2d(x, w_h, groups=C)
    x = F.conv2d(x, w_v, groups=C)
    return x

def run_startup_self_tests(upsampler, decoder, sample_dataset, device):
    print("\n=== STARTUP SELF-TESTS ===")
    # Test A: separable blur must equal torchvision's gaussian_blur.
    # TF32 is disabled during this test so rounding differences cannot mask or fake a mismatch.
    prev_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    try:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(1234)
        test_img = torch.rand(1, 3, 518, 518, generator=gen).to(device)
        for sigma in (100.0, 50.0, 5.0):
            k = int(sigma * 4) | 1
            kernel1d = make_gaussian_kernel1d(k, sigma, device)
            with torch.no_grad():
                ours = separable_gaussian_blur(test_img, kernel1d)
                ref = TF.gaussian_blur(test_img, [k, k], [sigma, sigma])
            max_diff = (ours - ref).abs().max().item()
            ok = max_diff < 1e-4
            print(f"[SELF-TEST A] sigma={sigma:6.1f} kernel={k:4d} max|diff|={max_diff:.3e} {'PASS' if ok else 'FAIL'}")
            if not ok:
                raise RuntimeError(f"Self-test A failed at sigma={sigma}: max|diff|={max_diff}")
    finally:
        torch.backends.cudnn.allow_tf32 = prev_tf32

    # Test B: one batched 4-camera render must equal four separate renders.
    was_up, was_dec = upsampler.training, decoder.training
    upsampler.eval()
    decoder.eval()
    try:
        sample = move_to_device(sample_dataset[0], device)
        with torch.no_grad():
            up_feat = upsampler(sample["patch_tokens"])
            p_list, _, _, _, _ = decoder(up_feat, sample["base_depth"], sample["patch_tokens"], rgb148_of(sample), sample["gt_rgb_A"])
            m, q, s, o, c = flatten_decoder_outputs_camera_space(
                p_list[0], p_list[1], p_list[2], sample["intrinsics_tuple_A"], device, mask_148=sample["mask_148_A"]
            )
            singles = []
            for v in ["A", "B", "C", "D"]:
                r, _, _ = rasterization(
                    means=m, quats=q, scales=s, opacities=o, colors=c,
                    viewmats=sample[f"viewmats_{v}"], Ks=sample[f"Ks_{v}"], width=518, height=518
                )
                singles.append(r)
            singles = torch.cat(singles, dim=0)

            vm = torch.cat([sample[f"viewmats_{v}"] for v in ["A", "B", "C", "D"]], dim=0)
            kk = torch.cat([sample[f"Ks_{v}"] for v in ["A", "B", "C", "D"]], dim=0)
            batched, _, _ = rasterization(
                means=m, quats=q, scales=s, opacities=o, colors=c,
                viewmats=vm, Ks=kk, width=518, height=518
            )
        if singles.shape != batched.shape:
            raise RuntimeError(f"Self-test B failed: shapes differ {tuple(singles.shape)} vs {tuple(batched.shape)}")
        if not (torch.isfinite(singles).all() and torch.isfinite(batched).all()):
            raise RuntimeError("Self-test B failed: non-finite values in renders")
        max_diff = (singles - batched).abs().max().item()
        ok = max_diff < 1e-4
        print(f"[SELF-TEST B] batched vs separate renders max|diff|={max_diff:.3e} {'PASS' if ok else 'FAIL'}")
        if not ok:
            raise RuntimeError(f"Self-test B failed: max|diff|={max_diff}")
    finally:
        upsampler.train(was_up)
        decoder.train(was_dec)
    print("=== ALL SELF-TESTS PASSED ===\n")

# ==========================================
# 3. Helpers (Flatten & Evaluate)
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

        raw_sh_dc = params["sh_dc"][0].permute(1, 2, 0).view(-1, 3)
        colors = function_preserving_activation(raw_sh_dc)

        opacities = params["opacity"][0].view(-1)
        if flat_mask is not None:
            opacities = opacities * flat_mask

        all_means.append(means)
        all_quats.append(quats)
        all_scales.append(scales)
        all_opacities.append(opacities)
        all_colors.append(colors)

    return (torch.cat(all_means, dim=0), torch.cat(all_quats, dim=0), torch.cat(all_scales, dim=0), torch.cat(all_opacities, dim=0), torch.cat(all_colors, dim=0))

def evaluate(dataloader, upsampler, decoder, device, restore_train=True):
    upsampler.eval()
    decoder.eval()
    scale_errors, shift_errors, novel_ssims, novel_psnrs, a_psnrs, per_sample_results = [], [], [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            for data in batch:
                data = move_to_device(data, device)
                upsampled_features = upsampler(data["patch_tokens"])
                params_list, calib_scale, calib_shift, _, attn_weights = decoder(upsampled_features, data["base_depth"], data["patch_tokens"], rgb148_of(data), data["gt_rgb_A"])

                pred_scale = calib_scale.item()
                pred_shift = calib_shift.item()

                scale_errors.append(abs(pred_scale - data["target_opt_scale"]))
                shift_errors.append(abs(pred_shift - data["target_opt_shift"]))

                m, q, s, o, c = flatten_decoder_outputs_camera_space(
                    params_list[0], params_list[1], params_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                )

                sample_ssims, sample_psnrs = [], []

                # View-A PSNR (input view). If this keeps rising while novel-view PSNR stalls,
                # the model is copying pixels onto A rather than learning colour.
                rA, _, _ = rasterization(
                    means=m, quats=q, scales=s, opacities=o, colors=c,
                    viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518
                )
                mask_A = data["mask_518_A"].float()
                sq_A = ((rA.permute(0, 3, 1, 2).clamp(0.0, 1.0) - data["gt_rgb_A"]) ** 2 * mask_A).sum()
                a_psnrs.append((-10.0 * torch.log10(sq_A / (mask_A.sum() * 3.0 + 1e-8) + 1e-10)).item())
                for view in ['B', 'C', 'D']:
                    r_colors, _, _ = rasterization(
                        means=m, quats=q, scales=s, opacities=o, colors=c,
                        viewmats=data[f"viewmats_{view}"], Ks=data[f"Ks_{view}"], width=518, height=518
                    )
                    pred_raw = r_colors.permute(0, 3, 1, 2)
                    ssim_val, _ = dycheck_ssim(pred_raw, data[f"gt_rgb_{view}"], data[f"mask_518_{view}"], window_size=7, cov_thresh=0.3)
                    sample_ssims.append(ssim_val.item())
                    mask_f = data[f"mask_518_{view}"].float()
                    sq_err = ((pred_raw.clamp(0.0, 1.0) - data[f"gt_rgb_{view}"]) ** 2 * mask_f).sum()
                    mse = sq_err / (mask_f.sum() * 3.0 + 1e-8)
                    sample_psnrs.append((-10.0 * torch.log10(mse + 1e-10)).item())

                novel_ssims.append(sum(sample_ssims) / 3.0)
                novel_psnrs.append(sum(sample_psnrs) / 3.0)

                per_sample_results.append({
                    "sample": os.path.basename(data["dir"]),
                    "pred_scale": pred_scale,
                    "target_scale": data["target_opt_scale"],
                    "pred_shift": pred_shift,
                    "target_shift": data["target_opt_shift"],
                    "attn_max": attn_weights.max().item(),
                    "attn_std": attn_weights.std().item(),
                })

    if restore_train:
        set_train_mode(upsampler, decoder)

    mean_scale_err = sum(scale_errors) / len(scale_errors) if scale_errors else 0.0
    mean_shift_err = sum(shift_errors) / len(shift_errors) if shift_errors else 0.0
    mean_novel_ssim = sum(novel_ssims) / len(novel_ssims) if novel_ssims else 0.0
    mean_novel_psnr = sum(novel_psnrs) / len(novel_psnrs) if novel_psnrs else 0.0
    mean_A_psnr = sum(a_psnrs) / len(a_psnrs) if a_psnrs else 0.0
    return mean_scale_err, mean_shift_err, mean_novel_ssim, per_sample_results, mean_novel_psnr, mean_A_psnr

# ==========================================
# 4. Checkpoint Selection
# ==========================================
# Calibration error is an ELIGIBILITY GATE, not the selector: a checkpoint
# with poor calibration (scale/shift error) is unusable regardless of how
# sharp its renders look, but among checkpoints that pass the gate, novel
# view SSIM (structural/perceptual quality) is what actually distinguishes
# a good reconstruction from a degenerate one (e.g. a scale-collapsed model
# can still show low calibration error while producing garbage geometry).
# Picking purely by calibration error is blind to that failure mode.

def select_best_checkpoint(eval_history_csv, scale_err_gate=None, shift_err_gate=None):
    """
    Reads eval_history.csv (written every epoch during training) and returns
    the epoch number of the best checkpoint under: calibration-error gate,
    then novel-view SSIM as the tiebreaker/selector among eligible epochs.

    If no gate values are given, the gate is computed adaptively as
    median + 1 MAD (median absolute deviation) of each error series, so a
    checkpoint has to be within a reasonable calibration range of the rest
    of the run to even be considered, and then the highest-SSIM checkpoint
    among those survivors wins.
    """
    df = pd.read_csv(eval_history_csv)
    if df.empty:
        raise RuntimeError(f"{eval_history_csv} is empty — no checkpoints to select from.")

    if scale_err_gate is None:
        med_s = df["mean_scale_err"].median()
        mad_s = (df["mean_scale_err"] - med_s).abs().median()
        scale_err_gate = med_s + mad_s
    if shift_err_gate is None:
        med_t = df["mean_shift_err"].median()
        mad_t = (df["mean_shift_err"] - med_t).abs().median()
        shift_err_gate = med_t + mad_t

    eligible = df[(df["mean_scale_err"] <= scale_err_gate) & (df["mean_shift_err"] <= shift_err_gate)]
    if eligible.empty:
        print("[WARN] No epoch passed the calibration gate; falling back to the full history "
              "(best available calibration error) instead of leaving this undecided.")
        eligible = df.loc[[df["mean_scale_err"].idxmin()]]

    best_row = eligible.loc[eligible["mean_novel_ssim"].idxmax()]
    best_epoch = int(best_row["epoch"])
    print(f"\n[CHECKPOINT SELECTION] Gate: scale_err<={scale_err_gate:.4f}, shift_err<={shift_err_gate:.4f}")
    print(f"[CHECKPOINT SELECTION] Eligible epochs: {sorted(eligible['epoch'].astype(int).tolist())}")
    print(f"[CHECKPOINT SELECTION] Selected epoch {best_epoch} "
          f"(scale_err={best_row['mean_scale_err']:.4f}, shift_err={best_row['mean_shift_err']:.4f}, "
          f"novel_ssim={best_row['mean_novel_ssim']:.4f})\n")
    return best_epoch

def select_best_checkpoint_stage2(eval_history_csv, ssim_margin=0.005):
    """
    Stage 2 selection (calibration is no longer the selector; PSNR is).
    Gate: novel-view SSIM >= (step-0 baseline SSIM - ssim_margin) AND calibration errors no worse than
    110% of the step-0 baseline (constant, so a no-op, when the calibrator is frozen).
    Select: highest masked novel-view PSNR among eligible (epoch, weights) rows, raw or EMA.
    """
    df = pd.read_csv(eval_history_csv)
    base = df[df["epoch"] == 0]
    if base.empty:
        raise RuntimeError("No step-0 baseline row (epoch 0) in eval_history.csv.")
    base_ssim = float(base["mean_novel_ssim"].iloc[0])
    base_scale = float(base["mean_scale_err"].iloc[0])
    base_shift = float(base["mean_shift_err"].iloc[0])
    cand = df[df["epoch"] >= 1]
    if cand.empty:
        raise RuntimeError("No finetune epochs in eval_history.csv.")
    eligible = cand[(cand["mean_novel_ssim"] >= base_ssim - ssim_margin)
                    & (cand["mean_scale_err"] <= 1.10 * base_scale)
                    & (cand["mean_shift_err"] <= 1.10 * base_shift)]
    if eligible.empty:
        print("[WARN] No epoch passed the SSIM / calibration gate; selecting by PSNR over all epochs.")
        eligible = cand
    best = eligible.loc[eligible["mean_novel_psnr"].idxmax()]
    print(f"\n[CHECKPOINT SELECTION] Baseline (step 0) SSIM={base_ssim:.4f}; floor={base_ssim - ssim_margin:.4f}; scale_err<={1.10 * base_scale:.4f}, shift_err<={1.10 * base_shift:.4f}")
    print(f"[CHECKPOINT SELECTION] Selected epoch {int(best['epoch'])} ({best['weights']} weights): "
          f"PSNR={best['mean_novel_psnr']:.3f}, SSIM={best['mean_novel_ssim']:.4f}, A-PSNR={best['mean_A_psnr']:.3f}\n")
    return int(best["epoch"]), str(best["weights"])

# ==========================================
# 5. Main Training Script
# ==========================================

def main():
    random.seed(42)
    torch.manual_seed(42)

    parser = argparse.ArgumentParser(description="TypoSplat Stage 2: RGB-skip colour finetune")
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Parent directory containing train sample folders")
    parser.add_argument("--diag_csv", type=str, default="/content/master_diagnostics.csv", help="Path to single master diagnostic CSV")
    parser.add_argument("--eval_data_dir", type=str, default=None, help="Directory containing separate eval sample folders (REQUIRED for true validation)")
    parser.add_argument("--checkpoint_dir", type=str, default="/content/drive/MyDrive/TypoSplat/stage1_checkpoints_v2", help="Persistent storage directory on Google Drive. Use a NEW/empty directory for this from-scratch run.")
    parser.add_argument("--batch_size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--num_epochs", type=int, default=20, help="Total finetune epochs (the cosine schedule is set for this length)")
    parser.add_argument("--init_checkpoint", type=str, default=None, help="Stage 1 checkpoint to warm-start from (required for a fresh finetune; ignored when resuming)")
    parser.add_argument("--lr_new", type=float, default=5e-4, help="Peak LR for the new RGB weights")
    parser.add_argument("--lr_base", type=float, default=4e-5, help="Peak LR for the existing (non-calibrator) weights")
    parser.add_argument("--warmup_new", type=int, default=300, help="Warmup steps for the new RGB weights")
    parser.add_argument("--warmup_base", type=int, default=500, help="Warmup steps for the existing weights")
    parser.add_argument("--freeze_calibrator", action="store_true", help="Freeze the depth calibrator (default: it is trained with the calibrator-regression loss, as in Stage 1)")
    parser.add_argument("--lr_calib", type=float, default=1e-5, help="Peak LR for the calibrator (Stage 1 value); same warmup/cosine as the other weights")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay for the weight average")
    parser.add_argument("--ema_start", type=int, default=1000, help="Iteration at which EMA averaging starts (before that EMA = raw weights)")
    parser.add_argument("--viz_every", type=int, default=1, help="Save render PNGs every N epochs")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of multi-processing workers for dataloading")
    parser.add_argument("--select_best_at_end", action="store_true", help="After training, run checkpoint selection (calibration gate + SSIM tiebreaker) and print the chosen epoch.")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== TypoSplat: Stage 2 RGB-Skip Finetune ===")
    print(f"Checkpoints and Logs routing to: {args.checkpoint_dir}\n")

    # --- Load Master Diagnostic CSV ---
    try:
        master_diag_df = pd.read_csv(args.diag_csv).set_index("Sample")
    except FileNotFoundError:
        print(f"ERROR: {args.diag_csv} not found!")
        sys.exit(1)

    # --- Discover and Filter Train Directories ---
    raw_train_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                      if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    train_dirs = sorted(d for d in raw_train_dirs if int(os.path.basename(d)) in master_diag_df.index)

    # --- Discover and Filter Eval Directories ---
    if args.eval_data_dir and args.eval_data_dir != args.data_dir:
        raw_eval_dirs = [os.path.join(args.eval_data_dir, d) for d in os.listdir(args.eval_data_dir)
                         if os.path.isdir(os.path.join(args.eval_data_dir, d)) and os.path.exists(os.path.join(args.eval_data_dir, d, "metadata.json"))]
        eval_dirs = sorted(d for d in raw_eval_dirs if int(os.path.basename(d)) in master_diag_df.index)
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

    # --- Construct Datasets ---
    train_dataset = TypoSplatDataset(train_dirs, master_diag_df)
    eval_dataset = TypoSplatDataset(eval_dirs, master_diag_df)

    loader_kwargs = {
        'collate_fn': collate_fn,
        'num_workers': args.num_workers,
        'pin_memory': True,
    }
    if args.num_workers > 0:
        loader_kwargs['persistent_workers'] = True
        loader_kwargs['prefetch_factor'] = 2

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    eval_dataloader = DataLoader(eval_dataset, batch_size=16, shuffle=False, **loader_kwargs)

    fixed_train_eval_dirs = train_dirs[:50]
    fixed_train_eval_dataset = TypoSplatDataset(fixed_train_eval_dirs, master_diag_df)
    fixed_train_eval_dataloader = DataLoader(fixed_train_eval_dataset, batch_size=16, shuffle=False, **loader_kwargs)

    fixed_vis_dirs = train_dirs[:2]
    fixed_vis_dataset = TypoSplatDataset(fixed_vis_dirs, master_diag_df)
    fixed_vis_dataloader = DataLoader(fixed_vis_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoderRGB(in_channels=258).to(device)

    calib_params = list(decoder.calibrator.parameters())
    if args.freeze_calibrator:
        for p in calib_params:
            p.requires_grad_(False)
    new_params = [p for n, p in decoder.named_parameters() if n.startswith("rgb_")]
    base_params = list(upsampler.parameters()) + [
        p for n, p in decoder.named_parameters() if 'calibrator' not in n and not n.startswith("rgb_")
    ]

    # Group 0 = new RGB weights, group 1 = existing weights, group 2 = calibrator (unless frozen).
    # LRs are set every step below.
    param_groups = [
        {'params': new_params, 'lr': args.lr_new},
        {'params': base_params, 'lr': args.lr_base},
    ]
    if not args.freeze_calibrator:
        param_groups.append({'params': calib_params, 'lr': args.lr_calib})
    optimizer = optim.Adam(param_groups)

    num_epochs = args.num_epochs
    iters_per_epoch = len(train_dataloader)
    total_iters = num_epochs * iters_per_epoch
    bootstrap_iters = int(iters_per_epoch * 4.0)  # unused in Stage 2 (Stage 1 ramps are finished); kept for the banner
    anneal_iters = int(iters_per_epoch * 4.0)

    W_SSIM_MAX = 0.2
    W_SCALE_MAG = 1.0
    W_ANISO = 0.1  # Scoped to in-letter Gaussians only (see valid_scales below).
                   # Kept at 0.1, not 1.0, to preserve the original effective
                   # training strength now that the loss is scoped to a
                   # smaller (in-letter-only) subset of Gaussians than before.

    start_epoch = 1
    global_iter = 0
    val_err_history = []
    fresh_start = False

    existing_checkpoints = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_epoch_*.pt"))
    if existing_checkpoints:
        latest_ckpt = max(existing_checkpoints, key=lambda p: int(p.split('_epoch_')[-1].split('.pt')[0]))
        print(f"\n[INFO] Found existing finetune checkpoint: {latest_ckpt}")
        print("Restoring weights and resuming training...")
        ckpt_data = torch.load(latest_ckpt, map_location=device)
        upsampler.load_state_dict(ckpt_data['upsampler'])
        decoder.load_state_dict(ckpt_data['decoder'])
        optimizer.load_state_dict(ckpt_data['optimizer'])
        start_epoch = ckpt_data['epoch'] + 1
        global_iter = ckpt_data['global_iter']
        print(f"Resuming from Epoch {start_epoch}, Global Iteration {global_iter}\n")
    else:
        if not args.init_checkpoint:
            print("ERROR: no finetune checkpoint found in --checkpoint_dir and --init_checkpoint was not given.")
            sys.exit(1)
        print(f"\n[INFO] Warm-starting from Stage 1 checkpoint: {args.init_checkpoint}")
        init_ckpt = torch.load(args.init_checkpoint, map_location=device)
        upsampler.load_state_dict(init_ckpt['upsampler'])
        missing, unexpected = decoder.load_state_dict(init_ckpt['decoder'], strict=False)
        expected_missing = {"rgb_trunk_0_1.weight", "rgb_trunk_2.weight"} | {
            f"rgb_skip.{i}.{t}" for i in range(3) for t in ("weight", "bias")
        }
        assert not unexpected, f"Unexpected keys in Stage 1 checkpoint: {unexpected}"
        assert set(missing) == expected_missing, f"Missing keys differ from the new RGB modules: {sorted(set(missing) ^ expected_missing)}"
        print(f"[INFO] Loaded Stage 1 weights (epoch {init_ckpt.get('epoch', '?')}); {len(missing)} new RGB tensors zero-initialised.")
        fresh_start = True

    ema_upsampler = copy.deepcopy(upsampler)
    ema_decoder = copy.deepcopy(decoder)
    for _m in (ema_upsampler, ema_decoder):
        _m.eval()
        for _p in _m.parameters():
            _p.requires_grad_(False)
    clip_thresh = None
    grad_norm_buf = []
    if not fresh_start:
        if 'ema_upsampler' in ckpt_data and 'ema_decoder' in ckpt_data:
            ema_upsampler.load_state_dict(ckpt_data['ema_upsampler'])
            ema_decoder.load_state_dict(ckpt_data['ema_decoder'])
        clip_thresh = ckpt_data.get('clip_thresh', None)
        print(f"[INFO] Resumed EMA weights and clip threshold ({clip_thresh}).")

    print("\n=======================================================")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Iters per Epoch: {iters_per_epoch}")
    print(f"Total Epochs:    {num_epochs} (Starting at {start_epoch})  |  Total iters: {total_iters}")
    print(f"LR new/base:     {args.lr_new} / {args.lr_base}  (warmup {args.warmup_new}/{args.warmup_base}, cosine to 10%)")
    print(f"W_ANISO:         {W_ANISO}")
    print("=======================================================\n")

    run_startup_self_tests(upsampler, decoder, fixed_vis_dataset, device)
    if fresh_start:
        run_zero_init_equivalence_test(upsampler, decoder, fixed_vis_dataset, device)

    set_train_mode(upsampler, decoder)

    # Step-0 evaluation = the Stage 1 baseline (identical model). Logged as epoch 0.
    if fresh_start:
        b_se, b_te, b_ssim, b_ps, b_psnr, b_apsnr = evaluate(eval_dataloader, upsampler, decoder, device)
        append_eval_row(os.path.join(args.checkpoint_dir, "eval_history.csv"), 0, "raw", b_se, b_te, b_ps, b_ssim, b_psnr, b_apsnr)
        print(f"[STEP-0 BASELINE] Val ScaleErr: {b_se:.4f} | ShiftErr: {b_te:.4f} | SSIM: {b_ssim:.4f} | PSNR: {b_psnr:.2f} | A-PSNR: {b_apsnr:.2f}\n")

    for epoch in range(start_epoch, num_epochs + 1):
        epoch_total_loss = 0.0
        epoch_gnorms, epoch_clipped, epoch_skipped = [], 0, 0

        # Initialize component loss tracking for this epoch
        epoch_component_losses = {k: 0.0 for k in [
            "rgb", "ssim", "depth", "extrusion", "aniso", "scale_mag",
            "normal", "novel_view", "centroid", "zreg", "opacity_sparsity", "calib_reg"
        ]}

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            global_iter = (epoch - 1) * iters_per_epoch + batch_idx

            cos_factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, global_iter / max(1, total_iters))))
            optimizer.param_groups[0]['lr'] = args.lr_new * min(1.0, (global_iter + 1) / args.warmup_new) * cos_factor
            optimizer.param_groups[1]['lr'] = args.lr_base * min(1.0, (global_iter + 1) / args.warmup_base) * cos_factor
            if not args.freeze_calibrator:
                optimizer.param_groups[2]['lr'] = args.lr_calib * min(1.0, (global_iter + 1) / args.warmup_base) * cos_factor

            optimizer.zero_grad()
            batch_total_loss = 0.0

            # Stage 1 bootstrap ramps are finished: full SSIM weight, no centroid term, no blur.
            ssim_weight = W_SSIM_MAX
            centroid_weight = 0.0
            current_sigma = 0.0
            blur_active = current_sigma > 0.5
            if blur_active:
                blur_kernel_size = int(current_sigma * 4) | 1
                blur_kernel1d = make_gaussian_kernel1d(blur_kernel_size, current_sigma, device)
            else:
                blur_kernel1d = None

            for sample_idx, data in enumerate(batch):
                data = move_to_device(data, device)
                upsampled_features = upsampler(data["patch_tokens"])

                params_list, calib_scale, calib_shift, _, attn_weights = decoder(upsampled_features, data["base_depth"], data["patch_tokens"], rgb148_of(data), data["gt_rgb_A"])
                params_0, params_1, params_2 = params_list

                means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
                    params_0, params_1, params_2, data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                )

                viewmats_all = torch.cat([data["viewmats_A"], data["viewmats_B"], data["viewmats_C"], data["viewmats_D"]], dim=0)
                Ks_all = torch.cat([data["Ks_A"], data["Ks_B"], data["Ks_C"], data["Ks_D"]], dim=0)
                renders_all, _, _ = rasterization(
                    means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
                    viewmats=viewmats_all, Ks=Ks_all, width=518, height=518,
                )
                assert renders_all.shape == (4, 518, 518, 3), f"Unexpected batched render shape {tuple(renders_all.shape)}"
                pred_rgb_A_raw = renders_all[0:1].permute(0, 3, 1, 2)
                pred_rgb_A_masked = pred_rgb_A_raw * data["mask_518_A"]
                loss_rgb_A = compute_l1_rgb_loss(pred_rgb_A_masked, data["gt_rgb_A"], mask=data["mask_518_A"])
                M_loss_A = F.max_pool2d(data["mask_518_A"].float(), kernel_size=11, stride=1, padding=5)
                ssim_A, _ = dycheck_ssim(pred_rgb_A_raw, data["gt_rgb_A"], M_loss_A, window_size=7, cov_thresh=0.5)
                layer_1_depth = params_0["true_depth"] + params_1["z_offset"]
                loss_depth = compute_scale_invariant_depth_loss(layer_1_depth, data["gt_depth_148_A"], data["mask_148_A"].bool())
                loss_extrusion = compute_extrusion_loss(params_1, params_2, data["meta"]["extrusion_depth"], data["mask_148_A"])
                loss_normal = compute_normal_loss(layer_1_depth, data["gt_depth_148_A"], data["intrinsics_dict_148_A"], data["mask_148_A"])

                # --- Novel-view losses (B, C, D), with the coarse-to-fine Gaussian
                # blur curriculum applied to L1 only. This widens the basin of
                # attraction early in training (sigma starts at MAX_BLUR_SIGMA and
                # anneals to 0 over bootstrap_iters), exactly mirroring the original
                # compute_novel_view_loss behavior but in cheap separable form and
                # batched across all 3 novel views' renders (already produced above
                # in the single 4-camera rasterization call). SSIM is always computed
                # on the RAW, unblurred renders — blurring would corrupt the
                # structural-similarity statistics, so only the L1 term sees the blur.
                loss_rgb_novel = 0.0
                loss_ssim_novel = 0.0
                loss_centroid = torch.zeros((), device=device)
                for view_idx, view in enumerate(['B', 'C', 'D'], start=1):
                    pred_raw_v = renders_all[view_idx:view_idx + 1].permute(0, 3, 1, 2)
                    gt_v = data[f"gt_rgb_{view}"]
                    mask_v = data[f"mask_518_{view}"]
                    if blur_active:
                        # Both sides are "letters on black": GT is masked BEFORE blurring, the render is
                        # blurred raw (no mask multiply, no clamp so gradients never die). The blurred
                        # mask is used only as a weight inside compute_l1_rgb_loss.
                        pred_blur_v = separable_gaussian_blur(pred_raw_v, blur_kernel1d)
                        with torch.no_grad():
                            gt_mask_blur = separable_gaussian_blur(torch.cat([gt_v * mask_v, mask_v], dim=1), blur_kernel1d)
                            gt_blur_v = gt_mask_blur[:, 0:3].clamp(0.0, 1.0)
                            mask_blur_v = gt_mask_blur[:, 3:4].clamp(0.0, 1.0)
                        l1_v = compute_l1_rgb_loss(pred_blur_v, gt_blur_v, mask=mask_blur_v)
                    else:
                        pred_masked_v = pred_raw_v * mask_v
                        l1_v = compute_l1_rgb_loss(pred_masked_v, gt_v, mask=mask_v)
                    M_loss_v = F.max_pool2d(mask_v.float(), kernel_size=11, stride=1, padding=5)
                    ssim_v, _ = dycheck_ssim(pred_raw_v, gt_v, M_loss_v, window_size=7, cov_thresh=0.5)
                    if centroid_weight > 0.0:
                        loss_centroid = loss_centroid + compute_centroid_loss(means, data[f"viewmats_{view}"], data[f"Ks_{view}"], mask_v, device)
                    loss_rgb_novel = loss_rgb_novel + l1_v
                    loss_ssim_novel = loss_ssim_novel + (1.0 - ssim_v)
                loss_rgb_novel = loss_rgb_novel / 3.0
                loss_ssim_novel = loss_ssim_novel / 3.0
                loss_centroid = loss_centroid / 3.0

                flat_mask_bool = data["mask_148_A"][0, 0].bool().view(-1)
                full_mask = torch.cat([flat_mask_bool]*3, dim=0)
                valid_scales = scales[full_mask]  # in-letter Gaussians only

                loss_aniso = compute_anisotropy_loss(valid_scales, r_bound=10.0)
                loss_scale_mag = compute_scale_magnitude_penalty(valid_scales)

                loss_zreg = compute_zoffset_regularization(params_1, params_2)
                loss_opacity_sparsity = compute_opacity_sparsity_loss(opacities)

                calib_reg_weight = 0.2  # Stage 1 final value (weight is only applied when the calibrator is trained)

                loss_calib_target = compute_calibrator_regression_loss(
                    calib_scale, calib_shift,
                    data["target_opt_scale_t"],
                    data["target_opt_shift_t"]
                )

                if batch_idx == len(train_dataloader) - 1 and sample_idx == 0:
                    L_novel_for_grad = loss_rgb_novel + ssim_weight * loss_ssim_novel
                    nv_grad = torch.autograd.grad(L_novel_for_grad, means, retain_graph=True, allow_unused=True)[0]
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

                # Real tensor loss for backpropagation
                sample_loss_tensor = (
                    1.0 * loss_rgb_A +
                    1.0 * loss_rgb_novel +
                    ssim_weight * (1.0 - ssim_A) +
                    ssim_weight * loss_ssim_novel +
                    50.0 * loss_depth +
                    1000.0 * loss_extrusion +
                    W_ANISO * loss_aniso +
                    W_SCALE_MAG * loss_scale_mag +
                    1.0 * loss_normal +
                    (centroid_weight * 0.05 * loss_centroid) +
                    0.05 * loss_zreg +
                    1.0 * loss_opacity_sparsity +
                    ((0.0 if args.freeze_calibrator else calib_reg_weight * 2.0) * loss_calib_target)
                )  # calib_reg is in the loss only while the calibrator is trained; it is always logged below.

                (sample_loss_tensor / args.batch_size).backward()
                batch_total_loss = batch_total_loss + sample_loss_tensor.detach() / args.batch_size

                weighted_terms = {
                    "rgb": 1.0 * loss_rgb_A.detach(),
                    "ssim": ssim_weight * (1.0 - ssim_A).detach(),
                    "novel_view": 1.0 * loss_rgb_novel.detach() + ssim_weight * loss_ssim_novel.detach(),
                    "depth": 50.0 * loss_depth.detach(),
                    "extrusion": 1000.0 * loss_extrusion.detach(),
                    "aniso": W_ANISO * loss_aniso.detach(),
                    "scale_mag": W_SCALE_MAG * loss_scale_mag.detach(),
                    "normal": 1.0 * loss_normal.detach(),
                    "centroid": centroid_weight * 0.05 * loss_centroid.detach(),
                    "zreg": 0.05 * loss_zreg.detach(),
                    "opacity_sparsity": 1.0 * loss_opacity_sparsity.detach(),
                    "calib_reg": calib_reg_weight * 2.0 * loss_calib_target.detach()
                }

                for k, v in weighted_terms.items():
                    epoch_component_losses[k] += v / args.batch_size

                if global_iter == 0 and sample_idx == 0:
                    audit_terms = {k: float(v) for k, v in weighted_terms.items() if not (args.freeze_calibrator and k == "calib_reg")}
                    print("\n=== LOSS WEIGHT AUDIT (iter 0, first sample) ===")
                    total = sum(audit_terms.values())
                    for name, v in sorted(audit_terms.items(), key=lambda x: -x[1]):
                        print(f"  {name:20s}: contribution={v:10.4f}  ({v/total:.1%} of total)")
                    print(f"  {'TOTAL':20s}: {total:.4f}")
                    print("=" * 60 + "\n")

            clip_params = new_params + base_params + ([] if args.freeze_calibrator else calib_params)
            gnorm = torch.nn.utils.clip_grad_norm_(clip_params, max_norm=float("inf")).item()  # measures only (no clipping)
            if not math.isfinite(gnorm):
                print(f"[WARN] Non-finite gradient norm at iter {global_iter}; skipping this step.")
                optimizer.zero_grad(set_to_none=True)
                epoch_skipped += 1
            else:
                epoch_gnorms.append(gnorm)
                if clip_thresh is None:
                    grad_norm_buf.append(gnorm)
                    if len(grad_norm_buf) >= 200:
                        clip_thresh = 3.0 * float(np.median(grad_norm_buf))
                        print(f"\n[CLIP] Threshold set to {clip_thresh:.4f} (3x median grad norm of first 200 steps)")
                elif gnorm > clip_thresh:
                    torch.nn.utils.clip_grad_norm_(clip_params, max_norm=clip_thresh)
                    epoch_clipped += 1
                optimizer.step()
                ema_update(ema_upsampler, upsampler, args.ema_decay, copy_weights=(global_iter < args.ema_start))
                ema_update(ema_decoder, decoder, args.ema_decay, copy_weights=(global_iter < args.ema_start))
            epoch_total_loss += batch_total_loss

        # --- Aggregate and Log Per-Epoch Stats ---
        avg_epoch_loss = float(epoch_total_loss) / len(train_dataloader)
        train_log_path = os.path.join(args.checkpoint_dir, "training_log.csv")
        write_header = not os.path.exists(train_log_path)
        with open(train_log_path, "a") as f:
            if write_header:
                f.write("epoch,global_iter,avg_epoch_loss,last_batch_nv_grad_mag,last_batch_off_screen_frac\n")
            f.write(f"{epoch},{global_iter},{avg_epoch_loss:.6f},{last_batch_metrics[0]:.10f},{last_batch_metrics[1]:.6f}\n")

        gn_path = os.path.join(args.checkpoint_dir, "grad_clip_log.csv")
        gn_header = not os.path.exists(gn_path)
        mean_gn = sum(epoch_gnorms) / max(1, len(epoch_gnorms))
        max_gn = max(epoch_gnorms) if epoch_gnorms else 0.0
        with open(gn_path, "a") as f:
            if gn_header:
                f.write("epoch,clip_thresh,mean_grad_norm,max_grad_norm,n_clipped,n_skipped\n")
            f.write(f"{epoch},{clip_thresh if clip_thresh is not None else float('nan')},{mean_gn:.4f},{max_gn:.4f},{epoch_clipped},{epoch_skipped}\n")
        print(f"[GRAD] epoch {epoch}: mean norm {mean_gn:.4f} | max {max_gn:.4f} | clipped {epoch_clipped} | skipped {epoch_skipped} | threshold {clip_thresh}")

        # Log per-component loss averages
        avg_components = {k: float(v) / len(train_dataloader) for k, v in epoch_component_losses.items()}
        comp_log_path = os.path.join(args.checkpoint_dir, "loss_components_log.csv")
        comp_write_header = not os.path.exists(comp_log_path)
        with open(comp_log_path, "a") as f:
            headers = ["rgb", "ssim", "depth", "extrusion", "aniso", "scale_mag", "normal", "novel_view", "centroid", "zreg", "opacity_sparsity", "calib_reg"]
            if comp_write_header:
                f.write("epoch,global_iter," + ",".join(headers) + "\n")
            f.write(f"{epoch},{global_iter}," + ",".join(f"{avg_components[k]:.6f}" for k in headers) + "\n")

        # --- Periodic Visualizations ---
        if args.viz_every > 0 and epoch % args.viz_every == 0:
            upsampler.eval()
            decoder.eval()
            with torch.no_grad():
                for vis_batch in fixed_vis_dataloader:
                    data = vis_batch[0]
                    data = move_to_device(data, device)
                    sample_name = os.path.basename(data["dir"])

                    up_feat = upsampler(data["patch_tokens"])
                    p_list, _, _, _, attn_w = decoder(up_feat, data["base_depth"], data["patch_tokens"], rgb148_of(data), data["gt_rgb_A"])
                    m, q, s, o, c = flatten_decoder_outputs_camera_space(
                        p_list[0], p_list[1], p_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
                    )

                    r_colors_A, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_A"], Ks=data["Ks_A"], width=518, height=518)
                    r_colors_B, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_B"], Ks=data["Ks_B"], width=518, height=518)
                    r_colors_C, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_C"], Ks=data["Ks_C"], width=518, height=518)
                    r_colors_D, _, _ = rasterization(means=m, quats=q, scales=s, opacities=o, colors=c, viewmats=data["viewmats_D"], Ks=data["Ks_D"], width=518, height=518)

                    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

                    for i, view in enumerate(['A', 'B', 'C', 'D']):
                        mask_v = data[f"mask_518_{view}"][0].permute(1, 2, 0).cpu().numpy()
                        gt_v = data[f"gt_rgb_{view}"][0].permute(1, 2, 0).cpu().numpy() * mask_v
                        if view == 'A':
                            r_v = (r_colors_A.permute(0,3,1,2) * data[f"mask_518_{view}"])[0].permute(1,2,0).detach().cpu().numpy()
                        elif view == 'B':
                            r_v = (r_colors_B.permute(0,3,1,2) * data[f"mask_518_{view}"])[0].permute(1,2,0).detach().cpu().numpy()
                        elif view == 'C':
                            r_v = (r_colors_C.permute(0,3,1,2) * data[f"mask_518_{view}"])[0].permute(1,2,0).detach().cpu().numpy()
                        else:
                            r_v = (r_colors_D.permute(0,3,1,2) * data[f"mask_518_{view}"])[0].permute(1,2,0).detach().cpu().numpy()

                        axes[0, i].imshow(gt_v)
                        axes[0, i].set_title(f"GT Camera {view}")
                        axes[0, i].axis('off')

                        axes[1, i].imshow(r_v.clip(0, 1))
                        axes[1, i].set_title(f"Render {view} (Epoch {epoch})")
                        axes[1, i].axis('off')

                    plt.tight_layout()
                    plt.savefig(os.path.join(args.checkpoint_dir, f"{sample_name}_render_ep{epoch}.png"), dpi=150)
                    plt.close(fig)

                    # Attention visualization
                    fig_attn, ax_attn = plt.subplots(figsize=(5, 5))
                    im = ax_attn.imshow(attn_w[0, 0].view(37, 37).cpu().numpy(), cmap='viridis')
                    ax_attn.set_title(f"Attention (Epoch {epoch})")
                    fig_attn.colorbar(im, ax=ax_attn)
                    plt.savefig(os.path.join(args.checkpoint_dir, f"{sample_name}_attn_ep{epoch}.png"), dpi=150)
                    plt.close(fig_attn)

            set_train_mode(upsampler, decoder)

        # --- Checkpoint Saving ---
        checkpoint_path = os.path.join(args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save({
            'upsampler': upsampler.state_dict(),
            'decoder': decoder.state_dict(),
            'ema_upsampler': ema_upsampler.state_dict(),
            'ema_decoder': ema_decoder.state_dict(),
            'clip_thresh': clip_thresh,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'global_iter': global_iter
        }, checkpoint_path)

        # --- Evaluate (raw weights and EMA weights) ---
        v_se, v_te, v_ssim, v_ps, v_psnr, v_apsnr = evaluate(eval_dataloader, upsampler, decoder, device)
        e_se, e_te, e_ssim, e_ps, e_psnr, e_apsnr = evaluate(eval_dataloader, ema_upsampler, ema_decoder, device, restore_train=False)
        t_se, t_te, t_ssim, _, t_psnr, t_apsnr = evaluate(fixed_train_eval_dataloader, upsampler, decoder, device)

        print(f"\n[EPOCH {epoch} EVAL raw] Val ScaleErr: {v_se:.4f} | ShiftErr: {v_te:.4f} | SSIM: {v_ssim:.4f} | PSNR: {v_psnr:.2f} | A-PSNR: {v_apsnr:.2f}")
        print(f"[EPOCH {epoch} EVAL EMA] Val ScaleErr: {e_se:.4f} | ShiftErr: {e_te:.4f} | SSIM: {e_ssim:.4f} | PSNR: {e_psnr:.2f} | A-PSNR: {e_apsnr:.2f}")
        print(f"[EPOCH {epoch} TRAIN raw] SSIM: {t_ssim:.4f} | PSNR: {t_psnr:.2f} | A-PSNR: {t_apsnr:.2f}")

        train_vs_val_csv = os.path.join(args.checkpoint_dir, "train_vs_val_history.csv")
        tv_write_header = not os.path.exists(train_vs_val_csv)
        with open(train_vs_val_csv, "a") as f:
            if tv_write_header:
                f.write("epoch,train_scale_err,train_shift_err,val_scale_err,val_shift_err,train_ssim,val_ssim,train_psnr,val_psnr,train_A_psnr,val_A_psnr\n")
            f.write(f"{epoch},{t_se:.6f},{t_te:.6f},{v_se:.6f},{v_te:.6f},{t_ssim:.6f},{v_ssim:.6f},{t_psnr:.4f},{v_psnr:.4f},{t_apsnr:.4f},{v_apsnr:.4f}\n")

        # Log ALL validation samples (raw weights)
        per_sample_csv = os.path.join(args.checkpoint_dir, "eval_per_sample_history.csv")
        ps_write_header = not os.path.exists(per_sample_csv)
        with open(per_sample_csv, "a") as f:
            if ps_write_header:
                f.write("epoch,sample,pred_scale,target_scale,pred_shift,target_shift,attn_max,attn_std\n")
            for r in v_ps:
                f.write(f"{epoch},{r['sample']},{r['pred_scale']:.6f},{r['target_scale']:.6f},{r['pred_shift']:.6f},{r['target_shift']:.6f},{r['attn_max']:.6f},{r['attn_std']:.6f}\n")

        eval_csv = os.path.join(args.checkpoint_dir, "eval_history.csv")
        append_eval_row(eval_csv, epoch, "raw", v_se, v_te, v_ps, v_ssim, v_psnr, v_apsnr)
        append_eval_row(eval_csv, epoch, "ema", e_se, e_te, e_ps, e_ssim, e_psnr, e_apsnr)

        print(f"[!] End of Epoch {epoch} -> Checkpoint saved to Drive.\n")

    print(f"\n[SUCCESS] Stage 2 RGB-skip finetune complete ({num_epochs} epochs)!")

    if args.select_best_at_end:
        eval_csv = os.path.join(args.checkpoint_dir, "eval_history.csv")
        best_epoch, best_weights = select_best_checkpoint_stage2(eval_csv)
        keys = "'ema_upsampler' / 'ema_decoder'" if best_weights == "ema" else "'upsampler' / 'decoder'"
        print(f"[FINAL] Best: epoch {best_epoch}, {best_weights} weights -> "
              f"{os.path.join(args.checkpoint_dir, f'checkpoint_epoch_{best_epoch}.pt')} (use the keys {keys})")

if __name__ == "__main__":
    main()