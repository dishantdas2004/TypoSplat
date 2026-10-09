"""
TypoSplat inference on a test set (Google Colab)
================================================
Runs a finetuned TypoSplat checkpoint on every sample in a test folder and saves, per sample,
ONE image containing the input view (A) and the three novel views (B, C, D): ground truth on the
top row, prediction on the bottom row, with PSNR / SSIM printed on every view and the average
PSNR / SSIM of the three novel views in the figure title. Also writes metrics.csv and summary.txt.

Metrics are the SAME as the training-time evaluation: computed inside the letter mask only
(prediction clamped to [0,1] for PSNR; masked dycheck SSIM, window 7, coverage threshold 0.3).

Requirements (Colab):  pip install gsplat==1.5.3 matplotlib pillow numpy tqdm   (torch is preinstalled)
Each test sample folder must contain: metadata.json, *view_A/B/C/D*.png and cached_features.pt
(the output of cache_features_1400.py for that folder). No VGGT or repo files are needed here.

NOTE ON THE MASK: exactly as in training/validation, the predicted opacities are multiplied by the
ground-truth letter mask of view A (mask_148_A from cached_features.pt). Results are therefore
comparable to your validation numbers, but they are not mask-free.
"""

import os
import sys
import glob
import json
import math
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gsplat import rasterization

# ==========================================================================
# Model definitions (copied verbatim from upsampler.py / decoder.py / the finetune script)
# ==========================================================================

def icnr_init(tensor, upscale_factor=4, initializer=nn.init.kaiming_normal_):
    """
    ICNR (Initialized Convolution Neural Network Resize) Initialization.
    Ensures that the weights for the r^2 sub-pixels are perfectly identical 
    at initialization. This mathematically prevents checkerboard artifacts 
    before the network has time to learn.
    """
    out_c_r2, in_c, k_h, k_w = tensor.shape
    out_c = out_c_r2 // (upscale_factor ** 2)
    
    # Initialize a smaller tensor representing the base channels
    base_tensor = torch.empty(out_c, in_c, k_h, k_w)
    initializer(base_tensor)
    
    # Repeat the base weights for all sub-pixels
    repeated_tensor = base_tensor.repeat_interleave(upscale_factor ** 2, dim=0)
    
    with torch.no_grad():
        tensor.copy_(repeated_tensor)


class TypoSplatUpsampler(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, upscale_factor=4):
        """
        in_channels: 2048 (from VGGT DPT-head features)
        out_channels: The base channel dimension for the downstream Gaussian decoder (e.g., 256)
        upscale_factor: 4 (37x37 -> 148x148)
        """
        super().__init__()
        self.upscale_factor = upscale_factor
        self.out_channels = out_channels
        
        # To output 'out_channels' after PixelShuffle(4), the conv needs 
        # out_channels * (4^2) output dimensions.
        conv_out_channels = out_channels * (upscale_factor ** 2)
        
        self.conv = nn.Conv2d(
            in_channels, 
            conv_out_channels, 
            kernel_size=3, 
            padding=1, 
            bias=True
        )
        
        # Apply the anti-checkerboard initialization to weights
        icnr_init(self.conv.weight, upscale_factor=self.upscale_factor)
        
        # CRITICAL FIX: Zero the bias so the sub-channels are truly identical at init
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
        
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
        self.activation = nn.GELU()

    def forward(self, patch_tokens):
        """
        patch_tokens: [B, 2048, 37, 37]
        Returns: [B, out_channels + 2, 148, 148] (e.g., 258 channels)
        """
        B, C, H, W = patch_tokens.shape
        
        # 1. Upsample: [B, 2048, 37, 37] -> [B, out_channels, 148, 148]
        x = self.conv(patch_tokens)
        x = self.pixel_shuffle(x)
        x = self.activation(x)
        
        _, _, out_H, out_W = x.shape
        
        # 2. Generate 2D Coordinate Grid (-1 to 1)
        with torch.no_grad():
            y_coords = torch.linspace(-1, 1, steps=out_H, device=x.device, dtype=x.dtype)
            x_coords = torch.linspace(-1, 1, steps=out_W, device=x.device, dtype=x.dtype)
            
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            
            # Stack into [2, 148, 148] and expand to match Batch size [B, 2, 148, 148]
            grid = torch.stack([grid_x, grid_y], dim=0) 
            grid = grid.unsqueeze(0).expand(B, -1, -1, -1)
        
        # 3. Inject: Concatenate the base features with the spatial coordinates
        # Output shape: [B, out_channels + 2, 148, 148]
        out = torch.cat([x, grid], dim=1)
        
        return out


class DepthCalibrator(nn.Module):
    """
    Predicts a per-sample (scale, shift) correction for VGGT's depth, since
    VGGT normalizes depth per-scene (not true metric scale) — a single
    dataset-wide constant would not generalize across samples with
    different scene geometry.
    Uses attention pooling instead of global average pooling: a learnable
    query token attends over the full 37x37 spatial patch grid, so the
    network can learn to search for and weight the specific spatial
    locations (ground-plane, structural anchors) that actually carry
    scale information, rather than averaging that information away.
    """
    def __init__(self, feature_dim=2048, num_heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, feature_dim) * 0.02)
        self.attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads, batch_first=True)
        self.fc = nn.Linear(feature_dim, 2)
        
        # Zero-init both weight and bias -> exact identity at iteration 0
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, patch_tokens):
        # patch_tokens: [B, 2048, 37, 37] -> flatten spatial dims for attention
        B, C, H, W = patch_tokens.shape
        tokens_flat = patch_tokens.view(B, C, H * W).permute(0, 2, 1)  # [B, 1369, 2048]
        q = self.query.expand(B, -1, -1)  # [B, 1, 2048]
        
        pooled, attn_weights = self.attn(q, tokens_flat, tokens_flat)  # [B, 1, 2048], [B, 1, 1369]
        pooled = pooled.squeeze(1)  # [B, 2048]
        
        out = self.fc(pooled)
        scale = 1.0 + out[:, 0:1]
        shift = out[:, 1:2]
        return scale, shift, out, attn_weights

class GaussianParameterHead(nn.Module):
    def __init__(self, in_channels, scale_bias=-5.0, target_total_extrusion=0.1124): 
        super().__init__()
        self.scale_bias = scale_bias
        
        # Outputs per anchor: 14 channels minimum
        self.head = nn.Conv2d(in_channels, 14, kernel_size=3, padding=1)
        
        # Safe Initialization
        nn.init.zeros_(self.head.bias)
        
        # 1. Identity Rotation (w = 1.0)
        self.head.bias.data[7] = 1.0 
        
        # 2. Cumulative Z-Offset Bias
        per_layer_offset = target_total_extrusion / 2.0
        self.head.bias.data[2] = math.log(per_layer_offset)

    def forward(self, x):
        out = self.head(x)
        
        xy_raw      = out[:, 0:2, :, :]
        z_raw       = out[:, 2:3, :, :]
        opacity_raw = out[:, 3:4, :, :]
        scale_raw   = out[:, 4:7, :, :]
        rot_raw     = out[:, 7:11, :, :]
        sh_dc       = out[:, 11:14, :, :]
        
        # Activations
        xy_offset = 3.5 * torch.tanh(xy_raw)
        z_offset = torch.exp(torch.clamp(z_raw, min=-10.0, max=6.0))
        opacity = torch.sigmoid(opacity_raw)
        scale = torch.exp(torch.clamp(scale_raw + self.scale_bias, max=math.log(0.06)))
        rot = F.normalize(rot_raw, p=2, dim=1)
        
        return {
            "xy_offset": xy_offset,
            "z_offset": z_offset,
            "opacity": opacity,
            "scale": scale,
            "rot": rot,
            "sh_dc": sh_dc
        }

class TypoSplatDecoder(nn.Module):
    def __init__(self, in_channels=258, trunk_channels=256):
        super().__init__()
        
        # Shared trunk for Observed Geometry (Front Face & Sides)
        self.trunk_0_1 = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.head_0 = GaussianParameterHead(trunk_channels)
        self.head_1 = GaussianParameterHead(trunk_channels)
        
        # Independent trunk for Inferred Geometry (Back Face)
        self.trunk_2 = nn.Sequential(
            nn.Conv2d(in_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(trunk_channels, trunk_channels, kernel_size=3, padding=1),
            nn.GELU()
        )
        self.head_2 = GaussianParameterHead(trunk_channels)
        
        # Global depth calibration to pull Gaussians into metric space
        self.calibrator = DepthCalibrator(feature_dim=2048)

    def forward(self, features, base_depth_518, patch_tokens):
        """
        features: [B, 258, 148, 148] from TypoSplatUpsampler
        base_depth_518: [B, 1, 518, 518] frozen VGGT base depth
        patch_tokens: [B, 2048, 37, 37] from VGGTWrapper
        """
        # 1. Feature Extraction
        feat_0_1 = self.trunk_0_1(features)
        feat_2   = self.trunk_2(features)
        
        # 2. Predict raw Gaussian parameters
        params_0 = self.head_0(feat_0_1)
        params_1 = self.head_1(feat_0_1)
        params_2 = self.head_2(feat_2)
        
        # 3. Downsample VGGT depth to match the anchor grid resolution
        base_depth_148 = F.interpolate(
            base_depth_518, 
            size=(148, 148), 
            mode='bilinear', 
            align_corners=False
        )
        
        scale, shift, raw_calib_out, attn_weights = self.calibrator(patch_tokens)
        scale_view = scale.view(-1, 1, 1, 1)
        shift_view = shift.view(-1, 1, 1, 1)
        
        base_depth_148 = scale_view * base_depth_148 + shift_view
        
        params_0["true_depth"] = base_depth_148
        params_1["true_depth"] = base_depth_148 + params_1["z_offset"]
        params_2["true_depth"] = base_depth_148 + params_1["z_offset"] + params_2["z_offset"]
        
        return [params_0, params_1, params_2], scale, shift, raw_calib_out, attn_weights


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


# ==========================================================================
# Helpers (copied verbatim from the training script)
# ==========================================================================

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

def _get_relative_viewmat(c2w_A_list, c2w_B_list, device):
    """Helper to calculate OpenCV viewmat from Camera A to Camera B."""
    c2w_A_blender = torch.tensor(c2w_A_list, dtype=torch.float32, device=device)
    c2w_B_blender = torch.tensor(c2w_B_list, dtype=torch.float32, device=device)
    
    S = torch.tensor([
        [1,  0,  0, 0],
        [0, -1,  0, 0],
        [0,  0, -1, 0],
        [0,  0,  0, 1]
    ], dtype=torch.float32, device=device)
    
    c2w_A_cv = c2w_A_blender @ S
    c2w_B_cv = c2w_B_blender @ S
    
    w2c_B_cv = torch.linalg.inv(c2w_B_cv)
    return (w2c_B_cv @ c2w_A_cv).unsqueeze(0)


# ==========================================================================
# Inference
# ==========================================================================

def load_image_tensor(path):
    img = Image.open(path).convert("RGB").resize((518, 518))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.copy()).permute(2, 0, 1).unsqueeze(0)  # [1,3,518,518]


def cache_file(sample_dir, cache_dir=None):
    """cached_features.pt location: <cache_dir>/<sample_name>/ if --cache_dir is given, else inside the sample folder."""
    base = os.path.join(cache_dir, os.path.basename(sample_dir)) if cache_dir else sample_dir
    return os.path.join(base, "cached_features.pt")


def load_sample(sample_dir, cache_dir=None):
    """Same tensors as TypoSplatDataset.__getitem__, minus the depth EXR / diagnostics CSV (not needed)."""
    with open(os.path.join(sample_dir, "metadata.json"), "r") as f:
        meta = json.load(f)

    paths = {}
    for v in "ABCD":
        found = sorted(glob.glob(os.path.join(sample_dir, f"*view_{v}*.png")))
        if not found:
            raise FileNotFoundError(f"No *view_{v}*.png in {sample_dir}")
        paths[v] = found[0]

    cache_path = cache_file(sample_dir, cache_dir)
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"cached_features.pt missing: {cache_path}")
    cached = torch.load(cache_path, map_location="cpu", weights_only=True)
    for k in ("patch_tokens", "base_depth", "mask_148_A", "mask_148_B", "mask_148_C", "mask_148_D"):
        if k not in cached:
            raise KeyError(f"cached_features.pt in {sample_dir} is missing '{k}'")

    data = {
        "dir": sample_dir,
        "patch_tokens": cached["patch_tokens"].float(),
        "base_depth": cached["base_depth"].float(),
        "mask_148_A": cached["mask_148_A"].float(),
        "intrinsics_tuple_A": (meta["fx"], meta["fy"], meta["cx"], meta["cy"]),
    }
    for v in "ABCD":
        m148 = cached[f"mask_148_{v}"].float()
        data[f"mask_518_{v}"] = F.interpolate(m148, size=(518, 518), mode="nearest")
        data[f"gt_rgb_{v}"] = load_image_tensor(paths[v])

    data["Ks_A"] = torch.tensor([[[meta["fx"], 0, meta["cx"]], [0, meta["fy"], meta["cy"]], [0, 0, 1]]], dtype=torch.float32)
    data["viewmats_A"] = torch.eye(4).unsqueeze(0)
    for v in "BCD":
        cam = meta[f"camera_{v}"]
        data[f"Ks_{v}"] = torch.tensor([[[cam["fx"], 0, cam["cx"]], [0, cam["fy"], cam["cy"]], [0, 0, 1]]], dtype=torch.float32)
        data[f"viewmats_{v}"] = _get_relative_viewmat(meta["camera_to_world_matrix"], cam["camera_to_world_matrix"], "cpu")
    return data


def move_to_device(data, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in data.items()}


def load_models(ckpt, tag, device):
    up_key, dec_key = ("ema_upsampler", "ema_decoder") if tag == "ema" else ("upsampler", "decoder")
    for k in (up_key, dec_key):
        if k not in ckpt:
            raise KeyError(f"Checkpoint has no '{k}'. Keys present: {list(ckpt.keys())}")
    upsampler = TypoSplatUpsampler(in_channels=2048, out_channels=256).to(device)
    decoder = TypoSplatDecoderRGB(in_channels=258).to(device)
    upsampler.load_state_dict(ckpt[up_key], strict=True)
    decoder.load_state_dict(ckpt[dec_key], strict=True)
    upsampler.eval()
    decoder.eval()
    return upsampler, decoder


@torch.no_grad()
def predict_and_score(data, upsampler, decoder, device):
    """Returns (renders [4,518,518,3] for A,B,C,D, metrics dict)."""
    feats = upsampler(data["patch_tokens"])
    params_list, calib_scale, calib_shift, _, _ = decoder(
        feats, data["base_depth"], data["patch_tokens"], rgb148_of(data), data["gt_rgb_A"]
    )
    means, quats, scales, opacities, colors = flatten_decoder_outputs_camera_space(
        params_list[0], params_list[1], params_list[2], data["intrinsics_tuple_A"], device, mask_148=data["mask_148_A"]
    )
    viewmats = torch.cat([data[f"viewmats_{v}"] for v in "ABCD"], dim=0)
    Ks = torch.cat([data[f"Ks_{v}"] for v in "ABCD"], dim=0)
    renders, _, _ = rasterization(
        means=means, quats=quats, scales=scales, opacities=opacities, colors=colors,
        viewmats=viewmats, Ks=Ks, width=518, height=518,
    )
    assert renders.shape == (4, 518, 518, 3), f"Unexpected render shape {tuple(renders.shape)}"

    metrics = {}
    for i, v in enumerate("ABCD"):
        pred = renders[i:i + 1].permute(0, 3, 1, 2)             # [1,3,518,518]
        gt = data[f"gt_rgb_{v}"]
        mask = data[f"mask_518_{v}"]
        sq_err = ((pred.clamp(0.0, 1.0) - gt) ** 2 * mask).sum()
        mse = sq_err / (mask.sum() * 3.0 + 1e-8)
        psnr = (-10.0 * torch.log10(mse + 1e-10)).item()
        ssim, _ = dycheck_ssim(pred, gt, mask, window_size=7, cov_thresh=0.3)
        metrics[v] = {"psnr": psnr, "ssim": float(ssim)}
    metrics["novel_psnr"] = sum(metrics[v]["psnr"] for v in "BCD") / 3.0
    metrics["novel_ssim"] = sum(metrics[v]["ssim"] for v in "BCD") / 3.0
    metrics["calib_scale"] = float(calib_scale.item())
    metrics["calib_shift"] = float(calib_shift.item())
    return renders, metrics


def save_figure(path, name, data, renders, metrics, tag, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(20, 11.5), gridspec_kw={"hspace": 0.30, "wspace": 0.05})
    titles = {"A": "Input view A", "B": "Novel view B", "C": "Novel view C", "D": "Novel view D"}
    for i, v in enumerate("ABCD"):
        mask = data[f"mask_518_{v}"][0].permute(1, 2, 0).cpu().numpy()
        gt = (data[f"gt_rgb_{v}"][0].permute(1, 2, 0).cpu().numpy() * mask).clip(0, 1)
        pr = (renders[i].cpu().numpy().clip(0, 1) * mask)
        axes[0, i].imshow(gt)
        axes[0, i].set_title(f"GT - {titles[v]}", fontsize=14)
        axes[1, i].imshow(pr)
        axes[1, i].set_title(f"Prediction - {titles[v]}\nPSNR {metrics[v]['psnr']:.2f} dB | SSIM {metrics[v]['ssim']:.4f}", fontsize=14)
        for r in (0, 1):
            axes[r, i].axis("off")
    fig.suptitle(
        f"Sample {name}  ({tag} weights)\n"
        f"Novel-view average (B, C, D):  PSNR {metrics['novel_psnr']:.2f} dB   |   SSIM {metrics['novel_ssim']:.4f}",
        fontsize=18, fontweight="bold",
    )
    fig.text(0.5, 0.005, "Images and metrics are restricted to the letter region (mask); background shown black.",
             ha="center", fontsize=10, color="gray")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.04)
    plt.savefig(path, dpi=dpi)
    plt.close(fig)


CSV_COLS = ["sample", "A_psnr", "A_ssim", "B_psnr", "B_ssim", "C_psnr", "C_ssim", "D_psnr", "D_ssim",
            "novel_psnr", "novel_ssim", "calib_scale", "calib_shift"]


def metrics_row(name, m):
    vals = [name]
    for v in "ABCD":
        vals += [f"{m[v]['psnr']:.4f}", f"{m[v]['ssim']:.6f}"]
    vals += [f"{m['novel_psnr']:.4f}", f"{m['novel_ssim']:.6f}", f"{m['calib_scale']:.6f}", f"{m['calib_shift']:.6f}"]
    return ",".join(vals)


def read_done(csv_path):
    done = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.read().strip().splitlines()
        for ln in lines[1:]:
            parts = ln.split(",")
            if len(parts) == len(CSV_COLS):
                done[parts[0]] = [float(x) for x in parts[1:]]
    return done


def write_summary(out_dir, tag, ckpt_path, done):
    arr = np.array(list(done.values()), dtype=np.float64)  # columns follow CSV_COLS[1:]
    col = {c: arr[:, i] for i, c in enumerate(CSV_COLS[1:])}
    lines = [f"TypoSplat test-set summary ({tag} weights)", f"checkpoint: {ckpt_path}", f"samples: {len(arr)}", ""]
    for v in "ABCD":
        lines.append(f"View {v}:  PSNR {col[f'{v}_psnr'].mean():.3f} dB | SSIM {col[f'{v}_ssim'].mean():.4f}")
    lines += ["", "Novel views (B,C,D) averaged per sample, then over samples:",
              f"  PSNR {col['novel_psnr'].mean():.3f} dB (std {col['novel_psnr'].std():.3f}) | SSIM {col['novel_ssim'].mean():.4f} (std {col['novel_ssim'].std():.4f})",
              "", "Metrics are masked to the letter region (same protocol as the training-time evaluation)."]
    text = "\n".join(lines)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print("\n" + text + "\n")


def main():
    parser = argparse.ArgumentParser(description="TypoSplat test-set inference (renders + PSNR/SSIM figures)")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint_epoch_N.pt")
    parser.add_argument("--test_dir", required=True, help="Folder with the test sample folders")
    parser.add_argument("--cache_dir", default=None, help="Folder holding <sample>/cached_features.pt (default: inside each sample folder)")
    parser.add_argument("--out_dir", required=True, help="Output folder (e.g. on Google Drive)")
    parser.add_argument("--weights", required=True, choices=["raw", "ema", "both"],
                        help="Which weights in the checkpoint to use. Use the one printed on the [FINAL] line of your training log; 'both' runs each into its own subfolder.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N samples (quick test)")
    parser.add_argument("--dpi", type=int, default=130, help="Figure resolution")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU found. In Colab: Runtime > Change runtime type > GPU.")
        sys.exit(1)
    device = torch.device("cuda")
    # gsplat needs FP32; keep TF32 off so metrics match the training-time evaluation.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    sample_dirs = [os.path.join(args.test_dir, d) for d in os.listdir(args.test_dir)
                   if os.path.isdir(os.path.join(args.test_dir, d)) and os.path.exists(os.path.join(args.test_dir, d, "metadata.json"))]
    sample_dirs.sort(key=lambda p: (0, int(os.path.basename(p))) if os.path.basename(p).isdigit() else (1, os.path.basename(p)))
    if args.limit > 0:
        sample_dirs = sample_dirs[:args.limit]
    if not sample_dirs:
        print(f"ERROR: no sample folders (with metadata.json) in {args.test_dir}")
        sys.exit(1)
    missing_cache = [d for d in sample_dirs if not os.path.exists(cache_file(d, args.cache_dir))]
    if missing_cache:
        print(f"ERROR: {len(missing_cache)} of {len(sample_dirs)} samples have no cached_features.pt (e.g. {missing_cache[0]}).")
        print("Run cache_features_1400.py on the test folder first (needs the repo, VGGT and each sample's mesh.ply).")
        sys.exit(1)

    print(f"Found {len(sample_dirs)} test samples in {args.test_dir}")
    print(f"Loading checkpoint {args.checkpoint}")
    try:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")

    tags = ["raw", "ema"] if args.weights == "both" else [args.weights]
    models, outs, dones = {}, {}, {}
    for tag in tags:
        models[tag] = load_models(ckpt, tag, device)
        outs[tag] = os.path.join(args.out_dir, tag)
        os.makedirs(os.path.join(outs[tag], "images"), exist_ok=True)
        dones[tag] = read_done(os.path.join(outs[tag], "metrics.csv"))
        if dones[tag]:
            print(f"[{tag}] resuming: {len(dones[tag])} samples already done, they will be skipped.")
    del ckpt

    t0 = time.time()
    failed = []
    for sd in tqdm(sample_dirs, desc="Test samples"):
        name = os.path.basename(sd)
        todo = [t for t in tags if not (name in dones[t] and os.path.exists(os.path.join(outs[t], "images", f"{name}.png")))]
        if not todo:
            continue
        try:
            data = move_to_device(load_sample(sd, args.cache_dir), device)
            for tag in todo:
                upsampler, decoder = models[tag]
                renders, m = predict_and_score(data, upsampler, decoder, device)
                save_figure(os.path.join(outs[tag], "images", f"{name}.png"), name, data, renders, m, tag, args.dpi)
                csv_path = os.path.join(outs[tag], "metrics.csv")
                write_header = not os.path.exists(csv_path)
                with open(csv_path, "a") as f:
                    if write_header:
                        f.write(",".join(CSV_COLS) + "\n")
                    f.write(metrics_row(name, m) + "\n")
                dones[tag][name] = [m["A"]["psnr"], m["A"]["ssim"], m["B"]["psnr"], m["B"]["ssim"], m["C"]["psnr"], m["C"]["ssim"],
                                    m["D"]["psnr"], m["D"]["ssim"], m["novel_psnr"], m["novel_ssim"], m["calib_scale"], m["calib_shift"]]
        except Exception as e:
            failed.append((name, str(e)))
            print(f"\n[SKIP] {name}: {e}")

    for tag in tags:
        if dones[tag]:
            write_summary(outs[tag], tag, args.checkpoint, dones[tag])
    if failed:
        print(f"[WARN] {len(failed)} samples failed:")
        for n, e in failed[:20]:
            print(f"   {n}: {e}")
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Outputs in {args.out_dir}")


if __name__ == "__main__":
    main()