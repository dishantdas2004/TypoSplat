"""
TypoSplat Gaussian Decoder
==========================
Takes the 258-channel upsampled feature map (148x148) and predicts the explicitly 
parameterized 3D Gaussian attributes for three depth layers.

Architecture:
- Layer 0 (Front Face) and Layer 1 (Mid/Side) share a convolutional trunk.
- Layer 2 (Back Face) uses an independent trunk due to differing (unobserved) supervision.
- Depth offsets are strictly cumulative (D -> D+d1 -> D+d1+d2) to enforce Z-ordering.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        
        pooled, _ = self.attn(q, tokens_flat, tokens_flat)  # [B, 1, 2048]
        pooled = pooled.squeeze(1)  # [B, 2048]
        
        out = self.fc(pooled)
        scale = 1.0 + out[:, 0:1]
        shift = out[:, 1:2]
        return scale, shift, out

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
        
        scale, shift, raw_calib_out = self.calibrator(patch_tokens)
        scale_view = scale.view(-1, 1, 1, 1)
        shift_view = shift.view(-1, 1, 1, 1)
        
        base_depth_148 = scale_view * base_depth_148 + shift_view
        
        params_0["true_depth"] = base_depth_148
        params_1["true_depth"] = base_depth_148 + params_1["z_offset"]
        params_2["true_depth"] = base_depth_148 + params_1["z_offset"] + params_2["z_offset"]
        
        return [params_0, params_1, params_2], scale, shift, raw_calib_out

if __name__ == "__main__":
    """
    Smoke Test for TypoSplatDecoder
    Run this via: python src/models/decoder.py
    """
    print("\n=== TypoSplat Decoder Smoke Test ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on: {device}\n")
    
    decoder = TypoSplatDecoder(in_channels=258).to(device)
    
    # Dummy outputs from upstream pipeline
    B, H, W = 2, 148, 148
    dummy_features = torch.randn(B, 258, H, W, device=device)
    dummy_base_depth = torch.ones(B, 1, 518, 518, device=device) * 5.0 # Flat 5m depth wall
    dummy_patch_tokens = torch.randn(B, 2048, 37, 37, device=device)
    
    # Forward pass
    layers, scale, shift, raw_calib_out = decoder(dummy_features, dummy_base_depth, dummy_patch_tokens)
    
    print(f"Generated {len(layers)} depth layers.")
    
    # Validate mathematical ordering of the cumulative depth logic
    depth_0 = layers[0]["true_depth"]
    depth_1 = layers[1]["true_depth"]
    depth_2 = layers[2]["true_depth"]
    
    ordering_valid = torch.all((depth_0 <= depth_1) & (depth_1 <= depth_2)).item()
    
    print("\n=== Cumulative Depth Check ===")
    print(f"Layer 0 Mean Depth: {depth_0.mean().item():.3f}m (Expected ~5.0m)")
    print(f"Layer 1 Mean Depth: {depth_1.mean().item():.3f}m")
    print(f"Layer 2 Mean Depth: {depth_2.mean().item():.3f}m")
    print(f"Strictly Increasing Z-Order: {ordering_valid}")
    
    # Validate bounding activations
    xy_bounds = (layers[0]["xy_offset"].min().item() >= -3.5) and (layers[0]["xy_offset"].max().item() <= 3.5)
    rot_norms = torch.norm(layers[0]["rot"], p=2, dim=1)
    rot_normalized = torch.allclose(rot_norms, torch.ones_like(rot_norms))
    
    print("\n=== Activation Constraints ===")
    print(f"XY Offsets bounded to +/- 3.5: {xy_bounds}")
    print(f"Rotations are Unit Quaternions: {rot_normalized}")
    print(f"No NaNs in output: {not torch.isnan(depth_2).any().item()}")
    
    assert ordering_valid, "Cumulative depth logic failed!"
    assert rot_normalized, "Quaternion normalization failed!"
    print("\n[SUCCESS] Decoder passes all topological and constraint checks.")