"""
VGGTWrapper
===========
Thin wrapper around the pretrained VGGT model.

Key facts confirmed from source:
- embed_dim = 1024 (ViT-L).
- Head input dim is 2 * embed_dim = 2048. Decoder consumes 2048-dim features.
- self.aggregator(images) returns (aggregated_tokens_list, patch_start_idx).
- depth output shape natively is [B, S, H, W, 1] -- converted to [B, 1, H, W] here.
- Camera output uses pose_encoding_to_extri_intri to get usable matrices.
"""

import torch
import torch.nn as nn

from vggt.models.vggt import VGGT
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


class VGGTWrapper(nn.Module):
    def __init__(self, pretrained_name="facebook/VGGT-1B", image_size=518):
        super().__init__()
        self.image_size = image_size

        # Load pretrained weights via HuggingFace hub mixin
        self.vggt = VGGT.from_pretrained(pretrained_name)

        # Freeze everything by default for Stages 1-3.
        for param in self.vggt.parameters():
            param.requires_grad = False

    def set_dpt_trainable(self, trainable: bool):
        """
        Selectively unfreezes the contiguous refinenet fusion blocks and 
        final output convolutions for Stage 4 fine-tuning. Keeps structural 
        scaffolding (projects, resize_layers) completely frozen.
        """
        count = 0
        for name, param in self.vggt.named_parameters():
            if "scratch.refinenet" in name or "scratch.output_conv" in name:
                param.requires_grad = trainable
                count += 1
        
        status = "UNFROZEN (Trainable)" if trainable else "FROZEN"
        print(f"[STAGE 4 CONFIG] Set {count} parameters in DPT fusion/output to {status}")

    def forward(self, image):
        """
        image: [B, 3, 518, 518] float32, values in [0, 1] (single-view input)

        Returns dict with:
          depth:         [B, 1, 518, 518] float32 
          depth_conf:    [B, 518, 518] float32
          extrinsic:     [B, 3, 4] float32 (OpenCV convention)
          intrinsic:     [B, 3, 3] float32
        """
        # Insert S=1 sequence dim ourselves
        if image.dim() == 4:
            image = image.unsqueeze(1)  # [B, 1, 3, H, W]

        predictions = self.vggt(image)

        # depth: [B, S, H, W, 1] -> squeeze S=1 -> permute to [B, 1, H, W]
        depth = predictions["depth"]  
        depth = depth.squeeze(1).permute(0, 3, 1, 2)  

        depth_conf = predictions["depth_conf"].squeeze(1)  

        # Camera: decode pose_enc into usable extrinsic/intrinsic matrices
        pose_enc = predictions["pose_enc"]  # [B, S, 9]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(
            pose_enc, image.shape[-2:]
        )
        extrinsic = extrinsic.squeeze(1)  # [B, 3, 4]
        intrinsic = intrinsic.squeeze(1)  # [B, 3, 3]

        return {
            "depth": depth,
            "depth_conf": depth_conf,
            "extrinsic": extrinsic,
            "intrinsic": intrinsic,
        }

    def forward_with_features(self, image):
        """
        Exposes intermediate tokens for the Gaussian decoder.

        Returns everything from forward() PLUS:
          patch_tokens:    [B, 2048, 37, 37] float32 -- 2D spatial feature map
          patch_start_idx: int
        """
        if image.dim() == 4:
            image_seq = image.unsqueeze(1)  # [B, 1, 3, H, W]
        else:
            image_seq = image

        aggregated_tokens_list, patch_start_idx = self.vggt.aggregator(image_seq)

        # Use the LAST layer's tokens as the decoder's feature input.
        last_layer_tokens = aggregated_tokens_list[-1]  # [B, S, T, 2*embed_dim]
        last_layer_tokens = last_layer_tokens.squeeze(1)  # [B, T, 2048]

        # Drop camera/register tokens
        patch_tokens_1d = last_layer_tokens[:, patch_start_idx:, :]  # [B, 1369, 2048]
        
        # Unflatten 1D patch sequence back to a 2D spatial grid [B, C, H, W]
        B, T, C = patch_tokens_1d.shape
        grid_size = int(T ** 0.5) # sqrt(1369) = 37
        patch_features_2d = patch_tokens_1d.permute(0, 2, 1).view(B, C, grid_size, grid_size) # [B, 2048, 37, 37]

        out = self.forward(image)
        out["patch_tokens"] = patch_features_2d 
        out["patch_start_idx"] = patch_start_idx
        return out


if __name__ == "__main__":
    """
    Smoke test -- run on CPU or MPS.
    """
    import sys
    from PIL import Image
    import numpy as np

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    wrapper = VGGTWrapper().to(device)
    wrapper.eval()

    img_path = sys.argv[1] if len(sys.argv) > 1 else "./output/0/view.png"
    
    # Gracefully exit if test image doesn't exist yet
    import os
    if not os.path.exists(img_path):
        print(f"[SKIP] Smoke test image {img_path} not found. Wrapper compiled successfully.")
        sys.exit(0)

    img = Image.open(img_path).convert("RGB").resize((518, 518))
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        out = wrapper.forward_with_features(img_tensor)

    print("\n=== VGGT Wrapper Output Check ===")
    for k, v in out.items():
        if torch.is_tensor(v):
            has_nan = torch.isnan(v).any().item()
            print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} has_nan={has_nan}")
        else:
            print(f"  {k}: {v}")

    assert out["patch_tokens"].shape == (1, 2048, 37, 37), \
        f"Expected [1, 2048, 37, 37], got {out['patch_tokens'].shape}"
    assert out["depth"].shape == (1, 1, 518, 518), \
        f"Expected [1, 1, 518, 518], got {out['depth'].shape}"

    print("\nShape checks passed.")