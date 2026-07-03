"""
TypoSplat Upsampler
===================
Upsamples the 37x37 VGGT patch tokens to a 148x148 anchor grid using a 4x 
PixelShuffle. Injects a normalized 2D coordinate grid (-1 to 1) to prevent 
feature cloning and provide explicit spatial grounding for the Gaussian heads.
"""

import torch
import torch.nn as nn

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


if __name__ == "__main__":
    """
    Smoke Test for TypoSplatUpsampler
    Run this via: python src/models/upsampler.py
    """
    print("\n=== TypoSplat Upsampler Smoke Test ===")
    
    # Test on M1 MPS if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Executing on: {device}\n")
    
    # 1. Initialize module
    in_dim = 2048
    out_dim = 256
    upsampler = TypoSplatUpsampler(in_channels=in_dim, out_channels=out_dim).to(device)
    
    # 2. Create dummy VGGT output [B, 2048, 37, 37]
    B, H, W = 2, 37, 37
    dummy_input = torch.randn(B, in_dim, H, W, device=device)
    print(f"Input shape (VGGT patch_tokens): {list(dummy_input.shape)}")
    
    # 3. Forward pass
    output = upsampler(dummy_input)
    print(f"Output shape (For Decoder):      {list(output.shape)}")
    
    # 4. Assertions
    expected_shape = [B, out_dim + 2, H * 4, W * 4] # [2, 258, 148, 148]
    assert list(output.shape) == expected_shape, f"Shape mismatch! Expected {expected_shape}, got {list(output.shape)}"
    
    assert not torch.isnan(output).any(), "NaNs detected in output!"
    
    # 5. Verify Positional Coordinates vary correctly
    # X coordinates are at channel index -2, Y coordinates at -1
    x_channel = output[0, -2, :, :]
    y_channel = output[0, -1, :, :]
    
    x_varies_horizontally = not torch.allclose(x_channel[:, 0], x_channel[:, -1])
    y_varies_vertically = not torch.allclose(y_channel[0, :], y_channel[-1, :])
    
    print("\n=== Coordinate Injection Checks ===")
    print(f"X-coordinates span: [{x_channel.min().item():.2f} to {x_channel.max().item():.2f}] | Varies Horizontally: {x_varies_horizontally}")
    print(f"Y-coordinates span: [{y_channel.min().item():.2f} to {y_channel.max().item():.2f}] | Varies Vertically:   {y_varies_vertically}")
    
    assert x_varies_horizontally and y_varies_vertically, "Positional injection failed: coordinates are static!"
    print("\n[SUCCESS] Upsampler is mathematically sound and ready for the Decoder.")