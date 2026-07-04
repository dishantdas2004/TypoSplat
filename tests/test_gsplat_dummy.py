"""
gsplat Dummy Forward Pass Verification
======================================
Tests the raw gsplat.rasterization API with dummy tensors to confirm 
expected input/output shapes and backend compatibility before wiring 
real model predictions.
"""

import torch
from gsplat import rasterization

def run_gsplat_smoke_test():
    # Attempting to run on Mac (MPS or CPU), but gsplat may demand CUDA.
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n=== gsplat Dummy Forward Pass Verification ===")
    print(f"Executing on: {device}")

    N = 100  # Number of dummy Gaussians
    C = 1    # Number of cameras
    H, W = 518, 518

    # 1. Construct Dummy Gaussians
    means = torch.randn(N, 3, device=device)
    
    # Quaternions must be normalized (w, x, y, z)
    quats = torch.randn(N, 4, device=device)
    quats = quats / quats.norm(dim=-1, keepdim=True)
    
    # Scales (kept small to avoid filling the whole screen with 1 Gaussian)
    scales = torch.rand(N, 3, device=device) * 0.1
    
    # Opacities [0, 1]
    opacities = torch.rand(N, device=device)
    
    # Colors (RGB format) [0, 1]
    colors = torch.rand(N, 3, device=device)

    # 2. Construct Dummy Camera (Identity viewmat)
    # gsplat expects viewmats of shape [C, 4, 4]
    viewmats = torch.eye(4, device=device).unsqueeze(0)
    
    # 3. Construct Dummy Intrinsics
    # Using the standard TypoSplat focal length and optical center
    Ks = torch.tensor([
        [[719.4, 0.0,   259.0],
         [0.0,   719.4, 259.0],
         [0.0,   0.0,   1.0  ]]
    ], device=device)

    print("\n--- Generated Input Tensors ---")
    print(f"Means:     {means.shape} | dtype: {means.dtype}")
    print(f"Quats:     {quats.shape} | dtype: {quats.dtype}")
    print(f"Scales:    {scales.shape} | dtype: {scales.dtype}")
    print(f"Opacities: {opacities.shape} | dtype: {opacities.dtype}")
    print(f"Colors:    {colors.shape} | dtype: {colors.dtype}")
    print(f"Viewmats:  {viewmats.shape} | dtype: {viewmats.dtype}")
    print(f"Ks:        {Ks.shape} | dtype: {Ks.dtype}")

    # 4. Execute the Rasterizer
    try:
        print("\nCalling gsplat.rasterization()...")
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=W,
            height=H,
        )
        
        print("\n=== Output Verification ===")
        print(f"Rendered RGB shape:   {render_colors.shape} (Expected: [{C}, {H}, {W}, 3])")
        print(f"Rendered Alpha shape: {render_alphas.shape}")
        
        assert render_colors.shape == (C, H, W, 3), f"Shape mismatch! Got {render_colors.shape}"
        assert not torch.isnan(render_colors).any(), "NaNs detected in rendered RGB!"
        
        print("\n[SUCCESS] gsplat dummy forward pass completed perfectly.")
        
    except Exception as e:
        print(f"\n[CRASH] gsplat encountered an error:\n{e}")
        print("\n--- DIAGNOSTIC NOTE ---")
        print("If the error mentions a missing backend, CUDA constraint, or NotImplementedError,")
        print("your Mac cannot run gsplat's C++/CUDA kernels locally. You will need to execute")
        print("this test script in your Colab environment.")

if __name__ == "__main__":
    run_gsplat_smoke_test()