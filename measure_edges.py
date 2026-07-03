import glob
import json
import numpy as np
import torch
import torch.nn.functional as F
import os
import math
import OpenEXR
import Imath

def load_exr_depth(filepath):
    """Robust EXR loader matching your dataset.py implementation."""
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = header['channels'].keys()
    
    channel_name = next((c for c in ['V', 'Z', 'R'] if c in channels), list(channels)[0])
    
    depth_str = exr_file.channel(channel_name, FLOAT)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape(height, width)
    return depth

def apply_sobel(depth_tensor):
    """
    Applies a standard 3x3 Sobel filter to compute gradient magnitude.
    depth_tensor: [1, 1, H, W]
    """
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    grad_x = F.conv2d(depth_tensor, sobel_x, padding=1)
    grad_y = F.conv2d(depth_tensor, sobel_y, padding=1)
    
    grad_mag = torch.sqrt(grad_x**2 + grad_y**2)
    return grad_mag

def analyze_depth_gradients(sample_dir):
    """Loads a depth map, masks the background safely, and computes edge magnitudes."""
    meta_path = os.path.join(sample_dir, "metadata.json")
    depth_files = glob.glob(os.path.join(sample_dir, "*depth*.exr"))
    
    if not os.path.exists(meta_path) or not depth_files:
        return None
        
    depth_path = depth_files[0]
        
    with open(meta_path, 'r') as f:
        meta = json.load(f)
        
    depth_np = load_exr_depth(depth_path)
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).float()
    
    # --- RIGOROUS CAMERA DERIVATION ---
    x, y, z = meta["camera_translation"]
    
    radius = math.sqrt(x**2 + y**2 + z**2)
    elevation_deg = math.degrees(math.asin(z / radius))
    azimuth_deg = math.degrees(math.atan2(y, x))
    
    # Ensure azimuth is positive [0, 360]
    if azimuth_deg < 0:
        azimuth_deg += 360
        
    # Case 1 constraints: Elevation within +/- 15 deg, Azimuth [70, 110] deg
    if -16 <= elevation_deg <= 16 and 69 <= azimuth_deg <= 111:
        cam_case = "FRONTAL (Case 1)"
    else:
        cam_case = "OBLIQUE/STEEP (Cases 2-4)"
    
    # --- CRITICAL FIX: Mitigate Background BEFORE Sobel ---
    bg_mask = depth_tensor >= 50.0
    
    valid_depths = depth_tensor[~bg_mask]
    if valid_depths.numel() == 0:
        return None
    mean_valid_depth = valid_depths.mean()
    
    # Replace the 100.0 background with the mean valid depth
    safe_depth = torch.where(bg_mask, mean_valid_depth, depth_tensor)
    
    # Run Sobel on the safe depth map
    grad_mag = apply_sobel(safe_depth)
    
    # Extract Percentiles strictly inside the valid geometry mask
    valid_grads = grad_mag[~bg_mask]
    p99 = torch.quantile(valid_grads, 0.99).item()
    
    return {
        "p99": p99,
        "cam_case": cam_case
    }

if __name__ == "__main__":
    sample_dirs = glob.glob('/Users/dishantdas/3d-typography-generator/output/*')
    
    def extract_id(d):
        try:
            return int(os.path.basename(d))
        except:
            return 999999
            
    # Process the first 1000 valid folders
    sample_dirs = sorted([d for d in sample_dirs if os.path.isdir(d)], key=extract_id)[:1000]
    
    results = {"FRONTAL (Case 1)": [], "OBLIQUE/STEEP (Cases 2-4)": []}
    
    for d in sample_dirs:
        res = analyze_depth_gradients(d)
        if res:
            results[res["cam_case"]].append(res["p99"])
            
    print("\n=== Edge Magnitude Analysis (99th Percentile) ===")
    
    frontal_key = "FRONTAL (Case 1)"
    oblique_key = "OBLIQUE/STEEP (Cases 2-4)"
    
    if results[frontal_key]:
        print(f"Frontal Samples: {len(results[frontal_key])} | Avg P99: {np.mean(results[frontal_key]):.4f}")
    if results[oblique_key]:
        print(f"Oblique Samples: {len(results[oblique_key])} | Avg P99: {np.mean(results[oblique_key]):.4f}")
    
    all_p99 = results[frontal_key] + results[oblique_key]
    if all_p99:
        print(f"Global Avg P99: {np.mean(all_p99):.4f}")