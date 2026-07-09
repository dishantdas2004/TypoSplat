"""
TypoSplat Diagnostic: The Ultimate Geometry & Regression Check (Batch CSV Mode)
Validates if a 1D Depth Scale/Shift can mathematically project Camera A into Camera B.
Logs VGGT Confidence, Angular Separation, Metadata, and Least-Squares Residuals.
"""

import os
import sys
import glob as glob_module
import json
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import OpenEXR
import Imath

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper
from src.losses.typ_losses import _get_relative_viewmat
from src.data.mask_generator import get_letter_mask

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

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n=== TypoSplat Diagnostic: Geometry Check (CSV Mode) ===")
    
    parent_dir = sys.argv[1] if len(sys.argv) > 1 else "/content/data"
    sample_dirs = sorted([
        d for d in glob_module.glob(os.path.join(parent_dir, "*"))
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "metadata.json"))
    ])
    
    print(f"Discovered {len(sample_dirs)} samples in {parent_dir}...\n")
    
    vggt = VGGTWrapper().to(device)
    vggt.eval()

    stats = {"fixable": 0, "marginal": 0, "broken": 0, "errored": 0}
    results = []
    
    with torch.no_grad():
        for sample_dir in tqdm(sample_dirs, desc="Analyzing Geometry"):
            sample_name = os.path.basename(sample_dir)
            
            try:
                # 1. Load Data
                meta_path = os.path.join(sample_dir, "metadata.json")
                mesh_path = os.path.join(sample_dir, "mesh.ply")
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    
                view_A_paths = glob_module.glob(os.path.join(sample_dir, "*view_A*.png"))
                depth_A_paths = glob_module.glob(os.path.join(sample_dir, "*depth_A*.exr"))
                if not view_A_paths or not depth_A_paths:
                    raise FileNotFoundError("Missing Camera A image or depth EXR.")
                    
                gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
                
                gt_depth_518 = load_exr_depth(depth_A_paths[0], device)
                gt_depth_148 = torch.nn.functional.interpolate(gt_depth_518, size=(148, 148), mode='nearest')
                
                mask_148_A = get_letter_mask(mesh_path, meta, device=device)
                mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device=device)
                mask_518_B = torch.nn.functional.interpolate(mask_148_B, size=(518, 518), mode='nearest')
                
                target_px = torch.nonzero(mask_518_B.squeeze(), as_tuple=False).float()
                if len(target_px) == 0:
                    raise ValueError("Camera B mask is completely empty.")
                target_centroid = target_px.mean(dim=0).flip(0)
                
                # 2. Forward VGGT & Extract Confidence
                vggt_out = vggt.forward_with_features(gt_rgb_A)
                base_depth_518 = vggt_out["depth"]
                base_depth_148 = torch.nn.functional.interpolate(base_depth_518, size=(148, 148), mode='bilinear', align_corners=False)
                
                conf_val = "N/A"
                if "depth_conf" in vggt_out:
                    mask_518_A = torch.nn.functional.interpolate(mask_148_A, size=(518, 518), mode='nearest')
                    if mask_518_A.sum() > 0:
                        conf_map = vggt_out["depth_conf"]
                        conf_val = ((conf_map * mask_518_A).sum() / mask_518_A.sum()).item()
                
                # 3. GLM's Least-Squares Affine Fit (Masked)
                mask_bool = mask_148_A[0,0].bool()
                Z_vggt_masked = base_depth_148[0,0][mask_bool]
                Z_gt_masked = gt_depth_148[0,0][mask_bool]
                
                if Z_vggt_masked.shape[0] < 2:
                    raise ValueError("Mask too small for regression.")
                
                # Closed-form least squares: s, t minimizing ||s*Z_vggt + t - Z_gt||^2
                X_design = torch.stack([Z_vggt_masked, torch.ones_like(Z_vggt_masked)], dim=1)
                
                # Use try-except to catch degenerate rank matrices
                try:
                    theta = torch.linalg.lstsq(X_design, Z_gt_masked.unsqueeze(1)).solution
                    s_opt, t_opt = theta[0,0].item(), theta[1,0].item()
                except RuntimeError:
                    raise ValueError("Least Squares solver failed (degenerate variance).")
                
                calibrated_depth_148 = s_opt * base_depth_148 + t_opt
                
                # 4. Geometric Projection (Using Calibrated Depth)
                fx_A, fy_A = meta["fx"], meta["fy"]
                cx_A, cy_A = meta["cx"], meta["cy"]
                scale_factor = 518.0 / 148.0
                
                y_grid, x_grid = torch.meshgrid(torch.arange(148, device=device, dtype=torch.float32), 
                                                torch.arange(148, device=device, dtype=torch.float32), indexing='ij')
                
                u_518 = (x_grid + 0.5) * scale_factor
                v_518 = (y_grid + 0.5) * scale_factor
                
                # --- USE CALIBRATED DEPTH ---
                Z = calibrated_depth_148[0, 0]
                X = (u_518 - cx_A) * Z / fx_A
                Y = (v_518 - cy_A) * Z / fy_A
                
                points_A = torch.stack([X, Y, Z], dim=-1).view(-1, 3)
                
                meta_B = meta["camera_B"]
                fx_B, fy_B = meta_B["fx"], meta_B["fy"]
                cx_B, cy_B = meta_B["cx"], meta_B["cy"]
                
                viewmats_B = _get_relative_viewmat(meta["camera_to_world_matrix"], meta_B["camera_to_world_matrix"], device)
                
                points_A_h = torch.cat([points_A, torch.ones_like(points_A[:, :1])], dim=1)
                points_B = (viewmats_B[0] @ points_A_h.T).T
                
                # 5. Strict Bound Checks (Claude's fix)
                Z_MIN = 1.0 
                Z_B = points_B[:, 2]
                valid_z = Z_B > Z_MIN
                
                Z_safe = Z_B.clone()
                Z_safe[~valid_z] = 1.0 
                
                x_proj = (points_B[:, 0] / Z_safe) * fx_B + cx_B
                y_proj = (points_B[:, 1] / Z_safe) * fy_B + cy_B
                
                # MUST BE VALID Z AND IN 518x518 FRAME
                in_frame = valid_z & (x_proj >= 0) & (x_proj <= 518) & (y_proj >= 0) & (y_proj <= 518)
                frac_offscreen = (~in_frame).float().mean().item()
                
                if in_frame.sum() > 0:
                    pred_centroid_x = x_proj[in_frame].mean().item()
                    pred_centroid_y = y_proj[in_frame].mean().item()
                    dist = np.sqrt((pred_centroid_x - target_centroid[0].item())**2 + (pred_centroid_y - target_centroid[1].item())**2)
                else:
                    dist = 9999.0 
                
                # Categorization (Based on Calibrated Distance)
                if frac_offscreen > 0.95 or dist > 150: 
                    category = "broken"
                elif dist > 75:
                    category = "marginal"
                else:
                    category = "fixable"
                    
                stats[category] += 1
                
                # 6. Extract Angular Separation
                c2w_A = np.array(meta["camera_to_world_matrix"])
                c2w_B = np.array(meta_B["camera_to_world_matrix"])
                dir_A = -c2w_A[:3, 2]
                dir_B = -c2w_B[:3, 2]
                dot_prod = np.clip(np.dot(dir_A, dir_B) / (np.linalg.norm(dir_A) * np.linalg.norm(dir_B)), -1.0, 1.0)
                angle_deg = np.degrees(np.arccos(dot_prod))
                
                # 7. Append Row
                results.append({
                    "Sample": sample_name,
                    "Category": category,
                    "Calib_Dist_px": round(dist, 2),
                    "Calib_Offscreen": round(frac_offscreen, 4),
                    "Opt_Scale": round(s_opt, 4),
                    "Opt_Shift": round(t_opt, 4),
                    "Angle_Deg": round(angle_deg, 2),
                    "VGGT_Conf": round(conf_val, 4) if isinstance(conf_val, float) else conf_val,
                    "Mount_Style": meta.get("mount_style", "UNKNOWN"),
                    "Is_Cursive": meta.get("is_cursive", False),
                    "Extrusion_Depth": round(meta.get("extrusion_depth", 0.0), 4),
                    "HDRI_Source": meta.get("hdri_source", "UNKNOWN"),
                    "Error": ""
                })

            except Exception as e:
                stats["errored"] += 1
                results.append({
                    "Sample": sample_name,
                    "Category": "errored",
                    "Calib_Dist_px": "", "Calib_Offscreen": "",
                    "Opt_Scale": "", "Opt_Shift": "", "Angle_Deg": "", 
                    "VGGT_Conf": "", "Mount_Style": "", "Is_Cursive": "", 
                    "Extrusion_Depth": "", "HDRI_Source": "",
                    "Error": str(e)
                })

    # 8. Write to CSV
    csv_path = os.path.join(current_dir, "diagnostic_results.csv")
    if results:
        keys = results[0].keys()
        with open(csv_path, 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(results)

    # 9. Terminal Summary
    total_samples = len(sample_dirs)
    print("\n" + "="*50)
    print(f"=== BATCH SUMMARY ({total_samples} samples) ===")
    print("="*50)
    print(f"✅ Fixable (Dist < 75px):  {stats['fixable']} ({(stats['fixable']/total_samples):.1%})")
    print(f"⚠️ Marginal (75-150px):   {stats['marginal']} ({(stats['marginal']/total_samples):.1%})")
    print(f"❌ Broken (>150px):       {stats['broken']} ({(stats['broken']/total_samples):.1%})")
    print(f"💥 Errored (Crash):       {stats['errored']} ({(stats['errored']/total_samples):.1%})")
    print(f"\n[!] Full results saved to: {csv_path}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()