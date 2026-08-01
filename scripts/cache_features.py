import os
import sys
import glob
import json
import shutil
import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper
from src.data.mask_generator import get_letter_mask
from src.utils.cache_tier import get_cache_tier

def backup_disk_tier_to_drive(sample_dirs, drive_backup_root):
    os.makedirs(drive_backup_root, exist_ok=True)
    for sample_dir in tqdm(sample_dirs, desc="Backing up disk-tier cache to Drive"):
        cache_path = os.path.join(sample_dir, "cached_features.pt")
        if not os.path.exists(cache_path):
            continue
        sample_id = os.path.basename(sample_dir)
        dest = os.path.join(drive_backup_root, f"{sample_id}.pt")
        if not os.path.exists(dest):
            shutil.copy2(cache_path, dest)

def main():
    parser = argparse.ArgumentParser(description="One-Time Feature & Mask Caching (Disk Tier)")
    parser.add_argument("--data_dir", type=str, default="/content/data", help="Directory of samples to cache")
    parser.add_argument("--backup_to_drive", action="store_true", help="Backup cached files to Drive after generating")
    parser.add_argument("--drive_backup_root", type=str, default="/content/drive/MyDrive/TypoSplat/disk_cache_backup")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"=== TypoSplat: One-Time Feature & Mask Caching ===")
    print(f"Target Directory: {args.data_dir}\n")
    
    sample_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir) 
                   if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    
    print(f"Found {len(sample_dirs)} total samples. Loading models...")
    
    vggt = VGGTWrapper().to(device)
    vggt.eval()
    for param in vggt.parameters():
        param.requires_grad = False
        
    skipped = 0
    errors = 0
    ram_skipped = 0
    
    with torch.no_grad():
        for sample_dir in tqdm(sample_dirs, desc="Caching Features & Masks"):
            sample_id = int(os.path.basename(sample_dir))
            tier = get_cache_tier(sample_id)
            
            if tier != "disk":
                ram_skipped += 1
                continue
            
            out_path = os.path.join(sample_dir, "cached_features.pt")
            tmp_path = out_path + ".tmp"
            
            if os.path.exists(out_path):
                skipped += 1
                continue
                
            view_A_paths = glob.glob(os.path.join(sample_dir, "*view_A*.png"))
            if not view_A_paths:
                errors += 1
                continue
                
            try:
                meta_path = os.path.join(sample_dir, "metadata.json")
                mesh_path = os.path.join(sample_dir, "mesh.ply")
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                    
                gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
                
                vggt_out = vggt.forward_with_features(gt_rgb_A)
                mask_148_A = get_letter_mask(mesh_path, meta, device=device)
                mask_148_B = get_letter_mask(mesh_path, meta["camera_B"], device=device)
                
                torch.save({
                    "patch_tokens": vggt_out["patch_tokens"].half().cpu(),
                    "base_depth": vggt_out["depth"].half().cpu(),
                    "mask_148_A": mask_148_A.half().cpu(),
                    "mask_148_B": mask_148_B.half().cpu()
                }, tmp_path)
                
                os.replace(tmp_path, out_path)
                
            except Exception as e:
                print(f"\n[SKIP] Failed to process {sample_dir}: {e}")
                errors += 1
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                continue
            
    print(f"\n[SUCCESS] Caching run complete.")
    print(f"          New Disk-Tier Cached: {len(sample_dirs) - skipped - errors - ram_skipped}")
    print(f"          Skipped (Already cached): {skipped}")
    print(f"          Skipped (RAM Tier): {ram_skipped}")
    print(f"          Errors: {errors}")
    
    if args.backup_to_drive:
        backup_disk_tier_to_drive(sample_dirs, args.drive_backup_root)
        print("\n[SUCCESS] Drive backup complete.")

if __name__ == "__main__":
    main()