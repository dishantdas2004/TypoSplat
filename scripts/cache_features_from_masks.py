"""
Stage 2 of 2 (run on a GPU runtime): VGGT features + final cached_features.pt.
Identical to cache_features_1400.py except the four letter masks are LOADED from
<sample>/masks_148.pt (made by cache_masks_only.py) instead of being computed, so pytorch3d is not needed.
The output cached_features.pt has exactly the same keys and dtypes as before.
Safe to re-run: samples that already have cached_features.pt are skipped.
"""
import os
import sys
import glob
import argparse
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.models.vggt_wrapper import VGGTWrapper


def main():
    parser = argparse.ArgumentParser(description="Stage 2: VGGT features (+ masks from masks_148.pt)")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of samples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== Stage 2: VGGT features | device {device} ===")

    sample_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    print(f"Found {len(sample_dirs)} total samples. Loading models...")

    vggt = VGGTWrapper().to(device)
    vggt.eval()
    for param in vggt.parameters():
        param.requires_grad = False

    skipped = errors = 0
    with torch.no_grad():
        for sample_dir in tqdm(sample_dirs, desc="Caching Features"):
            out_path = os.path.join(sample_dir, "cached_features.pt")
            tmp_path = out_path + ".tmp"
            if os.path.exists(out_path):
                skipped += 1
                continue

            view_A_paths = glob.glob(os.path.join(sample_dir, "*view_A*.png"))
            masks_path = os.path.join(sample_dir, "masks_148.pt")
            if not view_A_paths or not os.path.exists(masks_path):
                print(f"\n[SKIP] {sample_dir}: missing view_A png or masks_148.pt (run stage 1 first)")
                errors += 1
                continue

            try:
                masks = torch.load(masks_path, map_location="cpu", weights_only=True)
                gt_rgb_A = transforms.ToTensor()(Image.open(view_A_paths[0]).convert("RGB").resize((518, 518))).unsqueeze(0).to(device)
                vggt_out = vggt.forward_with_features(gt_rgb_A)

                torch.save({
                    "patch_tokens": vggt_out["patch_tokens"].half().cpu(),
                    "base_depth": vggt_out["depth"].half().cpu(),
                    "mask_148_A": masks["mask_148_A"].half().cpu(),
                    "mask_148_B": masks["mask_148_B"].half().cpu(),
                    "mask_148_C": masks["mask_148_C"].half().cpu(),
                    "mask_148_D": masks["mask_148_D"].half().cpu(),
                }, tmp_path)
                os.replace(tmp_path, out_path)
            except Exception as e:
                print(f"\n[SKIP] Failed to process {sample_dir}: {e}")
                errors += 1
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    print(f"\n[SUCCESS] Stage 2 complete. New: {len(sample_dirs) - skipped - errors} | Skipped (already cached): {skipped} | Errors: {errors}")


if __name__ == "__main__":
    main()