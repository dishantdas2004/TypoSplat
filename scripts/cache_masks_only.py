"""
Stage 1 of 2 (run on a CPU runtime): letter masks only.
Writes <sample>/masks_148.pt = {mask_148_A, mask_148_B, mask_148_C, mask_148_D} (float16, CPU),
using the SAME get_letter_mask calls as cache_features_1400.py. Needs pytorch3d, not VGGT.
Safe to re-run: samples that already have masks_148.pt (or cached_features.pt) are skipped.
"""
import os
import sys
import json
import argparse
import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(root_dir)

from src.data.mask_generator import get_letter_mask


def main():
    parser = argparse.ArgumentParser(description="Stage 1: letter masks only")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory of samples")
    parser.add_argument("--device", type=str, default="cpu", help="cpu (default) or cuda")
    args = parser.parse_args()
    device = torch.device(args.device)

    sample_dirs = [os.path.join(args.data_dir, d) for d in os.listdir(args.data_dir)
                   if os.path.isdir(os.path.join(args.data_dir, d)) and os.path.exists(os.path.join(args.data_dir, d, "metadata.json"))]
    print(f"=== Stage 1: masks only | {len(sample_dirs)} samples | device {device} ===")

    done = skipped = errors = 0
    with torch.no_grad():
        for sample_dir in tqdm(sample_dirs, desc="Masks"):
            out_path = os.path.join(sample_dir, "masks_148.pt")
            tmp_path = out_path + ".tmp"
            if os.path.exists(out_path) or os.path.exists(os.path.join(sample_dir, "cached_features.pt")):
                skipped += 1
                continue
            try:
                with open(os.path.join(sample_dir, "metadata.json"), "r") as f:
                    meta = json.load(f)
                mesh_path = os.path.join(sample_dir, "mesh.ply")
                masks = {
                    "mask_148_A": get_letter_mask(mesh_path, meta, device=device),
                    "mask_148_B": get_letter_mask(mesh_path, meta["camera_B"], device=device),
                    "mask_148_C": get_letter_mask(mesh_path, meta["camera_C"], device=device),
                    "mask_148_D": get_letter_mask(mesh_path, meta["camera_D"], device=device),
                }
                torch.save({k: v.half().cpu() for k, v in masks.items()}, tmp_path)
                os.replace(tmp_path, out_path)
                done += 1
            except Exception as e:
                print(f"\n[SKIP] Failed to process {sample_dir}: {e}")
                errors += 1
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    print(f"\n[SUCCESS] Stage 1 complete. New: {done} | Skipped: {skipped} | Errors: {errors}")


if __name__ == "__main__":
    main()