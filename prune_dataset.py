import os
import sys
import shutil
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Delete local sample directories that fail diagnostic criteria.")
    parser.add_argument("--csv_file", type=str, default="diagnostics_train_id_0-5303.csv", help="Path to diagnostic CSV")
    parser.add_argument("--data_dir", type=str, default="data", help="Local directory containing the sample folders")
    parser.add_argument("--dry_run", action="store_true", help="Print what would be deleted without actually deleting")
    args = parser.parse_args()

    print(f"Loading diagnostics from {args.csv_file}...")
    try:
        df = pd.read_csv(args.csv_file)
    except FileNotFoundError:
        print(f"ERROR: Could not find '{args.csv_file}'. Make sure it's in the same folder as the script.")
        sys.exit(1)
    
    if 'Sample' not in df.columns:
        print("ERROR: Could not find a 'Sample' column in the CSV. Please check your column names.")
        sys.exit(1)

    # 1. Define the KEEP criteria exactly as requested
    abs_shift = df['Opt_Shift'].abs()
    cond_a = (abs_shift < 2.0) & (df['Calib_Dist_px'] < 95.0)
    cond_b = (abs_shift < 2.6) & (df['Calib_Dist_px'] < 75.0)
    cond_scale = df['Opt_Scale'] < 8.0
    
    keep_mask = (cond_a | cond_b) & cond_scale
    
    # 2. Invert the mask to find what needs to be DELETED
    discard_df = df[~keep_mask]
    rejected_sample_ids = discard_df['Sample'].astype(str).tolist()

    print(f"Total samples in CSV: {len(df)}")
    print(f"Samples meeting criteria (Keeping): {keep_mask.sum()}")
    print(f"Samples failing criteria (Deleting): {len(rejected_sample_ids)}\n")

    if args.dry_run:
        print("DRY RUN ENABLED. No files will be deleted. First 10 directories that would be deleted:")
        for sid in rejected_sample_ids[:10]:
            print(f"  -> {os.path.join(args.data_dir, sid)}")
        return

    # 3. Execute the deletion
    deleted_count = 0
    missing_count = 0
    
    for sample_id in rejected_sample_ids:
        sample_path = os.path.join(args.data_dir, sample_id)
        
        if os.path.exists(sample_path) and os.path.isdir(sample_path):
            try:
                shutil.rmtree(sample_path)
                deleted_count += 1
            except Exception as e:
                print(f"Failed to delete {sample_path}: {e}")
        else:
            missing_count += 1
            
    print("--- Pruning Summary ---")
    print(f"Successfully deleted: {deleted_count} sample directories")
    if missing_count > 0:
        print(f"Note: {missing_count} directories were already missing or not found in '{args.data_dir}'")

if __name__ == "__main__":
    main()