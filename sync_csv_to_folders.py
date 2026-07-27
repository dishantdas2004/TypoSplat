import os
import sys
import pandas as pd
import argparse

def main():
    parser = argparse.ArgumentParser(description="Sync diagnostic CSV with existing local folders and append to a master CSV.")
    parser.add_argument("--input_csv", type=str, default="diagnostics_train_id_0-5303.csv", help="Path to the new/raw diagnostic CSV")
    parser.add_argument("--data_dir", type=str, default="data", help="Local directory containing the filtered sample folders")
    parser.add_argument("--output_csv", type=str, default="master_diagnostics.csv", help="The master CSV to create or update")
    args = parser.parse_args()

    # 1. Get the list of all currently existing sample folders
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory '{args.data_dir}' not found.")
        sys.exit(1)
        
    existing_folders = [f for f in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, f))]
    print(f"Found {len(existing_folders)} sample folders in '{args.data_dir}'.")

    # 2. Load the input diagnostic CSV
    try:
        input_df = pd.read_csv(args.input_csv)
    except FileNotFoundError:
        print(f"ERROR: Could not find input CSV '{args.input_csv}'.")
        sys.exit(1)
        
    if 'Sample' not in input_df.columns:
        print("ERROR: Could not find a 'Sample' column in the input CSV.")
        sys.exit(1)

    # 3. Filter the input CSV to ONLY include rows where the folder exists
    # Convert 'Sample' to string to match folder names reliably
    input_df['Sample'] = input_df['Sample'].astype(str)
    filtered_df = input_df[input_df['Sample'].isin(existing_folders)]
    
    print(f"Extracted {len(filtered_df)} matching rows from '{args.input_csv}'.")

    # 4. Create or Update the Master CSV
    if os.path.exists(args.output_csv):
        print(f"Master CSV '{args.output_csv}' found. Merging new data...")
        master_df = pd.read_csv(args.output_csv)
        master_df['Sample'] = master_df['Sample'].astype(str)
        
        # Combine the old master and the new filtered data
        combined_df = pd.concat([master_df, filtered_df], ignore_index=True)
        
        # Drop duplicates based on 'Sample' ID, keeping the most recent entry
        combined_df.drop_duplicates(subset=['Sample'], keep='last', inplace=True)
        
        # Save back to disk
        combined_df.to_csv(args.output_csv, index=False)
        print(f"Successfully updated '{args.output_csv}'. It now contains {len(combined_df)} total samples.")
        
    else:
        print(f"Master CSV '{args.output_csv}' not found. Creating a new one...")
        filtered_df.to_csv(args.output_csv, index=False)
        print(f"Successfully created '{args.output_csv}' with {len(filtered_df)} samples.")

if __name__ == "__main__":
    main()