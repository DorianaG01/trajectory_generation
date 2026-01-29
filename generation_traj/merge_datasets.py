import pandas as pd
import os

print("Starting merge script...")

# --- File Names ---
# Input files for piecewise generation
file_p_clean = 'vehicle_piecewise_clean.csv'
file_p_noisy = 'vehicle_piecewise_noisy_err.csv'

# Input files for MPC generation
file_m_clean = 'vehicle_mpc_clean.csv'
file_m_noisy = 'vehicle_mpc_noisy_err.csv'

# Output combined files
output_clean = 'combined_dataset_clean.csv'
output_noisy = 'combined_dataset_noisy_err.csv'

# List of files to verify before processing
files_to_check = [file_p_clean, file_m_clean, file_p_noisy, file_m_noisy]

# Check if all required files exist in the current directory
all_files_exist = True
for f in files_to_check:
    if not os.path.exists(f):
        print(f"ERROR: File not found: '{f}'")
        all_files_exist = False

if not all_files_exist:
    print("\nOperation cancelled. Please ensure all CSV files are in the folder.")
else:
    try:
        # --- 1. Process CLEAN files ---
        print(f"Loading {file_p_clean}...")
        df_p_clean = pd.read_csv(file_p_clean)
        
        print(f"Loading {file_m_clean}...")
        df_m_clean = pd.read_csv(file_m_clean)

        # Find the maximum ID in the first file (e.g., 4999) to prevent overlaps
        max_id_file1 = df_p_clean['trajectory_id'].max()
        
        # Calculate the ID offset (e.g., 4999 + 1 = 5000)
        id_offset = max_id_file1 + 1
        
        print(f"Re-indexing IDs in {file_m_clean} (offset: +{id_offset})...")
        df_m_clean['trajectory_id'] = df_m_clean['trajectory_id'] + id_offset

        print("Merging 'clean' files...")
        combined_clean = pd.concat([df_p_clean, df_m_clean], ignore_index=True, sort=False)
        
        combined_clean.to_csv(output_clean, index=False)
        print(f"-> Combined 'clean' file saved as: '{output_clean}'")

        # --- 2. Process NOISY files ---
        print(f"\nLoading {file_p_noisy}...")
        df_p_noisy = pd.read_csv(file_p_noisy)
        
        print(f"Loading {file_m_noisy}...")
        df_m_noisy = pd.read_csv(file_m_noisy)
        
        # Re-index IDs using the same offset to maintain consistency with clean data
        print(f"Re-indexing IDs in {file_m_noisy} (offset: +{id_offset})...")
        df_m_noisy['trajectory_id'] = df_m_noisy['trajectory_id'] + id_offset

        print("Merging 'noisy' files...")
        combined_noisy = pd.concat([df_p_noisy, df_m_noisy], ignore_index=True, sort=False)

        combined_noisy.to_csv(output_noisy, index=False)
        print(f"-> Combined 'noisy' file saved as: '{output_noisy}'")

        # --- 3. Final Verification ---
        print("\n--- Verification Completed ---")
        print(f"Combined Clean File: {len(combined_clean)} rows.")
        print(f"  ID Range: from {combined_clean['trajectory_id'].min()} to {combined_clean['trajectory_id'].max()}")
        print(f"  Columns: {list(combined_clean.columns)}")
        
        print(f"\nCombined Noisy File: {len(combined_noisy)} rows.")
        print(f"  ID Range: from {combined_noisy['trajectory_id'].min()} to {combined_noisy['trajectory_id'].max()}")
        print(f"  Columns: {list(combined_noisy.columns)}")

    except Exception as e:
        print(f"\nAn error occurred during processing: {e}")
