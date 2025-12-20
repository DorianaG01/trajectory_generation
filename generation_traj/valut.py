import pandas as pd
import os

print("Inizio script di unione...")

# --- Nomi dei file ---
file_p_clean = 'vehicle_piecewise_clean.csv'
file_m_clean = 'vehicle_mpc_clean.csv'
output_clean = 'combined_dataset_clean.csv'

file_p_noisy = 'vehicle_piecewise_noisy_err005.csv'
file_m_noisy = 'vehicle_mpc_noisy_err005.csv'
output_noisy = 'combined_dataset_noisy_err005.csv'

# Lista di file da controllare
files_to_check = [file_p_clean, file_m_clean, file_p_noisy, file_m_noisy]

# Controlla se tutti i file esistono prima di iniziare
all_files_exist = True
for f in files_to_check:
    if not os.path.exists(f):
        print(f"ERRORE: File non trovato: '{f}'")
        all_files_exist = False

if not all_files_exist:
    print("\nOperazione annullata. Assicurati che tutti i file CSV siano nella cartella.")
else:
    try:
        # --- 1. Processa i file CLEAN ---
        print(f"Caricamento {file_p_clean}...")
        df_p_clean = pd.read_csv(file_p_clean)
        
        print(f"Caricamento {file_m_clean}...")
        df_m_clean = pd.read_csv(file_m_clean)

        # Trova l'ID massimo del primo file (dovrebbe essere 4999)
        max_id_file1 = df_p_clean['trajectory_id'].max()
        
        # Calcola l'offset (dovrebbe essere 4999 + 1 = 5000)
        id_offset = max_id_file1 + 1
        
        print(f"Ricalcolo degli ID in {file_m_clean} (offset: +{id_offset})...")
        df_m_clean['trajectory_id'] = df_m_clean['trajectory_id'] + id_offset

        print("Unione dei file 'clean'...")
        combined_clean = pd.concat([df_p_clean, df_m_clean], ignore_index=True, sort=False)
        
        combined_clean.to_csv(output_clean, index=False)
        print(f"-> File 'clean' combinato salvato come: '{output_clean}'")

        # --- 2. Processa i file NOISY ---
        print(f"\nCaricamento {file_p_noisy}...")
        df_p_noisy = pd.read_csv(file_p_noisy)
        
        print(f"Caricamento {file_m_noisy}...")
        df_m_noisy = pd.read_csv(file_m_noisy)
        
        # Ricalcolo degli ID (usando lo stesso offset)
        print(f"Ricalcolo degli ID in {file_m_noisy} (offset: +{id_offset})...")
        df_m_noisy['trajectory_id'] = df_m_noisy['trajectory_id'] + id_offset

        print("Unione dei file 'noisy'...")
        combined_noisy = pd.concat([df_p_noisy, df_m_noisy], ignore_index=True, sort=False)

        combined_noisy.to_csv(output_noisy, index=False)
        print(f"-> File 'noisy' combinato salvato come: '{output_noisy}'")

        # --- 3. Verifica Finale ---
        print("\n--- Verifica completata ---")
        print(f"File Clean: {len(combined_clean)} righe.")
        print(f"  ID da {combined_clean['trajectory_id'].min()} a {combined_clean['trajectory_id'].max()}")
        print(f"  Colonne: {list(combined_clean.columns)}")
        
        print(f"\nFile Noisy: {len(combined_noisy)} righe.")
        print(f"  ID da {combined_noisy['trajectory_id'].min()} a {combined_noisy['trajectory_id'].max()}")
        print(f"  Colonne: {list(combined_noisy.columns)}")

    except Exception as e:
        print(f"\nSi è verificato un errore: {e}")