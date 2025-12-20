import pandas as pd
import numpy as np

def print_csv_min_max(file_path):
    """
    Carica un file CSV e stampa i valori minimi e massimi per ogni colonna.
    """
    try:
        # Carica il file CSV
        df = pd.read_csv(file_path)
        
        # Seleziona solo le colonne numeriche per il calcolo
        # (altrimenti 'min'/'max' potrebbe dare errore su colonne di testo come 'mode')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if numeric_cols.empty:
            print(f"Errore: Nessuna colonna numerica trovata in '{file_path}'.")
            return

        # Calcola min e max solo per le colonne numeriche
        stats = df[numeric_cols].agg(['min', 'max'])
        
        # Formattazione per la stampa
        pd.set_option('display.precision', 4)
        pd.set_option('display.float_format', '{:,.4f}'.format)
        
        print(f"--- Statistiche Min/Max per il file: '{file_path}' ---")
        print(stats)

    except FileNotFoundError:
        print(f"Errore: File non trovato.")
        print(f"Assicurati che il file '{file_path}' sia nella stessa cartella dello script.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")

if __name__ == "__main__":

    # Esempio per il file del secondo script
    nome_file_smooth = "/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/generation_traj/combined_dataset_clean.csv"
    print_csv_min_max(nome_file_smooth)
    
    print("\n" + "="*80 + "\n")
    
    # Esempio per il file del primo script
    nome_file_piecewise = "/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/generation_traj/combined_dataset_noisy.csv"
    print_csv_min_max(nome_file_piecewise)