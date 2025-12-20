import pandas as pd
import matplotlib.pyplot as plt
import os

# --- 1. IMPOSTAZIONI ---
# Specifica il file CSV "pulito"
clean_csv_path = '/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/generation_traj/vehicle_piecewise_noisy.csv'

# Specifica quale ID di traiettoria vuoi vedere
# (Deve essere uno degli ID nell'ultima colonna del CSV)
TRAJECTORY_ID_TO_PLOT = 4

# Definiamo i nomi delle colonne in base ai tuoi dati (10 colonne)
# t, m=6 stati (x, y, phi, vx, vy, omega), d=2 controlli (d, delta), id
column_names = [
    'time', 'x', 'y', 'vx', 'vy', 'omega', 
    'control_d', 'control_delta', 'traj_id'
]

# -----------------------------------------------------------------

print(f"Caricamento dati da: {clean_csv_path}...")
try:
    # Carica l'intero CSV usando pandas
    df = pd.read_csv(clean_csv_path)
except FileNotFoundError:
    print(f"ERRORE: File non trovato. Controlla il percorso: {clean_csv_path}")
    exit()

print("Dati caricati.")

# --- 2. FILTRA PER ID ---
print(f"Filtraggio per TRAJECTORY_ID_TO_PLOT = {TRAJECTORY_ID_TO_PLOT}...")
trajectory_data = df[df['trajectory_id'] == TRAJECTORY_ID_TO_PLOT]

if trajectory_data.empty:
    print(f"ERRORE: Nessun dato trovato per l'ID traiettoria {TRAJECTORY_ID_TO_PLOT}.")
    print("Controlla che l'ID sia corretto.")
    exit()

print(f"Trovati {len(trajectory_data)} time step per questa traiettoria.")

# --- 3. PLOTTING ---
print("Generazione plot...")

# Lista dei 6 stati che vogliamo plottare
states_to_plot = ['X', 'Y', 'vx', 'vy', 'omega']
state_titles = [
    'Stato X [m]', 'Stato Y [m]', 
    'Stato vx [m/s]', 'Stato vy [m/s]', 'Stato omega [rad/s]'
]

fig, axs = plt.subplots(3, 2, figsize=(15, 12)) # 3x2 per 6 stati
fig.suptitle(f"Analisi Ground Truth - Traiettoria ID: {TRAJECTORY_ID_TO_PLOT}", fontsize=16)

# Itera e plotta ogni stato
for i, state_key in enumerate(states_to_plot):
    row = i // 2
    col = i % 2
    
    ax = axs[row, col]
    
    # Plotta i dati filtrati
    ax.plot(
        trajectory_data['t'],  # Asse X
        trajectory_data[state_key], # Asse Y
        'k-', # Linea nera solida
        label='Ground Truth (Pulito)'
    )
    
    ax.set_title(state_titles[i])
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Valore')
    ax.legend()
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Aggiusta per il titolo
plt.show()

print("Fatto.")