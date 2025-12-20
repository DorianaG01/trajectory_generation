import pandas as pd
import numpy as np
import torch 

def load_vehicle_dataset(noisy_csv_path, clean_csv_path, T_steps=600, train_split=0.7, val_split=0.15, seed=42):

    # Stato completo (target, 6D)
    state_cols = ['X', 'Y', 'phi', 'vx', 'vy', 'omega']
    # Misure rumorose (input, 5D - senza 'phi')
    measure_cols = ['X', 'Y', 'vx', 'vy', 'omega']
    # Controlli (input, 2D)
    control_cols = ['d', 'delta']
    
    print(f"Caricamento dati da {noisy_csv_path} e {clean_csv_path}...")
    try:
        df_noisy = pd.read_csv(noisy_csv_path)
        df_clean = pd.read_csv(clean_csv_path)
    except FileNotFoundError as e:
        print(f"ERRORE: File non trovato. {e}")
        return None
    
    # Trova il numero totale di traiettorie
    num_traj = df_noisy['trajectory_id'].nunique()
    
    # Inizializza gli array Numpy vuoti
    # y = misure [N, 5, T]
    y_all = np.zeros((num_traj, len(measure_cols), T_steps), dtype=np.float32)
    # u = controlli [N, 2, T]
    u_all = np.zeros((num_traj, len(control_cols), T_steps), dtype=np.float32)
    # x = stato (target) [N, 6, T]
    x_all = np.zeros((num_traj, len(state_cols), T_steps), dtype=np.float32)

    # Raggruppa i dataframe per un accesso più rapido
    grouped_noisy = df_noisy.groupby('trajectory_id')
    grouped_clean = df_clean.groupby('trajectory_id')

    # Itera su ogni traiettoria e riempi gli array
    for i in range(num_traj):
        try:
            traj_n = grouped_noisy.get_group(i)
            traj_c = grouped_clean.get_group(i)
            
            # Estrai i primi T_steps 
            # .values estrae come [T, C], quindi usiamo .T per ottenere [C, T]
            y_all[i, :, :] = traj_n[measure_cols].values[:T_steps, :].T
            u_all[i, :, :] = traj_n[control_cols].values[:T_steps, :].T
            x_all[i, :, :] = traj_c[state_cols].values[:T_steps, :].T
        except KeyError:
            print(f"ERRORE: Traiettoria {i} non trovata o corrotta. Interruzione.")
            return None
            
    print(f"Rimodellamento completato. Shape totali: y={y_all.shape}, u={u_all.shape}, x={x_all.shape}")

    
    # Crea e mescola gli indici delle traiettorie
    indices = np.arange(num_traj)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    # Applica gli indici mescolati
    y_all = y_all[indices]
    u_all = u_all[indices]
    x_all = x_all[indices]
    
    # Calcola gli indici di divisione
    n_train = int(num_traj * train_split)
    n_val = int(num_traj * val_split)
    
    # Suddivide in training, test e validation
    y_train = y_all[:n_train]
    u_train = u_all[:n_train]
    x_train = x_all[:n_train]
    
    y_val = y_all[n_train : n_train + n_val]
    u_val = u_all[n_train : n_train + n_val]
    x_val = x_all[n_train : n_train + n_val]
    
    y_test = y_all[n_train + n_val :]
    u_test = u_all[n_train + n_val :]
    x_test = x_all[n_train + n_val :]

    # Conversione in tensori pytorch
    
    train_data = (
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(u_train, dtype=torch.float32),
        torch.tensor(x_train, dtype=torch.float32)
    )
    val_data = (
        torch.tensor(y_val, dtype=torch.float32),
        torch.tensor(u_val, dtype=torch.float32),
        torch.tensor(x_val, dtype=torch.float32)
    )
    test_data = (
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(u_test, dtype=torch.float32),
        torch.tensor(x_test, dtype=torch.float32)
    )
    
    print("\nCaricamento completato.")
    print(f"  Train set (y, u, x): {train_data[0].shape}, {train_data[1].shape}, {train_data[2].shape}")
    print(f"  Val set   (y, u, x): {val_data[0].shape}, {val_data[1].shape}, {val_data[2].shape}")
    print(f"  Test set  (y, u, x): {test_data[0].shape}, {test_data[1].shape}, {test_data[2].shape}")

    return train_data, val_data, test_data

