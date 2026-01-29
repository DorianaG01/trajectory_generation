import pandas as pd
import numpy as np
import torch 

def load_vehicle_dataset(noisy_csv_path, clean_csv_path, T_steps=600, train_split=0.7, val_split=0.15, seed=42):
    """
    Loads, processes, and splits the vehicle dataset into Train, Validation, and Test sets.
    
    Returns:
        tuple: (train_data, val_data, test_data) where each is a tuple of (y, u, x) tensors.
    """

    # Full state (Targets/Labels, 6D: X, Y, phi, vx, vy, omega)
    state_cols = ['X', 'Y', 'phi', 'vx', 'vy', 'omega']
    # Noisy measurements (Inputs/Observations, 5D - excluding 'phi')
    measure_cols = ['X', 'Y', 'vx', 'vy', 'omega']
    # Control inputs (Inputs/Actions, 2D: duty cycle, steering)
    control_cols = ['d', 'delta']
    
    print(f"Loading data from {noisy_csv_path} and {clean_csv_path}...")
    try:
        df_noisy = pd.read_csv(noisy_csv_path)
        df_clean = pd.read_csv(clean_csv_path)
    except FileNotFoundError as e:
        print(f"ERROR: File not found. {e}")
        return None
    
    # Identify total number of unique trajectories
    num_traj = df_noisy['trajectory_id'].nunique()
    
    # Initialize empty Numpy arrays with shape [N_trajectories, N_features, T_time_steps]
    # y = measurements [N, 5, T]
    y_all = np.zeros((num_traj, len(measure_cols), T_steps), dtype=np.float32)
    # u = control inputs [N, 2, T]
    u_all = np.zeros((num_traj, len(control_cols), T_steps), dtype=np.float32)
    # x = ground truth states (targets) [N, 6, T]
    x_all = np.zeros((num_traj, len(state_cols), T_steps), dtype=np.float32)

    # Group dataframes by ID for faster sequential access
    grouped_noisy = df_noisy.groupby('trajectory_id')
    grouped_clean = df_clean.groupby('trajectory_id')

    # Iterate through each trajectory and fill the pre-allocated arrays
    for i in range(num_traj):
        try:
            traj_n = grouped_noisy.get_group(i)
            traj_c = grouped_clean.get_group(i)
            
            # Extract the first T_steps.
            # .values returns [T, C], so we use .T to transpose to [C, T]
            y_all[i, :, :] = traj_n[measure_cols].values[:T_steps, :].T
            u_all[i, :, :] = traj_n[control_cols].values[:T_steps, :].T
            x_all[i, :, :] = traj_c[state_cols].values[:T_steps, :].T
        except KeyError:
            print(f"ERROR: Trajectory {i} not found or corrupted. Aborting.")
            return None
            
    print(f"Reshaping completed. Total shapes: y={y_all.shape}, u={u_all.shape}, x={x_all.shape}")

    # Create and shuffle trajectory indices for dataset splitting
    indices = np.arange(num_traj)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    
    # Apply shuffled indices to all arrays simultaneously
    y_all = y_all[indices]
    u_all = u_all[indices]
    x_all = x_all[indices]
    
    # Calculate split points for training, validation, and testing
    n_train = int(num_traj * train_split)
    n_val = int(num_traj * val_split)
    
    # Split the data
    y_train = y_all[:n_train]
    u_train = u_all[:n_train]
    x_train = x_all[:n_train]
    
    y_val = y_all[n_train : n_train + n_val]
    u_val = u_all[n_train : n_train + n_val]
    x_val = x_all[n_train : n_train + n_val]
    
    y_test = y_all[n_train + n_val :]
    u_test = u_all[n_train + n_val :]
    x_test = x_all[n_train + n_val :]

    # Convert Numpy arrays to PyTorch tensors
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
    
    print("\nData loading completed successfully.")
    print(f"  Train set (y, u, x): {train_data[0].shape}, {train_data[1].shape}, {train_data[2].shape}")
    print(f"  Val set   (y, u, x): {val_data[0].shape}, {val_data[1].shape}, {val_data[2].shape}")
    print(f"  Test set  (y, u, x): {test_data[0].shape}, {test_data[1].shape}, {test_data[2].shape}")

    return train_data, val_data, test_data
