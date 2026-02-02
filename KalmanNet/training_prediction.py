import torch
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# --- Custom Modules ---
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset
from kalman_net import KalmanNetNN  

if __name__ == "__main__":

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Path to the trained model weights
    model_path = 'KNet_Results/best_model_KNet_Vehicle.pt' 
    
    noisy_csv = 'combined_dataset_clean.csv' # Need data to calc stats
    clean_csv = 'combined_dataset_clean.csv'
    
    plot_idx = 0        # Index of test trajectory to visualize
    save_plots = True   # Save figures to disk
    output_dir = 'Prediction_Results'
    
    if not os.path.exists(model_path):
        print(f"ERROR: Model file {model_path} not found.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)

    m, n = 6, 5         # States and Observations
    Ts = 0.01           # Sampling time
    T_full = 1200       # Trajectory length
    
  
    in_mult = 5
    out_mult = 40
    hidden_gru = 128

    print("\n Data Loading ")
    try:
        
        (train_data, val_data, test_data) = load_vehicle_dataset(
            noisy_csv, clean_csv, T_steps=T_full, train_split=0.7, val_split=0.15
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    y_train_full, u_train_full, x_train_full = train_data
    y_test, u_test, x_test = test_data
    
    y_test = y_test.to(device)
    u_test = u_test.to(device)
    x_test = x_test.to(device)
    
    print(f"Dataset Loaded. Test shape: {x_test.shape}")
    
    x_min_val = x_train_full[:, 0, :].min().item()
    x_max_val = x_train_full[:, 0, :].max().item()
    y_min_val = x_train_full[:, 1, :].min().item()
    y_max_val = x_train_full[:, 1, :].max().item()
    phi_min_val = x_train_full[:, 2, :].min().item()
    phi_max_val = x_train_full[:, 2, :].max().item()
    vx_min_val = x_train_full[:, 3, :].min().item()
    vx_max_val = x_train_full[:, 3, :].max().item()
    vy_min_val = x_train_full[:, 4, :].min().item()
    vy_max_val = x_train_full[:, 4, :].max().item()
    omega_min_val = x_train_full[:, 5, :].min().item()
    omega_max_val = x_train_full[:, 5, :].max().item()
    
    print(f"Position X Limits: [{x_min_val:.2f}, {x_max_val:.2f}]")
    print(f"Position Y Limits: [{y_min_val:.2f}, {y_max_val:.2f}]")


    print(" Calculating Normalization Statistics ")
    
    x_mean = torch.mean(x_train_full, dim=(0, 2)).reshape(1, m, 1).to(device)
    x_std  = torch.std(x_train_full, dim=(0, 2)).reshape(1, m, 1).to(device) + 1e-8
    
    y_mean = torch.mean(y_train_full, dim=(0, 2)).reshape(1, n, 1).to(device)
    y_std  = torch.std(y_train_full, dim=(0, 2)).reshape(1, n, 1).to(device) + 1e-8

    print("\n Model Initialization ")
    
    m1x_0 = torch.zeros(m, 1).to(device)
    sys_model = VehicleModel(Ts, T_full, T_full, m1x_0, torch.eye(m), torch.eye(m), torch.eye(n))
    
    sys_model.Params.update({
        "x_min": x_min_val, "x_max": x_max_val,
        "y_min": y_min_val, "y_max": y_max_val,
        "phi_min": phi_min_val, "phi_max": phi_max_val,
        "vx_min": vx_min_val, "vx_max": vx_max_val,
        "vy_min": vy_min_val, "vy_max": vy_max_val,
        "omega_min": omega_min_val, "omega_max": omega_max_val
    })

    knet_model = KalmanNetNN()
    knet_model.NNBuild(sys_model, in_mult_KNet=in_mult, out_mult_KNet=out_mult, hidden_dim_gru=hidden_gru)
    knet_model.set_normalization(x_mean, x_std, y_mean, y_std)
    
    print(f"Loading weights from {model_path}")
    knet_model.load_state_dict(torch.load(model_path, map_location=device))
    knet_model.to(device)
    knet_model.eval()

    print("\n START PREDICTION ")
    
    y_test_norm = (y_test - y_mean) / y_std
    
    N_test = y_test.shape[0]
    T_test = y_test.shape[2]
    x_est_list = []

    with torch.no_grad():
        knet_model.batch_size = N_test
        knet_model.init_hidden_KNet()
        
        init_noise = torch.randn_like(x_test[:, :, 0]) * 0.1 
        x_0_init_norm = (x_test[:, :, 0] - x_mean.squeeze(2)) / x_std.squeeze(2)
        m1x_0_batch = (x_0_init_norm + init_noise).unsqueeze(2)
        
        knet_model.InitSequence(m1x_0_batch, T_test)
        
        for t in range(T_test):
            y_in = y_test_norm[:, :, t].unsqueeze(2)
            u_in = u_test[:, :, t].unsqueeze(2)
            
            # Forward pass
            x_out_norm = knet_model(y_in, u_in)
            
            # Denormalize
            x_out_real = knet_model._denorm_x(x_out_norm)
            x_est_list.append(x_out_real.squeeze(2))

    x_est_tensor = torch.stack(x_est_list, dim=2)

    mse = torch.nn.functional.mse_loss(x_est_tensor, x_test)
    mse_db = 10 * torch.log10(mse)
    print(f"Test MSE: {mse.item():.6f}")
    print(f"Test MSE [dB]: {mse_db.item():.2f} dB")

    print(f"\n PLOTTING (Index {plot_idx})")
    
    GT = x_test[plot_idx].cpu().numpy()
    EST = x_est_tensor[plot_idx].cpu().numpy()
    t_axis = np.arange(T_test) * Ts

    # Plot 1: Trajectory
    plt.figure(figsize=(10, 8))
    plt.plot(GT[0, :], GT[1, :], 'k--', label='Ground Truth', linewidth=2)
    plt.plot(EST[0, :], EST[1, :], 'b-', label='KNet Estimate', linewidth=1.5)
    plt.title(f"Trajectory - MSE: {mse_db.item():.2f} dB")
    plt.xlabel("X [m]")
    plt.ylabel("Y [m]")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'traj_{plot_idx}.png'))
    plt.show()

    # Plot 2: States
    state_names = ['x', 'y', 'phi', 'vx', 'vy', 'omega']
    fig, axs = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle(f"State Estimation - Test ID {plot_idx}")
    
    axs = axs.ravel()
    for i in range(m):
        axs[i].plot(t_axis, GT[i, :], 'k--', label='GT')
        axs[i].plot(t_axis, EST[i, :], 'b-', label='Est')
        axs[i].set_title(state_names[i])
        axs[i].grid(True)
        if i == 0: axs[i].legend()
    
    plt.tight_layout()
    if save_plots:
        plt.savefig(os.path.join(output_dir, f'states_{plot_idx}.png'))
    plt.show()

    print(" END SCRIPT ")
