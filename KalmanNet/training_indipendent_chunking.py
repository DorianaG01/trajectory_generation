import torch
import numpy as np
import os
import wandb
from datetime import datetime
import random

from kalman_net import KalmanNetNN
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset
from pipeline_chunk_shuffle import Pipeline_Vehicle_KNet


def create_chunks(data_tensor, chunk_size):
    """
    Splits tensors [N_traj, features, T_full] into [N_total_chunks, features, T_chunk].
    """
    N_traj, n_feats, T_full = data_tensor.shape
    
    # Calculate integer number of chunks
    num_chunks_per_traj = T_full // chunk_size
    
    valid_len = num_chunks_per_traj * chunk_size
    data_truncated = data_tensor[:, :, :valid_len]
    
    # Reshape: [N, F, n_chunks, T_chunk]
    data_view = data_truncated.view(N_traj, n_feats, num_chunks_per_traj, chunk_size)
    
    # Permute: [N, n_chunks, F, T_chunk]
    data_permuted = data_view.permute(0, 2, 1, 3)
  
    data_final = data_permuted.reshape(-1, n_feats, chunk_size)
    
    return data_final

CONFIG = {
    # Paths
    "path_results": "KNet_Vehicle_Results",
    "noisy_csv": "combined_dataset_noisy.csv", 
    "clean_csv": "combined_dataset_clean.csv",       
    
    "m": 6, "n": 5, "d": 2, # States, Observations, Control
    "Ts": 0.01,
    "T_full": 1200,    # Total trajectory length
    "T_train": 200,    # Training chunk length
    "T_val": 1200,     # Validation length (full)
    
    # Training Hyperparameters
    "n_epochs": 100,      
    "batch_size": 64,
    "lr": 1e-4,
    "wd": 1e-5,
    "alpha": 0.8,         
    "use_composite": True,
    
    # KNet Architecture
    "in_mult": 5,
    "out_mult": 40,
    "hidden_gru": 128,
    
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

def main():

    print(f"Using device: {CONFIG['device']}")
    
    timestamp = datetime.now().strftime("%m.%d.%Y_%H-%M-%S")
    model_name = f"KNet_Vehicle_CompositeLoss_Chunked_{timestamp}"
    

    m1x_0 = torch.zeros(CONFIG['m'], 1)
    prior_Q = torch.eye(CONFIG['m'])
    prior_Sigma = torch.eye(CONFIG['m'])
    prior_S = torch.eye(CONFIG['n'])
    
    sys_model = VehicleModel(CONFIG['Ts'], CONFIG['T_full'], CONFIG['T_val'], 
                             m1x_0, prior_Q, prior_Sigma, prior_S)


    print("Loading dataset")
    train_data, val_data, test_data = load_vehicle_dataset(
        CONFIG['noisy_csv'], CONFIG['clean_csv'], 
        T_steps=CONFIG['T_full'], 
        train_split=0.7, val_split=0.15
    )
    
    y_train_full, u_train_full, x_train_full = train_data
    y_val, u_val, x_val = val_data
    y_test, u_test, x_test = test_data

    # Calculate Normalization Statistics (on full training set)
    print("Calculating normalization statistics")
    x_mean = torch.mean(x_train_full, dim=(0, 2)).reshape(1, CONFIG['m'], 1).to(CONFIG['device'])
    x_std = torch.std(x_train_full, dim=(0, 2)).reshape(1, CONFIG['m'], 1).to(CONFIG['device']) + 1e-8
    y_mean = torch.mean(y_train_full, dim=(0, 2)).reshape(1, CONFIG['n'], 1).to(CONFIG['device'])
    y_std = torch.std(y_train_full, dim=(0, 2)).reshape(1, CONFIG['n'], 1).to(CONFIG['device']) + 1e-8
    
    norm_stats = {
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std
    }
    
    # Calculate Min/Max (from full training set) 
    # State indices: 0(x), 1(y), 2(phi), 3(vx), 4(vy), 5(omega)
    x_min = x_train_full[:, 0, :].min().item(); x_max = x_train_full[:, 0, :].max().item()
    y_min = x_train_full[:, 1, :].min().item(); y_max = x_train_full[:, 1, :].max().item()
    phi_min = x_train_full[:, 2, :].min().item(); phi_max = x_train_full[:, 2, :].max().item()
    vx_min = x_train_full[:, 3, :].min().item(); vx_max = x_train_full[:, 3, :].max().item()
    vy_min = x_train_full[:, 4, :].min().item(); vy_max = x_train_full[:, 4, :].max().item()
    omega_min = x_train_full[:, 5, :].min().item(); omega_max = x_train_full[:, 5, :].max().item()

    # Update Physical Model Parameters
    sys_model.Params.update({
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "phi_min": phi_min, "phi_max": phi_max,
        "vx_min": vx_min, "vx_max": vx_max,
        "vy_min": vy_min, "vy_max": vy_max,
        "omega_min": omega_min, "omega_max": omega_max
    })
    
    print("Physical limits updated in VehicleModel:")
    print(f"  Pos X: [{x_min:.2f}, {x_max:.2f}] | Pos Y: [{y_min:.2f}, {y_max:.2f}]")
    print(f"  Vel:   [{vx_min:.2f}, {vx_max:.2f}] | [{vy_min:.2f}, {vy_max:.2f}]")

    print(f"Creating training chunks (Len: {CONFIG['T_train']})...")
    y_train_chunks = create_chunks(y_train_full, CONFIG['T_train'])
    u_train_chunks = create_chunks(u_train_full, CONFIG['T_train'])
    x_train_chunks = create_chunks(x_train_full, CONFIG['T_train'])
    
    print(f"Original Training Dataset: {y_train_full.shape}") # e.g. [700, 5, 1200]
    print(f"Chunked Training Dataset:   {y_train_chunks.shape}") # e.g. [4200, 5, 200]
    
    train_data_chunks = (y_train_chunks, u_train_chunks, x_train_chunks)
    val_data_full = (y_val, u_val, x_val)
    test_data_full = (y_test, u_test, x_test)

    knet_model = KalmanNetNN()
    knet_model.NNBuild(sys_model, 
                       in_mult_KNet=CONFIG['in_mult'], 
                       out_mult_KNet=CONFIG['out_mult'], 
                       hidden_dim_gru=CONFIG['hidden_gru'])
    knet_model.set_normalization(x_mean, x_std, y_mean, y_std)
    knet_model.to(CONFIG['device'])


    pipeline = Pipeline_Vehicle_KNet(CONFIG['path_results'], model_name)
    
    pipeline.set_models(sys_model, knet_model)
    
    pipeline.set_data(train_data_chunks, val_data_full, test_data_full, 
                      norm_stats, CONFIG['device'])
    
    pipeline.set_training_params(
        n_steps=CONFIG['n_epochs'],
        n_batch=CONFIG['batch_size'],
        lr=CONFIG['lr'],
        wd=CONFIG['wd'],
        T_chunk_train=CONFIG['T_train'],
        CompositionLoss=CONFIG['use_composite'],
        alpha=CONFIG['alpha']
    )
    try:
        pipeline.train()
        pipeline.plot_learning_curve()
        pipeline.test()
    except KeyboardInterrupt:
        print("\nTraining manually interrupted.")
        wandb.finish()

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    main()
