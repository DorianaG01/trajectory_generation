import torch
import sys
import os
import matplotlib.pyplot as plt
from datetime import datetime

from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset
from pipeline import Pipeline_Vehicle_KNet 
from kalman_net import KalmanNetNN  

if __name__ == "__main__":
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    noisy_csv = 'combined_dataset_clean.csv'
    clean_csv = 'combined_dataset_clean.csv'
    

    if not os.path.exists(clean_csv):
        print(f"WARNING: File {clean_csv} not found. Check the path.")

    
    strTime = datetime.now().strftime("%m%d_%H%M")
    path_results = f'KNet_TBPTT_Results_{strTime}'
    
    m, n = 6, 5         # Number of states and observations
    Ts = 0.01           # Sampling time
    T_full = 1200       # Full trajectory length
    
    CHUNK_SIZE = 200    # K_TBPTT: Chunk length for backpropagation
    N_EPOCHS = 500      # Number of Epochs
    BATCH_SIZE = 64     # Batch Size
    LR = 1e-4           # Learning Rate
    WD = 1e-5           # Weight Decay
    
    # Training Strategy: 
    # 'standard'     -> Updates weights every CHUNK_SIZE steps (Faster learning)
    # 'accumulation' -> Accumulates gradients and updates at end of trajectory (More stable)
    TRAIN_STRATEGY = 'standard' 

    #  DATA LOADING
    print("\n Data Loading ")

    try:
        (train_data, val_data, test_data) = load_vehicle_dataset(
            noisy_csv, clean_csv, T_steps=T_full, train_split=0.7, val_split=0.15
        )
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

    y_train_full, u_train_full, x_train_full = train_data
    print(f"Dataset Loaded. Train shape: {x_train_full.shape}")

    # PHYSICAL LIMITS EXTRACTION (MIN/MAX)

    x_min_val = x_train_full[:, 0, :].min().item()
    x_max_val = x_train_full[:, 0, :].max().item()
    y_min_val = x_train_full[:, 1, :].min().item()
    y_max_val = x_train_full[:, 1, :].max().item()
    vx_min_val = x_train_full[:, 3, :].min().item()
    vx_max_val = x_train_full[:, 3, :].max().item()
    vy_min_val = x_train_full[:, 4, :].min().item()
    vy_max_val = x_train_full[:, 4, :].max().item()
    phi_min_val = x_train_full[:, 2, :].min().item()
    phi_max_val = x_train_full[:, 2, :].max().item()
    omega_min_val = x_train_full[:, 5, :].min().item()
    omega_max_val = x_train_full[:, 5, :].max().item()

    
    print(f"Position X Limits: [{x_min_val:.2f}, {x_max_val:.2f}]")
    print(f"Position Y Limits: [{y_min_val:.2f}, {y_max_val:.2f}]")

    
    # NORMALIZATION
    
    print(" Calculating Normalization Statistics ")
    
    x_mean = torch.mean(x_train_full, dim=(0, 2)).reshape(1, m, 1)
    x_std  = torch.std(x_train_full, dim=(0, 2)).reshape(1, m, 1) + 1e-8
    
    y_mean = torch.mean(y_train_full, dim=(0, 2)).reshape(1, n, 1)
    y_std  = torch.std(y_train_full, dim=(0, 2)).reshape(1, n, 1) + 1e-8

    norm_stats = {
        "x_mean": x_mean.to(device), "x_std": x_std.to(device),
        "y_mean": y_mean.to(device), "y_std": y_std.to(device)
    }

    # MODEL INITIALIZATION
    print("\n Model Initialization ")
    
    m1x_0 = torch.zeros(m, 1).to(device)
    sys_model = VehicleModel(Ts, T_full, T_full, m1x_0, torch.eye(m), torch.eye(m), torch.eye(n))
    
    knet_model = KalmanNetNN()
    knet_model.to(device)
    
    knet_model.NNBuild(sys_model, in_mult_KNet=10, out_mult_KNet=40, hidden_dim_gru=128)
    
    # PIPELINE SETUP
    print("\n Pipeline Setup ")
    pipeline = Pipeline_Vehicle_KNet(path_results, "KNet_Vehicle_TBPTT")
    
    pipeline.set_models(sys_model, knet_model)
    pipeline.set_data(train_data, val_data, test_data, norm_stats, device)
    
    pipeline.set_training_params(
        n_steps=N_EPOCHS,
        n_batch=BATCH_SIZE,
        lr=LR,
        wd=WD,
        K_TBPTT=CHUNK_SIZE,   
        T=T_full,             
        training_strategy=TRAIN_STRATEGY, # 'standard' or 'accumulation'
        CompositionLoss=True, 
        alpha=0.8             # 0.8 * State_Loss + 0.2 * Observation_Loss
    )
    
    # TRAINING AND TEST
    print("\n START TRAINING ")
    pipeline.train()
    
    print("\n PLOTTING ")
    pipeline.plot_learning_curve()
    

    if hasattr(pipeline, 'plot_trajectories'):
         print("Generating trajectory plots")
         pipeline.plot_trajectories(sample_idx=0)
    
    print("\n START TEST ")
    pipeline.test()
    
    print(" END SCRIPT ")
