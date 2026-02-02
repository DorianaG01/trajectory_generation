import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time
import random

from kalman_net import KalmanNetNN
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset


PATH_RESULTS = "trajectory_generation/KNet_Vehicle_Results"
os.makedirs(PATH_RESULTS, exist_ok=True)

# Ensure this matches the specific model file you want to test
MODEL_FILENAME = "best_model_KNet" 
MODEL_PATH = os.path.join(PATH_RESULTS, MODEL_FILENAME)

# Dataset Paths
noisy_csv = "combined_dataset_noisy.csv"
clean_csv = "combined_dataset_clean.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

m, n = 6, 5
Ts = 0.01


H = 200             # Prediction Horizon (steps) -> 2 seconds
EVAL_STEP = 100     # Sliding Window step size
T_START_EVAL = 50   # Start step to avoid filter transients
T_OBSERVATION_VIS = 600 # Specific time step for visualization


VIS_IDXS = np.random.choice(1500, size=10, replace=False)  # Randomly select 10 indices


INIT_MODE = "noisy_gt"
INIT_NOISE_STD = 0.2
INIT_SEED = 0


@torch.no_grad()
def run_full_filter(model, y_norm, u, x0_norm):
    """
    Runs the filter on the full sequence.
    Returns the complete sequence of estimated states (normalized).
    """
    model.eval()
    model.batch_size = 1
    model.init_hidden_KNet()
    
    T_full = y_norm.shape[2]
    model.InitSequence(x0_norm, T_full)
    
    x_est_list = []
    for t in range(T_full):
        y_in = y_norm[:, :, t].unsqueeze(2)
        u_in = u[:, :, t].unsqueeze(2)
        x_out = model(y_in, u_in)
        x_est_list.append(x_out)
        
    return torch.cat(x_est_list, dim=2) # [1, m, T_full]

@torch.no_grad()
def rollout_open_loop(sys_model, x0_real, u, t_start_state, H):
    """
    Performs physical Open Loop prediction for H steps.
    x0_real: initial state (denormalized).
    """
    x_curr = x0_real
    preds = []
    T_max = u.shape[2]
    
    for k in range(H):
        t_u = t_start_state + k
        if t_u >= T_max: break
        
        u_k = u[:, :, t_u].unsqueeze(2)
        x_next = sys_model.f(x_curr, u_k)
        preds.append(x_next)
        x_curr = x_next
        
    if len(preds) == 0: return x0_real # Fallback
    return torch.cat(preds, dim=2)

def compute_metrics(pred_real, gt_real):
    """
    Computes ADE (Average Displacement Error) and FDE (Final Displacement Error) on XY.
    Returns scalars.
    """
    # Pred and GT are [1, m, H] -> take only x,y (channels 0 and 1)
    diff = pred_real[:, :2, :] - gt_real[:, :2, :] 
    
    # Euclidean distance for each step
    err_dist = torch.sqrt(torch.sum(diff**2, dim=1)) # [1, H]
    
    ade = err_dist.mean().item()
    fde = err_dist[0, -1].item()
    
    return ade, fde

def get_error_profile(pred_real, gt_real):
    """
    Computes error profile over time steps.
    Returns numpy array of length H.
    """
    diff = pred_real[:, :2, :] - gt_real[:, :2, :] 
    err_dist = torch.sqrt(torch.sum(diff**2, dim=1)) # [1, H]
    return err_dist.squeeze(0).cpu().numpy()


def main():
    print(f"--- Sliding Window Evaluation (ADE/FDE + Trends) ---")
    print(f"Model: {MODEL_FILENAME}")
    print(f"Horizon H={H}, Slide Step={EVAL_STEP}")
    print(f"Device: {device}")


    (train_full, _, test_full) = load_vehicle_dataset(
        noisy_csv, clean_csv, T_steps=1200, train_split=0.7, val_split=0.15
    )
    y_test, u_test, x_test = test_full
    y_tr, _, x_tr = train_full

    N_test = y_test.shape[0]

    x_mean = torch.mean(x_tr, dim=(0, 2)).reshape(1, m, 1).to(device)
    x_std  = torch.std(x_tr,  dim=(0, 2)).reshape(1, m, 1).to(device) + 1e-8
    y_mean = torch.mean(y_tr, dim=(0, 2)).reshape(1, n, 1).to(device)
    y_std  = torch.std(y_tr,  dim=(0, 2)).reshape(1, n, 1).to(device) + 1e-8
  
    dummy = torch.eye(m)
    sys_model = VehicleModel(Ts, 1200, 1200, torch.zeros(m, 1), dummy, dummy, torch.eye(n))
    
    sys_model.Params.update({
        "x_min": x_tr[:, 0, :].min().item(), "x_max": x_tr[:, 0, :].max().item(),
        "y_min": x_tr[:, 1, :].min().item(), "y_max": x_tr[:, 1, :].max().item(),
        "phi_min": x_tr[:, 2, :].min().item(), "phi_max": x_tr[:, 2, :].max().item(),
        "vx_min": x_tr[:, 3, :].min().item(), "vx_max": x_tr[:, 3, :].max().item(),
        "vy_min": x_tr[:, 4, :].min().item(), "vy_max": x_tr[:, 4, :].max().item(),
        "omega_min": x_tr[:, 5, :].min().item(), "omega_max": x_tr[:, 5, :].max().item()
    })

    model = KalmanNetNN()

    model.NNBuild(sys_model, in_mult_KNet=10, out_mult_KNet=40, hidden_dim_gru=128) 
    model.set_normalization(x_mean, x_std, y_mean, y_std)

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"ERROR loading weights: {e}")
        return

    model.to(device).eval()
    model.f = sys_model.f # Link physics function

    
    # SLIDING WINDOW LOOP
   
    all_ades = []
    all_fdes = []
    all_error_profiles = []
    
    torch.manual_seed(INIT_SEED)

    print(f"Evaluating {N_test} trajectories")
    
    for i in range(N_test):
        # Data Prep
        y = y_test[i].unsqueeze(0).to(device)
        u = u_test[i].unsqueeze(0).to(device)
        x_gt = x_test[i].unsqueeze(0).to(device)
        T_curr = y.shape[2]

        y_norm = (y - y_mean) / y_std
        x0_norm = (x_gt[:, :, 0] - x_mean.squeeze(2)) / x_std.squeeze(2)

  
        if INIT_MODE == "gt":
            x0n = x0_norm.unsqueeze(2)
        elif INIT_MODE == "noisy_gt":
            noise0 = torch.randn_like(x0_norm) * INIT_NOISE_STD
            x0n = (x0_norm + noise0).unsqueeze(2)
        else:
            raise ValueError("Invalid INIT_MODE")

        # 1. Closed Loop Filter
        x_est_full_norm = run_full_filter(model, y_norm, u, x0n)
        
        # 2. Sliding Window Prediction (Open Loop)
        for t in range(T_START_EVAL, T_curr - H, EVAL_STEP):
            
            # Start Point (Estimate at time t)
            x_start_norm = x_est_full_norm[:, :, t].unsqueeze(2)
            x_start_real = x_start_norm * x_std + x_mean
            
            # Future Ground Truth
            gt_future = x_gt[:, :, t+1 : t+1+H]
            
            # Rollout
            pred_real = rollout_open_loop(sys_model, x_start_real, u, t_start_state=t, H=H)
            
            # Dimensionality check (in case we are at the end of file and gt_future < H)
            current_H = pred_real.shape[2]
            if gt_future.shape[2] != current_H:
                continue

            ade, fde = compute_metrics(pred_real, gt_future)
            all_ades.append(ade)
            all_fdes.append(fde)
            
            err_profile = get_error_profile(pred_real, gt_future)
            all_error_profiles.append(err_profile)

    mean_ade = np.mean(all_ades)
    std_ade = np.std(all_ades)
    mean_fde = np.mean(all_fdes)
    std_fde = np.std(all_fdes)

    print("\n" + "="*40)
    print(f" RESULTS (Sliding Window over Test Set)")
    print("="*40)
    print(f"Total Prediction Windows: {len(all_ades)}")
    print(f"Horizon (H): {H} steps ({H*Ts} s)")
    print(f"ADE: {mean_ade:.4f} m  (std: {std_ade:.4f})")
    print(f"FDE: {mean_fde:.4f} m  (std: {std_fde:.4f})")
    print("="*40 + "\n")

    
    # PLOTTING DISTRIBUTIONS
  
    print("Generating statistical plots")

    # Boxplot
    plt.figure(figsize=(10, 6))
    plt.boxplot([all_ades, all_fdes], labels=["ADE", "FDE"], patch_artist=True, 
                boxprops=dict(facecolor="lightblue"), medianprops=dict(color="red"))
    plt.title("Error Distribution (ADE vs FDE)")
    plt.ylabel("Error [m]")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.savefig(os.path.join(PATH_RESULTS, "Error_Boxplot.png"), dpi=200)
    plt.show()

    # Histogram
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    ax1.hist(all_ades, bins=40, color="skyblue", edgecolor="black", alpha=0.7)
    ax1.set_title(f"ADE Distribution (Mean: {mean_ade:.2f}m)")
    ax1.set_xlabel("ADE [m]")
    ax1.axvline(mean_ade, color='r', linestyle='--')

    ax2.hist(all_fdes, bins=40, color="lightgreen", edgecolor="black", alpha=0.7)
    ax2.set_title(f"FDE Distribution (Mean: {mean_fde:.2f}m)")
    ax2.set_xlabel("FDE [m]")
    ax2.axvline(mean_fde, color='r', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATH_RESULTS, "Error_Histograms.png"), dpi=200)
    plt.show()

    # ERROR EVOLUTION OVER TIME
   
    error_matrix = np.array(all_error_profiles) 
    
    # Calculate mean and std deviation per time step
    mean_err_time = np.mean(error_matrix, axis=0) # [H]
    std_err_time = np.std(error_matrix, axis=0)   # [H]
    time_steps = np.arange(H) * Ts # X-axis in seconds

    plt.figure(figsize=(10, 6))
    
    plt.plot(time_steps, mean_err_time, label="Mean Error", color="blue", linewidth=2)
    
    plt.fill_between(time_steps, 
                     np.maximum(0, mean_err_time - std_err_time), 
                     mean_err_time + std_err_time, 
                     color="blue", alpha=0.2, label="Std Dev ($\pm 1\sigma$)")
    
    plt.title(f"Prediction Error Evolution (Horizon={H*Ts}s)")
    plt.xlabel("Prediction Time [s]")
    plt.ylabel("Euclidean Error [m]")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlim(0, H*Ts)
    
    plt.savefig(os.path.join(PATH_RESULTS, "Error_Evolution_Time.png"), dpi=200)
    print(f"Graph Saved: {os.path.join(PATH_RESULTS, 'Error_Evolution_Time.png')}")
    plt.show()


    # -----------------------
    # VISUALIZATION SINGLE TRAJ
    # -----------------------
    print(f"Visualizing specific trajectories")
    
    for idx in VIS_IDXS:
        if idx >= N_test: continue
        
        y = y_test[idx].unsqueeze(0).to(device)
        u = u_test[idx].unsqueeze(0).to(device)
        x_gt = x_test[idx].unsqueeze(0).to(device)
        
        y_norm = (y - y_mean) / y_std
        x0_norm = (x_gt[:, :, 0] - x_mean.squeeze(2)) / x_std.squeeze(2)
        
        torch.manual_seed(INIT_SEED + idx) 
        if INIT_MODE == "noisy_gt":
            x0n = (x0_norm + torch.randn_like(x0_norm)*INIT_NOISE_STD).unsqueeze(2)
        else:
            x0n = x0_norm.unsqueeze(2)

        # Filter
        x_est_full_norm = run_full_filter(model, y_norm, u, x0n)
        x_est_full_real = x_est_full_norm * x_std + x_mean
        
        # Prediction
        t_start = T_OBSERVATION_VIS
        if t_start + H >= x_gt.shape[2]: continue # Skip if out of bounds

        x_start_pred_real = x_est_full_real[:, :, t_start].unsqueeze(2)
        pred_real = rollout_open_loop(sys_model, x_start_pred_real, u, t_start, H)
        
  
        gt_np = x_gt.squeeze(0).cpu().numpy()
        meas_np = y.squeeze(0).cpu().numpy()
        est_np = x_est_full_real.squeeze(0).cpu().numpy()
        pred_np = pred_real.squeeze(0).cpu().numpy()
        
    
        plt.figure(figsize=(10, 6))
        
   
        plt.plot(gt_np[0, :], gt_np[1, :], 'k--', alpha=0.5, label="Ground Truth")

        plt.scatter(meas_np[0, :t_start], meas_np[1, :t_start], s=5, c='r', alpha=0.3, label="GPS Data")
 
        plt.plot(est_np[0, :t_start], est_np[1, :t_start], 'b-', linewidth=1.5, label="Filter Estimate")
      
        pred_x = np.concatenate(([est_np[0, t_start]], pred_np[0, :]))
        pred_y = np.concatenate(([est_np[1, t_start]], pred_np[1, :]))
        plt.plot(pred_x, pred_y, 'g-', linewidth=3, label=f"Prediction ({H*Ts}s)")
        
        plt.scatter(est_np[0, t_start], est_np[1, t_start], c='orange', s=100, marker='X', zorder=5)

        plt.title(f"Trajectory {idx} | Open Loop Prediction")
        plt.xlabel("X [m]")
        plt.ylabel("Y [m]")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.axis('equal')
        
        out_name = os.path.join(PATH_RESULTS, f"Vis_Traj{idx}.png")
        plt.savefig(out_name, dpi=200)
        plt.show()

if __name__ == "__main__":
    main()
