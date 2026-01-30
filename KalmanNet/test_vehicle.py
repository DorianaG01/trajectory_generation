import torch
import os
import time
import random
import matplotlib.pyplot as plt

from kalman_net import KalmanNetNN
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset

PHI_IDX = 2
EPS_DB = 1e-12


def _angular_mse_from_real(phi_pred_real, phi_tgt_real):
    """Angular MSE handling 2π periodicity."""
    diff = torch.atan2(torch.sin(phi_pred_real - phi_tgt_real),
                       torch.cos(phi_pred_real - phi_tgt_real))
    return (diff ** 2).mean()


def loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx):
    """
    MSE on non-phi channels (normalized space) +
    angular MSE on phi channel (real space).
    x_*_norm shape: [B, m, T]
    """
    diff2 = (x_out_norm - x_tgt_norm) ** 2
    idx_no_phi = [i for i in range(m) if i != phi_idx]
    mse_no_phi = diff2[:, idx_no_phi, :].mean()

    # Denormalize
    x_out_real = (x_out_norm * x_std) + x_mean
    x_tgt_real = (x_tgt_norm * x_std) + x_mean

    phi_pred = x_out_real[:, phi_idx, :]
    phi_tgt = x_tgt_real[:, phi_idx, :]
    mse_phi_ang = _angular_mse_from_real(phi_pred, phi_tgt)

    return (float(m - 1) / m) * mse_no_phi + (1.0 / m) * mse_phi_ang


# ------------------ PATHS ------------------
path_results = "trajectory_generation/KNet_Vehicle_Results"
os.makedirs(path_results, exist_ok=True)

# Model file name as saved by pipeline.py: best_model_{modelName}.pt
model_name = "KNet_Vehicle_Model" # TO EDIT
model_path = os.path.join(path_results, f"best_model_{model_name}.pt")

# ------------------ PARAMETERS ------------------
m, n = 6, 5
Ts = 0.01
T = 1200
T_test = 1200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

m1x_0 = torch.zeros(m, 1).to(device)
prior_Q = torch.eye(m).to(device)
prior_Sigma = torch.eye(m).to(device)
prior_S = torch.eye(n).to(device)

noisy_csv = "combined_dataset_noisy.csv"
clean_csv = "combined_dataset_clean.csv"

# ------------------ LOAD DATA ------------------
sys_model = VehicleModel(Ts, T, T_test, m1x_0, prior_Q, prior_Sigma, prior_S)
(train_data, val_data, test_data) = load_vehicle_dataset(
    noisy_csv, clean_csv, T_steps=T, train_split=0.7, val_split=0.15
)

y_test, u_test, x_test = test_data
print(f"Test data loaded: {y_test.shape[0]} samples.")

# Compute normalization statistics from TRAIN set
y_train, _, x_train = train_data
x_train_mean = torch.mean(x_train, dim=(0, 2)).reshape(1, m, 1).to(device)
x_train_std = torch.std(x_train, dim=(0, 2)).reshape(1, m, 1).to(device) + 1e-8
y_train_mean = torch.mean(y_train, dim=(0, 2)).reshape(1, n, 1).to(device)
y_train_std = torch.std(y_train, dim=(0, 2)).reshape(1, n, 1).to(device) + 1e-8
print("Normalization statistics computed from training set.")

# Set physical limits in the system model
sys_model.Params.update({
    "x_min": x_train[:, 0, :].min().item(), "x_max": x_train[:, 0, :].max().item(),
    "y_min": x_train[:, 1, :].min().item(), "y_max": x_train[:, 1, :].max().item(),
    "phi_min": x_train[:, 2, :].min().item(), "phi_max": x_train[:, 2, :].max().item(),
    "vx_min": x_train[:, 3, :].min().item(), "vx_max": x_train[:, 3, :].max().item(),
    "vy_min": x_train[:, 4, :].min().item(), "vy_max": x_train[:, 4, :].max().item(),
    "omega_min": x_train[:, 5, :].min().item(), "omega_max": x_train[:, 5, :].max().item()
})
print("Physical limits set in the system model.")

# ------------------ LOAD MODEL ------------------
model = KalmanNetNN()
model.NNBuild(sys_model)
model.to(device)
model.set_normalization(x_train_mean, x_train_std, y_train_mean, y_train_std)

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found: {model_path}")

model.load_state_dict(torch.load(model_path, map_location=device))
print(f"Model successfully loaded from {model_path}")

# ------------------ TEST ------------------
model.eval()

y_test = y_test.to(device)
u_test = u_test.to(device)
x_test = x_test.to(device)

test_input_norm = (y_test - y_train_mean) / y_train_std
test_target_norm = (x_test - x_train_mean) / x_train_std

N_TEST = y_test.shape[0]
T_TEST = y_test.shape[2]

start_time = time.time()

with torch.no_grad():
    model.batch_size = N_TEST
    model.init_hidden_KNet()

    # Hybrid initialization 
    y0 = test_input_norm[:, :, 0]
    m1x_0_hybrid = torch.zeros(N_TEST, m, 1).to(device)
    m1x_0_hybrid[:, 0, 0] = y0[:, 0]  # px
    m1x_0_hybrid[:, 1, 0] = y0[:, 1]  # py
    # phi remains 0
    m1x_0_hybrid[:, 3, 0] = y0[:, 2]  # vx
    m1x_0_hybrid[:, 4, 0] = y0[:, 3]  # vy
    m1x_0_hybrid[:, 5, 0] = y0[:, 4]  # omega

    model.InitSequence(m1x_0_hybrid, T_TEST)

    x_out_test_norm_list = []
    for t in range(T_TEST):
        x_out_t = torch.squeeze(
            model(torch.unsqueeze(test_input_norm[:, :, t], 2),
                  torch.unsqueeze(u_test[:, :, t], 2))
        )
        x_out_test_norm_list.append(x_out_t)

    x_out_test_norm = torch.stack(x_out_test_norm_list, dim=2)

    loss_x_test = loss_x_with_angular(
        x_out_test_norm, test_target_norm,
        x_train_mean, x_train_std, m, PHI_IDX
    )

end_time = time.time()
test_time = end_time - start_time

test_loss_linear = loss_x_test.item()
test_loss_db = 10 * torch.log10(torch.tensor(test_loss_linear).clamp_min(EPS_DB))

print("\n--- TEST RESULTS ---")
print(f"Model: {model_path}")
print(f"Total inference time: {test_time:.4f} s")
print(f"Average time per sample: {test_time / max(1, N_TEST):.6f} s")
print(f"Test MSE (linear): {test_loss_linear:.8f}")
print(f"Test MSE (dB):     {test_loss_db:.4f} dB")

# ------------------ PLOT ONE RANDOM TRAJECTORY ------------------
idx_to_plot = random.randint(0, N_TEST - 1)
print(f"Plotting trajectory index: {idx_to_plot}")

x_out_test_denorm = (x_out_test_norm * x_train_std) + x_train_mean

test_target_np = x_test.cpu().numpy()
test_input_np = y_test.cpu().numpy()
x_out_test_np = x_out_test_denorm.cpu().numpy()

fig, axs = plt.subplots(3, 2, figsize=(15, 12))
states = [
    "State X [m]", "State Y [m]", "State phi [rad]",
    "State vx [m/s]", "State vy [m/s]", "State omega [rad/s]"
]

obs_map = {0: 0, 1: 1, 3: 2, 4: 3, 5: 4}

for i in range(m):
    row, col = i // 2, i % 2

    axs[row, col].plot(range(T_TEST), test_target_np[idx_to_plot, i, :],
                       label="Ground Truth")
    axs[row, col].plot(range(T_TEST), x_out_test_np[idx_to_plot, i, :],
                       linestyle="--", label="KalmanNet Estimate")

    if i in obs_map:
        j = obs_map[i]
        axs[row, col].plot(
            range(T_TEST), test_input_np[idx_to_plot, j, :],
            linestyle="None", marker=".", markersize=3, alpha=0.35,
            label="Noisy Observation", zorder=0
        )

    axs[row, col].set_title(states[i])
    axs[row, col].set_xlabel("Time Step")
    axs[row, col].set_ylabel("Value")
    axs[row, col].legend()
    axs[row, col].grid(True)

plt.suptitle(
    f"Test Example - Trajectory {idx_to_plot} | Avg MSE: {test_loss_db:.2f} dB",
    fontsize=16
)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
out_plot = os.path.join(path_results, "test_trajectory_example.png")
plt.savefig(out_plot, dpi=150)
print(f"Example plot saved in: {out_plot}")
plt.show()
plt.close()
