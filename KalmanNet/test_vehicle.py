import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import os
import random 

from kalman_net import KalmanNetNN
from pipeline import Pipeline_EKF
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset

path_results = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/KNet_Vehicle_Results'

m = 6 # Dimensione Stato
n = 5 # Dimensione Osservazione
d = 2 # Dimensione Controllo

Ts = 0.02 
T = 600  
T_test = 600 

noisy_csv = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/piecewise_mpc_noisy.csv'  
clean_csv = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/piecewise_mpc_clean.csv'

# Stato iniziale e prior 
m1x_0 = torch.zeros(m, 1) 
prior_Q = torch.eye(m)
prior_Sigma = torch.eye(m)
prior_S = torch.eye(n)

sys_model = VehicleModel(Ts, T, T_test, m1x_0, prior_Q, prior_Sigma, prior_S)

# Modello KalmanNet
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model) 
print("Modello KalmanNet costruito.")

print("Caricamento dati.")
(train_data, _, test_data) = load_vehicle_dataset(
    noisy_csv, 
    clean_csv, 
    T_steps=T, 
    train_split=0.7, 
    val_split=0.15   
)

_, _, x_train = train_data
y_test, u_test, x_test = test_data

test_input = y_test
test_target = x_test
test_control = u_test
print(f"Dati di test caricati: {test_input.shape[0]} sequenze.")


print("Calcolo e impostazione dei limiti min/max")
# Indici: 0:X, 1:Y, 2:phi, 3:vx, 4:vy, 5:omega
x_min = x_train[:, 0, :].min().item()
x_max = x_train[:, 0, :].max().item()
y_min = x_train[:, 1, :].min().item()
y_max = x_train[:, 1, :].max().item()
phi_min = x_train[:, 2, :].min().item()
phi_max = x_train[:, 2, :].max().item()
vx_min = x_train[:, 3, :].min().item()
vx_max = x_train[:, 3, :].max().item()
vy_min = x_train[:, 4, :].min().item()
vy_max = x_train[:, 4, :].max().item()
omega_min = x_train[:, 5, :].min().item()
omega_max = x_train[:, 5, :].max().item()

# Imposta i limiti nel modello di sistema
sys_model.Params["x_min"] = x_min
sys_model.Params["x_max"] = x_max
sys_model.Params["y_min"] = y_min
sys_model.Params["y_max"] = y_max
sys_model.Params["phi_min"] = phi_min
sys_model.Params["phi_max"] = phi_max
sys_model.Params["vx_min"] = vx_min
sys_model.Params["vx_max"] = vx_max
sys_model.Params["vy_min"] = vy_min
sys_model.Params["vy_max"] = vy_max
sys_model.Params["omega_min"] = omega_min
sys_model.Params["omega_max"] = omega_max


KalmanNet_Pipeline = Pipeline_EKF("TestTime", path_results, "KalmanNet_Vehicle")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)

KalmanNet_Pipeline.setTrainingParams(n_steps=1, n_batch=1, lr=1e-5, wd=1e-5, alpha=0.5)

print(f"Esecuzione NNTest sul modello: {path_results + 'best_model.pt'}")
[MSE_test_linear_arr, MSE_test_linear_avg, MSE_test_dB_avg, 
 x_out_test, t, KalmanGainKN, MSESingleTrajectory] = \
    KalmanNet_Pipeline.NNTest(sys_model, 
                             test_input, 
                             test_target, 
                             test_control, 
                             path_results)

print(f"TEST COMPLETATO. MSE MEDIO: {MSE_test_dB_avg:.2f} dB")


print("Plotting di una traiettoria di esempio")

idx_to_plot = random.randint(0, test_input.shape[0] - 1) 

fig, axs = plt.subplots(3, 2, figsize=(15, 12)) # 3x2 per 6 stati
states = ['Stato X [m]', 'Stato Y [m]', 'Stato phi [rad]', 
          'Stato vx [m/s]', 'Stato vy [m/s]', 'Stato omega [rad/s]']

for i in range(m):
    row = i // 2
    col = i % 2
    
    # Target (pulito)
   
    axs[row, col].plot(range(T_test), test_target[idx_to_plot, i, :].detach().numpy(), 'k', label='Ground Truth (Pulito)')
    
    # Stima KNet
    axs[row, col].plot(range(T_test), x_out_test[idx_to_plot, i, :].detach().numpy(), 'r--', label='Stima KalmanNet')
    
    # Osservazioni rumorose
    if i == 0: # X
        axs[row, col].plot(range(T_test), test_input[idx_to_plot, 0, :].detach().numpy(), 'g:', alpha=0.5, label='Osservazione (Rumorosa)')
    elif i == 1: # Y
        axs[row, col].plot(range(T_test), test_input[idx_to_plot, 1, :].detach().numpy(), 'g:', alpha=0.5, label='Osservazione (Rumorosa)')
    elif i == 3: # vx
        axs[row, col].plot(range(T_test), test_input[idx_to_plot, 2, :].detach().numpy(), 'g:', alpha=0.5, label='Osservazione (Rumorosa)')
    elif i == 4: # vy
        axs[row, col].plot(range(T_test), test_input[idx_to_plot, 3, :].detach().numpy(), 'g:', alpha=0.5, label='Osservazione (Rumorosa)')
    elif i == 5: # omega
        axs[row, col].plot(range(T_test), test_input[idx_to_plot, 4, :].detach().numpy(), 'g:', alpha=0.5, label='Osservazione (Rumorosa)')

    axs[row, col].set_title(f'Stato: {states[i]}')
    axs[row, col].set_xlabel('Time Step')
    axs[row, col].set_ylabel('Valore')
    axs[row, col].legend()
    axs[row, col].grid(True)

plt.tight_layout()
plt.savefig(path_results + 'test_trajectory_example.png')
print(f"Plot di esempio salvato in {path_results}test_trajectory_example.png")
plt.show()
