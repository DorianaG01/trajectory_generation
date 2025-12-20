import torch
import torch.nn as nn
import os
import time
import random  
from datetime import datetime
import matplotlib.pyplot as plt

from kalman_net import KalmanNetNN 
from vehicle_model import VehicleModel 
from data_loader import load_vehicle_dataset 

PHI_IDX = 2
EPS_DB = 1e-12

def _angular_mse_from_real(phi_pred_real, phi_tgt_real):
    diff = torch.atan2(torch.sin(phi_pred_real - phi_tgt_real),
                       torch.cos(phi_pred_real - phi_tgt_real))
    return (diff ** 2).mean()

def loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx):
    diff2 = (x_out_norm - x_tgt_norm) ** 2
    idx_no_phi = [i for i in range(m) if i != phi_idx]
    mse_no_phi = diff2[:, idx_no_phi, :].mean() 
    x_out_real = (x_out_norm * x_std) + x_mean
    x_tgt_real = (x_tgt_norm * x_std) + x_mean
    phi_pred = x_out_real[:, phi_idx, :]
    phi_tgt  = x_tgt_real[:, phi_idx, :]
    mse_phi_ang = _angular_mse_from_real(phi_pred, phi_tgt)
    return (float(m-1) / m) * mse_no_phi + (1.0 / m) * mse_phi_ang


path_results = '/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/KalmanNet/KNet_Vehicle_Results'
model_path = os.path.join(path_results, "best_model_KNet_Vehicle_CompositeLoss_12.17.25_11-28-18.pt") 
#best_model_KNet_Vehicle_CompositeLoss_11.22.25_23-36-55.pt alpha=0.9
#best_model_KNet_Vehicle_CompositeLoss_11.22.25_15-05-10.pt alpha=0.7

m, n, d = 6, 5, 2
Ts = 0.01
T = 1200
T_test = 1200

device = torch.device('cpu')
print("Using CPU")

m1x_0 = torch.zeros(m, 1) 
prior_Q, prior_Sigma, prior_S = torch.eye(m), torch.eye(m), torch.eye(n)

noisy_csv = '/Users/dorianagiovarruscio/combined_dataset_noisy_err005.csv'  
clean_csv = '/Users/dorianagiovarruscio/combined_dataset_clean.csv'

sys_model = VehicleModel(Ts, T, T_test, m1x_0, prior_Q, prior_Sigma, prior_S)
(train_data, val_data, test_data) = load_vehicle_dataset(
    noisy_csv, clean_csv, T_steps=T, train_split=0.7, val_split=0.15   
)
y_test, u_test, x_test = test_data
print(f"Dati di test caricati: {y_test.shape[0]} campioni.")

# Calcola media/std SUL TRAIN SET
y_train, _, x_train = train_data
x_train_mean = torch.mean(x_train, dim=(0, 2)).reshape(1, m, 1).to(device) 
x_train_std = torch.std(x_train, dim=(0, 2)).reshape(1, m, 1).to(device) + 1e-8
y_train_mean = torch.mean(y_train, dim=(0, 2)).reshape(1, n, 1).to(device)
y_train_std = torch.std(y_train, dim=(0, 2)).reshape(1, n, 1).to(device) + 1e-8
print("Parametri di normalizzazione (da train set) calcolati.")

# Imposta i limiti del modello
x_min = x_train[:, 0, :].min().item(); x_max = x_train[:, 0, :].max().item()
y_min = x_train[:, 1, :].min().item(); y_max = x_train[:, 1, :].max().item()
phi_min = x_train[:, 2, :].min().item(); phi_max = x_train[:, 2, :].max().item()
vx_min = x_train[:, 3, :].min().item(); vx_max = x_train[:, 3, :].max().item()
vy_min = x_train[:, 4, :].min().item(); vy_max = x_train[:, 4, :].max().item()
omega_min = x_train[:, 5, :].min().item(); omega_max = x_train[:, 5, :].max().item()
sys_model.Params.update({
    "x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
    "phi_min": phi_min, "phi_max": phi_max, "vx_min": vx_min, "vx_max": vx_max,
    "vy_min": vy_min, "vy_max": vy_max, "omega_min": omega_min, "omega_max": omega_max
})
print("Limiti del modello impostati.")


model = KalmanNetNN()
model.NNBuild(sys_model) 
model.to(device)
model.set_normalization(
    x_train_mean, x_train_std, y_train_mean, y_train_std
)

try:
    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Modello V1 caricato con successo da {model_path}")
except Exception as e:
    print(f"ERRORE: Impossibile caricare il modello. {e}")
    exit()


print("Inizio Test.")
model.eval() 

N_TEST = y_test.shape[0]
T_TEST = y_test.shape[2] 

y_test = y_test.to(device)
u_test = u_test.to(device)
x_test = x_test.to(device)

test_input_norm = (y_test - y_train_mean) / y_train_std
test_target_norm = (x_test - x_train_mean) / x_train_std
y_0_norm = test_input_norm[:, :, 0]

start_time = time.time()

with torch.no_grad(): 
    model.batch_size = N_TEST
    model.init_hidden_KNet()
    m1x_0_hybrid = torch.zeros(N_TEST, m, 1).to(device)
    
    #m1x_0_test = test_target_norm[:, :, 0].unsqueeze(2)
    m1x_0_cold = torch.zeros(N_TEST, m, 1).to(device)
    # -- Posizione X, Y (Stati 0, 1) --
    m1x_0_hybrid[:, 0, 0] = y_0_norm[:, 0] # Assegna X misurato a X stato
    m1x_0_hybrid[:, 1, 0] = y_0_norm[:, 1] # Assegna Y misurato a Y stato

    # -- Angolo Phi (Stato 2) --
    # NON TOCCARE. Rimane 0.0.
    # Non avendo sensori per l'angolo, partiamo dalla media (incertezza massima).

    # -- Velocità vx, vy, omega (Stati 3, 4, 5) --
    m1x_0_hybrid[:, 3, 0] = y_0_norm[:, 2] # vx misurato -> vx stato
    m1x_0_hybrid[:, 4, 0] = y_0_norm[:, 3] # vy misurato -> vy stato
    m1x_0_hybrid[:, 5, 0] = y_0_norm[:, 4] # omega misurato -> omega stato

# 4. Inizializza il modello con questo stato ibrido
    #model.InitSequence(m1x_0_hybrid, T_TEST)
    model.InitSequence(m1x_0_hybrid , T_TEST)

    x_out_test_norm_list = []
    for t in range(T_TEST):
        x_out_t_norm_squeezed = torch.squeeze(
            model(torch.unsqueeze(test_input_norm[:, :, t], 2),
                  torch.unsqueeze(u_test[:, :, t], 2)) 
        )
        x_out_test_norm_list.append(x_out_t_norm_squeezed)
    
    x_out_test_norm = torch.stack(x_out_test_norm_list, dim=2) # [N_TEST, m, T_TEST]
    
    loss_x_test = loss_x_with_angular(
        x_out_test_norm, test_target_norm, 
        x_train_mean, x_train_std, m, PHI_IDX
    )

end_time = time.time()
test_time = end_time - start_time

test_loss_linear = loss_x_test.item()
test_loss_db = 10 * torch.log10(torch.tensor(test_loss_linear).clamp_min(EPS_DB))

print("\n--- RISULTATI TEST ---")
print(f"Modello: {model_path}")
print(f"Tempo di inferenza totale: {test_time:.4f} s")
print(f"Tempo medio per campione: {test_time / N_TEST:.6f} s")
print(f"MSE Test (Lineare): {test_loss_linear:.8f}")
print(f"MSE Test (dB):    {test_loss_db:.4f} dB")



# Scegli una traiettoria a caso da plottare
idx_to_plot = random.randint(0, N_TEST - 1) 
print(f"Plotting della traiettoria indice: {idx_to_plot}")

# Denormalizza l'output del modello per il plotting
x_out_test_denorm = (x_out_test_norm * x_train_std) + x_train_mean

# Converti in numpy per il plotting
test_target_np = x_test.cpu().numpy()
test_input_np = y_test.cpu().numpy() # y_test sono le osservazioni rumorose
x_out_test_np = x_out_test_denorm.cpu().numpy()

fig, axs = plt.subplots(3, 2, figsize=(15, 12)) # 3x2 per 6 stati
states = ['Stato X [m]', 'Stato Y [m]', 'Stato phi [rad]', 
          'Stato vx [m/s]', 'Stato vy [m/s]', 'Stato omega [rad/s]']

for i in range(m): # Itera sui 6 stati
    row = i // 2
    col = i % 2
    
    # Target (pulito)
    axs[row, col].plot(range(T_TEST), test_target_np[idx_to_plot, i, :], 'k', label='Ground Truth (Pulito)')
    
    # Stima KNet
    axs[row, col].plot(range(T_TEST), x_out_test_np[idx_to_plot, i, :], 'r--', label='Stima KalmanNet')
    
    # Osservazioni rumorose (Mappa gli indici di stato 0,1,3,4,5 agli indici di osservazione 0,1,2,3,4)
    obs_label = 'Osservazione (Rumorosa)'

    # Impostazioni grafiche per il rumore (modifica qui se vuoi cambiare stile)
    obs_style = {
        'color': 'limegreen',   # Un verde più acceso ma trasparente
        'linestyle': 'None',    # NESSUNA linea di collegamento
        'marker': '.',          # Usa puntini
        'markersize': 3,        # Dimensione puntino
        'alpha': 0.4,           # Molto trasparente per non disturbare
        'zorder': 0,            # Mette i punti SOTTO le altre linee
        'label': obs_label
    }

    if i == 0:  # X (indice osservazione 0)
        axs[row, col].plot(
            range(T_TEST), test_input_np[idx_to_plot, 0, :],
            **obs_style
        )
    elif i == 1:  # Y (indice osservazione 1)
        axs[row, col].plot(
            range(T_TEST), test_input_np[idx_to_plot, 1, :],
            **obs_style
        )
    elif i == 3:  # vx (indice osservazione 2)
        axs[row, col].plot(
            range(T_TEST), test_input_np[idx_to_plot, 2, :],
            **obs_style
        )
    elif i == 4:  # vy (indice osservazione 3)
        axs[row, col].plot(
            range(T_TEST), test_input_np[idx_to_plot, 3, :],
            **obs_style
        )
    elif i == 5:  # omega (indice osservazione 4)
        axs[row, col].plot(
            range(T_TEST), test_input_np[idx_to_plot, 4, :],
            **obs_style
        )
    axs[row, col].set_title(f'Stato: {states[i]}')
    axs[row, col].set_xlabel('Time Step')
    axs[row, col].set_ylabel('Valore')
    axs[row, col].legend()
    axs[row, col].grid(True)

plt.suptitle(f"Esempio di Test - Traiettoria {idx_to_plot} | MSE Medio: {test_loss_db:.2f} dB", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
plt.savefig(os.path.join(path_results, 'test_trajectory_example.png'))
print(f"Plot di esempio salvato in {os.path.join(path_results, 'test_trajectory_example.png')}")
plt.show()
plt.close()


def plot_ground_truth_vs_noisy_with_zoom(idx, x_clean, y_noisy, T):
    """
    x_clean : tensor/np array [N, 6, T]
    y_noisy : tensor/np array [N, 5, T]
    idx : traiettoria da plottare
    """

    x_clean_np = x_clean[idx]
    y_noisy_np = y_noisy[idx]

    # Mappatura osservazioni: stati [0,1,3,4,5] corrispondono a osservazioni [0..4]
    obs_map = {0:0, 1:1, 3:2, 4:3, 5:4}

    state_names = ["Stato X [m]", "Stato Y [m]", "phi NON osservabile",
                   "vx [m/s]", "vy [m/s]", "omega [rad/s]"]

    fig, axs = plt.subplots(3, 2, figsize=(15, 12))

    for i in range(6):
        r, c = i//2, i%2
        axs[r,c].plot(x_clean_np[i], "k", label="Ground Truth")

        if i in obs_map:
            axs[r,c].plot(y_noisy_np[obs_map[i]], "orange", alpha=0.6, 
                          linestyle="--", label="Osservazione (Rumorosa)")

        axs[r,c].set_title(state_names[i])
        axs[r,c].set_xlabel("Time Step")
        axs[r,c].set_ylabel("Valore")
        axs[r,c].grid(True)
        axs[r,c].legend()

    plt.suptitle(f"GT vs Osservazioni Noisy – Traiettoria {idx}")
    plt.tight_layout()
    plt.show()

    # -------------------------------
    #          PLOT ZOOMATI
    # -------------------------------
    zoom_start = 400
    zoom_end = 600
    print(f"\nZoom range: {zoom_start} → {zoom_end}")

    fig2, axs2 = plt.subplots(1, 3, figsize=(18, 5))
    zoom_states = [0, 1, 3]  # X, Y, vx

    for j, i in enumerate(zoom_states):
        axs2[j].plot(x_clean_np[i, zoom_start:zoom_end], "k", label="Ground Truth")

        if i in obs_map:
            axs2[j].plot(y_noisy_np[obs_map[i], zoom_start:zoom_end],
                         color="orange", linestyle="--", linewidth=2,
                         label="Osservazione (Rumorosa)")

        axs2[j].set_title("ZOOM – " + state_names[i])
        axs2[j].set_xlabel("Time Step")
        axs2[j].set_ylabel("Valore")
        axs2[j].grid(True)
        axs2[j].legend()

        # Zoom dinamico per far vedere bene la differenza
        gt_seg = x_clean_np[i, zoom_start:zoom_end]
        noisy_seg = y_noisy_np[obs_map[i], zoom_start:zoom_end]

        min_v = min(gt_seg.min(), noisy_seg.min())
        max_v = max(gt_seg.max(), noisy_seg.max())
        padding = 0.02 * (max_v - min_v)

        axs2[j].set_ylim(min_v - padding, max_v + padding)

    plt.suptitle(f"ZOOM Ground Truth vs Noisy – Traiettoria {idx}")
    plt.tight_layout()
    plt.show()

plot_ground_truth_vs_noisy_with_zoom(idx_to_plot, x_test.cpu().numpy(), y_test.cpu().numpy(), T_TEST)
