import torch
import torch.nn as nn
from datetime import datetime
import matplotlib.pyplot as plt
import os

import wandb 

from kalman_net import KalmanNetNN
from pipeline import Pipeline_EKF

from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset


today = datetime.today()
now = datetime.now()
strToday = today.strftime("%m.%d.%y")
strNow = now.strftime("%H-%M-%S") 
strTime = strToday + "_" + strNow


path_results = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/KNet_Vehicle_Results'
if not os.path.exists(path_results):
    os.makedirs(path_results)


m = 6 # Dimensione Stato: [X, Y, phi, vx, vy, omega]
n = 5 # Dimensione Osservazione: [X, Y, vx, vy, omega]
d = 2 # Dimensione Controllo: [d, delta]

Ts = 0.02 # Passo di campionamento (12s / 600 steps)
T = 600   # Lunghezza sequenze di training e validation
T_test = 600 # Lunghezza sequenze di test

noisy_csv = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/piecewise_mpc_noisy.csv'  
clean_csv = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/piecewise_mpc_clean.csv'

CompositionLoss = True

# Parametri di training
N_steps = 500 # Numero epoche
N_batch = 100 # Dimensione del singolo batch
lr = 5e-6    # Learning rate
wd = 5e-5     # Weight decay
alpha = 0.5   # Peso per la Composition Loss

use_cuda = False 
device = torch.device('cpu')
print("Using CPU")

# Stato iniziale
m1x_0 = torch.zeros(m, 1) # [6, 1]

# Prior per gli stati nascosti delle GRU di KalmanNet
prior_Q = torch.eye(m)     # [6, 6]
prior_Sigma = torch.eye(m) # [6, 6]
prior_S = torch.eye(n)     # [5, 5]


# Modello non lineare
sys_model = VehicleModel(Ts, T, T_test, m1x_0, prior_Q, prior_Sigma, prior_S)


(train_data, val_data, test_data) = load_vehicle_dataset(
    noisy_csv, 
    clean_csv, 
    T_steps=T,
    train_split=0.7, # 70% training (7000 traiettorie)
    val_split=0.15   # 15% validation (1500 traiettorie)
)


y_train, u_train, x_train = train_data
y_val, u_val, x_val = val_data
y_test, u_test, x_test = test_data


train_input = y_train     # Misure rumorose (input)
train_target = x_train    # Stato pulito (target)
train_control = u_train   # Controlli (input)

cv_input = y_val
cv_target = x_val
cv_control = u_val


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

print(f"  vx: [{vx_min:.4f}, {vx_max:.4f}]")
print(f"  vy: [{vy_min:.4f}, {vy_max:.4f}]")
print(f"  omega: [{omega_min:.4f}, {omega_max:.4f}]")

wandb.init(
    project="KalmanNet_Vehicle_Tesi",  
    name =f'KNet_Vehicle_{strTime}',            
    config={
        "learning_rate": lr,
        "weight_decay": wd,
        "epochs": N_steps,
        "batch_size": N_batch,
        "alpha_composition_loss": alpha,
        "loss_function": "CompositionLoss" if CompositionLoss else "MSELoss",
        "model_class": "KalmanNetNN",
    }
)

# KalmanNet Pipeline 

# KalmanNet Model
KalmanNet_model = KalmanNetNN()
KalmanNet_model.NNBuild(sys_model) 
print("Number of trainable parameters for KalmanNet:", sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))

# Pipeline training
KalmanNet_Pipeline = Pipeline_EKF(strTime, path_results, "KalmanNet_Vehicle")
KalmanNet_Pipeline.setssModel(sys_model)
KalmanNet_Pipeline.setModel(KalmanNet_model)
KalmanNet_Pipeline.setTrainingParams(N_steps, N_batch, lr, wd, alpha=0.5) # alpha per CompositionLoss


print(" Training")
[MSE_cv_linear_epoch, MSE_cv_dB_epoch, MSE_train_linear_epoch, MSE_train_dB_epoch] = \
    KalmanNet_Pipeline.NNTrain(sys_model, 
                             cv_input, cv_target, cv_control, 
                             train_input, train_target, train_control, 
                             path_results, CompositionLoss, loadModel=False) # loadModel=False per iniziare da zero

KalmanNet_Pipeline.save()


pipeline_artifact = wandb.Artifact(f'pipeline-{wandb.run.name}', type='pipeline')
pipeline_artifact.add_file(KalmanNet_Pipeline.PipelineName) 
wandb.log_artifact(pipeline_artifact)

print("Plotting learning curve")
plt.figure()
plt.plot(range(N_steps), MSE_train_dB_epoch, 'b', label="Training Loss")
plt.plot(range(N_steps), MSE_cv_dB_epoch, 'g', label="Validation Loss")
plt.xlabel("Training Epoch")
plt.ylabel("MSE Loss [dB]")
plt.legend()
plt.grid(True)
plt.title("Curva di Apprendimento KalmanNet - Veicolo")
plt.savefig(path_results + 'learning_curve.png')
wandb.log({"Learning Curve": wandb.Image(plt)})
plt.show()
plt.close()
wandb.finish()
