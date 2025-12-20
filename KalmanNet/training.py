import torch
import os
from datetime import datetime

from kalman_net import KalmanNetNN 
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset 

from pipeline import Pipeline_Vehicle_KNet


if __name__ == "__main__":
    

    # setup parametri
    strTime = datetime.now().strftime("%m.%d.%y_%H-%M-%S")
    path_results = '/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/KalmanNet/KNet_Vehicle_Results'
    
    
    # Dimensioni modello
    m, n, d = 6, 5, 2
    Ts = 0.01
    T = 1200   # Lunghezza sequenze train/val
    T_test = 1200 

    # Parametri di training
    N_steps = 500 # Epoche
    N_batch = 100 # Batch size
    lr = 5e-4   
    wd = 1e-5    
    K_TBPTT = 200 # lunghezza troncamento bptt

    
    use_composition_loss = True
    alpha = 0.8 # Peso: alpha * loss_stato + (1-alpha) * loss_osservazione
    # Se use_composition_loss = False, alpha viene ignorato
    
    loss_type = "CompositeLoss" if use_composition_loss else "StateLoss"
    model_name_auto = f'KNet_Vehicle_{loss_type}_{strTime}'

    # Device
    device = torch.device('cpu')
    print("Using CPU")

    # Stato iniziale e prior
    m1x_0 = torch.zeros(m, 1) 
    prior_Q, prior_Sigma, prior_S = torch.eye(m), torch.eye(m), torch.eye(n)

    # Path dataset
    noisy_csv = '/Users/dorianagiovarruscio/combined_dataset_noisy_err005.csv'  
    clean_csv = '/Users/dorianagiovarruscio/combined_dataset_clean.csv'

    sys_model = VehicleModel(Ts, T, T_test, m1x_0, prior_Q, prior_Sigma, prior_S)

    print("Caricamento dataset.")
    (train_data, val_data, test_data) = load_vehicle_dataset(
       noisy_csv, clean_csv, T_steps=T, train_split=0.7, val_split=0.15   
    )
    if train_data is None:
        print("Errore nel caricamento dati. Interruzione.")
        exit()
        
    y_train, u_train, x_train = train_data
    
    # Calcola media e std per la normalizzazione
    print("Calcolo statistiche di normalizzazione.")
    x_train_mean = torch.mean(x_train, dim=(0, 2)).reshape(1, m, 1).to(device) 
    x_train_std = torch.std(x_train, dim=(0, 2)).reshape(1, m, 1).to(device) + 1e-8
    y_train_mean = torch.mean(y_train, dim=(0, 2)).reshape(1, n, 1).to(device)
    y_train_std = torch.std(y_train, dim=(0, 2)).reshape(1, n, 1).to(device) + 1e-8
    
    norm_stats = {
        'x_mean': x_train_mean, 'x_std': x_train_std,
        'y_mean': y_train_mean, 'y_std': y_train_std
    }

    print("Impostazione limiti del modello (min/max) dal train set.")
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

    # Inizializza modello KNet
    
    KalmanNet_model = KalmanNetNN() 
    KalmanNet_model.NNBuild(sys_model, hidden_dim_gru=128) 
    KalmanNet_model.to(device)
    KalmanNet_model.set_normalization(
        x_train_mean, x_train_std,
        y_train_mean, y_train_std
    )
    print("Numero parametri:", sum(p.numel() for p in KalmanNet_model.parameters() if p.requires_grad))

    
    pipeline = Pipeline_Vehicle_KNet(folderName=path_results, modelName=model_name_auto)
    
    pipeline.set_models(
        sys_model=sys_model, 
        model=KalmanNet_model
    )
    
    pipeline.set_data(
        train_data=train_data, 
        val_data=val_data, 
        test_data=test_data, 
        norm_stats=norm_stats, 
        device=device
    )
    
    # Passa i nuovi parametri alla pipeline
    pipeline.set_training_params(
        n_steps=N_steps, 
        n_batch=N_batch, 
        lr=lr, 
        wd=wd, 
        K_TBPTT=K_TBPTT,
        T=T,
        CompositionLoss=use_composition_loss, 
        alpha=alpha                       
    )

    
    pipeline.train()
    
    pipeline.plot_learning_curve()
    
    pipeline.test()
    
    print("Pipeline terminata.")