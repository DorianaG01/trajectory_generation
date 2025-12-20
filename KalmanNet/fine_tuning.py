import os
import torch
from datetime import datetime
from kalman_net import KalmanNetNN
from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset
from pipeline import Pipeline_Vehicle_KNet


# Funzione necessaria per creare i pezzi corti
def create_shuffled_chunks(y, u, x, chunk_size):
    """
    Taglia le traiettorie lunghe in pezzi (chunk_size) e li mescola.
    Input: [N, m, T_full] -> Output: [N_chunks, m, chunk_size]
    """
    N, m, T_full = x.shape
    num_chunks = T_full // chunk_size
    
    # Taglia l'eccedenza
    limit = num_chunks * chunk_size
    y, u, x = y[:, :, :limit], u[:, :, :limit], x[:, :, :limit]
    
    # Dividi in chunk (dim=2 è il tempo)
    y_chunks = torch.cat(torch.split(y, chunk_size, dim=2), dim=0)
    u_chunks = torch.cat(torch.split(u, chunk_size, dim=2), dim=0)
    x_chunks = torch.cat(torch.split(x, chunk_size, dim=2), dim=0)
    
    # Shuffle casuale dei pezzi
    perm = torch.randperm(y_chunks.shape[0])
    
    print(f"Dataset Chunked creato: {y_chunks.shape[0]} chunks da {chunk_size} steps.")
    return y_chunks[perm], u_chunks[perm], x_chunks[perm]


if __name__ == "__main__":
    
    # ------------------------------------------------------------------
    # 0. CONFIGURAZIONE PATH E MODELLO PRE-ADDESTRATO
    # ------------------------------------------------------------------
    # PERCORSO DEL MODELLO "PARALLELO" (Quello che stimava bene la dinamica ma male l'angolo)
    PRETRAINED_MODEL_PATH = '/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/KalmanNet/KNet_Vehicle_Results/best_model_KNet_Vehicle_CompositeLoss_12.15.25_11-38-13.pt' 
    
    strTime = datetime.now().strftime("%m.%d.%y_%H-%M-%S")
    path_results = '/Users/dorianagiovarruscio/Desktop/tesi/trajectory_generation/KalmanNet/KNet_Vehicle_Results'
    
    # Dimensioni e Tempi
    m, n, d = 6, 5, 2
    Ts = 0.01
    
    # STRATEGIA IBRIDA:
    T_full = 1200       # Per Test e Validation (vediamo la verità lunga)
    T_chunk = 200       # Per il Training (Gradienti stabili e "Urgenza" di correzione)

    # Device
    device = torch.device('cpu') 
    print(f"Using Device: {device}")

    # ------------------------------------------------------------------
    # 1. CARICAMENTO DATI
    # ------------------------------------------------------------------
    # Path dataset
    noisy_csv = '/Users/dorianagiovarruscio/combined_dataset_noisy_err005.csv'  
    clean_csv = '/Users/dorianagiovarruscio/combined_dataset_clean.csv'

    # Setup Modello Fisico
    m1x_0 = torch.zeros(m, 1) 
    prior_Q, prior_Sigma, prior_S = torch.eye(m), torch.eye(m), torch.eye(n)
    sys_model = VehicleModel(Ts, T_full, T_full, m1x_0, prior_Q, prior_Sigma, prior_S)

    print("Caricamento dataset (Sequenze Complete)...")
    (train_data, val_data, test_data) = load_vehicle_dataset(
       noisy_csv, clean_csv, T_steps=T_full, train_split=0.7, val_split=0.15   
    )
    
    # Estraiamo i dati FULL per Val/Test
    y_train_full, u_train_full, x_train_full = train_data
    
    # CREIAMO I CHUNK PER IL TRAINING (La parte che mancava nel tuo codice precedente)
    print(f"Creazione Chunk da {T_chunk} steps per il Fine-Tuning...")
    y_train_chunked, u_train_chunked, x_train_chunked = create_shuffled_chunks(
        y_train_full, u_train_full, x_train_full, chunk_size=T_chunk
    )
    
    # Calcolo statistiche di normalizzazione (su tutto il dataset)
    print("Calcolo statistiche di normalizzazione.")
    x_train_mean = torch.mean(x_train_full, dim=(0, 2)).reshape(1, m, 1).to(device) 
    x_train_std = torch.std(x_train_full, dim=(0, 2)).reshape(1, m, 1).to(device) + 1e-8
    y_train_mean = torch.mean(y_train_full, dim=(0, 2)).reshape(1, n, 1).to(device)
    y_train_std = torch.std(y_train_full, dim=(0, 2)).reshape(1, n, 1).to(device) + 1e-8
    
    norm_stats = {
        'x_mean': x_train_mean, 'x_std': x_train_std,
        'y_mean': y_train_mean, 'y_std': y_train_std
    }

    # Setup limiti modello (Min/Max)
    sys_model.Params.update({
        "x_min": x_train_full[:, 0, :].min().item(), "x_max": x_train_full[:, 0, :].max().item(),
        "y_min": x_train_full[:, 1, :].min().item(), "y_max": x_train_full[:, 1, :].max().item(),
        "phi_min": x_train_full[:, 2, :].min().item(), "phi_max": x_train_full[:, 2, :].max().item(),
        "vx_min": x_train_full[:, 3, :].min().item(), "vx_max": x_train_full[:, 3, :].max().item(),
        "vy_min": x_train_full[:, 4, :].min().item(), "vy_max": x_train_full[:, 4, :].max().item(),
        "omega_min": x_train_full[:, 5, :].min().item(), "omega_max": x_train_full[:, 5, :].max().item()
    })

    # ------------------------------------------------------------------
    # 2. COSTRUZIONE MODELLO E CARICAMENTO PESI ESISTENTI
    # ------------------------------------------------------------------
    print("Costruzione KalmanNet...")
    KalmanNet_model = KalmanNetNN() 
    KalmanNet_model.NNBuild(sys_model, hidden_dim_gru=128) # Assicurati sia lo stesso dim!
    KalmanNet_model.to(device)
    KalmanNet_model.set_normalization(x_train_mean, x_train_std, y_train_mean, y_train_std)
    
    # CARICAMENTO CHECKPOINT
    if os.path.exists(PRETRAINED_MODEL_PATH):
        print(f"--> CARICAMENTO PESI DA: {PRETRAINED_MODEL_PATH}")
        weights = torch.load(PRETRAINED_MODEL_PATH, map_location=device)
        KalmanNet_model.load_state_dict(weights)
        print("--> Pesi caricati. Inizio Fine-Tuning su CHUNK CORTI.")
    else:
        print(f"ERRORE: Il file {PRETRAINED_MODEL_PATH} non esiste.")
        exit()

    # ------------------------------------------------------------------
    # 3. SETUP PIPELINE PER FINE-TUNING SU CHUNK
    # ------------------------------------------------------------------
    model_name_finetune = f'KNet_Vehicle_TunedShort_{strTime}'
    
    pipeline = Pipeline_Vehicle_KNet(folderName=path_results, modelName=model_name_finetune)
    pipeline.set_models(sys_model=sys_model, model=KalmanNet_model)
    
    # Dati: USIAMO I CHUNK PER IL TRAINING!
    # Questo è il segreto: torniamo alla stabilità dei chunk, ma con i pesi già buoni.
    pipeline.set_data(
        train_data=(y_train_chunked, u_train_chunked, x_train_chunked), # <--- CHUNKS (T=200)
        val_data=val_data,    # <--- FULL (T=1200) per vedere se migliora globalmente
        test_data=test_data, 
        norm_stats=norm_stats, 
        device=device
    )
    
    # Parametri Training
    lr_tuning = 1e-4      # Un po' più alto del 5e-5, serve spinta per correggere l'angolo
    
    pipeline.set_training_params(
        n_steps=100,             # Numero epoche
        n_batch=64,              # Batch size
        lr=lr_tuning,          
        wd=1e-5, 
        K_TBPTT=T_chunk,         # <--- T=200
        T=T_chunk,               # <--- T=200
        CompositionLoss=True, 
        alpha=0.8                       
    )

    print("\n" + "="*50)
    print(f"AVVIO TUNING SU CHUNK (T={T_chunk}, LR={lr_tuning})")
    print("="*50)
    
    pipeline.train()
    pipeline.plot_learning_curve()
    
    print("Test Finale (Su sequenze intere):")
    pipeline.test()
    
    print("Fine.")