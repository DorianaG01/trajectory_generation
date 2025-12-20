import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import wandb
import matplotlib.pyplot as plt
import random
from torch.nn.utils import clip_grad_norm_

# Indice di 'phi' nello stato
PHI_IDX = 2  
EPS_DB = 1e-12

def _angular_mse_from_real(phi_pred_real, phi_tgt_real):
    """Calcola MSE angolare (gestisce periodicità 2pi)"""
    diff = torch.atan2(torch.sin(phi_pred_real - phi_tgt_real),
                       torch.cos(phi_pred_real - phi_tgt_real))
    return (diff ** 2).mean()

def loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx=PHI_IDX):
    """
    Combina MSE sui canali != phi (spazio normalizzato) + 
    MSE angolare su phi (spazio reale).
    x_out_norm, x_tgt_norm: [B, m, T_chunk]
    """
    # MSE sui canali != phi in spazio normalizzato
    diff2 = (x_out_norm - x_tgt_norm) ** 2
    idx_no_phi = [i for i in range(m) if i != phi_idx]
    
    mse_no_phi = diff2[:, idx_no_phi, :].mean() 

    # MSE angolare su phi in spazio reale
    x_out_real = (x_out_norm * x_std) + x_mean
    x_tgt_real = (x_tgt_norm * x_std) + x_mean
    
    phi_pred = x_out_real[:, phi_idx, :]
    phi_tgt  = x_tgt_real[:, phi_idx, :]
    mse_phi_ang = _angular_mse_from_real(phi_pred, phi_tgt)

    # pesi uniformi per i 6 canali
    return (float(m-1) / m) * mse_no_phi + (1.0 / m) * mse_phi_ang


def compute_composite_loss(x_out_norm, x_tgt_norm, y_tgt_norm, 
                           x_mean, x_std, m, n, 
                           alpha, phi_idx=PHI_IDX):
    
    # Loss 1: State Loss (x_out vs x_tgt)
    loss_x = loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx)
    
    # Loss 2: Observation Loss (h(x_out) vs y_tgt)
    idx_y = [i for i in range(m) if i != phi_idx] # Indici [0, 1, 3, 4, 5]
    y_pred_norm = x_out_norm[:, idx_y, :]
    loss_y = torch.nn.functional.mse_loss(y_pred_norm, y_tgt_norm, reduction='mean')
    
    # Combina le due loss
    return alpha * loss_x + (1.0 - alpha) * loss_y


def train_epoch(model, optimizer, y_train, u_train, x_train, norm_stats, params):
    """
    Esegue una singola epoca di training con logica TBPTT.
    Gestisce sia la loss normale che quella composita.
    """
    
    # Unpack parametri
    N_E, N_batch = params['N_E'], params['N_batch']
    K_TBPTT, T = params['K_TBPTT'], params['T']
    m, n = params['m'], params['n']
    device = params['device']
    use_composite = params['CompositionLoss']
    alpha = params['alpha']

    # Unpack statistiche di normalizzazione
    x_mean, x_std = norm_stats['x_mean'], norm_stats['x_std']
    y_mean, y_std = norm_stats['y_mean'], norm_stats['y_std']
    
    model.train()
    
    # Batch Sampling
    indices = random.sample(range(N_E), k=N_batch)
    y_batch = y_train[indices].to(device)
    u_batch = u_train[indices].to(device)
    x_batch = x_train[indices].to(device)

    # Normalizzazione batch
    y_train_norm = (y_batch - y_mean) / y_std
    x_train_norm = (x_batch - x_mean) / x_std
    
    # Init stati nascosti
    model.batch_size = N_batch
    model.init_hidden_KNet()

    init_noise_std = 0.2
    x_0_true_norm = x_train_norm[:, :, 0]
    # Genera rumore gaussiano e aggiungilo alla GT
    noise = torch.randn_like(x_0_true_norm) * init_noise_std
    m1x_0_batch = (x_0_true_norm + noise).unsqueeze(2)

    model.InitSequence(m1x_0_batch, T)

    optimizer.zero_grad()
    
    epoch_train_loss_linear = 0.0
    outputs_norm_chunk = []
    x_targets_norm_chunk = []
    y_targets_norm_chunk = [] 
    
    num_chunks = 0

    for t in range(T):
        # --- Forward step ---
        x_out_t_norm_squeezed = torch.squeeze(
            model(torch.unsqueeze(y_train_norm[:, :, t], 2),
                  torch.unsqueeze(u_batch[:, :, t], 2)) 
        )
        
        # Accumula output e target per il chunk
        outputs_norm_chunk.append(x_out_t_norm_squeezed)
        x_targets_norm_chunk.append(x_train_norm[:, :, t])
        if use_composite:
            y_targets_norm_chunk.append(y_train_norm[:, :, t])

        # --- TBPTT STEP ---
        if (t + 1) % K_TBPTT == 0 or (t + 1) == T:
            
            x_out_chunk = torch.stack(outputs_norm_chunk, dim=2)
            x_tgt_chunk = torch.stack(x_targets_norm_chunk, dim=2)
            
            # Calcola la loss 
            if use_composite:
                y_tgt_chunk = torch.stack(y_targets_norm_chunk, dim=2)
                loss = compute_composite_loss(
                    x_out_chunk, x_tgt_chunk, y_tgt_chunk,
                    x_mean, x_std, m, n, alpha, PHI_IDX
                )
            else:
                loss = loss_x_with_angular(
                    x_out_chunk, x_tgt_chunk, 
                    x_mean, x_std, m, PHI_IDX
                )
        
            loss.backward() 
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            optimizer.step()

            optimizer.zero_grad()

            # Logging
            epoch_train_loss_linear += loss.item()
            num_chunks += 1
            
            # Svuota accumulatori
            outputs_norm_chunk = []
            x_targets_norm_chunk = []
            y_targets_norm_chunk = []
            
            model.h_Q.detach_()
            model.h_Sigma.detach_()
            model.h_S.detach_()
            model.m1x_posterior.detach_()


    avg_train_loss_linear = epoch_train_loss_linear / max(1, num_chunks)
    train_loss_db = 10 * torch.log10(torch.tensor(avg_train_loss_linear).clamp_min(EPS_DB))
    
    return avg_train_loss_linear, train_loss_db

def validate_epoch(model, y_val_norm, u_val, x_val_norm, norm_stats, params):
    """
    Esegue una singola epoca di validazione (forward pass completo).
    """
    
    N_CV = params['N_CV'] 
    T, m, n = params['T'], params['m'], params['n']
    device = params['device']
    use_composite = params['CompositionLoss']
    alpha = params['alpha']
    
    x_mean, x_std = norm_stats['x_mean'], norm_stats['x_std']

    model.eval()
    with torch.no_grad():
        model.batch_size = N_CV
        model.init_hidden_KNet()

        init_noise_std_val = 0.2
        
        x_0_true_cv_norm = x_val_norm[:, :, 0]
        noise_cv = torch.randn_like(x_0_true_cv_norm) * init_noise_std_val
        m1x_0_cv = (x_0_true_cv_norm + noise_cv).unsqueeze(2)

        model.InitSequence(m1x_0_cv, T)

        x_out_cv_norm_list = []
        for t in range(T):
            x_out_cv_norm_list.append(torch.squeeze(
                model(torch.unsqueeze(y_val_norm[:, :, t], 2),
                      torch.unsqueeze(u_val[:, :, t], 2))
            ))
        
        x_out_cv_norm = torch.stack(x_out_cv_norm_list, dim=2) 

        if use_composite:
            loss_x_val = compute_composite_loss(
                x_out_cv_norm, x_val_norm, y_val_norm, 
                x_mean, x_std, m, n, alpha, PHI_IDX
            )
        else:
            loss_x_val = loss_x_with_angular(
                x_out_cv_norm, x_val_norm, 
                x_mean, x_std, m, PHI_IDX
            )

    val_loss_linear = loss_x_val.item()
    val_loss_db = 10 * torch.log10(torch.tensor(val_loss_linear).clamp_min(EPS_DB))
    
    return val_loss_linear, val_loss_db


class Pipeline_Vehicle_KNet:
    
    def __init__(self, folderName, modelName):
        super().__init__()
        self.folderName = folderName
        self.modelName = modelName
        self.best_model_path = os.path.join(self.folderName, f"best_model_{self.modelName}.pt")
        os.makedirs(self.folderName, exist_ok=True)
        print(f"Pipeline inizializzata. I risultati saranno salvati in: {self.folderName}")

    def set_models(self, sys_model, model):
        self.sys_model = sys_model
        self.model = model
        print("Modelli (VehicleModel e KalmanNetNN) impostati.")

    def set_data(self, train_data, val_data, test_data, norm_stats, device):
        self.y_train, self.u_train, self.x_train = train_data
        self.y_val, self.u_val, self.x_val = val_data
        self.y_test, self.u_test, self.x_test = test_data
        
        self.norm_stats = norm_stats
        self.device = device
        
        self.N_E = len(self.y_train)
        self.N_CV = len(self.y_val)
        self.N_T = len(self.y_test)
        
        print(f"Dati impostati. Train: {self.N_E}, Val: {self.N_CV}, Test: {self.N_T}")

    def set_training_params(self, n_steps, n_batch, lr, wd, K_TBPTT, T,
                            CompositionLoss=False, alpha=0.5):
        self.N_steps = n_steps
        self.N_batch = n_batch
        self.lr = lr
        self.wd = wd
        self.K_TBPTT = K_TBPTT
        self.T = T
        self.m = self.sys_model.m
        self.n = self.sys_model.n 
        
        self.CompositionLoss = CompositionLoss
        self.alpha = alpha

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        
        self.train_loss_history_db = []
        self.val_loss_history_db = []
        print(f"Parametri di training impostati. Composition Loss: {self.CompositionLoss}, Alpha: {self.alpha}")

    def train(self):
        
        wandb.init(
            project="KalmanNet_Vehicle_Tesi_TBPTT",  
            name=self.modelName,            
            config={
                "learning_rate": self.lr,
                "weight_decay": self.wd,
                "epochs": self.N_steps,
                "batch_size": self.N_batch,
                "TBPTT_K": self.K_TBPTT,
                "CompositionLoss": self.CompositionLoss, 
                "alpha": self.alpha,                     
                "model_class": "KalmanNetNN_Original",
            }
        )
        wandb.watch(self.model)

        best_val_loss = float('inf')

        # Prepara i dati di validazione
        y_val_dev = self.y_val.to(self.device)
        u_val_dev = self.u_val.to(self.device)
        x_val_dev = self.x_val.to(self.device)
        
        cv_input_norm = (y_val_dev - self.norm_stats['y_mean']) / self.norm_stats['y_std']
        cv_target_norm = (x_val_dev - self.norm_stats['x_mean']) / self.norm_stats['x_std']
        
        train_params = {
            'N_E': self.N_E, 'N_batch': self.N_batch, 'K_TBPTT': self.K_TBPTT, 
            'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }
        val_params = {
            'N_CV': self.N_CV, 'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }

        print(f"Inizio Training con TBPTT (K = {self.K_TBPTT})...")
        
        for epoch in range(self.N_steps):
            
            # --- Training Epoch ---
            avg_train_loss_linear, train_loss_db = train_epoch(
                self.model, self.optimizer, 
                self.y_train, self.u_train, self.x_train, 
                self.norm_stats, train_params
            )
            self.train_loss_history_db.append(train_loss_db)

            # --- Validation Epoch ---
            val_loss_linear, val_loss_db = validate_epoch(
                self.model, 
                cv_input_norm, u_val_dev, cv_target_norm, 
                self.norm_stats, val_params
            )
            self.val_loss_history_db.append(val_loss_db)
            
            # --- Logging e Scheduler ---
            self.scheduler.step(val_loss_linear)
            current_lr = self.optimizer.param_groups[0]['lr']

            wandb.log({
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss_x_linear": avg_train_loss_linear,
                "train_loss_x_dB": train_loss_db,
                "val_loss_x_linear": val_loss_linear,
                "val_loss_x_dB": val_loss_db,
            })

            print(f"{epoch:4d} | train_x: {train_loss_db:7.4f} dB | "
                  f"val_x: {val_loss_db:7.4f} dB | "
                  f"LR: {current_lr:.1e}")

            # Salva il modello migliore
            if val_loss_linear < best_val_loss:
                best_val_loss = val_loss_linear
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> Nuovo best model salvato. Val Loss: {best_val_loss:.6f}")

        wandb.finish()
        print("Training completato.")
        print(f"Miglior modello salvato in: {self.best_model_path} con loss {best_val_loss:.6f}")

    def test(self):
        print(f"Inizio test sul modello: {self.best_model_path}")
        
        # Carica il modello migliore
        try:
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        except FileNotFoundError:
            print("Errore: File del modello migliore non trovato. Eseguire prima il training.")
            return
            
        self.model.to(self.device)
        self.model.eval()

        # Prepara i dati di test
        y_test_dev = self.y_test.to(self.device)
        u_test_dev = self.u_test.to(self.device)
        x_test_dev = self.x_test.to(self.device)
        
        test_input_norm = (y_test_dev - self.norm_stats['y_mean']) / self.norm_stats['y_std']
        test_target_norm = (x_test_dev - self.norm_stats['x_mean']) / self.norm_stats['x_std']
        
        test_params = {
            'N_CV': self.N_T, 
            'T': self.T, 
            'm': self.m, 
            'n': self.n, 
            'device': self.device,
            'CompositionLoss': self.CompositionLoss, 
            'alpha': self.alpha
        }

        # Esegui la valutazione
        print("Esecuzione forward pass sul test set...")
        test_loss_linear, test_loss_db = validate_epoch(
            self.model,
            test_input_norm, u_test_dev, test_target_norm,
            self.norm_stats, test_params
        )

        print("\n--- Risultati del Test ---")
        print(f"  Loss Lineare (MSE): {test_loss_linear:.6f}")
        print(f"  Loss in dB (MSE):   {test_loss_db:.4f} dB")
        print("--------------------------")
        
        try:
            wandb.log({
                "test_loss_linear": test_loss_linear,
                "test_loss_dB": test_loss_db
            })
            print("Risultati di test loggati su W&B.")
        except Exception as e:
            print(f"Nota: W&B non attivo. Risultati stampati solo a console. ({e})")

    def plot_learning_curve(self):
        print("Plotting learning curve")
        plt.figure()
        plt.plot(range(self.N_steps), self.train_loss_history_db, 'b', label="Training Loss (TBPTT)")
        plt.plot(range(self.N_steps), self.val_loss_history_db, 'g', label="Validation Loss")
        plt.xlabel("Training Epoch")
        plt.ylabel("MSE Loss [dB]")
        plt.legend()
        plt.grid(True)
        plt.title(f"Curva di Apprendimento - {self.modelName}")
        plot_path = os.path.join(self.folderName, f'learning_curve_{self.modelName}.png')
        plt.savefig(plot_path)
        plt.show()
        plt.close()
        print(f"Curva di apprendimento salvata in: {plot_path}")