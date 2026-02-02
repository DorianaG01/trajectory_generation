import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import wandb
import matplotlib.pyplot as plt
import random
from torch.nn.utils import clip_grad_norm_

PHI_IDX = 2   # Index of Yaw angle in state vector
EPS_DB = 1e-12

# --- LOSS FUNCTIONS ---

def _angular_mse_from_real(phi_pred_real, phi_tgt_real):
    """Calculates angular MSE handling 2pi periodicity."""
    diff = torch.atan2(torch.sin(phi_pred_real - phi_tgt_real),
                       torch.cos(phi_pred_real - phi_tgt_real))
    return (diff ** 2).mean()

def loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx=PHI_IDX):
    """
    Computes State Loss.
    Combines standard MSE for linear channels and Angular MSE for phi.
    """
    # 1. MSE on linear channels (non-phi) in normalized space
    diff2 = (x_out_norm - x_tgt_norm) ** 2
    idx_no_phi = [i for i in range(m) if i != phi_idx]
    mse_no_phi = diff2[:, idx_no_phi, :].mean() 

    # 2. Angular MSE on phi in real space (denormalized)
    x_out_real = (x_out_norm * x_std) + x_mean
    x_tgt_real = (x_tgt_norm * x_std) + x_mean
    
    phi_pred = x_out_real[:, phi_idx, :]
    phi_tgt  = x_tgt_real[:, phi_idx, :]
    mse_phi_ang = _angular_mse_from_real(phi_pred, phi_tgt)

    return (float(m-1) / m) * mse_no_phi + (1.0 / m) * mse_phi_ang


def compute_composite_loss(x_out_norm, x_tgt_norm, y_tgt_norm, 
                           x_mean, x_std, m, n, 
                           alpha, phi_idx=PHI_IDX):
    
    # Loss 1: State Loss (x_out vs x_tgt)
    loss_x = loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx)
    
    # Loss 2: Observation Loss (approximated as subset of x vs y_tgt)
    # Indices corresponding to observations (everything except phi)
    idx_y = [i for i in range(m) if i != phi_idx] 
   
    y_pred_norm = x_out_norm[:, idx_y, :]
    
    loss_y = torch.nn.functional.mse_loss(y_pred_norm, y_tgt_norm, reduction='mean')
    
    return alpha * loss_x + (1.0 - alpha) * loss_y


def train_epoch(model, optimizer, y_train, u_train, x_train, norm_stats, params):
    """
    Unified training epoch.
    Supports:
    1. 'standard': TBPTT (update every K steps).
    2. 'accumulation': Gradient Accumulation (update at end of trajectory).
    """
 
    strategy = params['strategy']  # 'standard' or 'accumulation'
    N_E, N_batch = params['N_E'], params['N_batch']
    K_TBPTT, T = params['K_TBPTT'], params['T']
    m, n = params['m'], params['n']
    device = params['device']
    use_composite = params['CompositionLoss']
    alpha = params['alpha']
    
    x_mean = norm_stats['x_mean'].to(device)
    x_std = norm_stats['x_std'].to(device)
    
    model.train()
    
    indices = random.sample(range(N_E), k=N_batch)
    y_batch = y_train[indices].to(device)
    u_batch = u_train[indices].to(device)
    x_batch = x_train[indices].to(device)

    y_train_norm = (y_batch - norm_stats['y_mean'].to(device)) / norm_stats['y_std'].to(device)
    x_train_norm = (x_batch - x_mean) / x_std
   
    model.batch_size = N_batch
    model.init_hidden_KNet()

    # Sequence Initialization with noise
    init_noise_std = 0.2
    x_0_true_norm = x_train_norm[:, :, 0]
    noise = torch.randn_like(x_0_true_norm) * init_noise_std
    m1x_0_batch = (x_0_true_norm + noise).unsqueeze(2)
    model.InitSequence(m1x_0_batch, T)

    optimizer.zero_grad()
    
    epoch_loss_acc = 0.0
    
    outputs_chunk = []
    x_tgts_chunk = []
    y_tgts_chunk = []
 
    num_chunks = (T + K_TBPTT - 1) // K_TBPTT
    chunks_processed = 0

    for t in range(T):
        # --- Forward Step ---
        y_in = y_train_norm[:, :, t].unsqueeze(2)
        u_in = u_batch[:, :, t].unsqueeze(2)
        
        x_out = model(y_in, u_in) 
        
        outputs_chunk.append(x_out.squeeze(2))
        x_tgts_chunk.append(x_train_norm[:, :, t])
        
        if use_composite:
            y_tgts_chunk.append(y_train_norm[:, :, t])

        # --- TBPTT Checkpoint ---
        is_chunk_end = ((t + 1) % K_TBPTT == 0) or ((t + 1) == T)
        
        if is_chunk_end:
            x_out_chunk = torch.stack(outputs_chunk, dim=2) # [B, m, K]
            x_tgt_chunk = torch.stack(x_tgts_chunk, dim=2)
            
            # --- COMPUTE LOSS ---
            if use_composite:
                y_tgt_chunk = torch.stack(y_tgts_chunk, dim=2)
                loss_val = compute_composite_loss(
                    x_out_chunk, x_tgt_chunk, y_tgt_chunk,
                    x_mean, x_std, m, n, alpha, PHI_IDX
                )
            else:
                loss_val = loss_x_with_angular(
                    x_out_chunk, x_tgt_chunk, x_mean, x_std, m, PHI_IDX
                )

            # --- BACKWARD PASS ---
            if strategy == 'accumulation':
                # Scale loss to average correctly at the end
                loss_scaled = loss_val / num_chunks
                loss_scaled.backward()
               
            else: 
                # Strategy == 'standard'
                loss_val.backward()
                clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                optimizer.zero_grad()

            # Logging & Cleanup
            epoch_loss_acc += loss_val.item()
            chunks_processed += 1
            
            outputs_chunk = []
            x_tgts_chunk = []
            y_tgts_chunk = []
            
            # Detach Hidden States
            model.h_Q.detach_()
            model.h_Sigma.detach_()
            model.h_S.detach_()
            model.m1x_posterior.detach_()

    # --- END OF LOOP T ---
    
    # If accumulation mode, update weights now
    if strategy == 'accumulation':
        clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = epoch_loss_acc / max(1, chunks_processed)
    loss_db = 10 * torch.log10(torch.tensor(avg_loss).clamp_min(EPS_DB))
    
    return avg_loss, loss_db

def validate_epoch(model, y_val_norm, u_val, x_val_norm, norm_stats, params):
    """
    Validation Loop. Uses full forward pass (no gradient chunking).
    """
    N_CV = params['N_CV'] 
    T, m, n = params['T'], params['m'], params['n']
    use_composite = params['CompositionLoss']
    alpha = params['alpha']
    device = params['device']
    
    x_mean = norm_stats['x_mean'].to(device)
    x_std = norm_stats['x_std'].to(device)

    model.eval()
    with torch.no_grad():
        model.batch_size = N_CV
        model.init_hidden_KNet()

        # Init state with small noise for robustness
        x_0 = x_val_norm[:, :, 0]
        m1x_0 = (x_0 + torch.randn_like(x_0)*0.2).unsqueeze(2)
        model.InitSequence(m1x_0, T)

        # Full sequence forward
        x_out_list = []
        for t in range(T):
            out = model(y_val_norm[:, :, t].unsqueeze(2),
                        u_val[:, :, t].unsqueeze(2))
            x_out_list.append(out.squeeze(2))
        
        x_out_full = torch.stack(x_out_list, dim=2) 

        # Compute Loss
        if use_composite:
            loss_val = compute_composite_loss(
                x_out_full, x_val_norm, y_val_norm, 
                x_mean, x_std, m, n, alpha, PHI_IDX
            )
        else:
            loss_val = loss_x_with_angular(
                x_out_full, x_val_norm, 
                x_mean, x_std, m, PHI_IDX
            )

    loss_lin = loss_val.item()
    loss_db = 10 * torch.log10(torch.tensor(loss_lin).clamp_min(EPS_DB))
    return loss_lin, loss_db


class Pipeline_Vehicle_KNet:
    
    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName
        self.best_model_path = os.path.join(self.folderName, f"best_model_{self.modelName}.pt")
        os.makedirs(self.folderName, exist_ok=True)
        print(f"Pipeline initialized. Output folder: {self.folderName}")

    def set_models(self, sys_model, model):
        self.sys_model = sys_model
        self.model = model
        print("Models set.")

    def set_data(self, train_data, val_data, test_data, norm_stats, device):
        self.y_train, self.u_train, self.x_train = train_data
        self.y_val, self.u_val, self.x_val = val_data
        self.y_test, self.u_test, self.x_test = test_data
        
        self.norm_stats = norm_stats
        self.device = device
        
        self.N_E = len(self.y_train)
        self.N_CV = len(self.y_val)
        self.N_T = len(self.y_test)
        print(f"Data Loaded. Train: {self.N_E}, Val: {self.N_CV}, Test: {self.N_T}")

    def set_training_params(self, n_steps, n_batch, lr, wd, K_TBPTT, T,
                            training_strategy='standard', # 'standard' or 'accumulation'
                            CompositionLoss=False, alpha=0.5):
        """
        Configure training parameters.
        training_strategy: 
          - 'standard': updates weights every K_TBPTT steps.
          - 'accumulation': accumulates gradients over T and updates at the end.
        """
        self.N_steps = n_steps
        self.N_batch = n_batch
        self.lr = lr
        self.wd = wd
        self.K_TBPTT = K_TBPTT
        self.T = T
        self.m = self.sys_model.m
        self.n = self.sys_model.n
        
        self.strategy = training_strategy
        self.CompositionLoss = CompositionLoss
        self.alpha = alpha

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
        
        self.train_loss_history_db = []
        self.val_loss_history_db = []
        
        print(f"Training Config:")
        print(f" - Strategy: {self.strategy.upper()}")
        print(f" - TBPTT K: {self.K_TBPTT}")
        print(f" - Composition Loss: {self.CompositionLoss} (alpha={self.alpha})")

    def train(self):
        wandb.init(
            project="KalmanNet_Vehicle_Hybrid",  
            name=f"{self.modelName}_{self.strategy}",            
            config={
                "strategy": self.strategy,
                "epochs": self.N_steps,
                "batch_size": self.N_batch,
                "TBPTT_K": self.K_TBPTT,
                "CompositionLoss": self.CompositionLoss,
                "alpha": self.alpha
            }
        )
        wandb.watch(self.model)

        best_val_loss = float('inf')

        # Prepare validation data
        y_val_dev = self.y_val.to(self.device)
        u_val_dev = self.u_val.to(self.device)
        x_val_dev = self.x_val.to(self.device)
        
        # Normalize validation set
        val_in_norm = (y_val_dev - self.norm_stats['y_mean'].to(self.device)) / self.norm_stats['y_std'].to(self.device)
        val_tgt_norm = (x_val_dev - self.norm_stats['x_mean'].to(self.device)) / self.norm_stats['x_std'].to(self.device)

        params_train = {
            'N_E': self.N_E, 'N_batch': self.N_batch, 'K_TBPTT': self.K_TBPTT, 
            'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device,
            'strategy': self.strategy,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }
        params_val = {
            'N_CV': self.N_CV, 'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }

        print(f"Start Training ({self.strategy})")

        for epoch in range(self.N_steps):
            
            # 1. Train Epoch
            train_loss, train_db = train_epoch(
                self.model, self.optimizer, 
                self.y_train, self.u_train, self.x_train, 
                self.norm_stats, params_train
            )
            self.train_loss_history_db.append(train_db)

            # 2. Validation Epoch
            val_loss, val_db = validate_epoch(
                self.model, 
                val_in_norm, u_val_dev, val_tgt_norm, 
                self.norm_stats, params_val
            )
            self.val_loss_history_db.append(val_db)
            
            # 3. Step Scheduler & Log
            self.scheduler.step(val_loss)
            curr_lr = self.optimizer.param_groups[0]['lr']

            wandb.log({
                "epoch": epoch, "lr": curr_lr,
                "train_dB": train_db, "val_dB": val_db
            })

            print(f"{epoch:3d} | Train: {train_db:6.2f}dB | Val: {val_db:6.2f}dB | LR: {curr_lr:.1e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"   -> Saved Best Model: {best_val_loss:.5f}")

        wandb.finish()
        print("Training End.")

    def test(self):
        print(f"Testing Best Model: {self.best_model_path}")
        self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

     
        y_test_dev = self.y_test.to(self.device)
        u_test_dev = self.u_test.to(self.device)
        x_test_dev = self.x_test.to(self.device)

        test_in_norm = (y_test_dev - self.norm_stats['y_mean'].to(self.device)) / self.norm_stats['y_std'].to(self.device)
        test_tgt_norm = (x_test_dev - self.norm_stats['x_mean'].to(self.device)) / self.norm_stats['x_std'].to(self.device)

        params_test = {
            'N_CV': self.N_T, 'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }

        l_lin, l_db = validate_epoch(
            self.model, test_in_norm, u_test_dev, test_tgt_norm, 
            self.norm_stats, params_test
        )
        
        print(f"TEST RESULT -> Loss: {l_db:.4f} dB (MSE: {l_lin:.6f})")

    def plot_learning_curve(self):
        plt.figure(figsize=(10,5))
        plt.plot(self.train_loss_history_db, label="Train")
        plt.plot(self.val_loss_history_db, label="Val")
        plt.title(f"Learning Curve ({self.strategy})")
        plt.ylabel("MSE [dB]")
        plt.xlabel("Epoch")
        plt.grid(True)
        plt.legend()
        path = os.path.join(self.folderName, f'lc_{self.modelName}.png')
        plt.savefig(path)
        print(f"Plot saved to {path}")
        plt.close()
