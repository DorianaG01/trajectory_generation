import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
import os
import random
import wandb
import numpy as np
from datetime import datetime

PHI_IDX = 2  
EPS_DB = 1e-12

def _angular_mse_from_real(phi_pred_real, phi_tgt_real):
    """
    Computes angular MSE handling 2pi periodicity.
    Operates in REAL coordinates.
    """
    diff = torch.atan2(torch.sin(phi_pred_real - phi_tgt_real),
                       torch.cos(phi_pred_real - phi_tgt_real))
    return (diff ** 2).mean()

def loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx=PHI_IDX):
    """
    Combines MSE on linear channels (normalized space) + 
    Angular MSE on phi (converted to real space).
    """
    # 1. MSE on NON-angular channels (normalized space)
    diff2 = (x_out_norm - x_tgt_norm) ** 2
    idx_no_phi = [i for i in range(m) if i != phi_idx]
    mse_no_phi = diff2[:, idx_no_phi, :].mean() 

    # 2. Angular MSE on phi (real space)
    x_out_real = (x_out_norm * x_std) + x_mean
    x_tgt_real = (x_tgt_norm * x_std) + x_mean
    
    phi_pred = x_out_real[:, phi_idx, :]
    phi_tgt  = x_tgt_real[:, phi_idx, :]
    mse_phi_ang = _angular_mse_from_real(phi_pred, phi_tgt)

    # 3. Weighted combination
    return (float(m-1) / m) * mse_no_phi + (1.0 / m) * mse_phi_ang

def compute_composite_loss(x_out_norm, x_tgt_norm, y_tgt_norm, 
                           x_mean, x_std, m, n, 
                           alpha, phi_idx=PHI_IDX):
    """
    Composite Loss: alpha * State_Loss + (1-alpha) * Observation_Loss
    """
    # Loss 1: State Loss (Estimate vs Ground Truth)
    loss_x = loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx)
    
    # Loss 2: Observation Loss (h(x_estimate) vs y_measure)
    # Observable indices: everything except phi (idx 2)
    idx_y = [i for i in range(m) if i != phi_idx] 
    y_pred_norm = x_out_norm[:, idx_y, :]
    loss_y = torch.nn.functional.mse_loss(y_pred_norm, y_tgt_norm, reduction='mean')
    
    return alpha * loss_x + (1.0 - alpha) * loss_y


# TRAINING AND VALIDATION FUNCTIONS 

def train_epoch_chunked(model, optimizer, y_chunks, u_chunks, x_chunks, norm_stats, params):
    """
    Performs a training epoch using CHUNKS and SHUFFLING.
    """
    m, n = params['m'], params['n']
    device = params['device']
    use_composite = params['CompositionLoss']
    alpha = params['alpha']
    N_batch = params['N_batch']
    T_chunk = params['T_train'] 

    x_mean, x_std = norm_stats['x_mean'], norm_stats['x_std']
    y_mean, y_std = norm_stats['y_mean'], norm_stats['y_std']
    
    model.train()
    
    # SHUFFLE CHUNKS 
    N_total_chunks = y_chunks.shape[0]
    all_indices = torch.randperm(N_total_chunks)
    
    epoch_loss_accum = 0.0
    num_batches_processed = 0
    
    # BATCH LOOP 
    for i in range(0, N_total_chunks, N_batch):
        batch_indices = all_indices[i : i + N_batch]
        current_batch_size = len(batch_indices)
        
        y_batch = y_chunks[batch_indices].to(device)
        u_batch = u_chunks[batch_indices].to(device)
        x_batch = x_chunks[batch_indices].to(device) # GT for init and loss
        
        y_batch_norm = (y_batch - y_mean) / y_std
        x_batch_norm = (x_batch - x_mean) / x_std
        
        model.batch_size = current_batch_size
        model.init_hidden_KNet()

        x_0_true_norm = x_batch_norm[:, :, 0]
        init_noise_std = 0.3 
        noise = torch.randn_like(x_0_true_norm) * init_noise_std
        m1x_0_batch = (x_0_true_norm + noise).unsqueeze(2)

        model.InitSequence(m1x_0_batch, T_chunk)

        optimizer.zero_grad()
        
        outputs_norm_list = []
        x_tgt_norm_list = []
        y_tgt_norm_list = []
        
        for t in range(T_chunk):
            y_in = torch.unsqueeze(y_batch_norm[:, :, t], 2)
            u_in = torch.unsqueeze(u_batch[:, :, t], 2)
            
            x_out = model(y_in, u_in)
            
            outputs_norm_list.append(torch.squeeze(x_out, 2))
            x_tgt_norm_list.append(x_batch_norm[:, :, t])
            if use_composite:
                y_tgt_norm_list.append(y_batch_norm[:, :, t])
        
        x_out_stack = torch.stack(outputs_norm_list, dim=2) # [B, m, T]
        x_tgt_stack = torch.stack(x_tgt_norm_list, dim=2)
        
        if use_composite:
            y_tgt_stack = torch.stack(y_tgt_norm_list, dim=2)
            loss = compute_composite_loss(
                x_out_stack, x_tgt_stack, y_tgt_stack,
                x_mean, x_std, m, n, alpha, PHI_IDX
            )
        else:
            loss = loss_x_with_angular(
                x_out_stack, x_tgt_stack, 
                x_mean, x_std, m, PHI_IDX
            )
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        epoch_loss_accum += loss.item()
        num_batches_processed += 1
        
        # Detach hidden states
        model.h_Q.detach_()
        model.h_Sigma.detach_()
        model.h_S.detach_()
        model.m1x_posterior.detach_()

    avg_loss = epoch_loss_accum / max(1, num_batches_processed)
    loss_db = 10 * torch.log10(torch.tensor(avg_loss).clamp_min(EPS_DB))
    
    return avg_loss, loss_db


def validate_epoch(model, y_val_norm, u_val, x_val_norm, norm_stats, params):
    """
    Performs validation on FULL trajectories (no chunking).
    """
    m, n = params['m'], params['n']
    T_val = params['T_val'] 
    N_CV = params['N_CV']
    use_composite = params['CompositionLoss']
    alpha = params['alpha']
    
    x_mean, x_std = norm_stats['x_mean'], norm_stats['x_std']

    model.eval()
    with torch.no_grad():
        model.batch_size = N_CV
        model.init_hidden_KNet()
      
        x_0_true_cv_norm = x_val_norm[:, :, 0]
        noise_cv = torch.randn_like(x_0_true_cv_norm) * 0.2
        m1x_0_cv = (x_0_true_cv_norm + noise_cv).unsqueeze(2)

        model.InitSequence(m1x_0_cv, T_val)

        x_out_cv_norm_list = []
        
        for t in range(T_val):
            y_in = torch.unsqueeze(y_val_norm[:, :, t], 2)
            u_in = torch.unsqueeze(u_val[:, :, t], 2)
            
            x_out = model(y_in, u_in)
            x_out_cv_norm_list.append(torch.squeeze(x_out, 2))
        
        x_out_cv_norm = torch.stack(x_out_cv_norm_list, dim=2) 

        if use_composite:
            loss_val = compute_composite_loss(
                x_out_cv_norm, x_val_norm, y_val_norm, 
                x_mean, x_std, m, n, alpha, PHI_IDX
            )
        else:
            loss_val = loss_x_with_angular(
                x_out_cv_norm, x_val_norm, 
                x_mean, x_std, m, PHI_IDX
            )

    val_loss_linear = loss_val.item()
    val_loss_db = 10 * torch.log10(torch.tensor(val_loss_linear).clamp_min(EPS_DB))
    
    return val_loss_linear, val_loss_db


class Pipeline_Vehicle_KNet:
    
    def __init__(self, folderName, modelName):
        super().__init__()
        self.folderName = folderName
        self.modelName = modelName
        self.best_model_path = os.path.join(self.folderName, f"best_model_{self.modelName}.pt")
        os.makedirs(self.folderName, exist_ok=True)
        print(f"Pipeline initialized. Results in: {self.folderName}")

    def set_models(self, sys_model, model):
        self.sys_model = sys_model
        self.model = model
        print("Models set.")

    def set_data(self, train_data_chunks, val_data_full, test_data_full, norm_stats, device):
        
        self.y_train_chunks, self.u_train_chunks, self.x_train_chunks = train_data_chunks
        self.y_val, self.u_val, self.x_val = val_data_full
        self.y_test, self.u_test, self.x_test = test_data_full
        
        self.norm_stats = norm_stats
        self.device = device
        
        self.N_train_chunks = self.y_train_chunks.shape[0]
        self.N_CV = self.y_val.shape[0]
        self.N_T = self.y_test.shape[0]
        
        print(f"Data Loaded:")
        print(f"  - Train Chunks: {self.N_train_chunks} (Shape: {self.y_train_chunks.shape})")
        print(f"  - Val Traj:     {self.N_CV} (Shape: {self.y_val.shape})")

    def set_training_params(self, n_steps, n_batch, lr, wd, T_chunk_train,
                            CompositionLoss=False, alpha=0.5):
        self.N_steps = n_steps
        self.N_batch = n_batch
        self.lr = lr
        self.wd = wd
        
        self.T_train = T_chunk_train 
        
        self.m = self.sys_model.m
        self.n = self.sys_model.n 
        
        self.CompositionLoss = CompositionLoss
        self.alpha = alpha

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
        
        self.train_loss_history_db = []
        self.val_loss_history_db = []
        print(f"Training parameters set. T_train={self.T_train}")

    def train(self):
        
        wandb.init(
            project="KalmanNet_Vehicle_Chunked",  
            name=self.modelName,            
            config={
                "learning_rate": self.lr,
                "weight_decay": self.wd,
                "epochs": self.N_steps,
                "batch_size": self.N_batch,
                "T_chunk": self.T_train,
                "CompositionLoss": self.CompositionLoss, 
                "alpha": self.alpha,                     
                "strategy": "Chunking+Shuffle"
            }
        )
        wandb.watch(self.model)

        best_val_loss = float('inf')

        y_val_dev = self.y_val.to(self.device)
        u_val_dev = self.u_val.to(self.device)
        x_val_dev = self.x_val.to(self.device)
      
        cv_input_norm = (y_val_dev - self.norm_stats['y_mean']) / self.norm_stats['y_std']
        cv_target_norm = (x_val_dev - self.norm_stats['x_mean']) / self.norm_stats['x_std']
        
        T_val_real = self.y_val.shape[2]

        train_params = {
            'N_batch': self.N_batch, 
            'T_train': self.T_train, 
            'm': self.m, 'n': self.n, 'device': self.device,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }
        
        val_params = {
            'N_CV': self.N_CV, 
            'T_val': T_val_real, 
            'm': self.m, 'n': self.n, 'device': self.device,
            'CompositionLoss': self.CompositionLoss, 'alpha': self.alpha
        }

        print(f"Starting Training (Strategy: Chunked Shuffle)...")
        
        for epoch in range(self.N_steps):
            
            # --- Training Epoch (on Chunks) ---
            avg_train_loss, train_loss_db = train_epoch_chunked(
                self.model, self.optimizer, 
                self.y_train_chunks, self.u_train_chunks, self.x_train_chunks, 
                self.norm_stats, train_params
            )
            self.train_loss_history_db.append(train_loss_db)

            # --- Validation Epoch (on Full Trajectories) ---
            val_loss_linear, val_loss_db = validate_epoch(
                self.model, 
                cv_input_norm, u_val_dev, cv_target_norm, 
                self.norm_stats, val_params
            )
            self.val_loss_history_db.append(val_loss_db)
            
            # --- Logging and Scheduler ---
            self.scheduler.step(val_loss_linear)
            current_lr = self.optimizer.param_groups[0]['lr']

            wandb.log({
                "epoch": epoch,
                "learning_rate": current_lr,
                "train_loss_dB": train_loss_db,
                "val_loss_dB": val_loss_db,
            })

            print(f"{epoch:4d} | Train: {train_loss_db:7.4f} dB | "
                  f"Val: {val_loss_db:7.4f} dB | "
                  f"LR: {current_lr:.1e}")

            if val_loss_linear < best_val_loss:
                best_val_loss = val_loss_linear
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> Best model saved. Val Loss: {best_val_loss:.6f}")

        wandb.finish()
        print("Training completed.")

    def test(self):
        print(f"Starting test on model: {self.best_model_path}")
        
        try:
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        except FileNotFoundError:
            print("Error: Model file not found.")
            return
            
        self.model.to(self.device)
        self.model.eval()

        # Test Data
        y_test_dev = self.y_test.to(self.device)
        u_test_dev = self.u_test.to(self.device)
        x_test_dev = self.x_test.to(self.device)
        
        test_input_norm = (y_test_dev - self.norm_stats['y_mean']) / self.norm_stats['y_std']
        test_target_norm = (x_test_dev - self.norm_stats['x_mean']) / self.norm_stats['x_std']
        
        T_test_real = self.y_test.shape[2]

        test_params = {
            'N_CV': self.N_T, 
            'T_val': T_test_real, 
            'm': self.m, 
            'n': self.n, 
            'device': self.device,
            'CompositionLoss': self.CompositionLoss, 
            'alpha': self.alpha
        }

        print("Executing forward pass on Test Set...")
        test_loss_linear, test_loss_db = validate_epoch(
            self.model,
            test_input_norm, u_test_dev, test_target_norm,
            self.norm_stats, test_params
        )

        print("\n--- Test Results ---")
        print(f"  MSE Linear: {test_loss_linear:.6f}")
        print(f"  MSE dB:     {test_loss_db:.4f} dB")
        print("--------------------")
        
    def plot_learning_curve(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_loss_history_db, label="Train Loss (Chunks)")
        plt.plot(self.val_loss_history_db, label="Val Loss (Full)")
        plt.xlabel("Epoch")
        plt.ylabel("MSE [dB]")
        plt.title(f"Learning Curve - {self.modelName}")
        plt.legend()
        plt.grid(True)
        plot_path = os.path.join(self.folderName, f'learning_curve.png')
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
