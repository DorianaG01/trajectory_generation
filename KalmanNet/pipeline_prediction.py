import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils import clip_grad_norm_
import os
import wandb
import matplotlib.pyplot as plt
import random
import math

PHI_IDX = 2  
EPS_DB = 1e-12

def _angular_mse_from_real(phi_pred_real, phi_tgt_real):
    """
    Computes angular MSE handling 2pi periodicity.
    Expects inputs in radians (real space).
    """
    diff = torch.atan2(torch.sin(phi_pred_real - phi_tgt_real),
                       torch.cos(phi_pred_real - phi_tgt_real))
    return (diff ** 2).mean()

def loss_x_with_angular(x_out_norm, x_tgt_norm, x_mean, x_std, m, phi_idx=PHI_IDX):
  
    # 1. MSE on NON-phi channels (normalized)
    diff2 = (x_out_norm - x_tgt_norm) ** 2
    idx_no_phi = [i for i in range(m) if i != phi_idx]
    
    mse_no_phi = diff2[:, idx_no_phi, :].mean() 

    # 2. Angular MSE on phi (real space)
    # Denormalize only phi channel or full tensor
    x_out_real = (x_out_norm * x_std) + x_mean
    x_tgt_real = (x_tgt_norm * x_std) + x_mean
    
    phi_pred = x_out_real[:, phi_idx, :]
    phi_tgt  = x_tgt_real[:, phi_idx, :]
    
    mse_phi_ang = _angular_mse_from_real(phi_pred, phi_tgt)

    # 3. Weighted combination (uniform weights)
    return (float(m-1) / m) * mse_no_phi + (1.0 / m) * mse_phi_ang

def _compute_step_loss_real(x_next_real, x_gt_next_real, phi_idx=PHI_IDX):
  
    diff = x_next_real - x_gt_next_real
    sq_diff = diff ** 2

    # Angular correction for phi channel
    phi_pred = x_next_real[:, phi_idx, :]
    phi_tgt  = x_gt_next_real[:, phi_idx, :]
    
    diff_phi = torch.atan2(torch.sin(phi_pred - phi_tgt),
                           torch.cos(phi_pred - phi_tgt))
    
    sq_diff[:, phi_idx, :] = diff_phi ** 2
    
    return sq_diff.mean()

def compute_prediction_loss(model, x_est_norm_t, t, u_train_full, x_train_norm_full, horizon):
    """
    Performs physical rollout (Open Loop) for 'horizon' steps and computes error.
    """
    T_max = x_train_norm_full.shape[2]
    total_loss = 0.0
    valid_steps = 0
    
    # 1. Denormalize initial state -> Real Space
    x_curr_real = model._denorm_x(x_est_norm_t) # [B, m, 1]

    for k in range(1, horizon + 1):
        idx_next = t + k
        if idx_next >= T_max:
            break
            
        # 2. Get control at time t+k-1
        u_step = u_train_full[:, :, t + k - 1].unsqueeze(2) # [B, d, 1]
        u_step_real = model._denorm_u(u_step)
        
        # 3. Physical Step: x_{k} = f(x_{k-1}, u_{k-1})
        x_next_real = model.f(x_curr_real, u_step_real)
        
        # 4. Get future GT and denormalize
        x_gt_next_norm = x_train_norm_full[:, :, idx_next].unsqueeze(2)
        x_gt_next_real = model._denorm_x(x_gt_next_norm)
        
        # 5. Compute Step Loss
        step_loss = _compute_step_loss_real(x_next_real, x_gt_next_real, phi_idx=PHI_IDX)
        total_loss += step_loss
        
        # 6. Update state for next step
        x_curr_real = x_next_real
        valid_steps += 1
        
    if valid_steps > 0:
        return total_loss / valid_steps
    else:
        return torch.tensor(0.0, device=x_est_norm_t.device)


def train_epoch(model, optimizer, y_train, u_train, x_train, norm_stats, params):
    
    N_E, N_batch = params['N_E'], params['N_batch']
    K_TBPTT, T = params['K_TBPTT'], params['T']
    m = params['m']
    device = params['device']
    
    lambda_pred = params.get('lambda_pred', 0.0)
    pred_horizon = params.get('pred_horizon', 0)
    
    x_mean, x_std = norm_stats['x_mean'], norm_stats['x_std']
    y_mean, y_std = norm_stats['y_mean'], norm_stats['y_std']
    
    model.train()
    
    indices = random.sample(range(N_E), k=N_batch)
    y_batch = y_train[indices].to(device)
    u_batch = u_train[indices].to(device)
    x_batch = x_train[indices].to(device)

    y_train_norm = (y_batch - y_mean) / y_std
    x_train_norm = (x_batch - x_mean) / x_std
    
    model.batch_size = N_batch
    model.init_hidden_KNet() 

    init_noise_std = 0.2
    x_0_true_norm = x_train_norm[:, :, 0]
    noise = torch.randn_like(x_0_true_norm) * init_noise_std
    m1x_0_batch = (x_0_true_norm + noise).unsqueeze(2)

    model.InitSequence(m1x_0_batch, T)
    
    optimizer.zero_grad()
    
    epoch_loss_filter = 0.0
    epoch_loss_pred = 0.0 
    num_chunks = 0

    outputs_norm_chunk = []
    x_targets_norm_chunk = []
    pred_losses_chunk = []

    for t in range(T):
        x_out_t_norm_squeezed = torch.squeeze(
            model(torch.unsqueeze(y_train_norm[:, :, t], 2),
                  torch.unsqueeze(u_batch[:, :, t], 2)) 
        )
        
        outputs_norm_chunk.append(x_out_t_norm_squeezed)
        x_targets_norm_chunk.append(x_train_norm[:, :, t])
        
        # Prediction Loss
        if lambda_pred > 0 and pred_horizon > 0:
            x_est_curr = x_out_t_norm_squeezed.unsqueeze(2)
            p_loss_t = compute_prediction_loss(
                model=model,
                x_est_norm_t=x_est_curr,
                t=t,
                u_train_full=u_batch,       
                x_train_norm_full=x_train_norm,
                horizon=pred_horizon
            )
            pred_losses_chunk.append(p_loss_t)

        # TBPTT Update
        if (t + 1) % K_TBPTT == 0 or (t + 1) == T:
            
            x_out_chunk = torch.stack(outputs_norm_chunk, dim=2)
            x_tgt_chunk = torch.stack(x_targets_norm_chunk, dim=2)
            
            # Calculate State Loss (Only X)
            loss_filter = loss_x_with_angular(x_out_chunk, x_tgt_chunk, x_mean, x_std, m, PHI_IDX)
            
            # Calculate Prediction Loss
            loss_pred_chunk = torch.tensor(0.0, device=device)
            if lambda_pred > 0 and len(pred_losses_chunk) > 0:
                loss_pred_chunk = torch.stack(pred_losses_chunk).mean()
            
            # Total Loss and Backward
            total_loss = loss_filter + (lambda_pred * loss_pred_chunk)
            
            total_loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            # Logging and Reset
            epoch_loss_filter += loss_filter.item()
            if lambda_pred > 0:
                epoch_loss_pred += loss_pred_chunk.item()
            num_chunks += 1
            
            outputs_norm_chunk = []
            x_targets_norm_chunk = []
            pred_losses_chunk = []
            
            # Detach Hidden States
            if hasattr(model, 'h_Q'): model.h_Q.detach_()
            if hasattr(model, 'h_Sigma'): model.h_Sigma.detach_()
            if hasattr(model, 'h_S'): model.h_S.detach_()
            if hasattr(model, 'h_gru'): model.h_gru.detach_() 
            
            model.m1x_posterior = model.m1x_posterior.detach()
            model.m1x_prior = model.m1x_prior.detach()

    avg_filter_loss = epoch_loss_filter / max(1, num_chunks)
    avg_pred_loss   = epoch_loss_pred / max(1, num_chunks)
    filter_loss_db = 10 * torch.log10(torch.tensor(avg_filter_loss).clamp_min(1e-12))
    
    return avg_filter_loss, avg_pred_loss, filter_loss_db

def validate_epoch(model, y_val_norm, u_val, x_val_norm, norm_stats, params):
    
    N_CV = params['N_CV'] 
    T, m = params['T'], params['m']
    
    pred_horizon_test = params.get('pred_horizon', 50) 
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
        pred_losses_val = [] 

        for t in range(T):
            x_out_t = torch.squeeze(
                model(torch.unsqueeze(y_val_norm[:, :, t], 2),
                      torch.unsqueeze(u_val[:, :, t], 2))
            )
            x_out_cv_norm_list.append(x_out_t)
            
            # Physical Test (sample every 10 steps)
            if pred_horizon_test > 0 and t % 10 == 0 and (t + pred_horizon_test < T):
                x_est_curr = x_out_t.unsqueeze(2)
                p_loss = compute_prediction_loss(
                    model=model,
                    x_est_norm_t=x_est_curr,
                    t=t,
                    u_train_full=u_val,       
                    x_train_norm_full=x_val_norm,
                    horizon=pred_horizon_test
                )
                pred_losses_val.append(p_loss)

        x_out_cv_norm = torch.stack(x_out_cv_norm_list, dim=2) 

        # Validation Loss (State only)
        loss_x_val = loss_x_with_angular(
            x_out_cv_norm, x_val_norm, 
            x_mean, x_std, m, PHI_IDX
        )
            
        avg_pred_loss_val = 0.0
        if len(pred_losses_val) > 0:
            avg_pred_loss_val = torch.stack(pred_losses_val).mean().item()

    val_loss_linear = loss_x_val.item()
    val_loss_db = 10 * torch.log10(torch.tensor(val_loss_linear).clamp_min(EPS_DB))
    
    return val_loss_linear, val_loss_db, avg_pred_loss_val



class Pipeline_Vehicle_KNet:
    
    def __init__(self, folderName, modelName):
        self.folderName = folderName
        self.modelName = modelName
        self.best_model_path = os.path.join(self.folderName, f"best_model_{self.modelName}.pt")
        os.makedirs(self.folderName, exist_ok=True)
        print(f"Pipeline initialized. Output: {self.folderName}")

    def set_models(self, sys_model, model):
        self.sys_model = sys_model
        self.model = model

    def set_data(self, train_data, val_data, test_data, norm_stats, device):
        self.y_train, self.u_train, self.x_train = train_data
        self.y_val, self.u_val, self.x_val = val_data
        self.y_test, self.u_test, self.x_test = test_data
        self.norm_stats = norm_stats
        self.device = device
        self.N_E = len(self.y_train)
        self.N_CV = len(self.y_val)
        self.N_T = len(self.y_test)

    def set_training_params(self, n_steps, n_batch, lr, wd, K_TBPTT, T,
                            lambda_pred=0.0, pred_horizon=0):
        self.N_steps = n_steps
        self.N_batch = n_batch
        self.lr = lr
        self.wd = wd
        self.K_TBPTT = K_TBPTT
        self.T = T
        self.m = self.sys_model.m
        self.n = self.sys_model.n 
        
        self.lambda_pred = lambda_pred
        self.pred_horizon = pred_horizon

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-7)
        
        self.train_loss_history_db = []
        self.val_loss_history_db = []
        
        print(f"Params Set. PredLambda: {self.lambda_pred}, Horizon: {self.pred_horizon}")

    def train(self):
        
        wandb.init(
            project="KalmanNet_Vehicle_Prediction",  
            name=self.modelName,            
            config={
                "learning_rate": self.lr,
                "weight_decay": self.wd,
                "epochs": self.N_steps,
                "batch_size": self.N_batch,
                "TBPTT_K": self.K_TBPTT,
                "lambda_pred": self.lambda_pred,
                "pred_horizon": self.pred_horizon,
            }
        )
        wandb.watch(self.model)

        best_val_loss = float('inf')

        # Validation Data to GPU
        y_val_dev = self.y_val.to(self.device)
        u_val_dev = self.u_val.to(self.device)
        x_val_dev = self.x_val.to(self.device)
        
        cv_input_norm = (y_val_dev - self.norm_stats['y_mean']) / self.norm_stats['y_std']
        cv_target_norm = (x_val_dev - self.norm_stats['x_mean']) / self.norm_stats['x_std']
        
        train_params = {
            'N_E': self.N_E, 'N_batch': self.N_batch, 'K_TBPTT': self.K_TBPTT, 
            'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device,
            'lambda_pred': self.lambda_pred,
            'pred_horizon': self.pred_horizon
        }
        
        val_params = {
            'N_CV': self.N_CV, 'T': self.T, 'm': self.m, 'n': self.n, 'device': self.device
        }

        print(f"Start Training. Horizon: {self.pred_horizon}, Lambda: {self.lambda_pred}")
        
        for epoch in range(self.N_steps):
            
            avg_filt_loss, avg_pred_loss, train_loss_db = train_epoch(
                self.model, self.optimizer, 
                self.y_train, self.u_train, self.x_train, 
                self.norm_stats, train_params
            )
            self.train_loss_history_db.append(train_loss_db)

            val_loss_linear, val_loss_db, val_pred_loss = validate_epoch(
                self.model, 
                cv_input_norm, u_val_dev, cv_target_norm, 
                self.norm_stats, val_params
            )
            self.val_loss_history_db.append(val_loss_db)
            
            self.scheduler.step(val_loss_linear)
            current_lr = self.optimizer.param_groups[0]['lr']

            wandb.log({
                "epoch": epoch,
                "lr": current_lr,
                "train_loss_filter": avg_filt_loss,
                "train_loss_pred": avg_pred_loss,
                "train_loss_dB": train_loss_db,
                "val_loss_linear": val_loss_linear,
                "val_loss_dB": val_loss_db,
                "val_pred_loss": val_pred_loss
            })

            print(f"{epoch:4d} | Filt: {train_loss_db:7.2f}dB | "
                  f"Val Filt: {val_loss_db:7.2f}dB | "
                  f"Val PRED: {val_pred_loss:.4f}") 

            if val_loss_linear < best_val_loss:
                best_val_loss = val_loss_linear
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  -> Best model saved.")

        wandb.finish()
        print("Training Completed.")

    def test(self):
        print(f"Starting test on model: {self.best_model_path}")
        
        try:
            self.model.load_state_dict(torch.load(self.best_model_path, map_location=self.device))
        except FileNotFoundError:
            print("Error: Best model file not found.")
            return
            
        self.model.to(self.device)
        self.model.eval()

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
            'device': self.device
        }

        print("Executing forward pass on test set")
        
        test_loss_linear, test_loss_db, test_pred_loss = validate_epoch(
            self.model,
            test_input_norm, u_test_dev, test_target_norm,
            self.norm_stats, test_params
        )
    
        print("\n--- Test Results ---")
        print(f"  Linear Loss (MSE): {test_loss_linear:.6f}")
        print(f"  Loss in dB (MSE):  {test_loss_db:.4f} dB")
        print(f"  Physics Loss (Pred): {test_pred_loss:.4f}")

        try:
            wandb.log({
                "test_loss_linear": test_loss_linear,
                "test_loss_dB": test_loss_db,
            })
        except Exception:
            pass

    def plot_learning_curve(self):
        plt.figure()
        plt.plot(range(len(self.train_loss_history_db)), self.train_loss_history_db, 'b', label="Training Loss")
        plt.plot(range(len(self.val_loss_history_db)), self.val_loss_history_db, 'g', label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss [dB]")
        plt.legend()
        plt.grid(True)
        plt.title(f"Learning Curve - {self.modelName}")
        plot_path = os.path.join(self.folderName, f'learning_curve_{self.modelName}.png')
        plt.savefig(plot_path)
        plt.show()
        plt.close()
