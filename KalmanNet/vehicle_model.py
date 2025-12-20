import torch
import torch.nn as nn
import numpy as np

Params = {
    "Cm1": 0.287, "Cm2": 0.0545, "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737, "Bf": 2.579,
    "Cf": 1.2,    "Df": 0.192, "m": 0.041, "Iz": 27.8e-6,
    "lf": 0.029, "lr": 0.033, "g": 9.81, "maxAlpha": 0.6, "vx_zero": 0.3,

}

# Funzioni operano su batch di tensori

def pt_clamp(x, lo, hi):
   
    return torch.clamp(x, lo, hi)

def pt_tire_forces(vx, vy, omega, d, delta, p):
    """
    Calcola le forze degli pneumatici.
    Accetta vx, vy, omega clippati.
    """
  
    # Calcola vx_eff in modo batched
    vx_eff = torch.max(torch.abs(vx), torch.tensor(p["vx_zero"], device=vx.device))
    
    # Calcola angoli di deriva 
    alpha_f = -torch.atan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  torch.atan2(omega * p["lr"] - vy, vx_eff)
    
    # Clamp alpha_f
    alpha_f = pt_clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    
    # Calcola forze pnematici
    Fy_f = p["Df"] * torch.sin(p["Cf"] * torch.atan(p["Bf"] * alpha_f))
    Fy_r = p["Dr"] * torch.sin(p["Cr"] * torch.atan(p["Br"] * alpha_r))
    Frx = (p["Cm1"] - p["Cm2"] * vx_eff) * d - p["Cr0"] - p["Cr2"] * (vx_eff**2)
    
    return Fy_f, Fy_r, Frx


def pt_f_cont(x_batch, u_batch, p):
  
    X = x_batch[:, 0] 
    Y = x_batch[:, 1] 
    phi = torch.clamp(x_batch[:, 2], p["phi_min"], p["phi_max"])
    vx = torch.clamp(x_batch[:, 3], p["vx_min"], p["vx_max"])
    vy = torch.clamp(x_batch[:, 4], p["vy_min"], p["vy_max"])
    omega = torch.clamp(x_batch[:, 5], p["omega_min"], p["omega_max"])
    
    d = u_batch[:, 0]
    delta = u_batch[:, 1]
    
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]

    # Chiama pt_tire_forces con i valori clippati 
    Fy_f, Fy_r, Frx = pt_tire_forces(vx, vy, omega, d, delta, p)
    
    # Calcola derivate 
    Xdot = vx*torch.cos(phi) - vy*torch.sin(phi)
    Ydot = vx*torch.sin(phi) + vy*torch.cos(phi)
    phidot = omega
    
    vxdot = (Frx - Fy_f*torch.sin(delta) + m*vy*omega) / m
    vydot = (Fy_r + Fy_f*torch.cos(delta) - m*vx*omega) / m
    omegadot = (Fy_f*lf*torch.cos(delta) - Fy_r*lr) / Iz
    
    x_dot_batch = torch.stack([Xdot, Ydot, phidot, vxdot, vydot, omegadot], dim=1)
    return x_dot_batch

class VehicleModel:

    def __init__(self, Ts, T_train, T_test, m1x_0_real, prior_Q, prior_Sigma, prior_S):
    
        self.m = 6 # Dimensione Stato: [X, Y, phi, vx, vy, omega]
        self.n = 5 # Dimensione Osservazione: [X, Y, vx, vy, omega]
        self.d = 2 # Dimensione Controllo: [d, delta]
        
        # Parametri di sistema
        self.Ts = Ts         
        self.Params = Params
        
        # Lunghezze sequenze
        self.T = T_train     
        self.T_test = T_test
        
        # Stato iniziale 
        self.m1x_0 = m1x_0_real 
        
        # Priors per KalmanNet
        # Questi sono i pesi iniziali per le GRU di KNet
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S


    def f(self, x_batch_in, u_batch_in):
       
        x_prev = torch.squeeze(x_batch_in, 2) # [B, 6]
        u = torch.squeeze(u_batch_in, 2)      # [B, 2]
        
        # Calcola x_dot = f_cont(x_prev, u)
        x_dot = pt_f_cont(x_prev, u, self.Params)
        
        # x_next può ancora essere inf se x_prev era inf
        # (inf + finito = inf)
        x_next = x_prev + self.Ts * x_dot
        
        # Clampa ogni variabile di stato ai suoi limiti
        x_next_clamped = x_next.clone()
        x_next_clamped[:, 0] = torch.clamp(x_next[:, 0], self.Params["x_min"], self.Params["x_max"])
        x_next_clamped[:, 1] = torch.clamp(x_next[:, 1], self.Params["y_min"], self.Params["y_max"])
        x_next_clamped[:, 2] = torch.clamp(x_next[:, 2], self.Params["phi_min"], self.Params["phi_max"])
        x_next_clamped[:, 3] = torch.clamp(x_next[:, 3], self.Params["vx_min"], self.Params["vx_max"])
        x_next_clamped[:, 4] = torch.clamp(x_next[:, 4], self.Params["vy_min"], self.Params["vy_max"])
        x_next_clamped[:, 5] = torch.clamp(x_next[:, 5], self.Params["omega_min"], self.Params["omega_max"])
        
        # Ritorna nel formato atteso [B, 6, 1]
        return torch.unsqueeze(x_next_clamped, 2)


    def h(self, x_batch_in):
        """
        Funzione di osservazione h(x_t)
        x_batch_in: [B, 6, 1] (Stato stimato)
        Stato: [X, Y, phi, vx, vy, omega] (indici 0, 1, 2, 3, 4, 5)
        
        Ritorna la misura predetta (senza rumore)
        Misura: [X, Y, vx, vy, omega] (indici 0, 1, 3, 4, 5)
        """
        
        # Seleziona indici [0, 1, 3, 4, 5] dalla dimensione 1
        indices = torch.tensor([0, 1, 3, 4, 5], device=x_batch_in.device)
        y_hat_batch = torch.index_select(x_batch_in, 1, indices)
        
        # Ritorna [B, 5, 1]
        return y_hat_batch

  
