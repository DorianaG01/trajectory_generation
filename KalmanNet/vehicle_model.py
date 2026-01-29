import torch
import torch.nn as nn
import numpy as np

# Vehicle and simulation parameters
Params = {
    "Cm1": 0.287, "Cm2": 0.0545, "Cr0": 0.0518, "Cr2": 0.00035,
    "Br": 3.3852, "Cr": 1.2691, "Dr": 0.1737, "Bf": 2.579,
    "Cf": 1.2,    "Df": 0.192, "m": 0.041, "Iz": 27.8e-6,
    "lf": 0.029, "lr": 0.033, "g": 9.81, "maxAlpha": 0.6, "vx_zero": 0.3,
}

# Functions operate on batches of tensors for GPU/Parallel efficiency

def pt_clamp(x, lo, hi):
    """Utility to clamp values within a range using PyTorch."""
    return torch.clamp(x, lo, hi)

def pt_tire_forces(vx, vy, omega, d, delta, p):
    """
    Calculates tire forces based on the Pacejka Magic Formula.
    Accepts batched clipped values of vx, vy, and omega.
    """
  
    # Calculate effective longitudinal velocity in a batched manner
    vx_eff = torch.max(torch.abs(vx), torch.tensor(p["vx_zero"], device=vx.device))
    
    # Calculate tire slip angles (alpha)
    alpha_f = -torch.atan2(omega * p["lf"] + vy, vx_eff) + delta
    alpha_r =  torch.atan2(omega * p["lr"] - vy, vx_eff)
    
    # Apply safety clamp to slip angles
    alpha_f = pt_clamp(alpha_f, -p["maxAlpha"], p["maxAlpha"])
    
    # Calculate lateral and longitudinal forces
    Fy_f = p["Df"] * torch.sin(p["Cf"] * torch.atan(p["Bf"] * alpha_f))
    Fy_r = p["Dr"] * torch.sin(p["Cr"] * torch.atan(p["Br"] * alpha_r))
    Frx = (p["Cm1"] - p["Cm2"] * vx_eff) * d - p["Cr0"] - p["Cr2"] * (vx_eff**2)
    
    return Fy_f, Fy_r, Frx


def pt_f_cont(x_batch, u_batch, p):
    """
    Continuous-time system dynamics (ODEs) using PyTorch tensors.
    Inputs:
        x_batch: [B, 6] state batch
        u_batch: [B, 2] control input batch
    """
    # Extract states and apply bounds for numerical stability
    X = x_batch[:, 0] 
    Y = x_batch[:, 1] 
    # phi, vx, vy, omega are clamped to predefined physical limits
    phi = torch.clamp(x_batch[:, 2], p["phi_min"], p["phi_max"])
    vx = torch.clamp(x_batch[:, 3], p["vx_min"], p["vx_max"])
    vy = torch.clamp(x_batch[:, 4], p["vy_min"], p["vy_max"])
    omega = torch.clamp(x_batch[:, 5], p["omega_min"], p["omega_max"])
    
    # Extract control inputs: [duty cycle, steering angle]
    d = u_batch[:, 0]
    delta = u_batch[:, 1]
    
    m, Iz, lf, lr = p["m"], p["Iz"], p["lf"], p["lr"]

    # Calculate tire forces using clipped state values
    Fy_f, Fy_r, Frx = pt_tire_forces(vx, vy, omega, d, delta, p)
    
    # Compute state derivatives (Equations of Motion)
    Xdot = vx*torch.cos(phi) - vy*torch.sin(phi)
    Ydot = vx*torch.sin(phi) + vy*torch.cos(phi)
    phidot = omega
    
    vxdot = (Frx - Fy_f*torch.sin(delta) + m*vy*omega) / m
    vydot = (Fy_r + Fy_f*torch.cos(delta) - m*vx*omega) / m
    omegadot = (Fy_f*lf*torch.cos(delta) - Fy_r*lr) / Iz
    
    # Reassemble derivatives into a batch tensor [B, 6]
    x_dot_batch = torch.stack([Xdot, Ydot, phidot, vxdot, vydot, omegadot], dim=1)
    return x_dot_batch

class VehicleModel:
    """
    High-level Vehicle Model class compatible with KalmanNet/Bayesian filtering.
    """

    def __init__(self, Ts, T_train, T_test, m1x_0_real, prior_Q, prior_Sigma, prior_S):
    
        self.m = 6 # State Dimension: [X, Y, phi, vx, vy, omega]
        self.n = 5 # Observation Dimension: [X, Y, vx, vy, omega]
        self.d = 2 # Control Dimension: [d, delta]
        
        # System parameters
        self.Ts = Ts         
        self.Params = Params
        
        # Sequence lengths for training and testing
        self.T = T_train     
        self.T_test = T_test
        
        # Initial state estimate
        self.m1x_0 = m1x_0_real 
        
        # Priors for KalmanNet (used for initializing GRU weights)
        self.prior_Q = prior_Q
        self.prior_Sigma = prior_Sigma
        self.prior_S = prior_S


    def f(self, x_batch_in, u_batch_in):
        """
        State transition function: x_t = f(x_{t-1}, u_t)
        Input shapes: [B, 6, 1] and [B, 2, 1]
        """
        # Squeeze to [B, 6] and [B, 2] for processing
        x_prev = torch.squeeze(x_batch_in, 2) 
        u = torch.squeeze(u_batch_in, 2)      
        
        # Calculate derivative using the continuous model
        x_dot = pt_f_cont(x_prev, u, self.Params)
        
        # Discrete-time step using Explicit Euler integration: x_next = x_prev + Ts * x_dot
        x_next = x_prev + self.Ts * x_dot
        
        # Enforce physical constraints on the next state
        x_next_clamped = x_next.clone()
        x_next_clamped[:, 0] = torch.clamp(x_next[:, 0], self.Params["x_min"], self.Params["x_max"])
        x_next_clamped[:, 1] = torch.clamp(x_next[:, 1], self.Params["y_min"], self.Params["y_max"])
        x_next_clamped[:, 2] = torch.clamp(x_next[:, 2], self.Params["phi_min"], self.Params["phi_max"])
        x_next_clamped[:, 3] = torch.clamp(x_next[:, 3], self.Params["vx_min"], self.Params["vx_max"])
        x_next_clamped[:, 4] = torch.clamp(x_next[:, 4], self.Params["vy_min"], self.Params["vy_max"])
        x_next_clamped[:, 5] = torch.clamp(x_next[:, 5], self.Params["omega_min"], self.Params["omega_max"])
        
        # Reshape back to expected KalmanNet format [B, 6, 1]
        return torch.unsqueeze(x_next_clamped, 2)


    def h(self, x_batch_in):
        """
        Observation function: y_t = h(x_t)
        Input x_batch_in: [B, 6, 1] (Estimated State)
        State Indices: [X, Y, phi, vx, vy, omega] (0, 1, 2, 3, 4, 5)
        
        Returns the predicted measurement (noise-free)
        Measurement Indices: [X, Y, vx, vy, omega] (0, 1, 3, 4, 5)
        Note: Heading (phi) is hidden/not measured.
        """
        
        # Select specific indices [0, 1, 3, 4, 5] from the state dimension
        indices = torch.tensor([0, 1, 3, 4, 5], device=x_batch_in.device)
        y_hat_batch = torch.index_select(x_batch_in, 1, indices)
        
        # Returns [B, 5, 1]
        return y_hat_batch
