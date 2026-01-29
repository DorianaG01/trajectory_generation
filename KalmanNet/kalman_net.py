import torch
import torch.nn as nn
import torch.nn.functional as F

class KalmanNetNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')
        self.has_norm_params_xy = False
        self.has_norm_params_u  = False  

    def set_normalization(self, x_mean, x_std, y_mean, y_std, u_mean=None, u_std=None):
        """Sets parameters for standardizing inputs and outputs."""
        self.x_mean = x_mean.to(self.device)
        self.x_std  = x_std.to(self.device)
        self.y_mean = y_mean.to(self.device)
        self.y_std  = y_std.to(self.device)
        self.has_norm_params_xy = True

        if (u_mean is not None) and (u_std is not None):
            self.u_mean = u_mean.to(self.device)
            self.u_std  = u_std.to(self.device)
            self.has_norm_params_u = True
        else:
            self.has_norm_params_u = False

    def NNBuild(self, SysModel, in_mult_KNet=5, out_mult_KNet=40, hidden_dim_gru=128):
        """Initializes the Neural Network architecture for Kalman Gain estimation."""
        self.innov_logit = nn.Parameter(torch.tensor(0.0))  # Initial gamma ≈ 0.5
        
        # System Dynamics (Physical Model)
        self.f = SysModel.f
        self.h = SysModel.h
        self.m = SysModel.m  # State dimension (e.g., 6)
        self.n = SysModel.n  # Observation dimension (e.g., 5)

        # Kalman Gain Network Configuration
        self.seq_len_input = 1
        self.batch_size = 0

        # FC5 (Forward) - Prepares input for the process noise (Q) GRU
        self.d_input_FC5 = self.m
        self.d_output_FC5 = self.m * in_mult_KNet
        self.FC5 = nn.Sequential(
            nn.Linear(self.d_input_FC5, self.d_output_FC5),
            nn.ReLU(),
            nn.Dropout(p=0.05)
        ).to(self.device)

        # GRU for Process Noise (Q) tracking
        self.d_input_Q = self.d_output_FC5 
        self.d_hidden_Q = hidden_dim_gru
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU for State Uncertainty (Sigma) - Takes Q output as input
        self.d_input_Sigma = self.d_hidden_Q
        self.d_hidden_Sigma = hidden_dim_gru
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)

        # FC1 - Maps Sigma hidden state to observation space
        self.d_input_FC1 = self.d_hidden_Sigma
        self.d_output_FC1 = self.n ** 2
        self.FC1 = nn.Sequential(
            nn.Linear(self.d_input_FC1, self.d_output_FC1),
            nn.ReLU(),
            nn.Dropout(p=0.05)
        ).to(self.device)

        # FC7 - Processes the current Innovation (observation diff)
        self.d_input_FC7 = self.n
        self.d_output_FC7 = self.n
        self.FC7 = nn.Sequential(
            nn.Linear(self.d_input_FC7, self.d_output_FC7),
            nn.ReLU(),
            nn.Dropout(p=0.05)
        ).to(self.device)

        # GRU for Innovation Covariance (S) tracking
        self.d_input_S = self.d_output_FC1 + self.d_output_FC7
        self.d_hidden_S = hidden_dim_gru
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)

        # FC2 (Kalman Gain) - Computes the Gain KG
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma
        self.d_output_FC2 = self.n * self.m
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult_KNet
        self.FC2 = nn.Sequential(
            nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
            nn.ReLU(),
            nn.Linear(self.d_hidden_FC2, self.d_output_FC2),
            nn.Dropout(p=0.1)
        ).to(self.device)

        # FC3 (Backward Refresh) - Feeds information back from KG
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2
        self.d_output_FC3 = self.m ** 2
        self.FC3 = nn.Sequential(
            nn.Linear(self.d_input_FC3, self.d_output_FC3),
            nn.ReLU(),
            nn.Dropout(p=0.05)
        ).to(self.device)

        # FC4 (Backward Refresh) - Refreshes the Sigma GRU hidden state
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3
        self.d_output_FC4 = self.d_hidden_Sigma
        self.FC4 = nn.Sequential(
            nn.Linear(self.d_input_FC4, self.d_output_FC4),
            nn.ReLU(),
            nn.Dropout(p=0.05)
        ).to(self.device)

    def InitSequence(self, M1_0_norm, T):
        """Initializes the sequence with a normalized initial state."""
        self.T = T
        self.m1x_posterior = M1_0_norm.to(self.device)

    # ---------- BRIDGE: normalized <-> real-world units ----------
    def _denorm_x(self, x_norm):
        if self.has_norm_params_xy:
            return (x_norm * self.x_std) + self.x_mean
        return x_norm

    def _renorm_x(self, x_real):
        if self.has_norm_params_xy:
            return (x_real - self.x_mean) / self.x_std
        return x_real

    def _denorm_y(self, y_norm):
        if self.has_norm_params_xy:
            return (y_norm * self.y_std) + self.y_mean
        return y_norm

    def _renorm_y(self, y_real):
        if self.has_norm_params_xy:
            return (y_real - self.y_mean) / self.y_std
        return y_real

    def _denorm_u(self, u_in):
        """Denormalizes control inputs if normalization parameters were provided."""
        if self.has_norm_params_u:
            return (u_in * self.u_std) + self.u_mean
        return u_in
    # -------------------------------------------------------------------

    def step_prior(self, u):
        """Physical propagation step (Time Update)."""
        # Posterior state is kept in normalized space; convert to real units for physics
        x_post_real = self._denorm_x(self.m1x_posterior)
        u_real      = self._denorm_u(u)

        # Physics-based propagation in real units
        x_prior_real = self.f(x_post_real, u_real)   # [B,m,1]
        y_pred_real  = self.h(x_prior_real)          # [B,n,1]

        # Convert back to normalized space for the neural networks
        self.m1x_prior = self._renorm_x(x_prior_real)
        self.m1y       = self._renorm_y(y_pred_real)

    def step_KGain_est(self, y):
        """Estimates the Kalman Gain KG based on innovation and prior state."""
        # y and m1y are in normalized space
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2)  # [B,n]
        m1x_prior_sq   = torch.squeeze(self.m1x_prior, 2)                  # [B,m]
        
        # Estimate KG using the Gain network
        KG = self.KGain_step(obs_innov_diff, m1x_prior_sq)
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n))

    def KNet_step(self, y, u):
        """Performs a single KalmanNet iteration (Prior -> Gain -> Posterior)."""
        self.step_prior(u)
        self.step_KGain_est(y)
        dy = y - self.m1y  # [B,n,1] in normalized space

        gamma = torch.sigmoid(self.innov_logit)  # Learned scaling factor in (0,1)
        INOV  = gamma * torch.bmm(self.KGain, dy)  # Calculated innovation correction
        self.m1x_posterior = self.m1x_prior + INOV
        return self.m1x_posterior

    def KGain_step(self, obs_innov_diff, m1x_prior):
        """Forward pass through the internal GRU-based Gain architecture."""
        def expand_dim(x):
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_innov_diff = expand_dim(obs_innov_diff)
        m1x_prior      = expand_dim(m1x_prior)

        # Forward Path
        in_FC5   = m1x_prior
        out_FC5  = self.FC5(in_FC5)
        out_Q, self.h_Q = self.GRU_Q(out_FC5, self.h_Q)
        
        # Simplified Version: Q output directly feeds into Sigma
        out_Sigma, self.h_Sigma = self.GRU_Sigma(out_Q, self.h_Sigma)
        
        out_FC1  = self.FC1(out_Sigma)
        out_FC7  = self.FC7(obs_innov_diff)
        in_S     = torch.cat((out_FC1, out_FC7), 2)
        out_S, self.h_S = self.GRU_S(in_S, self.h_S)
        in_FC2   = torch.cat((out_Sigma, out_S), 2)
        out_FC2  = self.FC2(in_FC2)

        # Backward Refresh (Hidden state maintenance)
        in_FC3   = torch.cat((out_S, out_FC2), 2)
        out_FC3  = self.FC3(in_FC3)
        in_FC4   = torch.cat((out_Sigma, out_FC3), 2)
        out_FC4  = self.FC4(in_FC4)
        self.h_Sigma = out_FC4

        return out_FC2

    def forward(self, y, u):
        """standard forward pass. Expects normalized y and real/normalized u."""
        return self.KNet_step(y.to(self.device), u.to(self.device))

    def init_hidden_KNet(self):
        """Initializes GRU hidden states with zeros."""
        weight = next(self.parameters()).data
        self.h_S     = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_Sigma = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Q     = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
