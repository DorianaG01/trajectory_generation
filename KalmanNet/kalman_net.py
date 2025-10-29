import torch
import torch.nn as nn
import torch.nn.functional as func

class KalmanNetNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
    
    def NNBuild(self, SysModel, in_mult_KNet=5, out_mult_KNet=40):
        
        self.device = torch.device('cpu')

        self.InitSystemDynamics(SysModel.f, SysModel.h, SysModel.m, SysModel.n)

        self.InitKGainNet(SysModel.prior_Q, SysModel.prior_Sigma, SysModel.prior_S, in_mult_KNet, out_mult_KNet)

    def InitKGainNet(self, prior_Q, prior_Sigma, prior_S, in_mult_KNet, out_mult_KNet):

        self.seq_len_input = 1 
        self.batch_size = 0 

        self.prior_Q = prior_Q.to(self.device)
        self.prior_Sigma = prior_Sigma.to(self.device)
        self.prior_S = prior_S.to(self.device)
        

        # GRU per tracciare Q (process noise)
        self.d_input_Q = self.m * in_mult_KNet # 6 * 5 = 30
        self.d_hidden_Q = self.m ** 2 # 6**2 = 36
        self.GRU_Q = nn.GRU(self.d_input_Q, self.d_hidden_Q).to(self.device)

        # GRU per tracciare Sigma (covarianza stato)
        self.d_input_Sigma = self.d_hidden_Q # 36
        self.d_hidden_Sigma = self.m ** 2 # 6**2 = 36
        self.GRU_Sigma = nn.GRU(self.d_input_Sigma, self.d_hidden_Sigma).to(self.device)
        
        # FC 1 (usato per GRU_S)
        self.d_input_FC1 = self.d_hidden_Sigma # 36
        self.d_output_FC1 = self.n ** 2 # 5**2 = 25
        self.FC1 = nn.Sequential(
                    nn.Linear(self.d_input_FC1, self.d_output_FC1),
                    nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

        # FC 7 (usato per GRU_S)
        self.d_input_FC7 = self.n # 5
        self.d_output_FC7 = self.n # 5
        self.FC7 = nn.Sequential(
                    nn.Linear(self.d_input_FC7, self.d_output_FC7),
                    nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

        # GRU per tracciare S (covarianza innovazione)
        self.d_input_S = self.d_output_FC1 + self.d_output_FC7 # 25 + 5 = 30
        self.d_hidden_S = self.n ** 2 # 5**2 = 25
        self.GRU_S = nn.GRU(self.d_input_S, self.d_hidden_S).to(self.device)
        
        # FC 2 (calcolo K-Gain, parte 1)
        self.d_input_FC2 = self.d_hidden_S + self.d_hidden_Sigma # 25 + 36 = 61
        self.d_output_FC2 = self.n * self.m # 5 * 6 = 30
        self.d_hidden_FC2 = self.d_input_FC2 * out_mult_KNet # 61 * 40 = 2440
        self.FC2 = nn.Sequential(
                    nn.Linear(self.d_input_FC2, self.d_hidden_FC2),
                    nn.ReLU(),
                    nn.Linear(self.d_hidden_FC2, self.d_output_FC2), nn.Dropout(p=0.3)).to(self.device)

        # FC 3 (backward flow, parte 1)
        self.d_input_FC3 = self.d_hidden_S + self.d_output_FC2 # 25 + 30 = 55
        self.d_output_FC3 = self.m ** 2 # 6**2 = 36
        self.FC3 = nn.Sequential(
                    nn.Linear(self.d_input_FC3, self.d_output_FC3),
                    nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)

        # FC 4 (backward flow, parte 2)
        self.d_input_FC4 = self.d_hidden_Sigma + self.d_output_FC3 # 36 + 36 = 72
        self.d_output_FC4 = self.d_hidden_Sigma # 36
        self.FC4 = nn.Sequential(
                    nn.Linear(self.d_input_FC4, self.d_output_FC4),
                    nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)
        
        # FC 5 (usato per GRU_Q)
        self.d_input_FC5 = self.m # 6
        self.d_output_FC5 = self.m * in_mult_KNet # 6 * 5 = 30
        self.FC5 = nn.Sequential(
                    nn.Linear(self.d_input_FC5, self.d_output_FC5),
                    nn.ReLU(), nn.Dropout(p=0.1)).to(self.device)


    def InitSystemDynamics(self, f, h, m, n):
        
        # Salva la funzione di transizione (non lineare)
         #self.f deve accettare (x, u)
        self.f = f
        self.m = m # 6

        # Salva la funzione di osservazione (lineare o non lineare)
        # self.h accetta (x) e restituisce (y)
        self.h = h
        self.n = n # 5

 
    def InitSequence(self, M1_0, T):
        #input M1_0 (torch.tensor): stima stato iniziale [batch_size, m, 1]
        self.T = T
        self.m1x_posterior = M1_0.to(self.device)


    def step_prior(self, u):
        """
        Esegue la predizione usando l'input di controllo u
        """
        #La funzione f  richiede anche l'input di controllo u
        self.m1x_prior = self.f(self.m1x_posterior, u)

        # Predizione dell'osservazione (non richiede u)
        self.m1y = self.h(self.m1x_prior)

    def step_KGain_est(self, y):
        # Calcola l'innovazione (differenza tra misura e predizione)
        obs_innov_diff = torch.squeeze(y, 2) - torch.squeeze(self.m1y, 2) # [batch_size, n]
        
        # Prende lo stato predetto
        m1x_prior_sq = torch.squeeze(self.m1x_prior, 2) # [batch_size, m]

        # Normalizza gli input per la rete
        obs_innov_diff = func.normalize(obs_innov_diff, p=2, dim=1, eps=1e-12, out=None)
        m1x_prior_sq = func.normalize(m1x_prior_sq, p=2, dim=1, eps=1e-12, out=None)

        # Esegue la rete neurale per calcolare K
        KG = self.KGain_step(obs_innov_diff, m1x_prior_sq)

        # Rimodella l'output della rete [batch_size, m*n] in una matrice [batch_size, m, n]
        self.KGain = torch.reshape(KG, (self.batch_size, self.m, self.n)) # [B, 6, 5]


    def KNet_step(self, y, u):
        #Esegue un passo completo di stima (predizione + correzione)
        
        #Passa u alla fase di predizione
        self.step_prior(u)

        # Calcola il Guadagno di Kalman ottimale usando la rete
        self.step_KGain_est(y)

        # Calcola l'innovazione residua
        dy = y - self.m1y # [batch_size, n, 1]

        # Fase di Correzione: aggiorna la stima
        INOV = torch.bmm(self.KGain, dy)
        self.m1x_posterior = self.m1x_prior + INOV

        return self.m1x_posterior

    def KGain_step(self, obs_innov_diff, m1x_prior):
    
        #Architettura della rete neurale per il calcolo del K-Gain
    
        def expand_dim(x):
            # Aggiunge la dimensione temporale (seq_len = 1) per la GRU
            expanded = torch.empty(self.seq_len_input, self.batch_size, x.shape[-1]).to(self.device)
            expanded[0, :, :] = x
            return expanded

        obs_innov_diff = expand_dim(obs_innov_diff) # [1, B, n=5]
        m1x_prior = expand_dim(m1x_prior)           # [1, B, m=6]


        #Forward Flow 

        # FC 5
        in_FC5 = m1x_prior # [1, B, 6]
        out_FC5 = self.FC5(in_FC5) # [1, B, 30]

        # Q-GRU
        in_Q = out_FC5
        out_Q, self.h_Q = self.GRU_Q(in_Q, self.h_Q) # out_Q [1, B, 36]

        # Sigma_GRU
        in_Sigma = out_Q
        out_Sigma, self.h_Sigma = self.GRU_Sigma(in_Sigma, self.h_Sigma) # out_Sigma [1, B, 36]

        # FC 1
        in_FC1 = out_Sigma
        out_FC1 = self.FC1(in_FC1) # [1, B, 25]

        # FC 7
        in_FC7 = obs_innov_diff # [1, B, 5]
        out_FC7 = self.FC7(in_FC7) # [1, B, 5]

        # S-GRU
        in_S = torch.cat((out_FC1, out_FC7), 2) # [1, B, 25+5=30]
        out_S, self.h_S = self.GRU_S(in_S, self.h_S) # out_S [1, B, 25]

        # FC 2
        in_FC2 = torch.cat((out_Sigma, out_S), 2) # [1, B, 36+25=61]
        out_FC2 = self.FC2(in_FC2) # [1, B, 30] ( = m*n = 6*5)

        # Backward Flow 

        # FC 3
        in_FC3 = torch.cat((out_S, out_FC2), 2) # [1, B, 25+30=55]
        out_FC3 = self.FC3(in_FC3) # [1, B, 36]

        # FC 4
        in_FC4 = torch.cat((out_Sigma, out_FC3), 2) # [1, B, 36+36=72]
        out_FC4 = self.FC4(in_FC4) # [1, B, 36]

        # Aggiorna lo stato nascosto di Sigma_GRU per il prossimo passo
        self.h_Sigma = out_FC4

        return out_FC2 # [1, B, 30]


    # Forward 

    def forward(self, y, u):
      
        y = y.to(self.device)
        u = u.to(self.device)
        return self.KNet_step(y, u)


    # Init Hidden State 

    def init_hidden_KNet(self):
        # Inizializza gli stati nascosti delle 3 GRU
        
        weight = next(self.parameters()).data
        
        # Init h_S
        hidden_S = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_S).zero_()
        self.h_S = hidden_S.data
        if self.prior_S is not None:
             self.h_S = self.prior_S.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

        # Init h_Sigma
        hidden_Sigma = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Sigma).zero_()
        self.h_Sigma = hidden_Sigma.data
        if self.prior_Sigma is not None:
            self.h_Sigma = self.prior_Sigma.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)

        # Init h_Q
        hidden_Q = weight.new(self.seq_len_input, self.batch_size, self.d_hidden_Q).zero_()
        self.h_Q = hidden_Q.data
        if self.prior_Q is not None:
            self.h_Q = self.prior_Q.flatten().reshape(1, 1, -1).repeat(self.seq_len_input, self.batch_size, 1)