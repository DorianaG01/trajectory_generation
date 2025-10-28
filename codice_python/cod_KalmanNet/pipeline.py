import torch
import torch.nn as nn
import random
import time
import numpy as np
import wandb 
from matplotlib import pyplot as plt
from torch_optimizer import Lookahead
import torch.optim.lr_scheduler as lr_scheduler



class Pipeline_EKF:
 
    def __init__(self, Time, folderName, modelName):
        super().__init__()
        self.Time = Time
        self.folderName = folderName + '/'
        self.modelName = modelName
        self.modelFileName = self.folderName + "model_" + self.modelName + ".pt" 
        self.PipelineName = self.folderName + "pipeline_" + self.modelName + ".pt"


    def save(self):
        torch.save(self, self.PipelineName)


    def setssModel(self, ssModel):
        self.ssModel = ssModel 

    def setModel(self, model):
        self.model = model 

    def setTrainingParams(self, n_steps, n_batch, lr, wd, alpha):
        self.device = torch.device('cpu')
        self.N_steps = n_steps  # Number of Training Steps
        self.N_B = n_batch # Number of Samples in Batch
        self.learningRate = lr # Learning Rate
        self.weightDecay = wd # L2 Weight Regularization - Weight Decay
        self.alpha = alpha # Composition loss factor
        # MSE LOSS Function
        self.loss_fn = nn.MSELoss(reduction='mean')

        base_optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learningRate, weight_decay=self.weightDecay)
        self.optimizer = Lookahead(base_optimizer)

        
    def NNTrain(self, SysModel, cv_input, cv_target, cv_control, train_input, train_target, train_control, path_results, CompositionLoss, loadModel):

        if loadModel:
            checkpoint = torch.load(path_results+'.pt', map_location=torch.device('cpu'))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        startEpoch = 0


        self.N_E = len(train_input)
        self.N_CV = len(cv_input)

        self.MSE_cv_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_cv_dB_epoch = torch.zeros([self.N_steps])

        #scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.N_steps, eta_min=1e-7)
    
        """scheduler = lr_scheduler.CyclicLR(self.optimizer, 
                                      base_lr=1e-6,           # Il LR minimo dell'oscillazione
                                      max_lr=self.learningRate, #  5e-5)
                                      step_size_up=30,        # N. di epoche per salire
                                      step_size_down=30,      # N. di epoche per scendere
                                      mode='triangular2')     # Politica di oscillazione"""
        """scheduler = lr_scheduler.OneCycleLR(self.optimizer, 
                                    max_lr=self.learningRate, 
                                    epochs=self.N_steps, 
                                    steps_per_epoch=1,
                                    pct_start=0.2)"""

        self.MSE_train_linear_epoch = torch.zeros([self.N_steps])
        self.MSE_train_dB_epoch = torch.zeros([self.N_steps])


        self.MSE_cv_dB_opt = 1000
        self.MSE_cv_idx_opt = 0

        for ti in range(startEpoch, self.N_steps):

            if ti == 200: # Step manuale di riduzione del LR
                print(f"--- STEP MANUALE: Epoca {ti}, taglio LR a 1e-6 ---")
                new_lr = 1e-6 
               
                for param_group in self.optimizer.optimizer.param_groups:
                    param_group['lr'] = new_lr

        # Training Sequence Batch 

            self.optimizer.zero_grad()
            
            # Training Mode
            self.model.train()
            self.model.batch_size = self.N_B
            # Init Hidden State
            self.model.init_hidden_KNet()

            # Init Training Batch tensors
            y_training_batch = torch.zeros([self.N_B, SysModel.n, SysModel.T]).to(self.device) # Misure [B, 5, T]
            u_training_batch = torch.zeros([self.N_B, SysModel.d, SysModel.T]).to(self.device) # Controlli [B, 2, T] 
            train_target_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device) # Stato [B, 6, T]
            x_out_training_batch = torch.zeros([self.N_B, SysModel.m, SysModel.T]).to(self.device) # Output [B, 6, T]
            
            # Randomly select N_B training sequences
            assert self.N_B <= self.N_E  # N_B must be smaller than N_E
            n_e = random.sample(range(self.N_E), k=self.N_B)
            ii = 0
            for index in n_e:
                y_training_batch[ii, :, :] = train_input[index]
                train_target_batch[ii, :, :] = train_target[index]
                u_training_batch[ii, :, :] = train_control[index] 
                ii += 1

            # Init Sequence
            self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_B, 1, 1), SysModel.T)

            # Forward Computation
            for t in range(0, SysModel.T):
                y_t = torch.unsqueeze(y_training_batch[:, :, t], 2)
                u_t = torch.unsqueeze(u_training_batch[:, :, t], 2)
                x_out_training_batch[:, :, t] = torch.squeeze(self.model(y_t, u_t))


            # Compute Training Loss
            MSE_trainbatch_linear_LOSS = 0
            if CompositionLoss:
                y_hat = torch.zeros([self.N_B, SysModel.n, SysModel.T]) # [B, 5, T]
                for t in range(SysModel.T):
                    # SysModel.h prende [B, 6, 1] e restituisce [B, 5, 1]
                    y_hat[:, :, t] = torch.squeeze(SysModel.h(torch.unsqueeze(x_out_training_batch[:, :, t], dim=2)), dim=2)
                
                # loss1 = loss(stima_6D, target_6D)
                loss1 = self.loss_fn(x_out_training_batch, train_target_batch)
                # loss2 = loss(h(stima)_5D, misura_5D)
                loss2 = self.loss_fn(y_hat, y_training_batch)
                
                MSE_trainbatch_linear_LOSS = self.alpha * loss1 + (1 - self.alpha) * loss2
                
            else:  # no composition loss
                MSE_trainbatch_linear_LOSS = self.loss_fn(x_out_training_batch, train_target_batch)

            # dB Loss
            self.MSE_train_linear_epoch[ti] = MSE_trainbatch_linear_LOSS.item()
            epsilon =  1e-10
            self.MSE_train_dB_epoch[ti] = 10 * torch.log10(self.MSE_train_linear_epoch[ti]) + epsilon

            # Optimizing 
            MSE_trainbatch_linear_LOSS.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=1.0)
            self.optimizer.step()

            #scheduler.step()

            # Validation Sequence Batch 
            self.model.eval()
            self.model.batch_size = self.N_CV
            # Init Hidden State
            self.model.init_hidden_KNet()

            with torch.no_grad():

                SysModel.T_test = cv_input.size()[-1] 
                x_out_cv_batch = torch.empty([self.N_CV, SysModel.m, SysModel.T_test]).to(self.device)

                # Init Sequence
                self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_CV, 1, 1), SysModel.T_test)

                for t in range(0, SysModel.T_test):
                    y_t = torch.unsqueeze(cv_input[:, :, t], 2)
                    u_t = torch.unsqueeze(cv_control[:, :, t], 2) 
                    x_out_cv_batch[:, :, t] = torch.squeeze(self.model(y_t, u_t))

                # Compute CV Loss
                # La CV loss è calcolata solo sullo stato (6D vs 6D)
                MSE_cvbatch_linear_LOSS = self.loss_fn(x_out_cv_batch, cv_target)

                # dB Loss
                self.MSE_cv_linear_epoch[ti] = MSE_cvbatch_linear_LOSS.item()
                epsilon =  1e-10
                self.MSE_cv_dB_epoch[ti] = 10 * torch.log10(self.MSE_cv_linear_epoch[ti]) + epsilon
                #scheduler.step(MSE_cvbatch_linear_LOSS.item())

                if (self.MSE_cv_dB_epoch[ti] < self.MSE_cv_dB_opt):
                    self.MSE_cv_dB_opt = self.MSE_cv_dB_epoch[ti]
                    self.MSE_cv_idx_opt = ti

                    # Salva il percorso in una variabile
                    model_path = path_results + 'best_model.pt'

                    torch.save({
                        'epoch': ti, 'model_state_dict': self.model.state_dict(), 'optimizer_state_dict': self.optimizer.optimizer.state_dict()
                    }, model_path) 

                    artifact = wandb.Artifact(f'model-{wandb.run.name}', type='model')
                    artifact.add_file(model_path)
                    
                    wandb.log_artifact(artifact, aliases=['best', f'epoch_{ti}'])

            # Training Summary 
            print(ti, "MSE Training :", self.MSE_train_dB_epoch[ti], "[dB]", "MSE Validation :",
                  self.MSE_cv_dB_epoch[ti],
                  "[dB]")
            
            wandb.log({
                "epoch": ti,
                "train_loss_dB": self.MSE_train_dB_epoch[ti],
                "val_loss_dB": self.MSE_cv_dB_epoch[ti],
                "train_loss_linear": self.MSE_train_linear_epoch[ti],
                "val_loss_linear": self.MSE_cv_linear_epoch[ti],
                "learning_rate": self.optimizer.optimizer.param_groups[0]['lr']
            })

            if (ti > 1):
                d_train = self.MSE_train_dB_epoch[ti] - self.MSE_train_dB_epoch[ti - 1]
                d_cv = self.MSE_cv_dB_epoch[ti] - self.MSE_cv_dB_epoch[ti - 1]
                print("diff MSE Training :", d_train, "[dB]", "diff MSE Validation :", d_cv, "[dB]")

            print("Optimal idx:", self.MSE_cv_idx_opt, "Optimal :", self.MSE_cv_dB_opt, "[dB]")

        #self.learningCurve(self.MSE_train_dB_epoch, self.MSE_cv_dB_epoch)


        return [self.MSE_cv_linear_epoch, self.MSE_cv_dB_epoch, self.MSE_train_linear_epoch, self.MSE_train_dB_epoch]

    def NNTest(self, SysModel, test_input, test_target, test_control, path_results, load_model=False, load_model_path=None):
    
        checkpoint = torch.load(path_results + 'best_model.pt', map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.N_T = test_input.shape[0]
        SysModel.T_test = test_input.size()[-1]
        self.MSE_test_linear_arr = torch.zeros([self.N_T])
        x_out_test = torch.zeros([self.N_T, SysModel.m, SysModel.T_test]).to(self.device) # [N, 6, T]

        # MSE LOSS Function
        loss_fn = nn.MSELoss(reduction='mean')

        # Test mode
        self.model.eval()
        self.model.batch_size = self.N_T
        # Init Hidden State
        self.model.init_hidden_KNet()
        torch.no_grad()
        MSESingleTrajectory = torch.zeros([self.N_T, SysModel.T_test])

        start = time.time()


        self.model.InitSequence(SysModel.m1x_0.reshape(1, SysModel.m, 1).repeat(self.N_T, 1, 1), SysModel.T_test)

        for t in range(0, SysModel.T_test):
            y_t = torch.unsqueeze(test_input[:, :, t], 2)
            u_t = torch.unsqueeze(test_control[:, :, t], 2) 
            x_out_test[:, :, t] = torch.squeeze(self.model(y_t, u_t))

        end = time.time()
        t = end - start

        # MSE loss
        for j in range(self.N_T):  # cannot use batch due to different length and std computation
            self.MSE_test_linear_arr[j] = loss_fn(x_out_test[j, :, :], test_target[j, :, :]).item()
            for k in range(SysModel.T_test):
                MSESingleTrajectory[j][k] = loss_fn(x_out_test[j, :, k], test_target[j, :, k]).item()


        # Average
        self.MSE_test_linear_avg = torch.mean(self.MSE_test_linear_arr)
        self.MSE_test_dB_avg = 10 * torch.log10(self.MSE_test_linear_avg)

        # Standard deviation
        self.MSE_test_linear_std = torch.std(self.MSE_test_linear_arr, unbiased=True)

        # Confidence interval
        self.test_std_dB = 10 * torch.log10(self.MSE_test_linear_std + self.MSE_test_linear_avg) - self.MSE_test_dB_avg

        # Print MSE and std
        str_name = self.modelName + "-" + "MSE LOSS:"
        print("--", str_name, self.MSE_test_dB_avg, "[dB]")
        str_name = self.modelName + "-" + "STD:"
        print("--", str_name, self.test_std_dB, "[dB]")
        # Print Run Time
        #print("Inference Time:", t)

        KalmanGainKN = self.model.KGain
        MSESingleTrajectory = 10*torch.log10(MSESingleTrajectory)

        return [self.MSE_test_linear_arr, self.MSE_test_linear_avg, self.MSE_test_dB_avg, x_out_test, t, KalmanGainKN, MSESingleTrajectory]


    def learningCurve(self, trainLoss, validationLoss):
        print(trainLoss.shape)
        print(validationLoss.shape)
        plt.plot(range(1, len(trainLoss)+1), trainLoss, label='Train Loss', marker='o')
        plt.plot(range(1, len(validationLoss)+1), validationLoss, label='Validation Loss', marker='x')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close('all')