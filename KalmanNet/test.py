import torch
import torch.nn as nn
from datetime import datetime
import os
import random

from vehicle_model import VehicleModel
from data_loader import load_vehicle_dataset
from data_loader import plot_results, plotSquaredError, plotBoxPlot, plotAverageTrajectories
from kalman_net import KalmanNetNN
from pipeline import Pipeline_EKF

def test(T_test=600):
    print("Pipeline Test Start")

    today = datetime.today()
    now = datetime.now()
    strToday = today.strftime("%m.%d.%y")
    strNow = now.strftime("%H:%M:%S")
    strTime = strToday + "_" + strNow
    print("Current Time =", strTime)
    
    path_results = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_KalmanNet/KNet_Vehicle_Results'
    if not os.path.exists(path_results):
        print(f"ERRORE: La cartella dei risultati '{path_results}' non esiste. Esegui prima il training.")
        return


    m = 6
    n = 5
    d = 2
    Ts = 0.02

    noisy_csv = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_gen_tra/mpc_piecewise_2.csv'  
    clean_csv = '/Users/dorianagiovarruscio/Desktop/tesi/codice python/cod_gen_tra/mpc_piecewise_.csv'

    device = torch.device('cpu')

    m1x_0 = torch.zeros(m, 1)
    prior_Q = torch.eye(m)
    prior_Sigma = torch.eye(m)
    prior_S = torch.eye(n)

    sys_model = VehicleModel(Ts, T_test, T_test, m1x_0, prior_Q, prior_Sigma, prior_S)

    print("Data Load Start...")
    (_, _, test_data) = load_vehicle_dataset(
        noisy_csv, 
        clean_csv, 
        T_steps=T_test,
        train_split=0.7,
        val_split=0.15
    )
    test_input, test_control, test_target = test_data
    N_T = test_input.shape[0]
    print(f"Test data loaded: {N_T} trajectories.")

    
    loss_obs = nn.MSELoss(reduction='mean')
    MSE_obs_linear_arr = torch.empty(N_T)
    
    # h(x) estrae i 5 stati misurabili dal target 6D
    test_target_5d = sys_model.h(test_target.transpose(1, 2).unsqueeze(-1)).squeeze(-1).transpose(1, 2)

    for i in range(N_T):
        MSE_obs_linear_arr[i] = loss_obs(test_input[i], test_target_5d[i]).item()
    
    MSE_obs_dB_avg = 10 * torch.log10(torch.mean(MSE_obs_linear_arr))
    print(f"-- Observation Noise Floor - MSE LOSS: {MSE_obs_dB_avg:.2f} [dB]")


    print("Loading pre-trained model from pipeline.")
    PipelineFile = path_results + 'pipeline_KalmanNet_Vehicle.pt'
    if os.path.exists(PipelineFile):
        KalmanNet_Pipeline = torch.load(PipelineFile, map_location=device)
        KalmanNet_Pipeline.model.eval()
    else:
        print(f"ERRORE: File pipeline '{PipelineFile}' non trovato. Esegui prima il training.")
        return

    
    print("Running KalmanNet Test.")
    [MSE_test_linear_arr, _, _, knet_out, _, _, squaredErrorKNet] = \
        KalmanNet_Pipeline.NNTest(sys_model, 
                                  test_input, test_target, test_control, 
                                  path_results)

    
    print("Generating plots.")
    
    # Converti MSE a dB per i plot
    MSE_obs_db_arr = 10 * torch.log10(MSE_obs_linear_arr)
    MSE_test_db_arr = 10 * torch.log10(MSE_test_linear_arr)

    # Scegli 5 traiettorie a caso da plottare
    indexes = random.sample(range(N_T), 5)
    
    # Plots
    plot_results(test_target, KNet_trajectories=knet_out, EKF_trajectories=None, indexes=indexes)
    plotBoxPlot(MSE_KNet=MSE_test_db_arr, MSE_EKF=None, MSE_obs=MSE_obs_db_arr)
    plotSquaredError(squaredErrorKNet=squaredErrorKNet, squaredErrorEKF=None, indexes=indexes)
    plotAverageTrajectories(squaredErrorKNet=squaredErrorKNet, squaredErrorEKF=None)



if __name__ == '__main__':
    test()