#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 14:32:18 2025

@author: sayan
"""

from station_model_util import predictor
import torch
import pandas as pd

####################### prediction ############################################
def predict(x,t,params,state_dict='./out/final_model.pth',
            scales='./data/processed/scaling_params.csv',
            device='cpu'):
    """
    Run temperature prediction using a trained LSTM model.

    Parameters
    ----------
    x : np.ndarray
        Input features of shape (samples, timesteps, features).
    t : np.ndarray
        Corresponding time array for predictions (forecast times).
    params : int or dict
        If int, selects the `params`-th row from CV results CSV as model hyperparameters.
        If dict, uses the provided hyperparameter values directly.
        Required keys: 'hidden_size', 'learning_rate', 'num_layers', 'weight_decay'.
    state_dict : str, optional
        Path to the trained model weights (.pth file). Default is './out/final_model.pth'.
    scales : str, optional
        Path to the CSV containing scaling parameters (mean and std for temperature).
        Default is './data/processed/scaling_params.csv'.
    device : str, optional
        Device to run inference on ('cpu' or 'gpu'). Default is 'gpu'.

    Returns
    -------
    None
        Writes predicted temperatures to './out/predicted_ts.csv' with columns:
        'time' (forecast timestamp) and 'predicted' (temperature in original scale).
    """
    if isinstance(params, int):
        df=pd.read_csv('./out/train_test/cv_result.csv')
        row=df.loc[params-1]
        params = row[['hidden_size', 'learning_rate', 'num_layers', 'weight_decay']].to_dict()
    elif isinstance(params,dict):
        keys=['hidden_size','learning_rate','num_layers','weight_decay']
        params={k:params[k] for k in keys if k in params}
    model=predictor(input_size=x.shape[2],
                    hidden_size=int(params['hidden_size']),
                    num_layers=int(params['num_layers'])).to(device)
    model.load_state_dict(torch.load(state_dict,map_location=device))
    x=torch.tensor(x,dtype=torch.float32).to(device)
    model.eval()
    y_pr=model(x).detach().cpu().numpy().ravel()
    scaling_df=pd.read_csv(scales,index_col=0)
    temp_mean=scaling_df.loc['temp','mean']
    temp_std=scaling_df.loc['temp','std']
    y_pr=(y_pr*temp_std)+temp_mean
    df_pr=pd.DataFrame({'time':t.ravel(),'predicted':y_pr})
    df_pr.to_csv('./out/predicted_ts.csv')