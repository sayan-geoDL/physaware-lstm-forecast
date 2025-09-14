#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prediction script for LSTM-based weather forecasting models.

Features:
- Loads trained LSTM models (single or ensemble) for inference.
- Supports flexible parameter selection:
  - Single run (final_model.pth or specified CV rank).
  - Ensemble mode (multiple CV-ranked models or all).
- Rescales predictions from standardized units to physical variables
  (dewpoint, temperature, relative humidity, pressure).
- For ensemble mode: computes mean forecasts with bootstrap confidence intervals (placeholder).
- Saves outputs as CSVs in ./out/predictions/ for downstream analysis.

Notes:
- Requires scaling parameters in ./data/processed/scaling_params.csv.
- Designed to integrate with station_model_util (predictor, batcher, bootstrap_ci).
"""

from station_model_util import predictor,bootstrap_ci
import numpy as np
import torch
import pandas as pd
import os
import logging
logger=logging.getLogger(__name__)

####################### prediction ############################################
def predict(x,t,params,
            scales='./data/processed/scaling_params.csv',
            device='cpu',mode='single',B=1000,
            low=2.5,high=97.5):
    """
    Generates predictions for unseen (future) data using trained LSTM model(s).

    Args:
        x (np.ndarray): Input features (time, batch, features).
        t (np.ndarray): Timestamps corresponding to input data.
        params (int, dict, list, or 'all'): LSTM hyperparameters or rank(s) to use for prediction.
        scales (str): Path to scaling parameters CSV.
        device (str): Device to use ('cpu' or 'cuda').
        mode (str): 'single' or 'ensemble'.
        B (int): Number of bootstrap samples for confidence intervals (ensemble).
        low, high (float): Percentiles for confidence intervals (ensemble).

    Saves:
        - For single mode:
            - Predictions to ./out/predictions/single_prediction.csv
        - For ensemble mode:
            - Predictions for each model to ./out/predictions/rank{rank}_prediction.csv
            - Ensemble mean and confidence intervals to ./out/predictions/ensemble_prediction.csv
    """
    if isinstance(params,int):
        df=pd.read_csv('./out/train_test/cv_result.csv')
        rank=[params]
        row=df.loc[params-1]
        params=row[['hidden_size','learning_rate','num_layers','weight_decay','lambda_physics']].to_dict()
        params=[params]
    elif isinstance(params,dict):
        keys=['hidden_size','learning_rate','num_layers','weight_decay','lambda_physics']
        rank=[0]
        params={k:params[k] for k in keys if k in params}
        params=[params]
    elif isinstance(params,list) and mode=='ensemble':
        df=pd.read_csv('./out/train_test/cv_result.csv')
        rank=params
        rows=df.iloc[[r-1 for r in rank]]
        params=rows[['hidden_size','learning_rate','num_layers','weight_decay','lambda_physics']].to_dict(orient='records')
    elif params=='all':
        df=pd.read_csv('./out/train_test/cv_result.csv')
        rank=list(range(1,len(df)+1))
        rows=df.iloc[[r-1 for r in rank]]
        params=rows[['hidden_size','learning_rate','num_layers','weight_decay','lambda_physics']].to_dict(orient='records')
    os.makedirs('./out/predictions',exist_ok=True)
    var_name=['dwpt','temp','rhum','pres']
    scaling_df=pd.read_csv(scales,index_col=0)
    temp_mean=scaling_df.loc['temp','mean']
    temp_std=scaling_df.loc['temp','std']
    dwpt_mean=scaling_df.loc['dwpt','mean']
    dwpt_std=scaling_df.loc['dwpt','std']
    pres_mean=scaling_df.loc['pres','mean']
    pres_std=scaling_df.loc['pres','std']
    x=torch.tensor(x,dtype=torch.float32).to(device)
    for i,param in enumerate(params):
        if mode=='single':
            state_dict='./out/models/final_model.pth'
        elif mode=='ensemble':
            state_dict=os.path.join("./out/models",f"rank{rank[i]}_model.pth")
            logger.info(f'Loading model from {state_dict}')
        model=predictor(input_size=x.shape[2],
                        hidden_size=int(param['hidden_size']),
                        num_layers=int(param['num_layers'])).to(device)
        model.load_state_dict(torch.load(state_dict,map_location=device))
        model.eval()
        y_pr=model(x).detach().cpu().numpy()
        # dwpt
        y_pr[:,0]=(y_pr[:,0]*dwpt_std)+dwpt_mean
        #temperature
        y_pr[:,1]=(y_pr[:,1]*temp_std)+temp_mean
        #rhum
        y_pr[:,2]=y_pr[:,2]*100
        #pres
        y_pr[:,3]=(y_pr[:,3]*pres_std)+pres_mean
        df_pr=pd.DataFrame({"time": t.ravel()})
        for j,var in enumerate(var_name):
            df_pr[var]=y_pr[:,j]
        df_pr=df_pr.round(2)
        if mode=='single':
            df_pr.to_csv('./out/predictions/single_prediction.csv',index=False)
        elif mode=='ensemble':
            df_pr.to_csv(f'./out/predictions/rank{rank[i]}_prediction.csv',index=False)
            logger.info(f'Saved predictions to ./out/predictions/rank{rank[i]}_prediction.csv')
    if mode=='ensemble':
        files=[f'./out/predictions/rank{r}_prediction.csv' for r in rank]
        dfs=[pd.read_csv(f) for f in files]
        df_ens=dfs[0][['time']].copy()
        for j,var in enumerate(var_name):
            preds=np.column_stack([df[var].values for df in dfs])
            df_ens[f'{var}_mean']=preds.mean(axis=1)
            df_ens[f'{var}_{str(low)}'],df_ens[f'{var}_{str(high)}']=bootstrap_ci(preds,B=B,low=low,high=high)
        df_ens=df_ens.round(2)
        df_ens.to_csv('./out/predictions/ensemble_prediction.csv',index=False)