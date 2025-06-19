#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 14:23:51 2025

@author: sayan
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
from station_model_util import predictor
import logging
logger = logging.getLogger(__name__)


################### traininig #################################################
def trainer(x,y,params,epochs=400,device='cpu'):
    """
    Train an LSTM model on the provided time series data.

    This function initializes and trains a PyTorch LSTM model using MSE loss
    and the Adam optimizer. It optionally retrieves hyperparameters from a
    saved cross-validation result file if `params` is an integer. It also 
    plots and saves the training loss curve and stores the trained model 
    as a `.pth` file.

    Parameters
    ----------
    x : numpy.ndarray
        Input features of shape (samples, sequence_length, features).
    y : numpy.ndarray
        Target values of shape (samples,).
    params : int or dict
        If int, treated as rank/index into a CSV file containing hyperparameters.
        If dict, should contain 'hidden_size', 'learning_rate', 
        'num_layers', and optionally 'weight_decay'.
    epochs : int, optional
        Number of training epochs. Default is 400.
    device : str, optional
        Device to run training on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    model : torch.nn.Module
        The trained LSTM model.

    Saves
    -----
    - A plot of the training loss curve to './plots/Train_loss_curve.png'.
    - The trained model's weights to './out/final_model.pth'.
    """
    phi=(1+5**(0.5))/2
    if isinstance(params, int):
        df=pd.read_csv('./out/train_test/cv_result.csv')
        row=df.loc[params-1]
        params = row[['hidden_size', 'learning_rate', 'num_layers', 'weight_decay']].to_dict()
    elif isinstance(params,dict):
        keys=['hidden_size','learning_rate','num_layers','weight_decay']
        params={k:params[k] for k in keys if k in params}
    logger.info(f"Training model with parameters: {params}")
    x_train=torch.tensor(x,dtype=torch.float32).to(device)
    y_train=torch.tensor(y,dtype=torch.float32).to(device)
    model=predictor(input_size=x.shape[2],
                    hidden_size=int(params['hidden_size']),
                    num_layers=int(params['num_layers'])).to(device)
    criterion=nn.MSELoss()
    if params.get('weight_decay') is not None:
        optimizer=optim.Adam(model.parameters(),
                             lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])
    else:
        optimizer=optim.Adam(model.parameters(),
                             lr=params['learning_rate'])
    losses=[]
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out=model(x_train)
        loss=criterion(out.squeeze(),y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    fig,axs=plt.subplots(figsize=(7*phi,7),dpi=150)
    axs.plot(range(epochs),losses,lw=3,color='black',label='Train loss')
    axs.set_xlabel('Epoch',fontsize=18)
    axs.set_ylabel('MSE Loss',fontsize=18)
    axs.tick_params(axis='both',which='both',length=5,width=3,labelsize=20)
    axs.tick_params(axis='x',rotation=45)
    axs.legend(loc='best',prop={'size':20})
    fig.tight_layout()
    fig_name="Train_loss_curve"
    ext='png'
    filename=f"{fig_name}.{ext}"
    full_path=os.path.join('./plots',filename)
    counter=1
    while os.path.exists(full_path):
        filename=f"{fig_name}_{counter}.{ext}"
        full_path=os.path.join('./plots',filename)
        counter+=1
    fig.savefig(full_path)
    plt.show()
    torch.save(model.state_dict(), "./out/final_model.pth")
############### testing #######################################################
def tester(x_tr,y_tr,t_tr,
           x_tst,y_tst,t_tst,
           params,scales='./data/processed/scaling_params.csv',
           state_dict='./out/final_model.pth',
           device='cpu'):
    """
    Evaluate the trained LSTM temperature prediction model on training and testing datasets,
    generate visual diagnostics, and save performance metrics.

    Parameters
    ----------
    x_tr : np.ndarray
        Input training features with shape (n_samples, sequence_length, n_features).
    y_tr : np.ndarray
        Target training values (scaled).
    t_tr : np.ndarray
        Time vector for training data.
    x_tst : np.ndarray
        Input testing features with shape (n_samples, sequence_length, n_features).
    y_tst : np.ndarray
        Target testing values (scaled).
    t_tst : np.ndarray
        Time vector for testing data.
    params : int or dict
        Model hyperparameters. If `int`, loads the corresponding row from a CSV file.
        If `dict`, should include keys: 'hidden_size', 'learning_rate', 'num_layers', 'weight_decay'.
    scales : str, optional
        Path to CSV file containing mean and std values for inverse transformation (default is './data/processed/scaling_params.csv').
    state_dict : str, optional
        Path to saved model weights (.pth file).
    device : str, optional
        Device to run the model on ('cpu' or 'cuda').

    Returns
    -------
    None

    Saves
    -----
    - './plots/train_test_distribution.png' : Histogram comparison of observed and predicted temperature distributions.
    - './plots/train_test_all_ts.png'       : Time series plot showing full observed and predicted data.
    - './plots/train_test_ts.png'           : Side-by-side time series plots for training and testing periods.
    - './out/train_test/test_summary.csv'   : Summary table with mean, std, RMSE, and R² for observed and predicted values.

    Notes
    -----
    - RMSE and R² are only computed for predicted values.
    - Assumes 'temp' row exists in the scaling CSV for de-normalizing temperature predictions.
    - Designed for post-training model evaluation and visualization.
    """
    phi=(1+5**(0.5))/2
    if isinstance(params, int):
        df=pd.read_csv('./out/train_test/cv_result.csv')
        row=df.loc[params-1]
        params = row[['hidden_size', 'learning_rate', 'num_layers', 'weight_decay']].to_dict()
    elif isinstance(params,dict):
        keys=['hidden_size','learning_rate','num_layers','weight_decay']
        params={k:params[k] for k in keys if k in params}
    logger.info(f"Evaluating model with parameters: {params}")
    model=predictor(input_size=x_tr.shape[2],
                    hidden_size=int(params['hidden_size']),
                    num_layers=int(params['num_layers'])).to(device)
    model.load_state_dict(torch.load(state_dict,map_location=device))
    x_train=torch.tensor(x_tr,dtype=torch.float32).to(device)
    x_test=torch.tensor(x_tst,dtype=torch.float32).to(device)
    model.eval()
    train_y_pr=model(x_train).detach().cpu().numpy().ravel()
    test_y_pr=model(x_test).detach().cpu().numpy().ravel()
    scaling_df=pd.read_csv(scales,index_col=0)
    temp_mean=scaling_df.loc['temp','mean']
    temp_std=scaling_df.loc['temp','std']
    train_y_pr=(train_y_pr*temp_std)+temp_mean
    test_y_pr=(test_y_pr*temp_std)+temp_mean
    train_y_obs=(y_tr*temp_std)+temp_mean
    test_y_obs=(y_tst*temp_std)+temp_mean
    df_tr=pd.DataFrame({'time':t_tr.ravel(),
                           'observed':train_y_obs,
                           'predicted':train_y_pr})
    df_tr=df_tr.round(2)
    df_tst=pd.DataFrame({'time':t_tst.ravel(),
                           'observed':test_y_obs,
                           'predicted':test_y_pr})
    df_tst=df_tst.round(2)
    rmse_tr=np.sqrt(((df_tr['observed']-df_tr['predicted'])**2).mean())
    rmse_tst=np.sqrt(((df_tst['observed']-df_tst['predicted'])**2).mean())
    r2_tr=1-((((df_tr['predicted']-df_tr['observed'])**2).sum())/(((df_tr['observed']-df_tr['observed'].mean())**2).sum()))
    r2_tst=1-((((df_tst['predicted']-df_tst['observed'])**2).sum())/(((df_tst['observed']-df_tst['observed'].mean())**2).sum()))
    fig_dist,axs_dist=plt.subplots(1,2,figsize=(10*phi,10),dpi=150)
    for ax,df, title in zip(axs_dist,[df_tr,df_tst],['training','testing']):
        counts,bins=np.histogram(df['observed'],bins='auto')
        ax.hist(df['observed'], bins=bins, alpha=0.8, color='gray', label='Observed')
        ax.hist(df['predicted'], bins=bins, alpha=0.4, color='red', label='predicted')
        if title=='training':
            ax.set_title(title+f"\n rmse= {round(rmse_tr,4)}$^\\circ$C \nR2={round(r2_tr,4)}", fontsize=20)
        elif title=='testing':
            ax.set_title(title+f"\n rmse= {round(rmse_tst,4)}$^\\circ$C \nR2={round(r2_tst,4)}", fontsize=20)
        ax.set_xlabel(r'Temperature ($^\circ$C)',fontsize=18)
        ax.set_ylabel('Frequency',fontsize=18)
        ax.tick_params(axis='both',which='both',length=5,width=3,labelsize=18)
        ax.tick_params(axis='x',which='both',rotation=45)
        ax.legend(loc='best',prop={'size':20})
    fig_dist.tight_layout()
    fig_dist.savefig('./plots/train_test_distribution.png')
    df_all=pd.concat([df_tr,df_tst],ignore_index=True)
    df_all=df_all.sort_values(by='time')
    fig_ts,axs_ts=plt.subplots(figsize=(10*phi,10),dpi=150)
    axs_ts.plot(df_tr['time'],df_tr['observed'],color='navy',lw=2,label='Observed Training')
    axs_ts.plot(df_tst['time'],df_tst['observed'],color='forestgreen',lw=2,label='Observed Testing')
    axs_ts.plot(df_all['time'],df_all['predicted'],color='crimson',lw=1.5,label='Predicted', linestyle='--')
    split_date = df_tst['time'].min()
    axs_ts.axvline(split_date, color='gray', linestyle='--', lw=1)
    axs_ts.grid(True, linestyle='--', alpha=0.8)
    axs_ts.set_ylabel(r'Temperature ($^\circ$C)',fontsize=18)
    axs_ts.tick_params(axis='both',which='both',length=5,width=3,labelsize=18)
    axs_ts.tick_params(axis='x',which='both',rotation=45)
    axs_ts.legend(loc='best',prop={'size':20})
    fig_ts.tight_layout()
    fig_ts.savefig('./plots/train_test_all_ts.png')
    fig_ts_indi,axs_ts_indi=plt.subplots(1,2,figsize=(12*phi,12),dpi=150)
    for ax, df,title in zip(axs_ts_indi,[df_tr,df_tst],['Training','Testing']):
        if title=='Training':
            ax.set_title(title+f"\n rmse= {round(rmse_tr,4)}$^\\circ$C\nR2={round(r2_tr,4)}", fontsize=25)
            ax.plot(df['time'],df['observed'],color='navy',lw=2,label='Observed Training')
        if title=='Testing':
            ax.set_title(title+f"\n rmse= {round(rmse_tst,4)}$^\\circ$C\nR2={round(r2_tst,4)}", fontsize=25)
            ax.plot(df['time'],df['observed'],color='forestgreen',lw=2,label='Observed Training')
        ax.plot(df['time'],df['predicted'],color='crimson',lw=1.5,label='Predicted', linestyle='--')
        ax.grid(True, linestyle='--', alpha=0.8)
        ax.set_ylabel(r'Temperature ($^\circ$C)',fontsize=24)
        ax.tick_params(axis='both',which='both',length=10,width=5,labelsize=20)
        ax.tick_params(axis='x',which='both',rotation=45)
        ax.legend(loc='best',prop={'size':20})
    fig_ts_indi.tight_layout()
    fig_ts_indi.savefig('./plots/train_test_ts.png')
    summary_df = pd.DataFrame({
    'mean training': [round(df_tr['observed'].mean(),2), round(df_tr['predicted'].mean(),2)],
    'std training': [round(df_tr['observed'].std(),2), round(df_tr['predicted'].std(),2)],
    'rmse training': [np.nan, rmse_tr],
    'R2 training': [np.nan, r2_tr],
    'mean test': [round(df_tst['observed'].mean(),2), round(df_tst['predicted'].mean(),2)],
    'std test': [round(df_tst['observed'].std(),2), round(df_tst['predicted'].std(),2)],
    'rmse test': [np.nan, rmse_tst],
    'R2 test': [np.nan, r2_tst]
    }, index=['Observed', 'Predicted'])

    summary_df = summary_df.round(4)
    summary_df.to_csv('./out/train_test/test_summary.csv')
    df_tst.to_csv('./out/train_test/test_ts.csv')
    df_tr.to_csv('./out/train_test/train_ts.csv')
