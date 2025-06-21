#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 17:46:40 2025

@author: sayan
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math as ma
from itertools import product
import matplotlib.pyplot as plt
import sys
import seaborn as sns
import os
from station_model_util import predictor
from station_extra_utilities import get_cpu_temperature,get_gpu_temperature
import logging
logger = logging.getLogger(__name__)


######################## cross validator ######################################
def cv_outs (x,y,param_grid,initial_train_size=None,
             val_size=10,k=5,device='cpu',epochs=300,es_patience=15):
    """
    Perform cross-validation over a grid of hyperparameters for an LSTM-based predictor.

    This function implements expanding window cross-validation. It trains and validates
    models for each hyperparameter combination across `k` folds, and evaluates each using:
    - MSE on validation set
    - A custom "overfit score" that penalizes divergence between train and validation MSE

    Parameters
    ----------
    x : ndarray
        3D array of input features of shape (samples, time_steps, features).
    y : ndarray
        1D array of target values.
    param_grid : dict
        Dictionary specifying grid search space with keys:
        'hidden_size', 'num_layers', 'learning_rate', and 'weight_decay'.
        Each value is either a list of values or a (start, stop, step) tuple.
    initial_train_size : int, optional
        Size of the initial training set. If None, it will be computed to allow `k` folds.
    val_size : int, optional
        Validation set size as a percentage of total data. Default is 10.
    k : int, optional
        Number of cross-validation folds. Default is 5.
    device : str, optional
        Either 'cpu' or 'cuda'. Default is 'cpu'.
    epochs : int, optional
        Maximum number of training epochs per fold. Default is 300.
    es_patience : int, optional
        Patience for early stopping. If None, no early stopping. Default is 15.

    Returns
    -------
    df_results : pandas.DataFrame
        A DataFrame sorted by mean overfit score containing:
        - hyperparameters
        - mean/std of overfit scores
        - mean RMSE across validation folds

    Notes
    -----
    - The results are saved to './out/train_test/cv_result.csv'.
    - If `device='cuda'`, GPU temperature is monitored to prevent overheating.
    - An "overfit score" is defined as:
        val_mse + (0.5 * |val_mse - train_mse|) / train_mse
      which penalizes poor generalization.
    """
    def get_param_combinations(param_grid):
        keys=list(param_grid.keys())
        values=list(param_grid.values())
        combinations=list(product(*values))
        return [dict(zip(keys,combo)) for combo in combinations]
    param_combos=get_param_combinations(param_grid)
    val_size=round((val_size/100)*len(y))
    if initial_train_size is None:
        initial_train_size = (x.shape[0] - k * val_size) // k
        if initial_train_size < 1:
            raise ValueError("Too many folds or large val_size for available data.")
    train_starts = np.linspace(initial_train_size
                               , np.shape(x)[0] - val_size, k).astype(int)
    folds=[]
    for train_end in train_starts:
        train_end=int(train_end)
        val_start=train_end
        val_end=val_start+val_size
        folds.append((0,train_end,val_start,val_end))
    results=[]
    count=0
    for i, params in enumerate(param_combos):
        ############### temporary function for temperature monitoring
        
        ###############################################################
        overfit_scores=[]
        val_rmses=[]
        for fold_id, (train_start,train_end,val_start,val_end) in enumerate(folds):
            x_train = torch.tensor(x[train_start:train_end,:,:], dtype=torch.float32).to(device)
            y_train = torch.tensor(y[train_start:train_end],dtype=torch.float32).to(device)
            x_val = torch.tensor(x[val_start:val_end,:,:],dtype=torch.float32).to(device)
            y_val = torch.tensor(y[val_start:val_end],dtype=torch.float32).to(device)
            model=predictor(input_size=x.shape[2],
                            hidden_size=params['hidden_size'],
                            num_layers=params['num_layers']).to(device)
            criterion=nn.MSELoss()
            if params.get('weight_decay') is not None:
                optimizer = optim.Adam(model.parameters(),
                                       lr=params['learning_rate'],
                                       weight_decay=params['weight_decay'])
            else:
                optimizer = optim.Adam(model.parameters(),
                           lr=params['learning_rate'])

            
            best_val_loss=float('inf')
            counter=0
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out=model(x_train)
                loss=criterion(out.squeeze(),y_train)
                loss.backward()
                optimizer.step()
                if es_patience!=None:
                    model.eval()
                    with torch.no_grad():
                        val_out = model(x_val).squeeze()
                        val_loss = criterion(val_out, y_val).item()
                    if val_loss< best_val_loss-1e-5:
                        best_val_loss=val_loss
                        counter=0
                        best_model_state = model.state_dict()
                    else:
                        counter+=1
                    if counter>=es_patience:
                        logger.info(f"Early stopping at epoch {epoch+1}, val_loss did not improve for {es_patience} epochs")
                        break
            if es_patience!=None:
                model.load_state_dict(best_model_state)
            model.eval()
            with torch.no_grad():
                train_pred=model(x_train).squeeze()
                val_pred=model(x_val).squeeze()
                train_mse=(torch.mean((train_pred - y_train) ** 2)).item()
                val_mse=(torch.mean((val_pred - y_val) ** 2)).item()
                overfit_score= val_mse+((0.5*(ma.fabs(val_mse-train_mse)))/train_mse) if train_mse>1e-6 else float('inf')
                overfit_scores.append(overfit_score)
                val_rmses.append(ma.sqrt(val_mse))
                if device=='cpu':
                    temp = get_cpu_temperature()
                elif device=='cuda':
                    temp = get_gpu_temperature()
                if temp is not None:
                    if temp > 90:
                        logger.info(f"Temperature {temp}°C exceeded threshold! Exiting.")
                        sys.exit(1)

        mean_score=np.mean(overfit_scores)
        std_score=np.std(overfit_scores)
        mean_rmse=np.mean(val_rmses)
        count+=1
        results.append({
            **params,
            'mean_overfit_score':mean_score,
            'std_overfit_score':std_score,
            'mean_rmse':mean_rmse})
        logger.info(f'grid combo {count}: {params} - completed')
    df_results=pd.DataFrame(results)
    df_results = df_results.sort_values('mean_overfit_score').reset_index(drop=True)
    df_results=df_results.round(4)
    df_results.to_csv('./out/train_test/cv_result.csv')  
    return df_results
def performance_box_plots(in_file='./out/train_test/cv_result.csv'):
    """
    Generate and save boxplots showing the relationship between hyperparameters 
    and model overfit scores.

    This function reads the cross-validation results from a CSV file, then creates 
    boxplots comparing the `mean_overfit_score` across various hyperparameter choices 
    (e.g., hidden units, number of layers, learning rate, weight decay). 
    The plots are saved to the ./plots directory with a unique filename.

    Parameters
    ----------
    in_file : str, optional
        Path to the CSV file containing the results of cross-validation. 
        The file must contain the columns: 
        ['hidden_size', 'num_layers', 'learning_rate', 'weight_decay', 'mean_overfit_score'].
        Default is './out/train_test/cv_result.csv'.

    Saves
    -----
    A PNG file named `performance_boxplots.png` (or `performance_boxplots_1.png`, etc. 
    if the filename already exists) in the `./plots` directory.

    Returns
    -------
    None
    """
    phi=(1+5**(0.5))/2
    df=pd.read_csv(in_file)
    score='mean_overfit_score'
    df['weight_decay_clean']=df['weight_decay'].fillna('No L2')
    params={'hidden_size':'Hidden Units',
            'num_layers': 'Number of Layers',
            'weight_decay_clean':'Weight Decay',
            'learning_rate': 'Learning Rate'}
    colors = ['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3']
    fig,axes=plt.subplots(2,2,figsize=(10*phi,10),dpi=150)
    axes=axes.flatten()
    for i, (param,label) in enumerate(params.items()):
        sns.boxplot(x=param,y=score,data=df,ax=axes[i],color=colors[i])
        axes[i].set_title(f'{label} vs Overfit Score')
        axes[i].tick_params(axis='x',rotation=45)
        axes[i].set_xlabel(label,fontsize=18)
        axes[i].set_ylabel('Overfit Score',fontsize=18)
        axes[i].tick_params(axis='both',labelsize=20)
    fig.tight_layout()
    fig_name='performance_boxplots'
    ext='png'
    filename=f"{fig_name}.{ext}"
    full_path=os.path.join('./plots',filename)
    counter=1
    while os.path.exists(full_path):
        filename=f"{fig_name}_{counter}.{ext}"
        full_path=os.path.join('./plots', filename)
        counter+=1
    fig.savefig(full_path)
    plt.show(block=False)

###################### plotter for cross validation ###########################
def cv_plot(x,y,cv_result='./out/train_test/cv_result.csv',initial_train_size=None,
             val_size=10,k=5,device='cpu',epochs=500,rank=1):
    """
    Visualize training and validation loss curves across k folds for a selected 
    hyperparameter configuration from cross-validation results.

    This function loads a specific set of hyperparameters from a cross-validation 
    result file (based on the provided rank), performs k-fold time series split, 
    trains the model on each fold, and plots the training and validation loss 
    curves over epochs.

    Parameters
    ----------
    x : np.ndarray
        Input feature array with shape (samples, time_steps, features).
    y : np.ndarray
        Target array with shape (samples,).
    cv_result : str, optional
        Path to the CSV file containing cross-validation results.
        Default is './out/train_test/cv_result.csv'.
    initial_train_size : int or None, optional
        Size of the initial training set. If None, it is computed automatically 
        based on k and val_size.
    val_size : int, optional
        Validation size as a percentage of total samples (default is 10).
    k : int, optional
        Number of cross-validation folds (default is 5).
    device : str, optional
        Device to train the model on ('cpu' or 'cuda'). Default is 'cpu'.
    epochs : int, optional
        Number of training epochs for each fold (default is 500).
    rank : int, optional
        Rank of the model in the `cv_result` file to use for training (1-based index). 
        Default is 1 (i.e., best performing).

    Saves
    -----
    One plot per fold showing training and validation loss curves. 
    The plots are saved in the `./plots` directory with filenames like `rank{rank}_fold_{i}.png`.

    Returns
    -------
    None
    """
    phi=(1+5**(0.5))/2
    params=pd.read_csv(cv_result).loc[rank-1][['hidden_size',
                                               'learning_rate',
                                               'num_layers',
                                               'weight_decay']]
    logger.info(f"Using hyperparameters (rank {rank}): {params.to_dict()}")
    val_size=round((val_size/100)*len(y))
    if initial_train_size is None:
        initial_train_size = (x.shape[0] - k * val_size) // k
        if initial_train_size < 1:
            raise ValueError("Too many folds or large val_size for available data.")
    train_starts = np.linspace(initial_train_size
                               , x.shape[0] - val_size, k).astype(int)
    folds=[]
    for train_end in train_starts:
        train_end=int(train_end)
        val_start=train_end
        val_end=val_start+val_size
        folds.append((0,train_end,val_start,val_end))
    train_fold=[]
    val_fold=[]
    for fold_id, (train_start,train_end,val_start,val_end) in enumerate(folds):
        x_train = torch.tensor(x[train_start:train_end,:,:], dtype=torch.float32).to(device)
        y_train = torch.tensor(y[train_start:train_end],dtype=torch.float32).to(device)
        x_val = torch.tensor(x[val_start:val_end,:,:],dtype=torch.float32).to(device)
        y_val = torch.tensor(y[val_start:val_end],dtype=torch.float32).to(device)
        model=predictor(input_size=x.shape[2],
                        hidden_size=int(params['hidden_size']),
                        num_layers=int(params['num_layers'])).to(device)
        criterion=nn.MSELoss()
        if ma.isnan(params.get('weight_decay'))==False:
            optimizer = optim.Adam(model.parameters(),
                                   lr=params['learning_rate'],
                                   weight_decay=params['weight_decay'])
        else:
            optimizer = optim.Adam(model.parameters(),
                       lr=params['learning_rate'])

        
        train_losses=[]
        val_losses=[]
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out=model(x_train)
            loss=criterion(out.squeeze(),y_train)
            loss.backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                train_pred=model(x_train).squeeze()
                val_pred=model(x_val).squeeze()
                train_loss=criterion(train_pred,y_train).item()
                val_loss=criterion(val_pred,y_val).item()
            train_losses.append(train_loss)
            val_losses.append(val_loss)
        train_fold.append(train_losses)
        val_fold.append(val_losses)
    for i in range(len(train_fold)):
        fig,axs=plt.subplots(figsize=(7*phi,7),dpi=150)
        axs.plot(range(epochs),train_fold[i],lw=3,color='black',label='Train loss')
        axs.plot(range(epochs),val_fold[i],lw=2,color='red',label='Val loss')
        axs.set_ylabel('MSE Loss')
        axs.set_xlabel('epochs')
        axs.set_title('fold= '+str(i+1))
        axs.legend(loc='best',prop={'size':15})
        plt.show(block=False)
        fig.tight_layout()
        fig_name=f"rank{rank}_fold_{i+1}"
        ext='png'
        filename=f"{fig_name}.{ext}"
        full_path=os.path.join('./plots',filename)
        counter=1
        while os.path.exists(full_path):
            filename=f"{fig_name}_{counter}.{ext}"
            full_path=os.path.join('./plots',filename)
            counter+=1
        fig.savefig(full_path)
