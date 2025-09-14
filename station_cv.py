#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cross-validation and hyperparameter optimization utilities for LSTM-based weather forecasting models.

Features:
- K-fold cross-validation with early stopping and physics-aware loss.
- Multi-objective hyperparameter optimization using Optuna (RMSE & overfit score).
- Pareto-efficient hyperparameter selection with CSV logging.
- Diagnostic plotting: RMSE vs overfit scatter grids and fold-wise loss curves.
- Automated PDF reporting of cross-validation outcomes.

Notes:
- Results saved under ./out/train_test/ (CSV, PDF).
- Plots saved under ./plots/cv/ (scatters, loss curves).
- Requires Optuna, PyTorch, Matplotlib, and ReportLab for full functionality.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import math as ma
import gc
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import sys
import os
from station_model_util import predictor,physics_aware_loss
from station_extra_utilities import get_cpu_temperature,get_gpu_temperature,is_pareto_efficient
import logging
logger=logging.getLogger(__name__)


######################## cross validator ######################################
def cv_outs(x,y,params,initial_train_size=None,
            val_size=10,k=5,device='cpu',epochs=300,es_patience=15,
            scaling_param_path='./data/processed/scaling_params.csv'):
    """
    Performs k-fold cross-validation for a given set of LSTM hyperparameters.

    Args:
        x (np.ndarray): Input features (time, batch, features).
        y (np.ndarray): Target values.
        params (dict): Hyperparameters for the LSTM model.
        initial_train_size (int, optional): Initial size of the training set.
        val_size (int): Validation set size as a percentage of total data.
        k (int): Number of folds.
        device (str): Device to use ('cpu' or 'cuda').
        epochs (int): Number of training epochs per fold.
        es_patience (int): Early stopping patience.
        scaling_param_path (str): Path to scaling parameters CSV.

    Returns:
        dict: Dictionary containing the input hyperparameters and cross-validation metrics:
            - mean_overfit_score
            - std_overfit_score
            - mean_rmse
    """
    val_size=round((val_size/100)*len(y))
    if initial_train_size is None:
        initial_train_size=(x.shape[0]-k*val_size)//k
        if initial_train_size < 1:
            raise ValueError("Too many folds or large val_size for available data.")

    train_starts=np.linspace(initial_train_size,np.shape(x)[0]-val_size,k).astype(int)
    folds=[(0,int(train_end),int(train_end),int(train_end)+val_size) for train_end in train_starts]

    count=0
    overfit_scores,val_rmses=[],[]

    for fold_id,(train_start,train_end,val_start,val_end) in enumerate(folds):
        # Train/val splits
        x_train=torch.tensor(x[train_start:train_end,:,:],dtype=torch.float32).to(device)
        y_train=torch.tensor(y[train_start:train_end],dtype=torch.float32).to(device)
        x_val=torch.tensor(x[val_start:val_end,:,:],dtype=torch.float32).to(device)
        y_val=torch.tensor(y[val_start:val_end],dtype=torch.float32).to(device)

        # Model
        model=predictor(input_size=x.shape[2],
                          hidden_size=params['hidden_size'],
                          num_layers=params['num_layers']).to(device)

        # Criterion
        if params.get('lambda_physics') is None or params['lambda_physics']==0:
            criterion=nn.MSELoss()
        else:
            criterion=physics_aware_loss(scaling_param_path=scaling_param_path,
                                           lambda_phys=params['lambda_physics'])

        # Optimizer (only model params now!)
        if params.get('weight_decay') is not None:
            optimizer=optim.AdamW(model.parameters(),
                                    lr=params['learning_rate'],
                                    weight_decay=params['weight_decay'])
        else:
            optimizer=optim.AdamW(model.parameters(),
                                    lr=params['learning_rate'])

        # Early stopping bookkeeping
        best_val_loss=float('inf')
        best_model_state=model.state_dict()
        counter=0

        # Training loop
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out=model(x_train)
            loss=criterion(out,y_train)
            loss.backward()
            optimizer.step()

            if es_patience is not None:
                model.eval()
                with torch.no_grad():
                    val_out=model(x_val)
                    val_loss=criterion(val_out,y_val).item()
                if val_loss < best_val_loss-1e-5:
                    best_val_loss=val_loss
                    counter=0
                    best_model_state=model.state_dict()
                else:
                    counter+=1
                if counter>=es_patience:
                    print(f"Early stopping at epoch {epoch+1},val_loss did not improve for {es_patience} epochs")
                    break

        # Restore best model
        if es_patience is not None:
            model.load_state_dict(best_model_state)

        # Eval
        model.eval()
        with torch.no_grad():
            train_pred=model(x_train)
            val_pred=model(x_val)
            train_mse=(torch.mean((train_pred-y_train) ** 2)).item()
            val_mse=(torch.mean((val_pred-y_val) ** 2)).item()
            overfit_score=(np.fabs(val_mse-train_mse))/train_mse if train_mse>1e-6 else float('inf')
            overfit_scores.append(overfit_score)
            val_rmses.append(ma.sqrt(val_mse))

            if device=='cpu':
                temp=get_cpu_temperature()
            elif device=='cuda':
                temp=get_gpu_temperature()
            if temp is not None and temp > 90:
                print(f"Temperature {temp}°C exceeded threshold! Exiting.")
                sys.exit(1)

    # Weighted averages across folds
    fold_weights=[train_end-train_start for (train_start,train_end,_,_) in folds]
    total_weights=sum(fold_weights)
    mean_score=sum(w*s for w,s in zip(fold_weights,overfit_scores))/total_weights
    std_score=np.sqrt(sum(w*(s-mean_score)**2 for w,s in zip(fold_weights,overfit_scores))/total_weights)
    mean_rmse=sum(w*s for w,s in zip(fold_weights,val_rmses))/total_weights

    count+=1
    print(f'grid combo: {params}-completed')

    return {
        **params,
        'mean_overfit_score': mean_score,
        'std_overfit_score': std_score,
        'mean_rmse': mean_rmse,
    }

def hyper_optim(x,y,search_space,initial_train_size=None,
             n_trials=100,njobs=4,val_size=10,k=5,device='cpu',epochs=300,es_patience=15,
             scaling_param_path='./data/processed/scaling_params.csv',force_lambda_zero=False,
             save_path='./out/train_test/cv_result.csv'):
    """
    Runs multi-objective hyperparameter optimization using Optuna for LSTM cross-validation.

    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): Target values.
        search_space (dict): Hyperparameter search space.
        initial_train_size (int, optional): Initial training set size.
        n_trials (int): Number of Optuna trials.
        njobs (int): Number of parallel jobs (CPU only).
        val_size (int): Validation set size as a percentage.
        k (int): Number of folds.
        device (str): Device to use ('cpu' or 'cuda').
        epochs (int): Number of epochs per trial.
        es_patience (int): Early stopping patience.
        scaling_param_path (str): Path to scaling parameters CSV.
        force_lambda_zero (bool): If True, sets lambda_physics to 0 for all trials.
        save_path (str): Path to save the cross-validation results CSV.

    Saves:
        - Pareto-optimal hyperparameters and metrics to {save_path} (default: ./out/train_test/cv_result.csv)
        - If force_lambda_zero is True, overwrites the CSV with only lambda_physics=0 results.
        - If force_lambda_zero is False, appends new results and keeps only Pareto-efficient rows.
    """

    def objective(trial):
        params={}
        for key,val in search_space.items():
            if key=='lambda_physics' and force_lambda_zero:
                continue
            if isinstance(val,dict):
                param_type=val.get("type")
                if param_type=="int":
                    step=val.get('step',1)
                    params[key]=trial.suggest_int(
                        name=key,
                        low=val['low'],
                        high=val['high'],
                        step=step
                    )
                elif param_type=="float":
                    step=val.get('step',None)
                    log=val.get('log',False)
                    if step is not None:
                        params[key]=trial.suggest_float(
                            name=key,
                            low=val['low'],
                            high=val['high'],
                            step=step,
                            log=log
                        )
                    else:
                        params[key]=trial.suggest_float(
                            name=key,
                            low=val['low'],
                            high=val['high'],
                            log=log
                        )
                else:
                    raise ValueError(f"Unsupported param type '{param_type}' for '{key}'")
            elif isinstance(val,list):
                params[key]=trial.suggest_categorical(key,val)
            else:
                raise ValueError(f"Invalid format for param '{key}': {val}")
        if force_lambda_zero:
            params["lambda_physics"]=0
        result=cv_outs(x,y,
                 params=params,
                 initial_train_size=initial_train_size,
                 val_size=val_size,
                 k=k,
                 device=device,
                 epochs=epochs,
                 es_patience=es_patience,
                 scaling_param_path=scaling_param_path)
        torch.cuda.empty_cache()
        gc.collect()
        return result['mean_rmse'],result['mean_overfit_score']
    os.makedirs('./out/train_test',exist_ok=True)
    study=optuna.create_study(directions=["minimize","minimize"],
                                sampler=optuna.samplers.TPESampler())
    if device=='cpu':
        study.optimize(objective,n_trials=n_trials,n_jobs=njobs,gc_after_trial=True)
    elif device=='cuda':
        study.optimize(objective,n_trials=n_trials,gc_after_trial=True)
    if force_lambda_zero and os.path.exists(save_path):
        os.remove(save_path)
    pareto_trials=study.best_trials
    result=[]
    for trial in pareto_trials:
        trial_dict=trial.params.copy()
        trial_dict['mean_rmse']=trial.values[0]
        trial_dict['mean_overfit_score']=trial.values[1]
        result.append(trial_dict)
    df=pd.DataFrame(result)
    os.makedirs(os.path.dirname(save_path),exist_ok=True)
    if force_lambda_zero:
        df['rank_score']=np.sqrt(df['mean_rmse']**2+df['mean_overfit_score']**2)
        df=df.sort_values('rank_score').reset_index(drop=True)
        cols_to_drop=['rank_score']
        df=df.drop(columns=[col for col in cols_to_drop if col in df.columns])
        if 'lambda_physics' not in df.columns:
            idx=df.columns.get_loc('weight_decay')+1
            df.insert(idx,'lambda_physics',0)
        df.insert(0, "rank", range(1, len(df) + 1))
        df.to_csv(save_path,index=False)
    else:
        if os.path.exists(save_path):
            existing=pd.read_csv(save_path)
            combined=pd.concat([existing,df],ignore_index=True).drop_duplicates()
        else:
            combined=df.copy()
        pareto_mask=is_pareto_efficient(combined[['mean_rmse','mean_overfit_score']].values)
        combined=combined[pareto_mask].reset_index(drop=True)
        combined['rank_score']=np.sqrt(combined['mean_rmse']**2+combined['mean_overfit_score']**2)
        combined=combined.sort_values('rank_score').reset_index(drop=True)
        cols_to_drop=['rank_score']
        combined=combined.drop(columns=[col for col in cols_to_drop if col in combined.columns])
        if "rank" in combined.columns:
            combined = combined.drop(columns=["rank"])
        combined.insert(0, "rank", range(1, len(combined) + 1))
        combined.to_csv(save_path,index=False)
        
def plot_rmse_overfit_colored(f_path='./out/train_test/cv_result.csv'):
    """
    Plots RMSE vs Overfit Score in a 3x2 grid, color-coded by each hyperparameter.

    Args:
        f_path (str): Path to the cross-validation results CSV.

    Saves:
        - Scatter plot grid to ./plots/cv/scatters.png
        - Creates ./plots/cv/ directory if it does not exist.
    """

    color_columns=['hidden_size','num_layers','learning_rate','weight_decay','lambda_physics']
    column_name=['hidden size','number of layers','learning rate','weight decay','λ physics']
    rmse_col='mean_rmse'
    overfit_col='mean_overfit_score'

    fig,axes=plt.subplots(3,2,figsize=(15,15))
    axes=axes.flatten()
    df=pd.read_csv(f_path)

    for i,col in enumerate(color_columns):
        if col in ['hidden_size','num_layers']:
            cmap='tab20'
            norm=None
        elif col in ['learning_rate','weight_decay','lambda_physics']:
            cmap='viridis'
            # Avoid zero/negative for LogNorm
            min_val=df[col].min()
            if min_val<=0:
                min_val=df.loc[df[col] > 0,col].min()
            norm=mcolors.LogNorm(vmin=min_val,vmax=df[col].max())
        else:
            cmap='viridis'
            norm=None
        
        sc=axes[i].scatter(
            df[rmse_col],
            df[overfit_col],
            c=df[col],
            cmap=cmap,
            norm=norm,
            s=50,
            alpha=0.8
        )
        axes[i].set_title(f"{column_name[i]}")
        axes[i].set_xlabel("Mean RMSE")
        axes[i].set_ylabel("Mean Overfit Score")
        plt.colorbar(sc,ax=axes[i],label=col)
    for j in range(len(color_columns),len(axes)):
        fig.delaxes(axes[j])
    fig.tight_layout()
    os.makedirs('./plots/cv',exist_ok=True)
    fig.savefig('./plots/cv/scatters.png')

###################### plotter for cross validation ###########################
def cv_plot(x,y,cv_result='./out/train_test/cv_result.csv',initial_train_size=None,
             val_size=10,k=5,device='cpu',epochs=500,rank=1,es_patience=20,
             scaling_param_path='./data/processed/scaling_params.csv'):
    """
    Plots training and validation loss curves for each fold and selected hyperparameter rank(s).

    Args:
        x (np.ndarray): Input features.
        y (np.ndarray): Target values.
        cv_result (str): Path to cross-validation results CSV.
        initial_train_size (int, optional): Initial training set size.
        val_size (int): Validation set size as a percentage.
        k (int): Number of folds.
        device (str): Device to use ('cpu' or 'cuda').
        epochs (int): Number of epochs per fold.
        rank (int, list, or 'all'): Hyperparameter rank(s) to plot.
        es_patience (int): Early stopping patience.
        scaling_param_path (str): Path to scaling parameters CSV.

    Saves:
        - For each rank and fold, saves a PNG loss curve to ./plots/cv/rank{rank}_fold_{fold}.png
        - Creates ./plots/cv/ directory if it does not exist.
    """
    phi=(1+5**(0.5))/2
    if isinstance(rank,int):
        rank=[rank]
    elif isinstance(rank,list):
        pass
    elif rank=='all':
    	rank=list(range(1,len(pd.read_csv(cv_result))+1))

    val_size=round((val_size/100)*len(y))
    if initial_train_size is None:
        initial_train_size=(x.shape[0]-k*val_size)//k
        if initial_train_size < 1:
            raise ValueError("Too many folds or large val_size for available data.")

    train_starts=np.linspace(initial_train_size,x.shape[0]-val_size,k).astype(int)
    folds=[]
    for train_end in train_starts:
        train_end=int(train_end)
        val_start=train_end
        val_end=val_start+val_size
        folds.append((0,train_end,val_start,val_end))

    for j,r in enumerate(rank):
        params=pd.read_csv(cv_result).loc[r-1][['hidden_size',
                                                'learning_rate',
                                                'num_layers',
                                                'weight_decay',
                                                'lambda_physics']]
        print(f"Using hyperparameters (rank {r}): {params.to_dict()}")

        train_fold=[]
        val_fold=[]
        for fold_id,(train_start,train_end,val_start,val_end) in enumerate(folds):
            x_train=torch.tensor(x[train_start:train_end,:,:],dtype=torch.float32).to(device)
            y_train=torch.tensor(y[train_start:train_end],dtype=torch.float32).to(device)
            x_val=torch.tensor(x[val_start:val_end,:,:],dtype=torch.float32).to(device)
            y_val=torch.tensor(y[val_start:val_end],dtype=torch.float32).to(device)

            model=predictor(input_size=x.shape[2],
                            hidden_size=int(params['hidden_size']),
                            num_layers=int(params['num_layers'])).to(device)

            if pd.isna(params['lambda_physics']) or params['lambda_physics']==0:
                criterion=nn.MSELoss()
                model_params=model.parameters()
            else:
                criterion=physics_aware_loss(scaling_param_path=scaling_param_path,
                                             lambda_phys=params['lambda_physics'])
                model_params=model.parameters()

            if params.get('weight_decay') is not None:
                optimizer=optim.AdamW(model_params,
                                       lr=params['learning_rate'],
                                       weight_decay=params['weight_decay'])
            else:
                optimizer=optim.AdamW(model_params,
                                       lr=params['learning_rate'])

            best_val_loss=float('inf')
            best_model_state=model.state_dict()
            counter=0
            train_losses=[]
            val_losses=[]

            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                out=model(x_train)
                loss=criterion(out,y_train)
                loss.backward()
                optimizer.step()

                model.eval()
                with torch.no_grad():
                    train_loss=criterion(model(x_train),y_train).item()
                    val_loss=criterion(model(x_val),y_val).item()
                train_losses.append(train_loss)
                val_losses.append(val_loss)

                if es_patience is not None:
                    if val_loss < best_val_loss-1e-5:
                        best_val_loss=val_loss
                        counter=0
                        best_model_state=model.state_dict()
                    else:
                        counter+=1
                    if counter>=es_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
                torch.cuda.empty_cache()
            train_fold.append(train_losses)
            val_fold.append(val_losses)
        os.makedirs('./plots/cv',exist_ok=True)
        # plot loss curves
        for i in range(len(train_fold)):
            fig,axs=plt.subplots(figsize=(7*phi,7),dpi=150)
            axs.plot(range(1,len(train_fold[i])+1),train_fold[i],lw=3,color='black',label='Train loss')
            axs.plot(range(1,len(val_fold[i])+1),val_fold[i],lw=2,color='red',label='Val loss')
            axs.set_xlim(1,epochs)
            axs.set_ylabel('MSE Loss')
            axs.set_xlabel('epochs')
            axs.set_title('fold= '+str(i+1))
            axs.legend(loc='best',prop={'size':15})
            fig.tight_layout()

            fig_name=f"rank{r}_fold_{i+1}"
            ext='png'
            filename=f"{fig_name}.{ext}"
            full_path=os.path.join('./plots/cv',filename)
            fig.savefig(full_path)
            plt.close()