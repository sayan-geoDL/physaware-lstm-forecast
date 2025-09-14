#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core utilities for LSTM-based weather station modeling.

Provides:
- batcher: Generates batched data for training or prediction.
- physics_aware_loss: Combines MSE with a physics-based RH consistency term.
- predictor: LSTM network for multi-output (dwpt, temp, rhum, pres).
- bootstrap_ci: Bootstrap confidence intervals (placeholder for EMOS).

Notes:
- Self-contained; does not write files.
- Bootstrap module is temporary until EMOS post-processing is integrated.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################ code for creating batched data for cv and training ###########
def batcher(strt,end,n_step,mode='train',inp='./data/processed/scaled_train.csv'):
    """
    Generates batched input and output arrays for LSTM training or prediction.

    Args:
        strt (str): Start date (YYYY-MM-DD).
        end (str): End date (YYYY-MM-DD).
        n_step (int): Number of time steps in each input sequence.
        mode (str): 'train' for training/validation, 'predict' for inference.
        inp (str): Path to input CSV file.

    Returns:
        - If mode == 'train':
            tuple: (x, y, t_y)
                x (np.ndarray): Input sequences of shape (samples, n_step, features).
                y (np.ndarray): Output targets of shape (samples, output_features).
                t_y (np.ndarray): Corresponding timestamps.
        - If mode == 'predict':
            tuple: (x, t_y)
                x (np.ndarray): Input sequences.
                t_y (np.ndarray): Forecast timestamps.

    Notes:
        - Does not save any files.
    """

    df_inp=pd.read_csv(inp,parse_dates=['time'],index_col='time')
    df_inp=df_inp.copy()
    df_inp=df_inp.sort_index()
    columns_in=['dwpt','temp','rhum','u','v','pres']
    columns_out=['dwpt','temp','rhum','pres']
    df_inp=df_inp.reindex(columns=columns_in)
    if mode=='train':
        x,y,t_y=[],[],[]
        for i in range(n_step,len(df_inp)-1):
            x_i=df_inp[columns_in].iloc[i-n_step:i].values
            y_i=df_inp[columns_out].iloc[i+1]
            t_y.append(df_inp.index[i+1])
            x.append(x_i)
            y.append(y_i)
        return np.array(x),np.array(y),np.array(t_y)
    if mode=='predict':
        x,t_y=[],[]
        for i in range(n_step,len(df_inp)):
            x_i=df_inp[columns_in].iloc[i-n_step:i].values
            forecast_time=df_inp.index[i]+pd.Timedelta(days=1)
            t_y.append(forecast_time)
            x.append(x_i)
        return np.array(x),np.array(t_y)
############ physics informed loss and utilities ##############################
class physics_aware_loss(nn.Module):
    """
    Custom PyTorch loss function combining MSE with a physics-based consistency loss for LSTM outputs.

    Args:
        scaling_param_path (str): Path to scaling parameters CSV.
        lambda_phys (float): Weight for the physics-based loss term.

    Methods:
        forward(y_pred, y_true): Computes total loss as MSE + lambda_phys * physics loss.

    Notes:
        - Uses dewpoint and temperature to compute a physics-based relative humidity.
        - Does not save any files.
    """
    def __init__(self,scaling_param_path,lambda_phys=1.0):
        super().__init__()
        self.stats=self.load_scaling_params(scaling_param_path)
        self.lambda_phys=lambda_phys

    def load_scaling_params(self,path):
        """
        Loads scaling parameters from CSV for unscaling predictions.

        Args:
            path (str): Path to scaling parameters CSV.

        Returns:
            dict: Dictionary of means and stds for each variable.
        """
        df=pd.read_csv(path,index_col=0)
        stats={
            var: {
                'mean': torch.tensor(df.loc[var,'mean'],dtype=torch.float32),
                'std': torch.tensor(df.loc[var,'std'],dtype=torch.float32)
            }
            for var in ['dwpt','temp','pres']
        }
        return stats

    def unscale(self,x_scaled,var):
        """
        Unscales a variable using loaded mean and std.

        Args:
            x_scaled (Tensor): Scaled tensor.
            var (str): Variable name.

        Returns:
            Tensor: Unscaled tensor.
        """
        return x_scaled*self.stats[var]['std']+self.stats[var]['mean']

    def es_water(self,T):
        """
        Computes saturation vapor pressure over water [hPa].

        Args:
            T (Tensor): Temperature in Celsius.

        Returns:
            Tensor: Saturation vapor pressure.
        """
        e=6.1121*torch.exp((17.502*T)/(240.97+T))
        return e

    def compute_rhum(self,temp,dwpt):
        """
        Computes relative humidity from dewpoint and temperature.

        Args:
            temp (Tensor): Temperature in Celsius.
            dwpt (Tensor): Dewpoint in Celsius.

        Returns:
            Tensor: Relative humidity.
        """
        e_t=self.es_water(temp,)# pres)
        e_td=self.es_water(dwpt)#,pres)
        rh=e_td/(e_t+1e-8)  # avoid division by zero
        return rh

    def forward(self,y_pred,y_true):
        """
        Computes the combined MSE and physics-based loss.

        Args:
            y_pred (Tensor): Model predictions.
            y_true (Tensor): Ground truth targets.

        Returns:
            Tensor: Total loss.
        """
        mse=F.mse_loss(y_pred,y_true)
        dwpt=self.unscale(y_pred[:,0],'dwpt')
        temp=self.unscale(y_pred[:,1],'temp')
        # RH from prediction
        rhum_pred=y_pred[:,2]
        rhum_phys=self.compute_rhum(temp,dwpt)
        # Physics-informed consistency loss
        phys_loss=F.mse_loss(rhum_pred,rhum_phys)
        # Total loss
        total_loss=mse+self.lambda_phys*phys_loss
        return total_loss

########### defininig model ###################################################
class predictor(nn.Module):
    """
    LSTM-based neural network for multi-output weather station prediction.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.

    Methods:
        forward(x): Runs the model forward pass.

    Notes:
        - Outputs 4 variables: dwpt, temp, rhum, pres.
        - Does not save any files.
    """
    def __init__(self,input_size,hidden_size,num_layers):
        super(predictor,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc=nn.Linear(hidden_size,4)
    def forward(self,x):
        """
        Forward pass of the LSTM model.

        Args:
            x (Tensor): Input tensor of shape (batch, n_step, input_size).

        Returns:
            Tensor: Output tensor of shape (batch, 4).
        """
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        out,_=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out
############### bootstrap ci ##################################################
def bootstrap_ci(preds: np.ndarray,B: int=1000,
                 low: float=2.5,high: float=97.5,
                 random_state=None):
    """
    Computes bootstrap confidence intervals for ensemble predictions.

    Args:
        preds (np.ndarray): Array of shape (T, M) with T timesteps and M models.
        B (int): Number of bootstrap samples.
        low (float): Lower percentile for confidence interval.
        high (float): Upper percentile for confidence interval.
        random_state (int, optional): Random seed.

    Returns:
        tuple: (lower, upper)
            lower (np.ndarray): Lower confidence bound (T,).
            upper (np.ndarray): Upper confidence bound (T,).

    Notes:
        - Does not save any files.
    """
    rng=np.random.default_rng(random_state)
    T,M=preds.shape
    means=np.zeros((B,T))
    for b in range(B):
        sample_idx=rng.integers(0,M,size=M)
        resample=preds[:,sample_idx]
        means[b,:]=resample.mean(axis=1)
    lower=np.percentile(means,low,axis=0)
    upper=np.percentile(means,high,axis=0)
    return lower,upper
