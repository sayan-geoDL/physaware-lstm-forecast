#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 17:50:09 2025

@author: sayan
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn

################ code for creating batched data for cv and training ###########
def batcher(strt,end,n_step,mode='train',inp='./data/processed/scaled_train.csv'):
    """
    Prepares input-output batches for training or prediction from a time-indexed CSV.

    Parameters
    ----------
    strt : str
        Start date (inclusive) in 'YYYY-MM-DD' format to slice the dataset.
    end : str
        End date (inclusive) in 'YYYY-MM-DD' format to slice the dataset.
    n_step : int
        Number of time steps to use as input for the LSTM model (look-back window).
    mode : str, optional
        Mode of operation: 'train' or 'predict'. Default is 'train'.
        - 'train': Returns (X, y, t_y), where y is the target temperature and t_y is the target time.
        - 'predict': Returns (X, t_y), where t_y is the forecast timestamp (1 day after last input).
    inp : str, optional
        Path to the input CSV file. The CSV must contain a 'time' column and the following variables:
        ['dwpt', 'temp', 'rhum', 'u', 'v', 'pres']. Default is './data/processed/scaled_train.csv'.

    Returns
    -------
    If mode == 'train':
        x : np.ndarray
            Input sequences of shape (samples, n_step, features).
        y : np.ndarray
            Target temperature values of shape (samples,).
        t_y : np.ndarray
            Target timestamps (1 day after the input window) of shape (samples,).
    If mode == 'predict':
        x : np.ndarray
            Input sequences of shape (samples, n_step, features).
        t_y : np.ndarray
            Forecast timestamps (1 day after the last input timestep) of shape (samples,).

    Notes
    -----
    - The function assumes a regular daily time index.
    - In 'predict' mode, the target temperature is not returned as it is not available.
    - This function will raise an error if the input CSV is missing required columns.

    """

    df=pd.read_csv(inp,parse_dates=['time'],index_col='time')
    df=df.copy()
    df=df.sort_index()
    df=df.loc[strt:end]
    columns=['dwpt','temp','rhum','u','v','pres']
    df=df.reindex(columns=columns)
    if mode=='train':
        x,y,t_y=[],[],[]
        for i in range(n_step,len(df)-1):
            x_i=df[columns].iloc[i-n_step:i].values
            y_i=df['temp'].iloc[i+1]
            t_y.append(df.index[i+1])
            x.append(x_i)
            y.append(y_i)
        return np.array(x),np.array(y),np.array(t_y)
    if mode=='predict':
        x,t_y=[],[]
        for i in range(n_step,len(df)):
            x_i=df[columns].iloc[i-n_step:i].values
            forecast_time = df.index[i - 1] + pd.Timedelta(days=1)
            t_y.append(forecast_time)
            x.append(x_i)
        return np.array(x),np.array(t_y)
########### defininig model ###################################################
class predictor(nn.Module):
    """
   LSTM-based neural network model for Temperature forecast at 1 day lag.

   This model uses a stack of LSTM layers followed by a fully connected (linear) layer
   to predict a single output value from a sequence of input features.

   Parameters
   ----------
   input_size : int
       The number of input features at each time step.
   hidden_size : int
       The number of features in the hidden state of the LSTM.
   num_layers : int
       The number of stacked LSTM layers to use.

   Attributes
   ----------
   lstm : nn.LSTM
       LSTM layer(s) for sequential modeling.
   fc : nn.Linear
       Fully connected layer to produce the final output.
   """
    def __init__(self,input_size,hidden_size,num_layers):
        super(predictor,self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size=input_size,
                          hidden_size=hidden_size,
                          num_layers=num_layers,
                          batch_first=True)
        self.fc=nn.Linear(hidden_size, 1)
    def forward(self,x):
        """
        Defines the forward pass of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, 1), representing the predicted value.
        """
        h0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        c0=torch.zeros(self.num_layers,x.size(0),self.hidden_size).to(x.device)
        out,_=self.lstm(x,(h0,c0))
        out=out[:,-1,:]
        out=self.fc(out)
        return out