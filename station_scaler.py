#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
station_scaler.py

Scales daily meteorological data for LSTM modeling, handling wind components, relative 
humidity, and other variables. Saves both the scaled dataset and normalization parameters 
for consistent training and prediction.

Includes the `scaler` function to:
- Compute and apply standardization during training.
- Reuse saved scaling statistics during prediction.
- Cap wind magnitude and normalize u/v components.
- Convert relative humidity to [0,1] range.
"""

import pandas as pd
import numpy as np

def scaler(strt,end,phase='train',
           inp='./data/processed/station_daily.csv',
           outp='./data/processed/scaled.csv',scales='./data/processed/scaling_params.csv',
           wspd_perc=90):
    """
    Scale daily meteorological data for training or prediction purposes.

    For the 'train' phase:
        - Computes normalization parameters (mean, std) for selected variables.
        - Scales the input data accordingly.
        - Caps wind speed magnitude at a given percentile and rescales u, v components.
        - Saves the computed statistics to a file for future use during inference.

    For the 'predict' (or any non-'train') phase:
        - Loads previously saved scaling statistics.
        - Applies the same scaling transformation to new unseen data.

    Parameters
    ----------
    strt : str
        Start date in 'YYYY-MM-DD' format to slice the input dataset.
    end : str
        End date in 'YYYY-MM-DD' format to slice the input dataset.
    phase : str, optional
        One of 'train' or 'predict'. Determines whether to compute or load scaling stats.
        Default is 'train'.
    inp : str, optional
        File path to the raw daily CSV data.
        Must include 'u', 'v', 'rhum', and other meteorological columns. Default is './data/processed/station_daily.csv'.
    outp : str, optional
        File path to save the scaled data. Default is './data/processed/scaled.csv'.
    scales : str, optional
        File path to save or load the scaling statistics (mean and std values).
        Default is './data/processed/scaling_params.csv'.
    wspd_perc : int, optional
        Percentile used to cap wind speed magnitude for normalization of 'u' and 'v'.
        Default is 90.

    Returns
    -------
    pandas.DataFrame
        The scaled DataFrame, also saved to disk at the path specified by `outp`.

    Notes
    -----
    - Wind components 'u' and 'v' are scaled by capping the wind magnitude at a given percentile,
      then normalizing directionally using the capped magnitude.
    - Relative humidity 'rhum' is divided by 100 to convert to [0, 1] range.
    - All other variables (excluding ['u', 'v', 'rhum']) are standardized using mean and std.
    - During prediction, the same statistics from the training phase are reused to ensure consistency.
    - Output is rounded to 4 decimal places.
    """

    df=pd.read_csv(inp,parse_dates=["time"],index_col="time")
    df.index = pd.to_datetime(df.index)
    df = df.loc[strt:end]
    df=df.copy()
    other_cols = [col for col in df.columns if col not in ['u', 'v','rhum']]
    if phase=='train':
        mag=np.sqrt(df['u']**2+df['v']**2)
        mag_safe=mag.replace(0,np.nan)
        mag_cap=mag.quantile(wspd_perc/100)
        mag_scaled=np.minimum(mag,3*mag_cap)/mag_cap
        df['u']=(df['u']/mag_safe)*mag_scaled
        df['v']=(df['v']/mag_safe)*mag_scaled
        df[['u','v']]=df[['u','v']].fillna(0)
        df['rhum']=df['rhum']/100
        mean_other=df[other_cols].mean()
        std_other=df[other_cols].std()
        df[other_cols]=(df[other_cols]-mean_other)/std_other
        df=df.round(4)
        stats_df=pd.DataFrame({
            'mean': pd.concat([pd.Series({'mag': mag_cap}), mean_other]),
            'std': pd.concat([pd.Series({'mag': np.nan}), std_other])})
        stats_df.to_csv(scales)
    elif phase!='train':
        df_scl=pd.read_csv(scales,index_col=0)
        other_cols = [col for col in df.columns if col not in ['u', 'v','rhum']]
        df=df.copy()
        mag=np.sqrt(df['u']**2+df['v']**2)
        mag_safe=mag.replace(0,np.nan)
        mag_cap=df_scl.loc['mag','mean']
        mag_scaled=np.minimum(mag,3*mag_cap)/mag_cap
        df['u']=(df['u']/mag_safe)*mag_scaled
        df['v']=(df['v']/mag_safe)*mag_scaled
        df[['u','v']]=df[['u','v']].fillna(0)
        df['rhum']=df['rhum']/100
        df[other_cols]=(df[other_cols]-df_scl.loc[other_cols,'mean'])/df_scl.loc[other_cols,'std']
        df=df.round(4)
    df.to_csv(outp,index=True)