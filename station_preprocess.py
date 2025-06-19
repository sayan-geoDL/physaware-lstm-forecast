#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 14:52:15 2025

@author: sayan
"""

from datetime import datetime
import pandas as pd
import meteostat
import os
import numpy as np

################## download data if necessary #################################
def download_data(station_id,start_date,end_date,name,sav_dir='./data'):
    """
    Download hourly weather data for a given station and time range using Meteostat API.

    Saves the data to a CSV file in the specified directory with the filename `{name}.csv`.

    Parameters
    ----------
    station_id : str or int
        Meteostat station ID (e.g., '03772'). Accepts either string or integer format.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    name : str
        Name to assign to the saved CSV file (without extension).
    sav_dir : str, optional
        Directory where the data CSV will be saved. Default is './data'.

    Returns
    -------
    pandas.DataFrame
        Hourly weather data for the specified station and time range.

    Raises
    ------
    ValueError
        If station_id is neither a string nor an integer.
    """
    start_date=datetime.strptime(start_date,'%Y-%m-%d')
    end_date=datetime.strptime(end_date,'%Y-%m-%d')
    if isinstance(station_id, int):
        station_id=str(station_id)
    elif isinstance(station_id,str):
        pass
    else:
        raise ValueError("station_id must be an integer or string")
    data=meteostat.Hourly(station_id,start_date,end_date).fetch()
    os.makedirs(sav_dir,exist_ok=True)
    data.to_csv(os.path.join(sav_dir,name+'.csv'))
    return data
############### handling missing values #######################################
def process_to_daily(name_in,name_out,data_path='./data/processed',
                     fill_method='smart',
                     phase='train',
                     gap_threshold=3):
    """
    Process raw hourly weather data into daily-averaged data with missing value handling.

    The function:
    - Fills missing values using climatology, interpolation, or a hybrid 'smart' strategy.
    - Converts wind speed and direction into zonal ('u') and meridional ('v') wind components.
    - Aggregates hourly data to daily means.
    - Saves intermediate and final outputs for reuse.

    Parameters
    ----------
    name_in : str
        Name of the raw input file (without '.csv') located in './data/'.
    name_out : str
        Base name of the output files (without '.csv').
    data_path : str, optional
        Directory where the processed daily file will be saved. Default is './data/processed'.
    fill_method : str, optional
        Method for filling missing values. One of:
        - 'climatology': Fill using long-term hourly climatology.
        - 'interpolate': Use time-based interpolation.
        - 'smart'      : Interpolate short gaps, fill long gaps with climatology.
        Default is 'smart'.
    phase : str, optional
        Either 'train' (compute and save climatology) or 'predict' (use existing climatology).
        Default is 'train'.
    gap_threshold : int, optional
        For 'smart' filling: maximum length (in hours) of a gap to be interpolated.
        Gaps larger than this will be filled using climatology. Default is 3.

    Returns
    -------
    pandas.DataFrame
        Daily-averaged DataFrame after gap-filling and wind component conversion.

    Saves
    -----
    - './data/{name_out}_filled.csv': Hourly data after filling and conversion.
    - '{data_path}/{name_out}.csv'  : Final daily-averaged output file.
    - './data/climatology.csv'      : Climatological averages (if phase == 'train').

    Notes
    -----
    - Input data must contain at least these columns: ['temp', 'dwpt', 'pres', 'rhum', 'wspd', 'wdir'].
    - Wind direction (`wdir`) and speed (`wspd`) are transformed into `u` and `v`, then dropped.
    - All numeric values are rounded to 2 decimal places.
    - Resampling is done with hourly frequency for gap detection and daily for final output.
    """
    infile=os.path.join('./data',name_in+'.csv')
    filled_names=os.path.join('./data',name_out+'_filled.csv')
    df=pd.read_csv(infile,parse_dates=["time"],index_col="time")
    df = df.resample('1h').asfreq()
    df=df.copy()
    df['u'] = -df['wspd'] * np.sin(np.deg2rad(df['wdir']))
    df['v'] = -df['wspd'] * np.cos(np.deg2rad(df['wdir']))
    df.drop(columns=['wspd', 'wdir'], inplace=True)
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    columns = ['temp', 'dwpt', 'pres', 'rhum','u','v']
    if phase=='train':
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['hour'] = df.index.hour
        clim = df.groupby(['month', 'day', 'hour'])[columns].mean().reset_index()
        clim.to_csv('./data/climatology.csv',index=False)
    elif phase=='predict':
        clim = pd.read_csv('./data/climatology.csv')
    if fill_method=='climatology':
        df = df.reset_index().merge(clim, on=['month', 'day', 'hour'], suffixes=('', '_clim'))
        for col in columns:
            df[col] = df[col].fillna(df[f'{col}_clim'])
        df[columns] = df[columns].round(2)
        df.drop(columns=['month', 'day', 'hour'] + [f'{col}_clim' for col in columns], inplace=True)
        df.set_index('time', inplace=True)
        df=df[columns]
        df.to_csv(filled_names)
    elif fill_method=='interpolate':
        df[columns] = df[columns].interpolate(method='time', limit_direction='both')
        df[columns]=df[columns].round(2)
        df=df[columns]
        df.to_csv(filled_names)
    elif fill_method=='smart':
        for col in columns:
            is_nan=df[col].isna()
            nan_groups=(is_nan != is_nan.shift()).cumsum()
            gap_ids=df[is_nan].groupby(nan_groups).apply(lambda group: (group.index[-1]-group.index[0]).total_seconds()/3600 + 1)
            gap_ids=gap_ids[gap_ids>= gap_threshold].index
            df[col]=df[col].interpolate(method='time',limit_direction='both')
            long_gap_mask=is_nan & nan_groups.isin(gap_ids)
            df.loc[long_gap_mask,col]=np.nan
            fill_df=df.loc[long_gap_mask,['month','day','hour']]
            fill_df=fill_df.merge(clim[['month','day','hour',col]],
                                  on=['month','day','hour'],
                                  how='left')
            df.loc[long_gap_mask, col] = fill_df[col].values
        df[columns] = df[columns].round(2)
        df.drop(columns=['month', 'day', 'hour'], inplace=True)
        df=df[columns]
        df.to_csv(filled_names)
    daily_df=df.resample('1D').mean()
    daily_df=daily_df.round(2)
    daily_df=daily_df[columns]
    os .makedirs(data_path,exist_ok=True)
    filename=os.path.join(data_path,name_out+'.csv')
    daily_df.to_csv(filename)
    return daily_df
#download_data('03772', '1973-01-01', '2024-12-31')
#df=process_to_daily(fill_method='smart',gap_threshold=3)