#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
station_preprocess.py

Preprocessing utilities for Meteostat weather station data.

Features:
- Downloads raw hourly weather observations from the Meteostat API.
- Gap-fills missing values using climatology, interpolation, or a hybrid "smart" method.
- Converts wind speed/direction into zonal (u) and meridional (v) components.
- Aggregates hourly records into daily means for modeling.
- Saves intermediate (hourly filled) and final (daily averaged) CSVs for downstream use.

Intended as the first step in the LSTM weather forecasting pipeline.
"""

from datetime import datetime
import pandas as pd
import meteostat
import os
import numpy as np

################## download data if necessary #################################
def download_data(station_id,start_date,end_date,name,sav_dir='./data'):
    """
    Download hourly weather data for a given station using the Meteostat API.

    Parameters
    ----------
    station_id : str or int
        Meteostat station identifier (e.g., '70261'). Accepts integer or string.
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    name : str
        Base name for the CSV file to save the downloaded data.
    sav_dir : str, optional
        Directory to save the CSV file. Default is './data'.

    Returns
    -------
    pandas.DataFrame
        Hourly weather data with columns such as ['temp', 'dwpt', 'pres', 'rhum', 'wspd', 'wdir'].

    Saves
    -----
    {sav_dir}/{name}.csv : CSV
        Hourly weather data stored for future use.

    Raises
    ------
    ValueError
        If `station_id` is not a string or integer.

    Notes
    -----
    - This function automatically converts integer station IDs to strings.
    - Creates the `sav_dir` if it does not exist.
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
def process_to_daily(name_in,name_out,
                     start_date=None,
                     end_date=None,
                     data_path='./data/processed',
                     fill_method='smart',
                     phase='train',
                     gap_threshold=3):
    """
    Convert raw hourly weather data into daily-averaged, analysis-ready datasets.

    This function prepares meteorological data for modeling by:
    - Resampling raw data to an hourly frequency
    - Converting wind speed/direction into zonal (u) and meridional (v) components
    - Handling missing values with user-specified strategies
    - Aggregating data to daily means
    - Optionally computing or applying climatology for gap-filling

    Parameters
    ----------
    name_in : str
        Base name of the input CSV file (located in './data/', without '.csv').
        File must contain at least: ['time', 'temp', 'dwpt', 'pres', 'rhum', 'wspd', 'wdir'].
    name_out : str
        Base name for output CSV files (without '.csv').
    start_date : str or pandas.Timestamp, optional
        Start date for subsetting data. If None, use earliest available date.
    end_date : str or pandas.Timestamp, optional
        End date for subsetting data. If None, use latest available date.
    data_path : str, optional
        Directory where daily processed CSV will be saved. Default: './data/processed'.
    fill_method : {'climatology', 'interpolate', 'smart'}, optional
        Gap-filling strategy:
        - 'climatology' : fill missing values using long-term hourly averages
        - 'interpolate' : time-based interpolation for all gaps
        - 'smart'       : interpolate short gaps; fill long gaps with climatology (default)
    phase : {'train', 'predict'}, optional
        Mode of operation for climatology-based filling:
        - 'train'   : compute climatology from input data and save to './data/climatology.csv'
        - 'predict' : load and apply climatology from './data/climatology.csv'
        Ignored if `fill_method='interpolate'`.
    gap_threshold : int, optional
        Maximum gap length (in hours) to fill with interpolation in 'smart' mode.
        Longer gaps fall back to climatology. Default: 3.

    Returns
    -------
    pandas.DataFrame
        Daily-averaged DataFrame containing columns:
        ['temp', 'dwpt', 'pres', 'rhum', 'u', 'v'].

    Saves
    -----
    - './data/{name_out}_filled.csv' : hourly gap-filled dataset
    - '{data_path}/{name_out}_daily.csv' : daily-averaged dataset
    - './data/climatology.csv' : climatology file (if `phase='train'`)

    Notes
    -----
    - Wind speed (`wspd`) and direction (`wdir`) are converted into u/v components
      and then dropped from the dataset.
    - All numeric values are rounded to 2 decimal places.
    - If climatology contains missing values, a nearest-neighbor interpolation is
      applied as a last resort.
    - Input data is assumed to have a datetime column named 'time'.
    """
    infile=os.path.join('./data',name_in+'.csv')
    filled_names=os.path.join('./data',name_out+'_filled.csv')
    df=pd.read_csv(infile,parse_dates=["time"],index_col="time")
    df = df.resample('1h').asfreq()
    if start_date or end_date:
        df=df.loc[pd.to_datetime(start_date) if start_date else df.index.min():
                  pd.to_datetime(end_date) if end_date else df.index.max()]
    df=df.copy()
    df['u'] = -df['wspd'] * np.sin(np.deg2rad(df['wdir']))
    df['v'] = -df['wspd'] * np.cos(np.deg2rad(df['wdir']))
    df.drop(columns=['wspd', 'wdir'], inplace=True)
    df['month'] = df.index.month
    df['day'] = df.index.day
    df['hour'] = df.index.hour
    columns = ['temp', 'dwpt', 'pres', 'rhum','u','v']
    if fill_method=='climatology' or fill_method=='smart':
        if phase=='train':
            df['month'] = df.index.month
            df['day'] = df.index.day
            df['hour'] = df.index.hour
            clim = df.groupby(['month', 'day', 'hour'])[columns].mean().reset_index()
            clim.to_csv('./data/climatology.csv',index=False)
        elif phase=='predict':
            clim = pd.read_csv('./data/climatology.csv')
        missing_combos=clim[columns].isna().any(axis=1)
        missing_combos=clim.loc[missing_combos, ['month', 'day', 'hour']]
    if fill_method=='climatology':
        df = df.reset_index().merge(clim, on=['month', 'day', 'hour'], suffixes=('', '_clim'))
        for col in columns:
            df[col] = df[col].fillna(df[f'{col}_clim'])
        df[columns] = df[columns].round(2)
        df.drop(columns=['month', 'day', 'hour'] + [f'{col}_clim' for col in columns], inplace=True)
        df.set_index('time', inplace=True)
        df=df[columns]
    elif fill_method=='interpolate':
        df[columns] = df[columns].interpolate(method='time', limit_direction='both')
        df[columns]=df[columns].round(2)
        df=df[columns]
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
    if fill_method=='smart' or fill_method=='climatology':
        if not missing_combos.empty:
            for col in columns:
                if df[col].isna().any():
                    df[col]=df[col].interpolate(method='nearest', limit_direction='both')
    df.to_csv(filled_names)
    daily_df=df.resample('1D').mean()
    daily_df=daily_df.round(2)
    daily_df=daily_df[columns]
    os .makedirs(data_path,exist_ok=True)
    filename=os.path.join(data_path,name_out+'_daily.csv')
    daily_df.to_csv(filename)
    return daily_df
if __name__ == "__main__":
    download_data('70261', '1946-01-01', '2024-12-31',name='fairbanks')
#df=process_to_daily(fill_method='smart',gap_threshold=3)
