#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training and evaluation script for LSTM-based weather forecasting models.

Capabilities:
- Trains LSTM predictors with physics-aware loss and early stopping.
- Supports walk-forward CV and multiple ensemble training runs.
- Evaluates models with RMSE and physics-consistency metrics.
- Generates ensemble forecasts with bootstrap confidence intervals
  (placeholder for future EMOS-based calibration).
- Rescales outputs to physical units for interpretability.
- Produces diagnostic plots and saves metrics/results to disk.

Notes:
- Models and metrics saved under ./out/ directories.
- Plots saved under ./plots/ directories.
- Designed for integration with station_model_util and station_cv.
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from station_model_util import predictor,physics_aware_loss,bootstrap_ci
import logging
import math as ma
logger=logging.getLogger(__name__)


################### traininig #################################################
def trainer(x,y,params,epochs=400,device='cpu',hold_out=10,
            mode='single',scaling_param_path='./data/processed/scaling_params.csv',
            es_patience=20):
    """
    Trains an LSTM-based neural network (single or ensemble) for weather station prediction.

    Args:
        x (np.ndarray): Input features (time, batch, features).
        y (np.ndarray): Target values.
        params (int, dict, list, or 'all'): LSTM hyperparameters or rank(s).
        epochs (int): Number of training epochs.
        device (str): Device to use ('cpu' or 'cuda').
        hold_out (float): Percentage of data for validation.
        mode (str): 'single' or 'ensemble'.
        scaling_param_path (str): Path to scaling parameters CSV.
        es_patience (int): Early stopping patience.

    Saves:
        - Model weights to:
            - ./out/models/final_model.pth (single mode)
            - ./out/models/rank{rank}_model.pth (ensemble mode)
        - Training/validation loss plot to:
            - ./plots/train/train_val_loss.png (single mode)
            - ./plots/train/rank{rank}_train_val_loss.png (ensemble mode)
    """

    phi=(1+5**(0.5))/2

    # --- param parsing ---
    if isinstance(params,int):
        df=pd.read_csv('./out/train_test/cv_result.csv')
        rank=[params]
        row=df.loc[params-1]
        params=row[['hidden_size','learning_rate','num_layers','weight_decay','lambda_physics']].to_dict()
        params=[params]
    elif isinstance(params,dict) and mode=='single':
        keys=['hidden_size','learning_rate','num_layers','weight_decay','lambda_physics']
        params={k:params[k] for k in keys if k in params}
        params=[params]
        rank=[1]  # dummy rank
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


    # --- train/val split ---
    train_end=ma.floor(x.shape[0]*(1-(hold_out/100)))
    if train_end < 1:
        raise ValueError("Hold-out split too large for dataset size")
    logger.info(f"Training model with parameters: {params}") 

    x_train=torch.tensor(x[0:train_end,:,:],dtype=torch.float32).to(device)
    y_train=torch.tensor(y[0:train_end,:],dtype=torch.float32).to(device)
    x_val=torch.tensor(x[train_end:,:,:],dtype=torch.float32).to(device)
    y_val=torch.tensor(y[train_end:,:],dtype=torch.float32).to(device)

    for i,param in enumerate(params):
        # --- model ---
        model=predictor(input_size=x.shape[2],
                        hidden_size=int(param['hidden_size']),
                        num_layers=int(param['num_layers'])).to(device)

        # --- loss ---
        if param['lambda_physics'] in [None,0]:
            criterion=nn.MSELoss()
        else:
            criterion=physics_aware_loss(scaling_param_path=scaling_param_path,
                                         lambda_phys=param['lambda_physics'])

        # --- optimizer ---
        if param.get('weight_decay') is not None:
            optimizer=optim.AdamW(model.parameters(),
                                   lr=param['learning_rate'],
                                   weight_decay=param['weight_decay'])
        else:
            optimizer=optim.AdamW(model.parameters(),
                                   lr=param['learning_rate'])

        # --- training loop ---
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

            # --- early stopping ---
            if es_patience is not None:
                if val_loss < best_val_loss - 1e-5:
                    best_val_loss=val_loss
                    counter=0
                    best_model_state=model.state_dict()
                else:
                    counter+=1
                if counter >= es_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        # --- final eval ---
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            train_pred=model(x_train)
            val_pred=model(x_val)
            train_mse=(torch.mean((train_pred - y_train)**2)).item()
            val_mse=(torch.mean((val_pred - y_val)**2)).item()
            overfit_score= (np.fabs(val_mse-train_mse))/train_mse if train_mse>1e-6 else float('inf')
            val_rmse=ma.sqrt(val_mse)

        # --- plot ---
        fig,axs=plt.subplots(figsize=(7*phi,7),dpi=150)
        axs.plot(range(1,len(train_losses)+1),train_losses,lw=3,color='black',label='Train loss')
        axs.plot(range(1,len(val_losses)+1),val_losses,lw=3,color='red',label='hold out loss')
        axs.set_xlabel('Epoch',fontsize=18)
        axs.set_ylabel('MSE Loss',fontsize=18)
        axs.set_title(f'Hold out RMSE: {round(val_rmse,3)},Overfit Score: {round(overfit_score,3)}',
                  fontsize=20)
        axs.tick_params(axis='both',which='both',length=5,width=3,labelsize=20)
        axs.tick_params(axis='x',rotation=45)
        axs.legend(loc='best',prop={'size':20})
        fig.tight_layout()

        # --- save plot ---
        if mode=='single':
            fig_name="train_val_loss"
        elif mode=='ensemble':
            fig_name=f'rank{rank[i]}_train_val_loss'
        os.makedirs('./plots/train',exist_ok=True)
        fig.savefig(f'./plots/train/{fig_name}.png')
        plt.close(fig)

        # --- save model ---
        os.makedirs('./out/models',exist_ok=True)
        if mode=='single':
            torch.save(model.state_dict(),"./out/models/final_model.pth")
        elif mode=='ensemble':
            torch.save(model.state_dict(),f"./out/models/rank{rank[i]}_model.pth")

############### testing #######################################################
def tester(x_tr,y_tr,t_tr,
           x_tst,y_tst,t_tst,
           params,scales='./data/processed/scaling_params.csv',
           device='cpu',
           mode='single',
           B=1000,low=2.5,high=97.5):
    """
    Evaluates trained LSTM model(s) on test data and computes metrics.

    Args:
        x_tr, y_tr, t_tr: Training data and timestamps.
        x_tst, y_tst, t_tst: Test data and timestamps.
        params: LSTM hyperparameters or rank(s).
        scales (str): Path to scaling parameters CSV.
        device (str): Device to use.
        mode (str): 'single' or 'ensemble'.
        B (int): Bootstrapping samples for confidence intervals (ensemble).
        low, high (float): Percentiles for confidence intervals (ensemble).

    Saves:
        - For single mode:
            - Metrics to ./out/train_test/metrics.csv
            - Training predictions to ./out/train_test/train_ts.csv
            - Test predictions to ./out/train_test/test_ts.csv
        - For ensemble mode:
            - Metrics for each model to ./out/train_test/ensemble_metrics.csv
            - Training predictions for each model to ./out/train_test/rank{rank}_train_ts.csv
            - Test predictions for each model to ./out/train_test/rank{rank}_test_ts.csv
            - Ensemble mean and confidence intervals to:
                - ./out/train_test/ensemble_train_ts.csv
                - ./out/train_test/ensemble_test_ts.csv
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
    logger.info(f"Evaluating model with parameters: {params}")
    var_name=['dwpt','temp','rhum','pres']
    scaling_df=pd.read_csv(scales,index_col=0)
    temp_mean=scaling_df.loc['temp','mean']
    temp_std=scaling_df.loc['temp','std']
    dwpt_mean=scaling_df.loc['dwpt','mean']
    dwpt_std=scaling_df.loc['dwpt','std']
    pres_mean=scaling_df.loc['pres','mean']
    pres_std=scaling_df.loc['pres','std']
    train_y_obs=y_tr.copy()
    test_y_obs=y_tst.copy()
    # dwpt
    train_y_obs[:,0]=(train_y_obs[:,0]*dwpt_std)+dwpt_mean
    test_y_obs[:,0]=(test_y_obs[:,0]*dwpt_std)+dwpt_mean
    # temp
    train_y_obs[:,1]=(train_y_obs[:,1]*temp_std)+temp_mean
    test_y_obs[:,1]=(test_y_obs[:,1]*temp_std)+temp_mean
    # rhum
    train_y_obs[:,2]=train_y_obs[:,2]*100
    test_y_obs[:,2]=test_y_obs[:,2]*100
    # pres
    train_y_obs[:,3]=(train_y_obs[:,3]*pres_std)+pres_mean
    test_y_obs[:,3]=(test_y_obs[:,3]*pres_std)+pres_mean
    if mode=='ensemble':
        all_metrics=[]
    for i,param in enumerate(params):
        if mode=='single':
            state_dict='./out/models/final_model.pth'
        elif mode=='ensemble':
            state_dict=os.path.join("./out/models",f"rank{rank[i]}_model.pth")
        model=predictor(input_size=x_tr.shape[2],
                        hidden_size=int(param['hidden_size']),
                        num_layers=int(param['num_layers'])).to(device)
        model.load_state_dict(torch.load(state_dict,map_location=device))
        x_train=torch.tensor(x_tr,dtype=torch.float32).to(device)
        x_test=torch.tensor(x_tst,dtype=torch.float32).to(device)
        model.eval()
        train_y_pr=model(x_train).detach().cpu().numpy()
        test_y_pr=model(x_test).detach().cpu().numpy()
        # dwpt
        train_y_pr[:,0]=(train_y_pr[:,0]*dwpt_std)+dwpt_mean
        test_y_pr[:,0]=(test_y_pr[:,0]*dwpt_std)+dwpt_mean
        #temperature
        train_y_pr[:,1]=(train_y_pr[:,1]*temp_std)+temp_mean
        test_y_pr[:,1]=(test_y_pr[:,1]*temp_std)+temp_mean
        # rhum
        train_y_pr[:,2]=train_y_pr[:,2]*100
        test_y_pr[:,2]=test_y_pr[:,2]*100
        # pres
        train_y_pr[:,3]=(train_y_pr[:,3]*pres_std)+pres_mean
        test_y_pr[:,3]=(test_y_pr[:,3]*pres_std)+pres_mean
        df_tr=pd.DataFrame({"time": t_tr.ravel()})
        df_tst=pd.DataFrame({"time": t_tst.ravel()})
        for j,var in enumerate(var_name):
            df_tr[f"{var}_observed"]=train_y_obs[:,j]
            df_tr[f"{var}_predicted"]=train_y_pr[:,j]
            df_tst[f"{var}_observed"]=test_y_obs[:,j]
            df_tst[f"{var}_predicted"]=test_y_pr[:,j]
        df_tr=df_tr.round(2)
        df_tst=df_tst.round(2)
        if mode=='single':
            metrics=[]
        elif mode=='ensemble':
            metrics={"rank":rank[i]}
        for var in var_name:
            rmse_tr=np.sqrt(((df_tr[f"{var}_observed"] - df_tr[f"{var}_predicted"])**2).mean())
            rmse_tst=np.sqrt(((df_tst[f"{var}_observed"] - df_tst[f"{var}_predicted"])**2).mean())
            r2_tr=1 - (((df_tr[f"{var}_predicted"] - df_tr[f"{var}_observed"])**2).sum()/
                 ((df_tr[f"{var}_observed"] - df_tr[f"{var}_observed"].mean())**2).sum())
            r2_tst=1 - (((df_tst[f"{var}_predicted"] - df_tst[f"{var}_observed"])**2).sum()/
                 ((df_tst[f"{var}_observed"] - df_tst[f"{var}_observed"].mean())**2).sum())
            if mode=='single':
                metrics.append({
                    "variable": var,
                    "rmse_train": rmse_tr,
                    "rmse_test": rmse_tst,
                    "r2_train": r2_tr,
                    "r2_test": r2_tst})
            elif mode=='ensemble':
                metrics[f"{var}_train_rmse"]=rmse_tr
                metrics[f"{var}_test_rmse"]=rmse_tst
                metrics[f"{var}_train_r2"]=r2_tr
                metrics[f"{var}_test_r2"]=r2_tst
        if mode=='single':
            metric_df=pd.DataFrame(metrics)
            metric_df=metric_df.round(2)
            metric_df.to_csv('./out/train_test/metrics.csv',index=False)
            df_tr.to_csv('./out/train_test/train_ts.csv',index=False)
            df_tst.to_csv('./out/train_test/test_ts.csv',index=False)
        elif mode=='ensemble':
            all_metrics.append(metrics)
            df_tr.to_csv(f'./out/train_test/rank{rank[i]}_train_ts.csv')
            df_tst.to_csv(f'./out/train_test/rank{rank[i]}_test_ts.csv')
    if mode=='ensemble':
        metric_df=pd.DataFrame(all_metrics)
        train_dfs=[]
        test_dfs=[]
        for r in rank:
            tr_path=f'./out/train_test/rank{r}_train_ts.csv'
            tst_path=f'./out/train_test/rank{r}_test_ts.csv'
            train_dfs.append(pd.read_csv(tr_path,parse_dates=['time']))
            test_dfs.append(pd.read_csv(tst_path,parse_dates=['time']))
        df_tr_ens=train_dfs[0][["time"]].copy()
        df_tst_ens=test_dfs[0][["time"]].copy()
        for var in var_name:
            tr_preds=np.column_stack([df[f"{var}_predicted"].values for df in train_dfs])
            tst_preds=np.column_stack([df[f"{var}_predicted"].values for df in test_dfs])
            df_tr_ens[f"{var}_observed"]=train_dfs[0][f"{var}_observed"]
            df_tst_ens[f"{var}_observed"]=test_dfs[0][f"{var}_observed"]
            df_tr_ens[f"{var}_predicted"]=tr_preds.mean(axis=1)
            df_tst_ens[f"{var}_predicted"]=tst_preds.mean(axis=1)
            df_tr_ens[f"{var}_low"],df_tr_ens[f"{var}_high"]=bootstrap_ci(preds=tr_preds,
                                                                         B=B,low=low,
                                                                         high=high)
            df_tst_ens[f"{var}_low"],df_tst_ens[f"{var}_high"]=bootstrap_ci(preds=tst_preds,
                                                                         B=B,low=low,
                                                                         high=high)
        df_tr_ens=df_tr_ens.round(2)
        df_tst_ens=df_tst_ens.round(2)
        df_tr_ens.to_csv('./out/train_test/ensemble_train_ts.csv',index=False)
        df_tst_ens.to_csv('./out/train_test/ensemble_test_ts.csv',index=False)
        ens_metrics={"rank": "ensemble"}
        for var in var_name:
            rmse_tr=np.sqrt(((df_tr_ens[f"{var}_observed"] - df_tr_ens[f"{var}_predicted"])**2).mean())
            rmse_tst=np.sqrt(((df_tst_ens[f"{var}_observed"] - df_tst_ens[f"{var}_predicted"])**2).mean())
            r2_tr=1 - (((df_tr_ens[f"{var}_predicted"] - df_tr_ens[f"{var}_observed"])**2).sum()/
                         ((df_tr_ens[f"{var}_observed"] - df_tr_ens[f"{var}_observed"].mean())**2).sum())
            r2_tst=1 - (((df_tst_ens[f"{var}_predicted"] - df_tst_ens[f"{var}_observed"])**2).sum()/
                          ((df_tst_ens[f"{var}_observed"] - df_tst_ens[f"{var}_observed"].mean())**2).sum())

            ens_metrics[f"{var}_train_rmse"]=rmse_tr
            ens_metrics[f"{var}_test_rmse"]=rmse_tst
            ens_metrics[f"{var}_train_r2"]=r2_tr
            ens_metrics[f"{var}_test_r2"]=r2_tst
        ens_df=pd.DataFrame([ens_metrics]).set_index("rank")
        metric_df=pd.concat([metric_df.set_index('rank'),ens_df])
        metric_df=metric_df.round(2)
        metric_df.to_csv('./out/train_test/ensemble_metrics.csv')
def train_plots(inp='./out/train_test',mode='single',low=2.5,high=97.5):
    """
    Generates and saves diagnostic plots for LSTM model predictions.

    Args:
        inp (str): Input directory for CSVs.
        mode (str): 'single' or 'ensemble'.
        low, high (float): Percentiles for confidence intervals (ensemble).

    Saves:
        - For single mode:
            - Distribution histograms to ./plots/train/distributions.png
            - Time series plots to ./plots/train/time_series.png
        - For ensemble mode:
            - Distribution histograms for each model to ./plots/train/rank{rank}_distributions.png
            - Time series plots for each model to ./plots/train/rank{rank}_time_series.png
            - Ensemble distribution histograms to ./plots/train/ensemble_distributions.png
            - For each variable (dwpt, temp, rhum, pres):
                - Ensemble train time series with confidence intervals to ./plots/train/ensemble_{var}_train_ts.png
                - Ensemble test time series with confidence intervals to ./plots/train/ensemble_{var}_test_ts.png
    """
    phi=(1+5**(0.5))/2
    var=['dwpt','temp','rhum','pres']
    unit=['°C','°C','%','hPa']
    if mode=='single':
        metrics=pd.read_csv('./out/train_test/metrics.csv',index_col='variable')
        if os.path.exists(os.path.join(inp,'train_ts.csv')):
            if os.path.exists(os.path.join(inp,'test_ts.csv')):
                df_tr=pd.read_csv(os.path.join(inp,'train_ts.csv'),parse_dates=['time'])
                df_tst=pd.read_csv(os.path.join(inp,'test_ts.csv'),parse_dates=['time'])
                fig_ts,axs_ts=plt.subplots(len(var),2,dpi=300,figsize=(8,8*phi))
                fig_dist,axs_dist=plt.subplots(len(var),2,dpi=300,figsize=(8,8*phi))
                for j,v in enumerate(var):
                    row=metrics.loc[v]
                    # histogram plots
                    counts,bins=np.histogram(df_tr[f"{v}_observed"],bins="auto")
                    axs_dist[j,0].hist(df_tr[f"{v}_observed"],bins=bins,alpha=0.8,color="gray",label="Observed")
                    axs_dist[j,0].hist(df_tr[f"{v}_predicted"],bins=bins,alpha=0.5,color="red",label="Predicted")
                    axs_dist[j,0].set_xlabel(f"{v} ({unit[j]})",fontsize=10)
                    axs_dist[j,0].set_ylabel("Frequency",fontsize=10)
                    title=f"{v} Train\nRMSE={row['rmse_train']:.2f}{unit[j]},R²={row['r2_train']:.2f}"
                    axs_dist[j,0].set_title(title,fontsize=14)
                    counts,bins=np.histogram(df_tst[f"{v}_observed"],bins="auto")
                    axs_dist[j,1].hist(df_tst[f"{v}_observed"],bins=bins,alpha=0.8,color="gray",label="Observed")
                    axs_dist[j,1].hist(df_tst[f"{v}_predicted"],bins=bins,alpha=0.5,color="red",label="Predicted")
                    axs_dist[j,1].set_xlabel(f"{v} ({unit[j]})",fontsize=10)
                    title=f"{v} Test\nRMSE={row['rmse_test']:.2f}{unit[j]},R²={row['r2_test']:.2f}"
                    axs_dist[j,1].set_title(title,fontsize=14)
                    axs_dist[j,0].tick_params(axis='both',which='both',length=5,width=3,labelsize=10)
                    axs_dist[j,1].tick_params(axis='both',which='both',length=5,width=3,labelsize=10)
                    #ts plots
                    axs_ts[j,0].plot(df_tr['time'],df_tr[f'{v}_observed'],linewidth=2,color='cornflowerblue',label='Observed Training')
                    axs_ts[j,0].plot(df_tr['time'],df_tr[f'{v}_predicted'],linewidth=1,color='red',linestyle='--',label='Predicted',)
                    axs_ts[j,0].set_ylabel(f"{v} ({unit[j]})",fontsize=10)
                    title=f"{v} Train\nRMSE={row['rmse_train']:.2f}{unit[j]},R²={row['r2_train']:.2f}"
                    axs_ts[j,0].set_title(title,fontsize=14)
                    axs_ts[j,1].plot(df_tst['time'],df_tst[f'{v}_observed'],linewidth=2,color='forestgreen',label='Observed Testing')
                    axs_ts[j,1].plot(df_tst['time'],df_tst[f'{v}_predicted'],linewidth=1,color='red',linestyle='--')
                    title=f"{v} Test\nRMSE={row['rmse_test']:.2f}{unit[j]},R²={row['r2_test']:.2f}"
                    axs_ts[j,1].set_title(title,fontsize=14)
                    axs_ts[j,0].tick_params(axis='both',which='both',length=5,width=3,labelsize=10,rotation=45)
                    axs_ts[j,0].xaxis.set_major_locator(mdates.AutoDateLocator())
                    axs_ts[j,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    axs_ts[j,1].tick_params(axis='both',which='both',length=5,width=3,labelsize=10,rotation=45)
                    axs_ts[j,1].xaxis.set_major_locator(mdates.AutoDateLocator())
                    axs_ts[j,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    
                handles,labels=axs_dist[0,0].get_legend_handles_labels()
                fig_dist.legend(handles,labels,loc="upper center",ncol=2,fontsize=12)
                fig_dist.subplots_adjust(top=0.88)
                fig_dist.tight_layout(rect=[0,0,1,0.95])  
                fig_dist.savefig('./plots/train/distributions.png')
                handles_train,labels_train=axs_ts[0,0].get_legend_handles_labels()
                handles_test,labels_test=axs_ts[0,1].get_legend_handles_labels()
                handles=handles_train+handles_test
                labels=labels_train+labels_test
                unique=dict(zip(labels,handles))
                fig_ts.legend(unique.values(),unique.keys(),loc='upper center',ncol=3,fontsize=12)
                fig_ts.subplots_adjust(top=0.88)
                fig_ts.tight_layout(rect=[0,0,1,0.95])  
                fig_ts.savefig('./plots/train/time_series.png')
                plt.close(fig_dist)
                plt.close(fig_ts)
    elif mode=='ensemble':
        files=[os.path.join('./out/train_test',f) for f in os.listdir('./out/train_test') if f.startswith('rank')]
        metrics=pd.read_csv('./out/train_test/ensemble_metrics.csv')
        files=sorted(files)
        for i in range(0,len(files),2):
            name=os.path.basename(files[i+1]).split("_",1)[0]
            logger.info(f'Plotting {name} histogram and time series')
            rnk=int(name.replace("rank",""))
            df_tr=pd.read_csv(files[i+1],parse_dates=['time'])
            df_tst=pd.read_csv(files[i],parse_dates=['time'])
            row=metrics.loc[metrics['rank']==str(rnk)].squeeze()
            fig_ts,axs_ts=plt.subplots(len(var),2,dpi=300,figsize=(8,8*phi))
            fig_dist,axs_dist=plt.subplots(len(var),2,dpi=300,figsize=(8,8*phi))
            
            for j,v in enumerate(var):
                # histogram plots
                counts,bins=np.histogram(df_tr[f"{v}_observed"],bins="auto")
                axs_dist[j,0].hist(df_tr[f"{v}_observed"],bins=bins,alpha=0.8,color="gray",label="Observed")
                axs_dist[j,0].hist(df_tr[f"{v}_predicted"],bins=bins,alpha=0.5,color="red",label="Predicted")
                axs_dist[j,0].set_xlabel(f"{v} ({unit[j]})",fontsize=10)
                axs_dist[j,0].set_ylabel("Frequency",fontsize=10)
                title=f"{v} Train\nRMSE={row.loc[f'{v}_train_rmse']}{unit[j]},R²={row.loc[f'{v}_train_r2']}"
                axs_dist[j,0].set_title(title,fontsize=14)
                counts,bins=np.histogram(df_tst[f"{v}_observed"],bins="auto")
                axs_dist[j,1].hist(df_tst[f"{v}_observed"],bins=bins,alpha=0.8,color="gray",label="Observed")
                axs_dist[j,1].hist(df_tst[f"{v}_predicted"],bins=bins,alpha=0.5,color="red",label="Predicted")
                axs_dist[j,1].set_xlabel(f"{v} ({unit[j]})",fontsize=10)
                title=title=f"{v} Test\nRMSE={row.loc[f'{v}_test_rmse']}{unit[j]},R²={row.loc[f'{v}_test_r2']}"
                axs_dist[j,1].set_title(title,fontsize=14)
                axs_dist[j,0].tick_params(axis='both',which='both',length=5,width=3,labelsize=10)
                axs_dist[j,1].tick_params(axis='both',which='both',length=5,width=3,labelsize=10)
                #ts plots
                axs_ts[j,0].plot(df_tr['time'],df_tr[f'{v}_observed'],linewidth=1,color='cornflowerblue',label='Observed Training')
                axs_ts[j,0].plot(df_tr['time'],df_tr[f'{v}_predicted'],linewidth=0.5,color='red',linestyle='--',label='Predicted',)
                axs_ts[j,0].set_ylabel(f"{v} ({unit[j]})",fontsize=10)
                title=title=f"{v} Train\nRMSE={row.loc[f'{v}_train_rmse']}{unit[j]},R²={row.loc[f'{v}_train_r2']}"
                axs_ts[j,0].set_title(title,fontsize=14)
                axs_ts[j,1].plot(df_tst['time'],df_tst[f'{v}_observed'],linewidth=1,color='forestgreen',label='Observed Testing')
                axs_ts[j,1].plot(df_tst['time'],df_tst[f'{v}_predicted'],linewidth=0.5,color='red',linestyle='--')
                title=title=f"{v} Test\nRMSE={row.loc[f'{v}_test_rmse']}{unit[j]},R²={row.loc[f'{v}_test_r2']}"
                axs_ts[j,1].set_title(title,fontsize=14)
                axs_ts[j,0].tick_params(axis='both',which='both',length=5,width=3,labelsize=10,rotation=45)
                axs_ts[j,0].xaxis.set_major_locator(mdates.AutoDateLocator())
                axs_ts[j,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                axs_ts[j,1].tick_params(axis='both',which='both',length=5,width=3,labelsize=10,rotation=45)
                axs_ts[j,1].xaxis.set_major_locator(mdates.AutoDateLocator())
                axs_ts[j,1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            handles,labels=axs_dist[0,0].get_legend_handles_labels()
            fig_dist.legend(handles,labels,loc="upper center",ncol=2,fontsize=12)
            fig_dist.subplots_adjust(top=0.88)
            fig_dist.tight_layout(rect=[0,0,1,0.95])
            fig_dist.savefig(f'./plots/train/{name}_distributions.png')
            handles_train,labels_train=axs_ts[0,0].get_legend_handles_labels()
            handles_test,labels_test=axs_ts[0,1].get_legend_handles_labels()
            handles=handles_train+handles_test
            labels=labels_train+labels_test
            unique=dict(zip(labels,handles))
            fig_ts.legend(unique.values(),unique.keys(),loc='upper center',ncol=3,fontsize=12)
            fig_ts.subplots_adjust(top=0.88)
            fig_ts.tight_layout(rect=[0,0,1,0.95]) 
            fig_ts.savefig(f'./plots/train/{name}_time_series.png')
            plt.close(fig_ts)
            plt.close(fig_dist)
        ens_tr=pd.read_csv('./out/train_test/ensemble_train_ts.csv',parse_dates=['time'])
        ens_tst=pd.read_csv('./out/train_test/ensemble_test_ts.csv',parse_dates=['time'])
        fig_dist,axs_dist=plt.subplots(len(var),2,dpi=300,figsize=(8,8*phi))
        row=metrics.loc[metrics['rank']=='ensemble'].squeeze()
        plt.close()
        plt.close()
        spread=float(high-low)
        for j,v in enumerate(var):
            # histogram plots
            counts,bins=np.histogram(ens_tr[f"{v}_observed"],bins="auto")
            axs_dist[j,0].hist(ens_tr[f"{v}_observed"],bins=bins,alpha=0.8,color="gray",label="Observed")
            axs_dist[j,0].hist(ens_tr[f"{v}_predicted"],bins=bins,alpha=0.5,color="red",label="Predicted")
            axs_dist[j,0].set_xlabel(f"{v} ({unit[j]})",fontsize=10)
            axs_dist[j,0].set_ylabel("Frequency",fontsize=10)
            title=f"{v} Train\nRMSE={row.loc[f'{v}_train_rmse']}{unit[j]},R²={row.loc[f'{v}_train_r2']}"
            axs_dist[j,0].set_title(title,fontsize=14)
            counts,bins=np.histogram(ens_tst[f"{v}_observed"],bins="auto")
            axs_dist[j,1].hist(ens_tst[f"{v}_observed"],bins=bins,alpha=0.8,color="gray",label="Observed")
            axs_dist[j,1].hist(ens_tst[f"{v}_predicted"],bins=bins,alpha=0.5,color="red",label="Predicted")
            axs_dist[j,1].set_xlabel(f"{v} ({unit[j]})",fontsize=10)
            title=f"{v} Test\nRMSE={row.loc[f'{v}_test_rmse']}{unit[j]},R²={row.loc[f'{v}_test_r2']}"
            axs_dist[j,1].set_title(title,fontsize=14)
            axs_dist[j,0].tick_params(axis='both',which='both',length=5,width=3,labelsize=10)
            axs_dist[j,1].tick_params(axis='both',which='both',length=5,width=3,labelsize=10)
            fig_tr,ax_tr=plt.subplots(dpi=300,figsize=(8*phi,8))
            ax_tr.plot(ens_tr['time'],ens_tr[f'{v}_observed'],color='cornflowerblue',linewidth=1,label='Observed')
            ax_tr.plot(ens_tr['time'],ens_tr[f'{v}_predicted'],color='red',linestyle='--',linewidth=0.5,label='Predicted')
            ax_tr.fill_between(ens_tr['time'],ens_tr[f'{v}_low'],ens_tr[f'{v}_high'],color='orange',alpha=0.5,label=f'{str(spread)}% confidence',zorder=3)
            ax_tr.set_title(f"{v} Train\nRMSE={row.loc[f'{v}_train_rmse']}{unit[j]},R²={row.loc[f'{v}_train_r2']}",fontsize=14)
            ax_tr.set_xlabel("Time")
            ax_tr.set_ylabel(f"{v} ({unit[j]})")
            ax_tr.tick_params(axis='x',rotation=45)
            ax_tr.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_tr.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_tr.legend(loc='best',prop={'size':10})
            fig_tr.tight_layout()
            fig_tr.savefig(f'./plots/train/ensemble_{v}_train_ts.png')
            plt.close(fig_tr)
            fig_tst,ax_tst=plt.subplots(dpi=300,figsize=(8*phi,8))
            ax_tst.plot(ens_tst['time'],ens_tst[f'{v}_observed'],color='forestgreen',linewidth=1,label='Observed')
            ax_tst.plot(ens_tst['time'],ens_tst[f'{v}_predicted'],color='red',linestyle='--',linewidth=0.5,label='Predicted')
            ax_tst.fill_between(ens_tst['time'],ens_tst[f'{v}_low'],ens_tst[f'{v}_high'],color='orange',alpha=0.5,label=f'{str(spread)}% confidence',zorder=3)
            ax_tst.set_title(f"{v} Test\nRMSE={row.loc[f'{v}_test_rmse']}{unit[j]},R²={row.loc[f'{v}_test_r2']}",fontsize=14)
            ax_tst.set_xlabel("Time")
            ax_tst.set_ylabel(f"{v} ({unit[j]})")
            ax_tst.tick_params(axis='x',rotation=45)
            ax_tst.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_tst.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax_tst.legend(loc='best',prop={'size':10})
            fig_tst.tight_layout()
            fig_tst.savefig(f'./plots/train/ensemble_{v}_test_ts.png')
            plt.close(fig_tst)
        handles,labels=axs_dist[0,0].get_legend_handles_labels()
        fig_dist.legend(handles,labels,loc="upper center",ncol=2,fontsize=12)
        fig_dist.subplots_adjust(top=0.88)
        fig_dist.tight_layout(rect=[0,0,1,0.95])
        fig_dist.savefig('./plots/train/ensemble_distributions.png')
        plt.close(fig_dist)
