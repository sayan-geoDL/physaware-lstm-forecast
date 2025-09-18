#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 19:28:45 2025

@author: sayan
"""

import yaml
import argparse
import logging
import sys
import os

import station_preprocess, station_scaler, station_extra_utilities
import station_model_util, station_cv, station_trainer, station_predict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("main.log"),
        logging.StreamHandler()
    ]
)
logger=logging.getLogger(__name__)

##################### processing of data ######################################
def process(config):
    """
    Handles the full data preparation pipeline for the station LSTM project.

    Steps:
        1. Optionally downloads raw station data for training, testing, and prediction periods.
        2. Processes raw data into daily format, filling gaps as specified in config.
        3. Scales the processed data for training, testing, and prediction using specified scaling parameters.

    Args:
        config (dict): Configuration dictionary loaded from YAML, containing all data, filling, and scaling parameters.

    Saves:
        - Processed and scaled CSV files in ./data/processed/
        - Logs progress and errors to main.log
    """
    logger.info("Starting data processing stage...")
    pred_dates=config.get("dates", {}).get("predict", {})
    pred_start=pred_dates.get("start")
    pred_end=pred_dates.get("end")

    if config.get("download_data", False):
        logger.info("Downloading station data as requested in config...")
        try:
            station_preprocess.download_data(
                station_id=config["station_id"],
                start_date=config["dates"]["train"]["start"],
                end_date=config["dates"]["test"]["end"],
                name='station_train_test'
            )
            if pred_start and pred_end:
                station_preprocess.download_data(
                    station_id=config["station_id"],
                    start_date=pred_start,
                    end_date=pred_end,
                    name='station_predict'
                )
        except Exception as e:
            logger.error(f"Failed to download station data: {e}")
            sys.exit(1)

    logger.info("Preprocessing and scaling training and testing data...")
    station_preprocess.process_to_daily(
        name_in='station_train_test',
        name_out='train',
        start_date=config['dates']['train']['start'],
        end_date=config['dates']['train']['end'],
        fill_method=config['filling_parameters']['type'],
        phase='train',
        gap_threshold=config['filling_parameters']['gap_threshold']
    )
    station_preprocess.process_to_daily(
        name_in='station_train_test',
        name_out='test',
        start_date=config['dates']['test']['start'],
        end_date=config['dates']['test']['end'],
        fill_method=config['filling_parameters']['type'],
        phase='predict',
        gap_threshold=config['filling_parameters']['gap_threshold']
    )

    station_scaler.scaler(
        strt=config['dates']['train']['start'],
        end=config['dates']['train']['end'],
        phase='train',
        inp='./data/processed/train_daily.csv',
        outp='./data/processed/train_scaled.csv',
        wspd_perc=config['scaling']['wind_threshold_percent']
    )

    station_scaler.scaler(
        strt=config['dates']['test']['start'],
        end=config['dates']['test']['end'],
        phase='predict',
        inp='./data/processed/test_daily.csv',
        outp='./data/processed/test_scaled.csv'
    )

    if pred_start and pred_end:
        logger.info("Preprocessing and scaling prediction data...")
        station_preprocess.process_to_daily(
            name_in='station_predict',
            name_out='predict',
            phase='predict',
            fill_method=config['filling_parameters']['type'],
            gap_threshold=config['filling_parameters']['gap_threshold']
        )
        station_scaler.scaler(
            strt=pred_start,
            end=pred_end,
            phase='predict',
            inp='./data/processed/predict_daily.csv',
            outp='./data/processed/predict_scaled.csv'
        )

    logger.info("Data processing completed.")

##################### grid search and cross validation ########################
def optimize_cv(config):
    """
    Performs hyperparameter optimization and cross-validation for the LSTM model.

    Steps:
        1. Loads and batches training data.
        2. Runs grid search or multi-objective optimization for hyperparameters, with or without physics-aware loss.
        3. Plots RMSE and overfitting diagnostics.
        4. Generates cross-validation plots and a PDF report of hyperparameter search results.

    Args:
        config (dict): Configuration dictionary with hyperparameter search space, CV settings, and device info.

    Saves:
        - Pareto-optimal hyperparameters to ./out/train_test/cv_result.csv
        - Diagnostic plots to ./plots/cv/
        - Hyperparameter report to ./out/train_test/cv_report.pdf
        - Logs progress and errors to main.log
    """
    logger.info("Starting grid search and cross-validation...")
    x, y, t=station_model_util.batcher(
        strt=config['dates']['train']['start'],
        end=config['dates']['train']['end'],
        n_step=config['lookback'],
        inp='./data/processed/train_scaled.csv'
    )
    conf_optim=config.get('hyper_optim', None)
    search_space= conf_optim['search_space']
    logger.info(f"Hyperparameter search space: {search_space}")
    for key, val in search_space.items():
        if val['type'] == 'float':
            val['low']  = float(val['low'])
            val['high'] = float(val['high'])
            if 'step' in val and val['step'] is not None:
                val['step'] = float(val['step'])
        elif val['type'] == 'int':
            val['low']  = int(val['low'])
            val['high'] = int(val['high'])
            if 'step' in val and val['step'] is not None:
                val['step'] = int(val['step'])
    if conf_optim['physics_mode']=='none':
        if os.path.exists('./out/train_test/cv_result.csv'):
            os.remove('./out/train_test/cv_result.csv')
        station_cv.hyper_optim(x, y,
                               search_space=search_space,
                               n_trials=conf_optim['n_trials'],
                               njobs=conf_optim['njobs'],
                               val_size=conf_optim['val_size'],
                               k=conf_optim['k_folds'],
                               device=conf_optim['device'],
                               epochs=conf_optim['epochs'],
                               es_patience=conf_optim['es_patience'],
                               force_lambda_zero=True)
    elif conf_optim['physics_mode']=='with':
        if os.path.exists('./out/train_test/cv_result.csv'):
            os.remove('./out/train_test/cv_result.csv')
        station_cv.hyper_optim(x, y,
                               search_space=search_space,
                               n_trials=conf_optim['n_trials'],
                               njobs=conf_optim['njobs'],
                               val_size=conf_optim['val_size'],
                               k=conf_optim['k_folds'],
                               device=conf_optim['device'],
                               epochs=conf_optim['epochs'],
                               es_patience=conf_optim['es_patience'],
                               force_lambda_zero=False)
    elif conf_optim['physics_mode']=='both':
        if os.path.exists('./out/train_test/cv_result.csv'):
            os.remove('./out/train_test/cv_result.csv')
        station_cv.hyper_optim(x, y,
                               search_space=search_space,
                               n_trials=conf_optim['n_trials'],
                               njobs=conf_optim['njobs'],
                               val_size=conf_optim['val_size'],
                               k=conf_optim['k_folds'],
                               device=conf_optim['device'],
                               epochs=conf_optim['epochs'],
                               es_patience=conf_optim['es_patience'],
                               force_lambda_zero=True)
        station_cv.hyper_optim(x, y,
                               search_space=search_space,
                               n_trials=conf_optim['n_trials'],
                               njobs=conf_optim['njobs'],
                               val_size=conf_optim['val_size'],
                               k=conf_optim['k_folds'],
                               device=conf_optim['device'],
                               epochs=conf_optim['epochs'],
                               es_patience=conf_optim['es_patience'],
                               force_lambda_zero=False)
    if os.path.exists('./plots/cv/scatters.png'):
        os.remove('./plots/cv/scatters.png')    
    station_cv.plot_rmse_overfit_colored(f_path='./out/train_test/cv_result.csv')
    if conf_optim['rank']=='all' or isinstance(conf_optim['rank'], int):
        rank=conf_optim['rank']
    elif isinstance(conf_optim['rank'], float):
        raise ValueError("Rank cannot be a float.")
    else:
        rank=station_extra_utilities.parse_range_or_list(conf_optim['rank'])
    station_extra_utilities.delete_files_with_prefix(directory='./plots/cv',
                                                     prefix='rank',logger=logger)
    
    station_cv.cv_plot(x,y,epochs=conf_optim['epochs'],val_size=conf_optim['val_size'],
                       device=conf_optim['device'],
                       rank=rank,k=conf_optim['k_folds'],
                       es_patience=conf_optim['es_patience'])
    station_extra_utilities.generate_hyperparam_report(csv_file='./out/train_test/cv_result.csv',
                                                       plot_dir='./plots/cv',
                                                       output_pdf='./out/train_test/cv_report.pdf')
    logger.info("Hyperparameter optimization and cross-validation completed.")

######################## training and testing stage ###########################
def train(config):
    """
    Trains and evaluates the LSTM model(s) using the specified configuration.

    Steps:
        1. Loads and batches training and test data.
        2. Trains a single model or an ensemble using selected hyperparameters.
        3. Evaluates model(s) on test data, computes metrics, and generates predictions.
        4. Produces diagnostic plots and summary reports.

    Args:
        config (dict): Configuration dictionary with training, evaluation, and plotting parameters.

    Saves:
        - Model weights to ./out/models/
        - Metrics and predictions to ./out/train_test/
        - Training and evaluation plots to ./plots/train/
        - PDF reports to ./out/train_test/
        - Logs progress and errors to main.log
    """
    logger.info("Starting model training and evaluation...")
    training_config=config['training']
    if training_config['mode']=='single':
        if training_config['use_rank']:
            logger.info(f"Using hyper parameters from CV rank: {training_config['rank']}")
            param=training_config['rank']
        else:
            logger.info("Using manually specified training hyper parameters.")
            param=training_config['params']
        station_extra_utilities.delete_files_with_prefix(directory='./out/models',prefix='final_model',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./plots/train',prefix='train_val_loss',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./out/train_test',prefix='metrics',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./out/train_test',prefix='train_ts',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./out/train_test',prefix='test_ts',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./plots/train',prefix='distributions',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./plots/train',prefix='time_series',logger=logger)
    elif training_config['mode']=='ensemble':
        logger.info("Using ensemble of models from specified CV ranks.")
        if not training_config['rank']=='all':
            param=station_extra_utilities.parse_range_or_list(training_config['rank'])
            logger.info(f"Using ensemble of models from CV ranks: {param}")
        else:
            param='all'
            logger.info("Using ensemble members from all CV ranks.")
        station_extra_utilities.delete_files_with_prefix(directory='./out/models',prefix='rank',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./plots/train',prefix='rank',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./out/train_test',prefix='rank',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./out/train_test',prefix='ensemble',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./plots/train',prefix='ensemble',logger=logger)
        
    x_tr, y_tr, t_tr=station_model_util.batcher(strt=config['dates']['train']['start'],
                                                  end=config['dates']['train']['end'],
                                                  n_step=config['lookback'],
                                                  inp='./data/processed/train_scaled.csv')

    station_trainer.trainer(x_tr, y_tr, param,
                            epochs=training_config['epochs'],
                            hold_out=training_config['hold_out'],
                            mode=training_config['mode'],
                            device=training_config['device'],
                            es_patience=training_config['es_patience']
                            )

    x_tst, y_tst, t_tst=station_model_util.batcher(strt=config['dates']['test']['start'],
                                                     end=config['dates']['test']['end'],
                                                     n_step=config['lookback'],
                                                     inp='./data/processed/test_scaled.csv')

    station_trainer.tester(x_tr, y_tr, t_tr,
                           x_tst, y_tst, t_tst,
                           params=param,
                           mode=training_config['mode'],
                           B=training_config['B'],
                           low=training_config['low'],
                           high=training_config['high'],
                           device=training_config['device'])
    station_trainer.train_plots(inp='./out/train_test',
                                mode=training_config['mode'],
                                low=training_config['low'],
                                high=training_config['high'])
    if training_config['mode']=='single':
        station_extra_utilities.generate_single_model_report()
    elif training_config['mode']=='ensemble':
        station_extra_utilities.generate_ensemble_report(var_names=['dwpt','temp','rhum','pres'],
                                                         ranks=param)
    logger.info("Model training and testing completed.\nReports and plots saved.")

############### prediction stage ##############################################
def predict(config):
    """
    Runs prediction on unseen (future) data using trained LSTM model(s).

    Steps:
        1. Loads and batches prediction data.
        2. Loads trained model(s) and generates predictions for the specified period.
        3. Saves prediction outputs and uncertainty estimates (if ensemble).

    Args:
        config (dict): Configuration dictionary with prediction settings, including device and ensemble parameters.

    Saves:
        - Prediction CSVs to ./out/predictions/
        - Logs progress and errors to main.log
    """
    if not config['prediction'].get('run_prediction', False):
        logger.warning("Prediction skipped: 'run_prediction' is set to False.")
        return

    logger.info("Starting prediction on unseen data...")
    training_config=config['training']
    prediction_config=config['prediction']
    if training_config['mode']=='single':
        if training_config['use_rank']:
            logger.info(f"Using hyper parameters from CV rank: {training_config['rank']}")
            param=training_config['rank']
        else:
            logger.info("Using manually specified training hyper parameters.")
            param=training_config['params']
        station_extra_utilities.delete_files_with_prefix(directory='./out/predictions',prefix='single_predictions',logger=logger)
    elif training_config['mode']=='ensemble':
        if training_config['rank']=="all":
            param='all'
            logger.info("Using ensemble members from all CV ranks.")
        else:
            logger.info("Using ensemble of models from specified CV ranks.")
            param=station_extra_utilities.parse_range_or_list(training_config['rank'])
        station_extra_utilities.delete_files_with_prefix(directory='./out/predictions',prefix='rank',logger=logger)
        station_extra_utilities.delete_files_with_prefix(directory='./out/predictions',prefix='ensemble',logger=logger)


    x_pr, t_pr=station_model_util.batcher(strt=config['dates']['predict']['start'],
                                            end=config['dates']['predict']['end'],
                                            n_step=config['lookback'],mode='predict',
                                            inp='./data/processed/predict_scaled.csv')

    station_predict.predict(x_pr, t_pr, param,
                            mode=training_config['mode'],
                            device=prediction_config['device'],
                            B=prediction_config['B'],
                            low=prediction_config['low'],
                            high=prediction_config['high'])
    logger.info("Prediction completed and saved.")

##################### main deployment #########################################
if __name__=="__main__":
    parser=argparse.ArgumentParser(description="Station LSTM Temperature Forecasting Pipeline")
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help="Path to configuration YAML file")
    parser.add_argument('--stage', type=str, required=True,
                        choices=['process', 'optimize_cv', 'train', 'predict', 'all'],
                        help="Stage of the pipeline to execute")

    args=parser.parse_args()

    with open(args.config, 'r') as f:
        config=yaml.safe_load(f)

    logger.info(f"Executing pipeline stage: {args.stage}")

    if args.stage=='process':
        process(config)
    elif args.stage=='optimize_cv':
        optimize_cv(config)
    elif args.stage=='train':
        train(config)
    elif args.stage=='predict':
        predict(config)
    elif args.stage=='all':
        process(config)
        optimize_cv(config)
        train(config)
        predict(config)

    logger.info("Pipeline execution completed.")
