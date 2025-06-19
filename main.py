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
logger = logging.getLogger(__name__)

##################### processing of data ######################################
def process(config):
    logger.info("Starting data processing stage...")
    pred_dates = config.get("dates", {}).get("predict", {})
    pred_start = pred_dates.get("start")
    pred_end = pred_dates.get("end")

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
        name_out='train_test_daily',
        fill_method=config['filling_parameters']['type'],
        phase='train',
        gap_threshold=config['filling_parameters']['gap_threshold']
    )

    station_scaler.scaler(
        strt=config['dates']['train']['start'],
        end=config['dates']['train']['end'],
        phase='train',
        inp='./data/processed/train_test_daily.csv',
        outp='./data/processed/train_scaled.csv',
        wspd_perc=config['scaling']['magnitude_threshold_percent']
    )

    station_scaler.scaler(
        strt=config['dates']['test']['start'],
        end=config['dates']['test']['end'],
        phase='predict',
        inp='./data/processed/train_test_daily.csv',
        outp='./data/processed/test_scaled.csv'
    )

    if pred_start and pred_end:
        logger.info("Preprocessing and scaling prediction data...")
        station_preprocess.process_to_daily(
            name_in='station_predict',
            name_out='predict_daily',
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
def grid_search_cv(config):
    logger.info("Starting grid search and cross-validation...")
    x, y, t = station_model_util.batcher(
        strt=config['dates']['train']['start'],
        end=config['dates']['train']['end'],
        n_step=config['input_window'],
        inp='./data/processed/train_scaled.csv'
    )

    param_grid = station_extra_utilities.grid(
        hidden_num=config['grid_search']['hidden_size'],
        layer_num=config['grid_search']['num_layers'],
        learn_rate=config['grid_search']['learning_rate'],
        weight_decay=config['grid_search']['weight_decay']
    )

    logger.info(f"Total hyperparameter combinations: {len(param_grid)}")
    station_cv.cv_outs(
        x, y, param_grid=param_grid,
        val_size=config['validation_size'], initial_train_size=None,
        k=config['fold_no'], device=config['device'],
        epochs=config['epochs'], es_patience=config['early_stop']
    )
    station_cv.performance_box_plots()
    logger.info("Grid search and cross-validation completed.")

###################### cross validation plots for selective rank ##############
def cv_plot(config):
    logger.info(f"Generating loss curve plots for rank {config['plot_cv_rank']}...")
    x, y, t = station_model_util.batcher(
        strt=config['dates']['train']['start'],
        end=config['dates']['train']['end'],
        n_step=config['input_window'],
        inp='./data/processed/train_scaled.csv'
    )

    station_cv.cv_plot(
        x, y, initial_train_size=None,
        val_size=config['validation_size'], k=config['fold_no'],
        device=config['device'], epochs=config['epochs'],
        rank=config['plot_cv_rank']
    )
    logger.info("Loss curve plots generated.")

######################## training and testing stage ###########################
def train(config):
    logger.info("Starting model training and evaluation...")
    training_config = config['training']
    if training_config['use_rank']:
        logger.info(f"Using parameters from CV rank: {training_config['rank']}")
        param = training_config['rank']
    else:
        logger.info("Using manually specified training parameters.")
        param = training_config['params']

    x_tr, y_tr, t_tr = station_model_util.batcher(
        strt=config['dates']['train']['start'],
        end=config['dates']['train']['end'],
        n_step=config['input_window'],
        inp='./data/processed/train_scaled.csv'
    )

    station_trainer.trainer(
        x_tr, y_tr, param,
        epochs=config['epochs'], device=config['device']
    )

    x_tst, y_tst, t_tst = station_model_util.batcher(
        strt=config['dates']['test']['start'],
        end=config['dates']['test']['end'],
        n_step=config['input_window'],
        inp='./data/processed/test_scaled.csv'
    )

    station_trainer.tester(
        x_tr, y_tr, t_tr,
        x_tst, y_tst, t_tst,
        params=param,
        device=config['device']
    )

    logger.info("Model training and testing completed.")

############### prediction stage ##############################################
def predict(config):
    if not config.get('run_prediction', False):
        logger.warning("Prediction skipped: 'run_prediction' is set to False.")
        return

    logger.info("Starting prediction on unseen data...")
    training_config = config['training']
    if training_config['use_rank']:
        logger.info(f"Using parameters from CV rank: {training_config['rank']}")
        param = training_config['rank']
    else:
        logger.info("Using manually specified prediction parameters.")
        param = training_config['params']

    x_pr, t_pr = station_model_util.batcher(
        strt=config['dates']['predict']['start'],
        end=config['dates']['predict']['end'],
        n_step=config['input_window'],
        mode='predict',
        inp='./data/processed/predict_scaled.csv'
    )

    station_predict.predict(
        x_pr, t_pr, param,
        device=config['device']
    )
    logger.info("Prediction completed and saved.")

##################### main deployment #########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Station LSTM Temperature Forecasting Pipeline")
    parser.add_argument('--config', type=str, default='./config.yaml',
                        help="Path to configuration YAML file")
    parser.add_argument('--stage', type=str, required=True,
                        choices=['process', 'grid_search_cv', 'cv_plot', 'train', 'predict', 'all'],
                        help="Stage of the pipeline to execute")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    logger.info(f"Executing pipeline stage: {args.stage}")

    if args.stage == 'process':
        process(config)
    elif args.stage == 'grid_search_cv':
        grid_search_cv(config)
    elif args.stage == 'cv_plot':
        cv_plot(config)
    elif args.stage == 'train':
        train(config)
    elif args.stage == 'predict':
        predict(config)
    elif args.stage == 'all':
        process(config)
        grid_search_cv(config)
        cv_plot(config)
        train(config)
        predict(config)

    logger.info("Pipeline execution completed.")