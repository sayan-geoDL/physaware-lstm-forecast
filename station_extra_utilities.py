#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 18:07:49 2025
Extra Utility functions
@author: sayan
"""
import numpy as np
import subprocess
import psutil
import ast
####################### getting temperature of the system #####################
def get_gpu_temperature():
    """
    Get the current GPU temperature using `nvidia-smi`.

    Returns
    -------
    int or None
        The GPU temperature in degrees Celsius if successfully retrieved, else None.

    Notes
    -----
    - Requires NVIDIA GPU and `nvidia-smi` to be available in the system PATH.
    - If multiple GPUs are present, only the first one is queried.
    - Returns None and prints an error message if the query fails or if parsing fails.
    """
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'])
        temp_str = output.decode('utf-8').strip()
        temp = int(temp_str.split('\n')[0])
        return temp
    except Exception as e:
        print(f"Error fetching GPU temperature: {e}")
        return None
def get_cpu_temperature():
    """
    Get the current CPU temperature using `psutil`.

    Returns
    -------
    float or None
        The CPU temperature in degrees Celsius if available, else None.

    Notes
    -----
    - Uses `psutil.sensors_temperatures()` to access sensor readings.
    - Searches for temperature under 'coretemp' or entries labeled 'Package id 0'.
    - Returns None if no suitable temperature entry is found or if sensors are not supported.
    """
    temps = psutil.sensors_temperatures()
    if not temps:
        return None
    for name, entries in temps.items():
        for entry in entries:
            if entry.label == 'Package id 0' or 'coretemp' in name:
                return entry.current
    return None
##################### generating dict for grid search #########################
##################### Parsing strings to tuples or accepting lists ############
def parse_range_or_list(entry):
    """
    Parse a string of format "(start, stop, step)" or accept a list directly.

    Parameters
    ----------
    entry : str or list
        Either a list of values or a string representing a 3-element tuple.

    Returns
    -------
    list
        A list of values generated using np.arange or range.

    Raises
    ------
    ValueError
        If parsing fails or format is invalid.
    """
    if isinstance(entry, list):
        return entry
    elif isinstance(entry, str):
        try:
            parsed = ast.literal_eval(entry)
            if not isinstance(parsed, tuple) or len(parsed) != 3:
                raise ValueError("Expected a 3-element tuple string, e.g., '(0.0, 0.01, 0.002)'")
            start, stop, step = parsed
            if any(isinstance(x, float) for x in parsed):
                return list(np.arange(start, stop, step))
            else:
                return list(range(start, stop, step))
        except Exception as e:
            raise ValueError(f"Failed to parse range from string '{entry}': {e}")
    else:
        raise ValueError("Only list or '(start, stop, step)' string formats are supported.")


##################### Generating dict for grid search #########################
def grid(hidden_num, layer_num, learn_rate, weight_decay):
    """
    Prepare a hyperparameter grid for LSTM training, accepting lists or string-tuple ranges.

    Parameters
    ----------
    hidden_num : list or str
        Hidden layer sizes, or a string like '(32, 128, 32)'.
    layer_num : list or str
        Number of layers, or a string like '(1, 4, 1)'.
    learn_rate : list or str
        Learning rates, or a string like '(0.0001, 0.001, 0.0002)'.
    weight_decay : list or str
        Weight decay values, or a string like '(0.0, 0.001, 0.0002)'.

    Returns
    -------
    grid_params : dict
        Dictionary containing lists of hyperparameter values:
        {
            'hidden_size': [...],
            'num_layers': [...],
            'learning_rate': [...],
            'weight_decay': [...]
        }
    """
    grid_params = {
        'hidden_size': parse_range_or_list(hidden_num),
        'num_layers': parse_range_or_list(layer_num),
        'learning_rate': parse_range_or_list(learn_rate),
        'weight_decay': parse_range_or_list(weight_decay)
    }
    return grid_params

