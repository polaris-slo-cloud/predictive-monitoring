import argparse
import os

from filelock import FileLock
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import TimeDistributed
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D

import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.keras import TuneReportCallback
from ray.tune.sample import Integer, Categorical

from sklearn.preprocessing import MinMaxScaler

from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import Timestamp

import numpy as np
from numpy import array

import json
import sys

from filelock import FileLock
import os

experiment_name = sys.argv[1]


from gcd_data_manipulation import (
    data_aggregation,
    load_data,
    scale_values,
    series_to_supervised,
)

def reshape_data(
    input_data,
    n_input_steps,
    n_out_steps,
    input_cols_interval=None,
    target_col_index=-1,
):
    X, y = list(), list()
    in_start = 0
    if input_cols_interval == None:
        lower_index = 0
        higher_index = input_data.shape[1]
    else:
        lower_index = input_cols_interval[0]
        higher_index = input_cols_interval[1]
    for _ in range(len(input_data)):
        in_end = in_start + n_input_steps
        out_end = in_end + n_out_steps
        if out_end <= len(input_data):
            x_input = input_data[in_start:in_end, lower_index:higher_index]
            x_input = x_input.reshape((x_input.shape[0], x_input.shape[1]))
            X.append(x_input)
            y.append(input_data[in_end:out_end, target_col_index])
        in_start += 1

    return array(X), array(y)


def read_data():
    job_id = 3418339

    columns_to_consider = [
        "end time",
        "CPU rate",
        "canonical memory usage",
        "assigned memory usage",
        "unmapped page cache",
        "total page cache",
        "maximum memory usage",
        "disk I/O time",
        "local disk space usage",
        "maximum CPU rate",
        "maximum disk IO time",
        "cycles per instruction",
        "memory accesses per instruction",
        "Efficiency",  # target metric
    ]

    readings_df = load_data(
        "/home/amorichetta/SLOC_predictive_monitoring/predictive_monitoring/data/task-usage_job-ID-%i_total.csv" % job_id, columns_to_consider
    )

    readings_df["datetime"] = timestamps_readings = [
    Timestamp(int(t / 1000000) + 1304233200, tz="US/Pacific", unit="s")
    for t in readings_df["end time"].values
    ]
    readings_df.set_index("datetime", inplace=True)
    readings_df.interpolate(method="time", inplace=True)
    
    
    def q95(x):
        return x.quantile(0.95)

    def q75(x):
        return x.quantile(0.75)

    def q25(x):
        return x.quantile(0.25)

    readings_df_x = readings_df.groupby('end time').agg([('mean', 'mean'), ('median', 'median'), ('min', 'min'), ('max','max'), ('q25', q25), ('q75', q75), ('q95', q95), ('std', 'std')])

    readings_df = readings_df_x[columns_to_consider[1:-1]]
    readings_df["Efficiency"] = readings_df_x["Efficiency"]["mean"].values

    values = readings_df.values

    scaled, scaler = scale_values(values)
    
    # split into train and test sets
    n_train = int(scaled.shape[0] * 0.6)
    n_validation = int(scaled.shape[0] * 0.2)
    train = scaled[:n_train, :]
    validation = scaled[n_train : (n_train + n_validation), :]
    test = scaled[(n_train + n_validation) :, :]
    
    return train, validation, test


def train_lstm(config):
    
    # Read data
    with FileLock(os.path.expanduser("~/.data.lock")):
        train, validation, test  = read_data()
        
    n_input = config["n_input"]
    n_out = 3
    
    train_X, train_y = reshape_data(
    train, n_input, n_out, input_cols_interval=(0, -1)
    )
    val_X, val_y = reshape_data(
        validation, n_input, n_out, input_cols_interval=(0, -1)
    )
    test_X, test_y = reshape_data(
        test, n_input, n_out, input_cols_interval=(0, -1)
    )    

    print("train_X", "train_y", train_X.shape, train_y.shape)
    print("val_X", "val_y", val_X.shape, val_y.shape)
    print("test_X", "test_y", test_X.shape, test_y.shape)

    model = Sequential()
    model.add(Conv1D(config["cnn_filters"], 
                     config["cnn_kernel_size"], strides=2, 
                     padding=config["cnn_padding"], input_shape=(None,train_X.shape[2])))
        
    for i in range(config["lstm_add_layers"]):
        model.add(Bidirectional(LSTM(config["neurons"], activation = config["activation"],
                                 input_shape=(train_X.shape[1], train_X.shape[2]),
                                dropout=config["dropout"], 
                                recurrent_dropout=config["recurrent_dropout"],
                                return_sequences=True)))
    model.add(Bidirectional(LSTM(config["neurons"], activation = config["activation"],
                                 input_shape=(train_X.shape[1], train_X.shape[2]),
                                dropout=config["dropout"], 
                                recurrent_dropout=config["recurrent_dropout"])))
    model.add(Dense(train_y.shape[1], activation = config["activation_out"]))
    
    model.compile(loss=config["loss"], optimizer="adam")
    
    history = model.fit(
        train_X,
        train_y,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        verbose=2,
        validation_data=(val_X, val_y),
        callbacks=[TuneReportCallback(
            metrics={"val_loss": "val_loss"})]
    )
    
    
def tune_lstm(num_training_iterations):
    
    
    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1)
    
    results = tune.run(
        train_lstm,
        name="ASHA",
        scheduler=asha_scheduler,
        metric="val_loss",
        mode="min",
        stop={"training_iteration": num_training_iterations},
        num_samples=5,
        config={
        "neurons": tune.qrandint(25, 100, 25),
        "batch_size": tune.choice([12, 16, 28, 32, 36, 48, 60, 64, 72]),
        "epochs": tune.qrandint(100, 400, 25),
        "n_input": tune.grid_search([1, 3, 6, 9, 12, 15, 18, 21, 24]),
        "loss": tune.grid_search(["mae", "mse"]),
        "dropout": tune.uniform(0.0, 0.25),
        "recurrent_dropout": tune.uniform(0.0, 0.25),
        "activation": tune.grid_search(['tanh', 'relu']),
        "activation_out": tune.grid_search(['linear',  'relu']),
        "cnn_filters": tune.qrandint(25, 100, 25),
        "cnn_kernel_size": tune.qrandint(3, 7, 1), 
        "cnn_padding": tune.grid_search(['valid', 'causal']),
        "lstm_add_layers": tune.qrandint(0, 5, 1), 
        },
        resources_per_trial={"gpu": 1},
        raise_on_failed_trial = False,
    )
    
    print("Best hyperparameters found were: ", results.best_config)

    results.results_df.to_csv("/home/amorichetta/SLOC_predictive_monitoring/predictive_monitoring/results/raytune/"+experiment_name)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke-test", action="store_true", help="Finish quickly for testing")

    args, _ = parser.parse_known_args()
    if args.smoke_test:
        ray.init()

    tune_lstm(num_training_iterations=5 if args.smoke_test else 300)