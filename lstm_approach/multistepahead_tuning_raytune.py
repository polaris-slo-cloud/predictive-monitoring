from __future__ import print_function

from typing import Dict, Union

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


from sklearn.preprocessing import MinMaxScaler

from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from pandas import Timestamp

import numpy as np

import json
import sys

from filelock import FileLock
import os

from ray import tune


from numpy import array


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

experiment_name = sys.argv[1]


from gcd_data_manipulation import (
    data_aggregation,
    load_data,
    scale_values,
    series_to_supervised,
)


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
        #"CPU ratio usage",
        #"memory ratio usage",
        #"disk ratio usage",
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
    
    
    def q95(x):
        return x.quantile(0.95)

    def q75(x):
        return x.quantile(0.75)

    def q25(x):
        return x.quantile(0.25)

    readings_df_x = readings_df.groupby('end time').agg([('mean', 'mean'), ('median', 'median'), ('min', 'min'), ('max','max'), ('q25', q25), ('q75', q75), ('q95', q95), ('std', 'std')])

    values = readings_df.values

    scaled, scaler = scale_values(values)
    
    # split into train and test sets
    n_train = int(scaled.shape[0] * 0.6)
    n_validation = int(scaled.shape[0] * 0.2)
    train = scaled[:n_train, :]
    validation = scaled[n_train : (n_train + n_validation), :]
    test = scaled[(n_train + n_validation) :, :]
    
    n_input = 24
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
    
    assert isinstance(test_y, object)
    return train_X, train_y, val_X, val_y, test_X, test_y


class lstmmodel(tune.Trainable):

    
    def build_model(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(self.neurons, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))) #
        model.add(Dense(self.train_y.shape[1]))

        return model

    def setup(self, config):
        with FileLock(os.path.expanduser("~/.tune.lock")):
            self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y = read_data()
        self.epochs = config['epochs']
        self.batch_size = config['batch_size']
        self.neurons = config['neurons']
        model = self.build_model()
        model.compile(loss='mae', optimizer='adam')
        self.model = model

    @property
    def step(self):
        self.model.fit(self.train_X, self.train_y,
                       epochs=self.epochs,
                       batch_size=self.batch_size,
                       validation_data=(self.val_X, self.val_y),
                       verbose=2,
                       shuffle=False)

    def save_checkpoint(self, checkpoint_dir):
        file_path = checkpoint_dir + "/model"
        self.model.save(file_path)
        return file_path

    def load_checkpoint(self, path):
        # See https://stackoverflow.com/a/42763323
        del self.model
        self.model = load_model(path)


if __name__ == "__main__":

    import ray
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.schedulers import PopulationBasedTraining
    from ray.tune.trial import ExportFormat

    ray.init(num_cpus=20, num_gpus=1)

    asha_scheduler = ASHAScheduler(
        time_attr='training_iteration',
        # metric="loss",
        # mode="min",
        max_t=100,
        grace_period=10,
        reduction_factor=3,
        brackets=1)

    config: Dict[str, Union[Integer, Dict[str, list], Categorical]] = {
        "neurons": tune.qrandint(25, 100, 25),
        "batch_size": tune.grid_search([12, 16, 28, 32, 36, 48, 60, 64, 72, 104, 136]),
        "epochs": tune.qrandint(100, 400, 25)

    }

    results = tune.run(
        lstmmodel,
        name="ASHA",
        scheduler=asha_scheduler,
        metric="val_loss",
        mode="min",
        stop={"training_iteration": 100},
        num_samples=100,
        config=config,
        resources_per_trial={"gpu": 1})

    print(results.best_result)

    results.best_result_df.to_csv(
        "/home/amorichetta/SLOC_predictive_monitoring/predictive_monitoring/data/results/raytune_LSTM_01_%s.csv" % experiment_name)
