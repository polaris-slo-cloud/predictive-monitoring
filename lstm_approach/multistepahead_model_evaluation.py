from math import sqrt
import numpy as np
import tensorflow as tf

from sklearn.metrics import mean_squared_error


# look also this https://community.dataquest.io/t/how-to-calculate-root-mean-squared-percentage-error-in-python/4142
# or also this https://s2.smu.edu/tfomby/eco5385_eco6380/lecture/Scoring%20Measures%20for%20Prediction%20Problems.pdf
def rmse_perc(y_true, y_pred):
    loss = np.sqrt(mean_squared_error(y_true, y_pred))
    loss = loss / np.mean(y_true)
    
    return loss

def rmse_perc_eps(y_true, y_pred):
    loss = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_target = np.mean(y_true)
    if abs(mean_target) <  np.finfo('float').eps:
        loss = loss / np.finfo('float').eps
    else:
        loss = loss / mean_target
    return loss

def rmse(y_true, y_pred):
    # calculate mse
    mse = mean_squared_error(y_true, y_pred)
    # calculate rmse
    rmse = sqrt(mse)
    return rmse

def true_val(y_true, y_pred):
    true_val = (y_pred - y_true).mean()
    return true_val

def true_val_max(y_true, y_pred):
    true_val_max = ((y_pred - y_true)/y_true).max()
    return true_val_max*100

def true_val_min(y_true, y_pred):
    true_val_min = ((y_pred - y_true)/y_true).min()
    return true_val_min*100

def true_val_ratio_pos(y_true, y_pred):
    t = (y_pred - y_true)
    true_val_ratio_pos = (t >0).sum() / t.shape[0]
    return true_val_ratio_pos


def evaluate_forecast(actual, predicted, func):
    scores = list()
    # calculate error for each day
    for i in range(actual.shape[1]):
        res = func(actual[:, i], predicted[:, i])
        scores.append(res)
    # calculate overall error
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (func(np.array([actual[row, col]]), np.array([predicted[row, col]]))) ** 2
    score = sqrt(s / (actual.shape[0] * actual.shape[1]))
    return score, scores


def walk_forward_validation(model, test_x, test_y, test_set, scaler_transf, n_out, gpu_id=0):
    predictions = list()
    real_values = list()
    start_val = 0
    for i, test_input_x in enumerate(test_x):
        test_input_x = test_input_x.reshape(
            (1, test_input_x.shape[0], test_input_x.shape[1])
        )
        
        yhat = model.predict(test_input_x, verbose=0)

            
        yhat_reshaped = np.array(yhat[0]).reshape((n_out, 1))
        end_val = start_val + n_out
        test_window = test_set[start_val:end_val, :-1]
        if test_window.shape[0] == yhat_reshaped.shape[0]:
            yhat_matrix = np.append(test_window, yhat_reshaped, axis=1)
            yhat_original_scale = scaler_transf.inverse_transform(yhat_matrix)
            inv_yhat = yhat_original_scale[:, -1]
            y_real = test_y[i, :]
            y_real_reshaped = y_real.reshape((n_out, 1))
            y_real_matrix = np.append(test_window, y_real_reshaped, axis=1)
            y_real_original_scale = scaler_transf.inverse_transform(y_real_matrix)
            inv_y = y_real_original_scale[:, -1]
            predictions.append(inv_yhat)
            real_values.append(inv_y)
        start_val += 1
    predictions = np.array(predictions)
    real_values = np.array(real_values)
    

    return predictions, real_values
