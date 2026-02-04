import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import time
import random
import pickle

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional,
    Conv1D, MaxPooling1D, Flatten, Concatenate
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

# ============================================================
# CONFIG
# ============================================================

train = True

normalize_flag = False
epochs = 400
batch_size = 32
validation_split = 0.2
timesteps = 3
patience = 100
dropout = 0.2
learning_rate = 0.0001

DB_file = "data/DB/train_DB.csv"
test_file = "data/DB/test_DB.csv"
model_file = 'temperature_prediction_model'
base_root = 'data/Test/'

model_types = ['CNN_LSTM']

data_header = [
    'sensor', 'dist_to_central_station', 'year',
    'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week', 'hour',
    'complete_timestamp(YYYY_M_DD_HH_M)', 'barometer_hpa', 'temp_centr',
    'hightemp_centr', 'lowtemp_centr', 'hum', 'dewpoint__c', 'wetbulb_c',
    'windspeed_km_h', 'windrun_km', 'highwindspeed_km_h', 'windchill_c',
    'heatindex_c', 'thwindex_c', 'thswindex_c', 'rain_mm', 'rain_rate_mm_h',
    'solar_rad_w_m_2', 'solar_energy_ly', 'high_solar_rad_w_m_2', 'ET_Mm',
    'heating_degree_days', 'cooling_degree_days', 'humidity_rh',
    'solar_klux', 'temp_to_estimate'
]

ts_col = 'complete_timestamp(YYYY_M_DD_HH_M)'
target_col = 'temp_to_estimate'


# ============================================================
# METRICS / UTILS
# ============================================================

def compute_metrics(predicted_values, actual_values):
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)

    loss = np.sum((predicted_values - actual_values) ** 2)
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    print(f"==============Results===============")
    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")


def create_sequences(dataset, seq_length):
    x, y = [], []
    # dataset: (N, n_features), last column = target
    for i in range(len(dataset) - seq_length):
        x.append(dataset[i:i+seq_length, :-1])      # all features except target
        y.append(dataset[i+seq_length, -1])         # 1-step ahead target
    return np.array(x), np.array(y)


# ============================================================
# DATA LOADING
# ============================================================

def read_DB(db_path):
    df = pd.read_csv(db_path)
    df = df[data_header].dropna()

    # parse and sort by timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], format='%Y_%m_%d_%H_%M')
    df.sort_values(ts_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # optional normalization
    if normalize_flag:
        scalers = {}
        for column in [target_col, 'temp_centr']:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[column] = scaler.fit_transform(df[[column]])
            scalers[column] = scaler

    # numeric-only for model input
    df_numeric = df.drop(columns=[ts_col])
    data = df_numeric.values.astype(np.float32)

    # build sequences
    x, y = create_sequences(data, timesteps)

    # keep full numeric data and timestamps for later reference
    return x, y, df_numeric, df


# ============================================================
# MODEL CLASS
# ============================================================

class ModelClass:
    def __init__(self, timesteps, n_features, dropout, learning_rate):
        self.timesteps = timesteps
        self.n_features = n_features
        self.dropout = dropout
        self.learning_rate = learning_rate

    def compile_model(self, model):
        opt = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=opt,
            loss='mean_absolute_percentage_error',
            metrics=['mape', 'mae', 'mse']
        )
        return model

    def create_LSTM(self):
        inp = Input(shape=(self.timesteps, self.n_features))
        x = Bidirectional(LSTM(256, return_sequences=True))(inp)
        x = Dropout(self.dropout)(x)
        x = Bidirectional(LSTM(128, return_sequences=True))(x)
        x = Dropout(self.dropout)(x)
        x = Bidirectional(LSTM(64, return_sequences=False))(x)
        x = Dropout(self.dropout)(x)
        out = Dense(1)(x)
        model = Model(inp, out)
        return self.compile_model(model)

    def create_cnn_lstm_model(self):
        inp = Input(shape=(self.timesteps, self.n_features))
        x = Conv1D(64, 3, activation='relu', padding='causal')(inp)
        x = MaxPooling1D(2)(x)
        x = LSTM(64)(x)
        x = Dropout(self.dropout)(x)
        out = Dense(1)(x)
        model = Model(inp, out)
        return self.compile_model(model)

    def create_lstm_cnn_model(self):
        inp = Input(shape=(self.timesteps, self.n_features))
        x = LSTM(64, return_sequences=True)(inp)
        x = Dropout(self.dropout)(x)
        x = Conv1D(64, 3, activation='relu', padding='causal')(x)
        x = Flatten()(x)
        out = Dense(1)(x)
        model = Model(inp, out)
        return self.compile_model(model)

    def create_parallel_cnn_lstm_model(self):
        inp = Input(shape=(self.timesteps, self.n_features))

        # LSTM branch
        lstm_branch = LSTM(64)(inp)

        # CNN branch
        cnn_branch = Conv1D(64, 3, activation='relu', padding='causal')(inp)
        cnn_branch = MaxPooling1D(2)(cnn_branch)
        cnn_branch = Flatten()(cnn_branch)

        merged = Concatenate()([lstm_branch, cnn_branch])
        x = Dropout(self.dropout)(merged)
        out = Dense(1)(x)

        model = Model(inp, out)
        return self.compile_model(model)


# ============================================================
# TRAINING / EVAL
# ============================================================

if train:
    # load training DB
    x, y, df_numeric_train, df_full_train = read_DB(DB_file)
    n_features = x.shape[2]

    modelClass = ModelClass(timesteps, n_features, dropout, learning_rate)

    # chronological split
    split_idx = int(len(x) * 0.8)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    for model_type in model_types:
        for i in range(1):
            if model_type == 'LSTM':
                root = base_root + 'Test_LSTM/'
                model = modelClass.create_LSTM()
            elif model_type == 'LSTM_CNN':
                root = base_root + 'Test_LSTM_FollowedBy_CNN/'
                model = modelClass.create_lstm_cnn_model()
            elif model_type == 'CNN_LSTM':
                root = base_root + 'Test_CNN_FollowedBy_LSTM/'
                model = modelClass.create_cnn_lstm_model()
            elif model_type == 'Parrarel_LSTM_CNN':
                root = base_root + 'Test_CNN_Parrarel_LSTM/'
                model = modelClass.create_parallel_cnn_lstm_model()
            else:
                continue

            checkpointer = ModelCheckpoint(
                filepath=root + model_file + str(i) + '.hdf5',
                verbose=0,
                save_best_only=True
            )
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True
            )

            print(model_type)
            start_time = time.time()

            history = model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                verbose=0,
                validation_split=validation_split,
                callbacks=[checkpointer, early_stopping]
            )

            end_time = time.time()
            train_time = end_time - start_time
            print(f"Training time: {train_time:.2f} seconds")

            # plots (optional)
            plt.figure()
            plt.plot(history.history['loss'], label='Train loss')
            plt.plot(history.history['val_loss'], label='Val loss')
            plt.legend()
            plt.title(f'Loss - {model_type}')
            plt.show()

            plt.figure()
            plt.plot(history.history['mse'], label='Train MSE')
            plt.plot(history.history['val_mse'], label='Val MSE')
            plt.legend()
            plt.title(f'MSE - {model_type}')
            plt.show()

            plt.figure()
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Val MAE')
            plt.legend()
            plt.title(f'MAE - {model_type}')
            plt.show()

            plt.figure()
            plt.plot(history.history['mape'], label='Train MAPE')
            plt.plot(history.history['val_mape'], label='Val MAPE')
            plt.legend()
            plt.title(f'MAPE - {model_type}')
            plt.show()

            # evaluate on held-out test
            test_results = model.evaluate(x_test, y_test, verbose=0)
            print('-------------- Test results for:', model_type, '-----------------')
            print(test_results)

            # save test data
            with open(root + 'x_test' + str(i), 'wb') as fout:
                pickle.dump(x_test, fout)
            with open(root + 'y_test' + str(i), 'wb') as fout:
                pickle.dump(y_test, fout)

            # also evaluate on separate test DB
            x_test2, y_test2, _, _ = read_DB(test_file)
            test_results2 = model.evaluate(x_test2, y_test2, verbose=0)
            print('-------------- External Test DB results for:', model_type, '-----------------')
            print(test_results2)

            with open(root + 'outputResultFor5runs.txt', 'a') as file:
                for item in test_results2:
                    file.write(str(item) + '__')
                file.write('\n')

else:
    # INFERENCE / COMPARISON MODE
    x_test, y_test, df_numeric_test, df_full_test = read_DB(test_file)
    data = df_numeric_test.values.astype(np.float32)  # (N, n_features)

    # load models
    lstm_model = load_model(base_root + 'Test_LSTM/' + 'temperature_prediction_model0.hdf5')
    LSTM_CNN_model = load_model(base_root + 'Test_LSTM_FollowedBy_CNN/' + 'temperature_prediction_model0.hdf5')
    CNN_LSTM_model = load_model(base_root + 'Test_CNN_FollowedBy_LSTM/' + 'temperature_prediction_model0.hdf5')
    Parrarel_LSTM_CNN_model = load_model(base_root + 'Test_CNN_Parrarel_LSTM/' + 'temperature_prediction_model0.hdf5')

    # choose valid indices where we have enough history
    valid_indices = np.arange(timesteps, len(data))
    random_indices = np.random.choice(valid_indices, 100, replace=False)

    result = []

    for idx in random_indices:
        window = data[idx - timesteps:idx, :-1]   # previous timesteps, all features except target
        x_seq = window.reshape(1, timesteps, -1)
        label = data[idx, -1]                     # target at current time

        predict_lstm = lstm_model.predict(x_seq, verbose=0)[0][0]
        predict_LSTM_CNN = LSTM_CNN_model.predict(x_seq, verbose=0)[0][0]
        predict_CNN_LSTM = CNN_LSTM_model.predict(x_seq, verbose=0)[0][0]
        predict_Parrarel = Parrarel_LSTM_CNN_model.predict(x_seq, verbose=0)[0][0]

        timestamp = df_full_test.iloc[idx][ts_col]
        result.append([
            timestamp, label,
            predict_lstm, predict_LSTM_CNN,
            predict_CNN_LSTM, predict_Parrarel
        ])

    # compute metrics per model on these sampled points
    labels = [r[1] for r in result]
    preds_lstm = [r[2] for r in result]
    preds_lstm_cnn = [r[3] for r in result]
    preds_cnn_lstm = [r[4] for r in result]
    preds_parallel = [r[5] for r in result]

    print("LSTM metrics on sampled points:")
    compute_metrics(preds_lstm, labels)
    print("LSTM-CNN metrics on sampled points:")
    compute_metrics(preds_lstm_cnn, labels)
    print("CNN-LSTM metrics on sampled points:")
    compute_metrics(preds_cnn_lstm, labels)
    print("Parallel CNN-LSTM metrics on sampled points:")
    compute_metrics(preds_parallel, labels)
print('')