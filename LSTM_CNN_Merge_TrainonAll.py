import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from ModelClass import *
# ============================================================
# CONFIG
# ============================================================

train = False

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

model_types = ['LSTM']

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
# POI ANALYSIS FUNCTIONS
# ============================================================

def sort_key(entry):
    """Sort by timestamp inside each POI."""
    timestamp = entry[3]  # timestamp stored in result list
    return timestamp

def compute_metrics_np(pred, actual):
    pred = np.array(pred)
    actual = np.array(actual)
    mse = np.mean((pred - actual)**2)
    mae = np.mean(np.abs(pred - actual))
    mape = np.mean(np.abs((actual - pred) / actual)) * 100
    return mse, mae, mape

def analyze_result_per_POIs(result):
    """
    result = [
        [POI, predicted, actual, timestamp],
        ...
    ]
    """

    print("\n================= OVERALL METRICS =================")
    all_pred = [r[1] for r in result]
    all_actual = [r[2] for r in result]
    mse, mae, mape = compute_metrics_np(all_pred, all_actual)
    print(f"Overall MSE={mse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}")

    # Unique POIs
    pois = sorted(set([r[0] for r in result]))

    for poi in pois:
        poi_items = [r for r in result if r[0] == poi]
        poi_items = sorted(poi_items, key=lambda x: x[3])  # sort by timestamp

        pred = [r[1] for r in poi_items]
        actual = [r[2] for r in poi_items]

        mse, mae, mape = compute_metrics_np(pred, actual)

        print(f"\n===== POI {poi} =====")
        print(f"MSE={mse:.3f}, MAE={mae:.3f}, MAPE={mape:.3f}")

        # Optional: plot
        plt.figure(figsize=(10,4))
        plt.plot(actual, label="Actual")
        plt.plot(pred, label="Predicted")
        plt.title(f"POI {poi} — Actual vs Predicted")
        plt.legend()
        plt.show()

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

    print("==============Results===============")
    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")


def create_sequences(dataset, seq_length):
    x, y = [], []
    for i in range(len(dataset) - seq_length):
        x.append(dataset[i:i+seq_length, :-1])
        y.append(dataset[i+seq_length, -1])
    return np.array(x), np.array(y)


# ============================================================
# DATA LOADING (Option C aware)
# ============================================================

def read_DB(db_path):
    df = pd.read_csv(db_path)
    df = df[data_header].dropna()

    df[ts_col] = pd.to_datetime(df[ts_col], format='%Y_%m_%d_%H_%M')
    df.sort_values(ts_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if normalize_flag:
        scalers = {}
        for column in [target_col, 'temp_centr']:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[column] = scaler.fit_transform(df[[column]])
            scalers[column] = scaler

    df_numeric = df.drop(columns=[ts_col])
    data = df_numeric.values.astype(np.float32)

    x, y = create_sequences(data, timesteps)

    return x, y, df_numeric, df


def read_DB_future_only(test_path, train_last_timestamp):
    df = pd.read_csv(test_path)
    df = df[data_header].dropna()

    df[ts_col] = pd.to_datetime(df[ts_col], format='%Y_%m_%d_%H_%M')
    df.sort_values(ts_col, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # KEEP ONLY FUTURE TIMESTAMPS
    df = df[df[ts_col] > train_last_timestamp]
    df.reset_index(drop=True, inplace=True)

    if len(df) < timesteps + 1:
        print("Warning: Not enough future data for sequence creation.")

    df_numeric = df.drop(columns=[ts_col])
    data = df_numeric.values.astype(np.float32)

    x, y = create_sequences(data, timesteps)

    return x, y, df_numeric, df



# ============================================================
# TRAINING / EVAL Train on all POIs, Test on Future Timestamps
# ============================================================

if train:
    x, y, df_numeric_train, df_full_train = read_DB(DB_file)
    n_features = x.shape[2]

    modelClass = ModelClass(timesteps, n_features, dropout, learning_rate)

    split_idx = int(len(x) * 0.8)
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_last_timestamp = df_full_train[ts_col].max()

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
            print(f"Training time: {end_time - start_time:.2f} seconds")

            test_results = model.evaluate(x_test, y_test, verbose=0)
            print('-------------- Test results for:', model_type, '-----------------')
            print(test_results)

            x_test2, y_test2, df_numeric_test, df_full_test = read_DB_future_only(
                test_file, train_last_timestamp
            )

            test_results2 = model.evaluate(x_test2, y_test2, verbose=0)
            print('-------------- External Future Test results for:', model_type, '-----------------')
            print(test_results2)

else:
    x_test, y_test, df_numeric_test, df_full_test = read_DB(test_file)
    data = df_numeric_test.values.astype(np.float32)

    lstm_model = load_model(base_root + 'Test_LSTM/' + 'temperature_prediction_model0.hdf5')
    LSTM_CNN_model = load_model(base_root + 'Test_LSTM_FollowedBy_CNN/' + 'temperature_prediction_model0.hdf5')
    CNN_LSTM_model = load_model(base_root + 'Test_CNN_FollowedBy_LSTM/' + 'temperature_prediction_model0.hdf5')
    Parrarel_LSTM_CNN_model = load_model(base_root + 'Test_CNN_Parrarel_LSTM/' + 'temperature_prediction_model0.hdf5')

    valid_indices = np.arange(timesteps, len(data))
    random_indices = np.random.choice(valid_indices, 100, replace=False)

    result = []

    for idx in random_indices:
        window = data[idx - timesteps:idx, :-1]
        x_seq = window.reshape(1, timesteps, -1)
        label = data[idx, -1]

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
