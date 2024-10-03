import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
import random
from datetime import datetime
from keras.layers import Conv1D, ReLU, MaxPooling1D, Flatten
from ModelClass import *
import operator

train = True
 
# =======RNN config
normalize_flag= False
epochs= 400
batch_size= 32
validation_split= 0.2
timesteps= 3
patience= 100
dropout= 0.2
learning_rate= 0.0001
with_out_process_time=False    
    
# Data preprocessing
# data_header = [
#     'sensor', 
#     'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
#     'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
# ]

data_header = ['sensor','dist_to_central_station', 'year',
       'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week', 'hour',
       'complete_timestamp(YYYY_M_DD_HH_M)', 'barometer_hpa', 'temp_centr',
       'hightemp_centr', 'lowtemp_centr', 'hum', 'dewpoint__c', 'wetbulb_c',
       'windspeed_km_h', 'windrun_km', 'highwindspeed_km_h', 'windchill_c',
       'heatindex_c', 'thwindex_c', 'thswindex_c', 'rain_mm', 'rain_rate_mm_h',
       'solar_rad_w_m_2', 'solar_energy_ly', 'high_solar_rad_w_m_2', 'ET_Mm',
       'heating_degree_days', 'cooling_degree_days', 'humidity_rh',
       'solar_klux', 'temp_to_estimate']

def localize_row(df_base,state,time_stamp):
    row = df_base[df_base['complete_timestamp(YYYY_M_DD_HH_M)']== 	time_stamp]   
    new_column_header = 'sensor'
    new_column_value = state[0]    
    # Add the new column to the DataFrame
    row = row.assign(**{new_column_header: new_column_value})
    row= row[data_header]
    return row
    
def sort_key(entry):
    timestamp_str = entry[3]  # Assuming timestamp is always at the 4th position
    timestamp_obj = datetime.strptime(timestamp_str, '%Y_%m_%d_%H_%M')
    return timestamp_obj

def analyze_result_per_POIs(resul):   
    predicted_values = [item[1] for item in resul]
    actual_values = [item[2] for item in resul]
    print('---Results For All ')
    compute_metrics(np.array(predicted_values), np.array(actual_values))
    for POI in range(1,8):
        result_items = [item for item in result if item[0] == POI]
        result_items = sorted(result_items, key=sort_key)
        predicted_values = [item[1] for item in result_items]
        actual_values = [item[2] for item in result_items]
        print('---Results For POI ',str(POI))
        compute_metrics(np.array(predicted_values), np.array(actual_values))
        plot_result_for_each_POI(actual_values,predicted_values,POI,result_items)

def change_lable(D):
    year=int(D[0])
    month=int(D[1])
    day=int(D[2])
    hours = int(D[3])
    minutes = int(D[4]) 
    # Map minutes to quarters
    minute_mapping = {
        0: 0,
        1: 15,
        2: 30,
        3: 45
    }
    # Get the corresponding minute value
    mapped_minutes = minute_mapping[minutes]
    # Format the result
    result = f"{year:04d}/{month:02d}/{day:02d}_{hours:02d}:{mapped_minutes:02d}"
    return result
            

 
def plot_result_for_each_month(actual_values,prediction_values,df):
    df=pd.DataFrame(df)
    unique_months = df[2].unique()
    for month in unique_months:
        plt.figure(figsize=(12, 6))        
        # Filter data for the specific month
        month_data = df[df.iloc[:, 2] == month]
        month_indices = month_data.index.values        
        # Index the actual and prediction values using the month indices
        month_actual_values = actual_values[:, month_indices]
        month_prediction_values = prediction_values[:, month_indices]    
        plt.scatter(range(len(month_actual_values.flatten())), month_actual_values.flatten(), label='Actual Values', marker='o', alpha=0.7)
        plt.scatter(range(len(month_prediction_values.flatten())), month_prediction_values.flatten(), label='Predicted Values', marker='x', alpha=0.7)
        plt.plot(month_actual_values.flatten(), label='_nolegend_', linestyle='-', color='blue', alpha=0.5)
        plt.plot(month_prediction_values.flatten(), label='_nolegend_', linestyle='-', color='orange', alpha=0.5)
        plt.xlabel('Data Points')
        plt.ylabel('Temperature')
        plt.title(f'Actual vs Predicted Values - Month {int(month)}')
        plt.legend()
        plt.show()
       
def compute_metrics(predicted_values,actual_values):
    # Convert lists to NumPy arrays
    predicted_values = np.array(predicted_values)
    actual_values = np.array(actual_values)
    # Calculate loss, MSE, MAE, and MAPE
    loss = np.sum((predicted_values - actual_values) ** 2)
    mse = np.mean((predicted_values - actual_values) ** 2)
    mae = np.mean(np.abs(predicted_values - actual_values))
    mape = np.mean(np.abs((actual_values - predicted_values) / actual_values)) * 100

    # Print the results
    print(f"==============Results===============")
    print(f"Loss: {loss}")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"MAPE: {mape}%")
    
def create_sequences(dataset, seq_length):
    x = []
    y = []
    for i in range(len(dataset)):
        x.append(dataset[i][:seq_length - 1])
        y.append(dataset[i][seq_length - 1:])
    return np.array(x), np.array(y)

def extract_timestamp(x_test_temp):
    y= int(x_test_temp[1])
    m= int(x_test_temp[2])
    d= int(x_test_temp[5])
    h= int(x_test_temp[7])
    t= int(x_test_temp[8])
    array_test_secondry=[]
    time_stamp=str(y)+'_'+str(m)+'_'+str(d)+'_'+str(h)+'_'+str(t)
    return time_stamp  # Just as an example

def create_LSTM_model():
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    model.add(Dropout(dropout))
    model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
    model.add(Dropout(dropout))
    model.add(Dense(units=1))
    custom_optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
    return model



def read_DB(db_path):
    # Load the data
    df = pd.read_csv(db_path)
    index = data_header.index('complete_timestamp(YYYY_M_DD_HH_M)')
    df.dropna(inplace=True)

    # Normalize data
    scalers = {}
    if normalize_flag:
        columns_to_scale = ['temp_to_estimate','temp_centr']
        for column in columns_to_scale:
            scaler = MinMaxScaler(feature_range=(0,1))
            df[column] = scaler.fit_transform(df[[column]])
            scalers[column] = scaler
        
    data = df[data_header].values
    for i in range(len(data)):
        data[i][index] = data[i][index][len(data[i][index]) - 1]
    data = np.asarray(data).astype(np.float32)
    x, y = create_sequences(data, len(data_header))
    original_x,original_y = x.copy(),y.copy()
    # Reshape the data
    samples = int(x.shape[0] / timesteps)
    x = x[:samples * timesteps].reshape(samples, timesteps, x.shape[1])
    y = y[:samples * timesteps].reshape(samples, timesteps, 1)
    return x,y,original_x,original_y
    

def plot_result_for_each_POI(result):
    # Sample data
    sample_num = len(result)
    result_items = [change_lable(item[0].split('_')) for item in result][:sample_num]
    actual_values = [item[1] for item in result[:sample_num]]
    lstm_values = [item[2] for item in result[:sample_num]]
    lstmcnn_values = [item[3] for item in result[:sample_num]]
    cnnlstm_values = [item[4] for item in result[:sample_num]]
    cnnParrarellstm_values = [item[5] for item in result[:sample_num]]

    # Plotting as a bar chart with different colors for each prediction type
    bar_width = 0.15
    index = np.arange(sample_num)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot bars with distinct colors and labels
    bar1 = ax.bar(index, actual_values, bar_width, label='Actual Values', color='dodgerblue')
    bar2 = ax.bar(index + bar_width, lstm_values, bar_width, label='LSTM', color='deeppink')
    bar3 = ax.bar(index + 2 * bar_width, lstmcnn_values, bar_width, label='LSTM-CNN', color='orange')
    bar4 = ax.bar(index + 3 * bar_width, cnnlstm_values, bar_width, label='CNN-LSTM', color='green')
    bar5 = ax.bar(index + 4 * bar_width, cnnParrarellstm_values, bar_width, label='CNN-Parallel-LSTM', color='purple')
    
    # Add labels on top of the bars with diagonal rotation
    def add_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2.0, height + 0.5, f'{value:.1f}', 
                    ha='center', va='bottom', fontsize=12, weight='bold', rotation=75)

    add_labels(bar1, actual_values)
    add_labels(bar2, lstm_values)
    add_labels(bar3, lstmcnn_values)
    add_labels(bar4, cnnlstm_values)
    add_labels(bar5, cnnParrarellstm_values)
    
    # Other plot configurations
    ax.set_ylabel('Temperature (°C)', fontsize=13, weight='bold')
    ax.set_title('Temperature Predictions for Different Models on Random Dates', fontsize=19, weight='bold')
    ax.set_xticks(index + 2 * bar_width)
    ax.set_xticklabels(result_items, rotation=45, ha='right', fontsize=13, weight='bold')
    ax.legend(fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(0, max(max(actual_values), max(lstm_values), max(lstmcnn_values), max(cnnlstm_values), max(cnnParrarellstm_values)) + 5)
    plt.tight_layout()
    plt.show()


DB_file = "data/DB/train_DB.csv"
test_file = "data/DB/test_DB.csv"
model_file = 'temperature_prediction_model'
base_root = 'data/Test/'


model_types = ['LSTM', 'CNN_LSTM','LSTM_CNN','Parrarel_LSTM_CNN']

if train:
    # Load the data
    x,y,_,_ = read_DB(DB_file)
    modelClass = ModelClass(timesteps, len(data_header)-1, dropout, learning_rate)
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)

    for model_type in model_types:
        for i in range(5):
            
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
                
            checkpointer = ModelCheckpoint(filepath = root + model_file+str(i)+'.hdf5', verbose = 0, save_best_only = True) 
            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
            
            print (model_type)
            # Train the model       
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,verbose=0, validation_split=validation_split,callbacks = [checkpointer,early_stopping])
            
            # Visualize the training loss
            plt.plot(history.history['loss'], label='Training Loss for: '+model_type)
            plt.plot(history.history['val_loss'], label='Validation Loss for: '+model_type)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
            
            # Visualize the training loss
            plt.plot(history.history['mse'], label='Training MSE for: '+model_type)
            plt.plot(history.history['val_mse'], label='Validation MSE for: '+model_type)
            plt.xlabel('Epochs')
            plt.ylabel('MSE')
            plt.legend()
            plt.show()
            
            # Visualize the training loss
            plt.plot(history.history['mae'], label='Training MAE for: '+model_type)
            plt.plot(history.history['val_mae'], label='Validation MAE for: '+model_type)
            plt.xlabel('Epochs')
            plt.ylabel('MAE')
            plt.legend()
            plt.show()
            
            plt.plot(history.history['mape'], label='Training MAPE for: '+model_type)
            plt.plot(history.history['val_mape'], label='Validation MAPE for: '+model_type)
            plt.xlabel('Epochs')
            plt.ylabel('MAPE')
            plt.legend()
            plt.show()
            
            # Evaluate the model on the test set
            test_results = model.evaluate(x_test, y_test)
            print('--------------Train results For : '+ model_type +' -----------------')
            print(test_results)
    
            with open(root+'x_test'+str(i), 'wb') as fout:
                pickle.dump(x_test, fout)
            fout.close()
            
            with open(root+'y_test'+str(i), 'wb') as fout:
                pickle.dump(y_test, fout)
            fout.close()
            
            x_test,y_test,original_x,original_y = read_DB(test_file)
            test_results = model.evaluate(x_test, y_test)
            print('--------------Test results For : '+ model_type +' -----------------')
            print(test_results)
            
            with open(root+'outputResultFor5runs.txt', 'a') as file:
                # Iterate over the list and write each item to the file
                for item in test_results:
                    file.write(str(item)+'__')
                file.write('\n')
                
else:
      
    x_test,y_test,original_x,original_y = read_DB(test_file)   
    # Generate random indices
    random_indices = np.random.choice(original_x.shape[0], 100, replace=False)    
    # Select the random rows from original_x and original_y
    sampled_x = original_x[random_indices]
    sampled_y = original_y[random_indices]
    result=list()  # time, label, lstm predicted value, LSTM_CNN predicted value ,CNN_LSTM predicted value, Parrarel_LSTM_CNN predicted value
        
        
    for model_type in model_types:
        
        if model_type == 'LSTM':
            root = base_root + 'Test_LSTM/'
            lstm_model = load_model(root + 'temperature_prediction_model2.hdf5')
            
        elif model_type == 'LSTM_CNN':
            root = base_root + 'Test_LSTM_FollowedBy_CNN/'
            LSTM_CNN_model = load_model(root + 'temperature_prediction_model4.hdf5')
            
        elif model_type == 'CNN_LSTM':
            root = base_root + 'Test_CNN_FollowedBy_LSTM/'
            CNN_LSTM_model = load_model(root + 'temperature_prediction_model3.hdf5')
            
        elif model_type == 'Parrarel_LSTM_CNN':
            root = base_root + 'Test_CNN_Parrarel_LSTM/'
            Parrarel_LSTM_CNN_model = load_model(root + 'temperature_prediction_model1.hdf5')
            
        
        

        
        
    for item,label in zip(sampled_x,sampled_y):
        
        x = np.tile(item, (1, 3, 1))
        predict_lstm =lstm_model.predict(x)[0][0]
        predict_LSTM_CNN =LSTM_CNN_model.predict(x) [0][0]
        predict_CNN_LSTM =CNN_LSTM_model.predict(x) [0][0]
        predict_Parrarel =Parrarel_LSTM_CNN_model.predict(x) [0][0]
        
        result.append([extract_timestamp(item),label[0],predict_lstm,predict_LSTM_CNN,predict_CNN_LSTM,predict_Parrarel])
        
    # Calculate differences and add them to the result
    for item in result:
        item.append(abs(item[1] - item[2]))  # Difference with predict_lstm , 6
        item.append(abs(item[1] - item[3]))  # Difference with predict_LSTM_CNN , 7
        item.append(abs(item[1] - item[4]))  # Difference with predict_CNN_LSTM , 8
        item.append(abs(item[1] - item[5]))  # Difference with predict_Parrarel , 9
    
    
    
    top_5_rows = random.sample(result,5)  
    plot_result_for_each_POI(top_5_rows)
    