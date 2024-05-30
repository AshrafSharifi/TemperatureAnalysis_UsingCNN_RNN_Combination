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



train = False
analyze_flag = True
test_flag = False 
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
data_header = [
    'sensor', 
    'year', 'month', 'week', 'day_of_year', 'day_of_month', 'day_of_week',
    'hour', 'complete_timestamp(YYYY_M_DD_HH_M)', 'temp_centr', 'temp_to_estimate'
]

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
            
def plot_result_for_each_POI(actual_values,prediction_values,POI,result):
    # Sample data
    sample_num = 10
    result_items = [change_lable(item[3].split('_')) for item in result][:sample_num]
    actual_values = actual_values[:sample_num]  # Extract values from the arrays
    prediction_values = prediction_values[:sample_num]   
    # Calculate the differences between actual and predicted values
    differences = np.array(actual_values) - np.array(prediction_values)  
    # Plotting as a bar chart with lines connecting actual and predicted values
    bar_width = 0.25
    index = np.arange(sample_num)
    fig, ax = plt.subplots()
    # Plot bars and set colors
    bar1 = ax.bar(index, actual_values, bar_width, label='Actual Values', linestyle='-', color='dodgerblue')
    bar2 = ax.bar(index + bar_width, prediction_values, bar_width, label='Predicted Values', color='deeppink')
    # Color specific bars differently
    # bar1[0].set_color('darkpink')  # Change the color of the first bar to dark pink
       # Add labels on top of the bars
    for i, v in enumerate(actual_values):
        ax.text(i, v + 0.6, str(round(v, 1)), color='blue', ha='center',rotation=90, va='bottom',  fontsize=8,weight='bold')
    
    for i, v in enumerate(prediction_values):
        ax.text(i + bar_width, v + 0.6, str(round(v, 1)), color='m', ha='center',rotation=90, va='bottom',  fontsize=8,weight='bold')

    # Other plot configurations
    ax.set_ylabel('Temperature °C')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(result_items, rotation=45, ha='right')
    ax.legend()
    plt.ylim(0, 35)
    plt.show()
    S=1
 
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

def create_CNN_model(hidden_size,num_classes):
    model = Sequential()
    model.add(Conv1D(64, 3, padding='same', input_shape=(hidden_size,)))
    model.add(ReLU())
    model.add(MaxPooling1D(2, 2))
    model.add(Conv1D(128, 3, padding='same'))
    model.add(ReLU())
    model.add(MaxPooling1D(2, 2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='linear'))
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
    

DB_file = "data/DB/train_DB.csv"
test_file = "data/DB/test_DB.csv"
root = 'data/Test/Test_LSTM/'

if normalize_flag:
    hyper_params_name = 'data/RNN_models/withNormalization/hyper_parameters'
    model_file = 'data/RNN_models/withNormalization/temperature_prediction_model.hdf5'
    best_result_path = 'data/RNN_models/withNormalization/best_trained_data'
else:
    hyper_params_name = 'data/RNN_models/without_Normalization/hyper_parameters'
    model_file = 'data/RNN_models/without_Normalization/temperature_prediction_model.hdf5'
    best_result_path = 'data/RNN_models/without_Normalization/best_trained_data'
        


if train:




    # Load the data
    x,y,_,_ = read_DB(DB_file)
    
    # Split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    # Build the model
    model = create_LSTM_model()
    # model.summary()
    checkpointer = ModelCheckpoint(filepath = model_file, verbose = 2, save_best_only = True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    # Train the model       
    history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split,callbacks = [checkpointer,early_stopping])
    
    # Visualize the training loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Visualize the training loss
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    # Visualize the training loss
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.show()
    
    plt.plot(history.history['mape'], label='Training MAPE')
    plt.plot(history.history['val_mape'], label='Validation MAPE')
    plt.xlabel('Epochs')
    plt.ylabel('MAPE')
    plt.legend()
    plt.show()
    
    # Evaluate the model on the test set
    test_results = model.evaluate(x_test, y_test)
    print('--------------Test results:-----------------')
    print(test_results)

    
    
    

   
                
                
    with open(root+'x_test', 'wb') as fout:
        pickle.dump(x_test, fout)
    fout.close()
    
    with open(root+'y_test', 'wb') as fout:
        pickle.dump(y_test, fout)
    fout.close()
    

    
    # # Make predictions and denormalize
    # y_pred = model.predict(x_test)
    # print("Prediction, real value")
    # initial_x_test=np.array([]).reshape(0, x_test.shape[2]) 
    # prediction_values=[]
    # actual_values=[]
    # result=list()
    # for i in range(len(y_pred)):
    #    temp=(x_test[i][timesteps-1].reshape(-1, 1)).T
    #    initial_x_test = np.vstack([initial_x_test, np.array(temp[0])])
  
    #    if normalize_flag:

elif test_flag:
    
        model = load_model(model_file)
        result=list()
        x_test,y_test,original_x,original_y = read_DB(test_file)
        test_results = model.evaluate(x_test, y_test)
        print('--------------Test results:-----------------')
        print(test_results)
        
        y_pred = model.predict(x_test)

    # Flatten the predictions and actual values for easier comparison
        y_pred_flat = y_pred.reshape(-1)
        y_test_flat = y_test.reshape(-1)
        
        # predicted_values =model.predict(x_test) 
        # result = [[original_x[i,0],model.predict(np.tile(original_x[i], (1, 3, 1))), original_y[i], extract_timestamp(original_x[i])] for i in range(len(original_x))]
        
        for j in range(0,original_x.shape[0]):
            print(j)
            x=np.tile(original_x[j], (1, 3, 1))
            samples = int(x.shape[0] / timesteps)
            x = x[:samples * timesteps].reshape(samples, timesteps, x.shape[1])
            predicted_values =model.predict(np.tile(original_x[j], (1, 3, 1))) 
            result.append([original_x[j,0],predicted_values[0][0], original_y[j][0], extract_timestamp(original_x[j])])
            
        with open(root+'prediction_result', 'wb') as fout:
            pickle.dump(result, fout)
        fout.close()  
        
        prediction_values = [item[1] for item in result]
        actual_values = [item[2] for item in result]
        


        # actual_values = np.reshape(actual_values, (1,len(actual_values)))
        # prediction_values = np.reshape(prediction_values, (1,len(prediction_values)))
        compute_metrics(prediction_values,actual_values)
        # plot_result_for_each_POI(actual_values,prediction_values,result)
        analyze_result_per_POIs(result)
        
elif analyze_flag:
    
        model = load_model(model_file)
        result=list()
        x_test,y_test,original_x,original_y = read_DB(test_file)
        test_results = model.evaluate(x_test, y_test)
        print('--------------Test results:-----------------')
        print(test_results)
        
    
        # with open(root+'prediction_result', 'rb') as fin:
        #     result = pickle.load(fin)
        # fin.close()
        # prediction_values = [item[1] for item in result]
        # actual_values = [item[2] for item in result]
        # # actual_values = np.reshape(actual_values, (1,len(actual_values)))
        # # prediction_values = np.reshape(prediction_values, (1,len(prediction_values)))
        # compute_metrics(prediction_values,actual_values)
        # # plot_result_for_each_POI(actual_values,prediction_values,result)
        # analyze_result_per_POIs(result)
    
    
