# Import necessary modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam

import numpy as np
from sklearn.metrics import accuracy_score


from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle


import random
from datetime import datetime

import tensorflow as tf

train_CNN_LSTM = True
analyze_flag = False
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

class CNN_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(CNN_LSTM, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=10, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=128, out_channels=10, kernel_size=3, stride=1, padding=1),  # Match output channels to input features
            nn.ReLU()
        )
        # hidden_sizes = [256, 128, 64]

        # self.lstm1 = nn.LSTM(10, hidden_sizes[0], batch_first=True, bidirectional=True)
        # self.dropout1 = nn.Dropout(dropout)
        
        # self.lstm2 = nn.LSTM(hidden_sizes[0]*2, hidden_sizes[1], batch_first=True, bidirectional=True)
        # self.dropout2 = nn.Dropout(dropout)
        
        # self.lstm3 = nn.LSTM(hidden_sizes[1]*2, hidden_sizes[2], batch_first=True, bidirectional=True)
        # self.dropout3 = nn.Dropout(dropout)
        
        # self.fc = nn.Linear(hidden_sizes[2]*2, 1)
        
        self.lstm = nn.LSTM(input_size=10, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Permute to match the input shape expected by Conv1d
        x_original = x
        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_len)
        out = self.cnn(x)
        # Reshape output of CNN to match the input shape expected by LSTM
        out = out.permute(0, 2, 1)  # (batch_size, seq_len, num_features)        
        # # Flatten the features to match LSTM input
        out = out.contiguous().view(out.size(0), out.size(1), -1)       
        # LSTM
        C = torch.cat((x_original, out), dim=2)
        # out, _ = self.lstm(C)        
        # batch_size, seq_length, hidden_size = out.shape
        # out = out.reshape(-1, hidden_size)        
        # # Apply the fully connected layer
        # out = self.fc(out)
        
        # # Reshape back to (batch_size, seq_length, output_size)
        # out = out.view(batch_size, seq_length, -1)
        # out, _ = self.lstm1(out)
        # out = self.dropout1(out)
        
        # out, _ = self.lstm2(out)
        # out = self.dropout2(out)
        
        # out, _ = self.lstm3(out)
        # out = self.dropout3(out)
        
        # # Since return_sequences=False in the last LSTM, we take the last output
        # # out = out[:, -1, :]
        # batch_size, seq_length, hidden_size = out.shape
        # out = out.reshape(-1, hidden_size)  
        # out = self.fc(out)
        # out = out.view(batch_size, seq_length, -1)
        # FC
        # out = self.fc(out[:, -1, :])
        out= self.lstm(out)
        out = self.fc(out[:, -1, :])
        
        return out





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
    return x,y,original_x,original_y,data

def train(models:List, train_loader:DataLoader, epochs:int):
    criterion = nn.CrossEntropyLoss()
    for model in models:
        print("Training model: ", model.__class__.__name__)
        
        optimizer = Adam(model.parameters(), lr=learning_rate)
        for epoch in range(epochs):
            model.train()
            for i, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                if (i+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        print("Training completed for model: ", model.__class__.__name__)    

DB_file = "data/DB/train_DB.csv"
test_file = "data/DB/test_DB.csv"
root = 'data/Test/'

if normalize_flag:
    hyper_params_name = 'data/RNN_models/withNormalization/hyper_parameters'
    model_file = 'data/RNN_models/withNormalization/temperature_prediction_model.hdf5'
    best_result_path = 'data/RNN_models/withNormalization/best_trained_data'
else:
    hyper_params_name = 'data/RNN_models/without_Normalization/hyper_parameters'
    model_file = 'data/RNN_models/without_Normalization/temperature_prediction_model.hdf5'
    best_result_path = 'data/RNN_models/without_Normalization/best_trained_data'
        


if train_CNN_LSTM:




    # Load the data
    x,y,x_o,y_o,data = read_DB(DB_file)
    

    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=23)
    
    # Convert the data to torch tensors
    X_train = torch.from_numpy(X_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_train = torch.from_numpy(y_train).float()
    y_test = torch.from_numpy(y_test).float()
   
   
    #Datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    #Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print(device)
    
    input_size = X_train.shape
    hidden_size = 128
    num_layers = 2
    num_classes = len(np.unique(y_train))
    
    
    
    cnn_lstm = CNN_LSTM(input_size, hidden_size, num_layers, num_classes).to(device)
    
    #train
    models = [cnn_lstm]
    num_epochs = 200
    train(models, train_loader, epochs=num_epochs)


   

