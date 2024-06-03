import tensorflow as tf
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout,Concatenate , Bidirectional, Conv1D,Flatten , ReLU, Input, Reshape, BatchNormalization, MaxPooling1D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, Adadelta
from tensorflow.keras.regularizers import l2
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle


class ModelClass:
    def __init__(self, timesteps, feature_size,dropout,learning_rate):
        self.timesteps = timesteps
        self.feature_size = feature_size
        self.dropout = dropout
        self.learning_rate = learning_rate
      
    def create_LSTM(self):
        model = Sequential()
        model.add(Bidirectional(LSTM(units=256, return_sequences=True, input_shape=(self.timesteps, self.feature_size))))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(units=64, return_sequences=False)))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))
        custom_optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=custom_optimizer, loss='mean_absolute_percentage_error', metrics=['mape', 'mae', 'mse'])
        return model
        
       
            
        
    

    