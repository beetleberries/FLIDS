import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

#data = pd.read_csv('monthly_milk_production.csv')
#data['Date'] = pd.to_datetime(data['Date'])
#data.set_index('Date', inplace=True)
#production = data['Production'].astype(float).values.reshape(-1, 1)

#scaler = MinMaxScaler(feature_range=(0,1))
#scaled_data = scaler.fit_transform(production)