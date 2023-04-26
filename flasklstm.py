import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense





# Define the number of days to use in the LSTM model
n_days = 10


# Scale the data to be between 0 and 1
scaler_close = MinMaxScaler(feature_range=(0, 1))

# Scaling values for the High 
scaler_high = MinMaxScaler(feature_range=(0, 1))


# Scaling values for the Low 
scaler_low = MinMaxScaler(feature_range=(0, 1))


# Scaling values for the Open 
scaler_open = MinMaxScaler(feature_range=(0, 1))



# Load the Models
from tensorflow.keras.models import load_model
# For closing
model_lstm_close = load_model('my_model.h5')
# For High 


# Select the last 20 values from the "close" column




def lstm_close_predict(input_data):
  
  # Scale the input data using the same scaler used for training
  scaled_input = scaler_close.transform(input_data.reshape(-1, 1))
  #print(scaled_input)

  # Reshape the input data for LSTM
  input_data = np.reshape(scaled_input, (1, n_days, 1))
  #print(input_data)
  # Get the model's predicted price value for the input data
  prediction = model_lstm_close.predict(input_data)

  # Unscale the predicted value
  prediction = scaler_close.inverse_transform(prediction)

  # Print the predicted value
  print("predicted value=",prediction[0][0])
  return prediction[0][0]


