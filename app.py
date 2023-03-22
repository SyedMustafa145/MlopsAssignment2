import pickle
import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request
#import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

scaler = MinMaxScaler(feature_range=(0, 1))

app = Flask(__name__)

# load the trained model from the pickle file
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['GET'])
def predict():
    url = 'https://www.alphavantage.co/query?function=FX_DAILY&from_symbol=EUR&to_symbol=USD&apikey=20MAD65B9CGJYW3K'
    r = requests.get(url)
    data = r.json()

    # extract the Time Series FX (Daily) data from the JSON response
    daily_data = data['Time Series FX (Daily)']
    # convert the daily_data dictionary into a pandas DataFrame
    df = pd.DataFrame.from_dict(daily_data, orient='index')
    # reset the index to make the date column a regular column
    df = df.reset_index()
    # rename the columns to more descriptive names
    df = df.rename(
        columns={'index': 'Date', '1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close'})
    # convert the Open, High, Low, and Close columns from strings to floats
    df['Open'] = df['Open'].astype(float)
    df['High'] = df['High'].astype(float)
    df['Low'] = df['Low'].astype(float)
    df['Close'] = df['Close'].astype(float)

    # get the input data for prediction from the first 10 rows of the DataFrame
    input_data = df.head(10)['Open'].values
    input_data = input_data.reshape(1, -1)

    # make a prediction using the model
    predictions = model.predict(input_data)
    predictions = scaler.inverse_transform(predictions)

    # create a response object
    response = {
        'prediction': predictions[0][0]
    }

    # return the response as a JSON object
    return jsonify(response)


@app.route('/')
# ‘/’ URL is bound with hello_world() function.
def hello_world():
    return 'Hello World'


if __name__ == '__main__':
    app.run(debug=True)
