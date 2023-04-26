import json

from collections import namedtuple
from json import JSONEncoder
from flask import Flask,request
from flasklstm import lstm_open_predict, lstm_low_predict, lstm_close_predict, lstm_high_predict
from flask_cors import CORS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
app = Flask(__name__)

CORS(app)
df = pd.read_csv('eurusd.csv')
data_close = df.filter(['close'])
data_high = df.filter(['high'])
data_low = df.filter(['low'])
data_open = df.filter(['open'])
import json
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/predictclose")
def predictclose():
    # get the last 10 closing prices from the request arguments
    cp1 = float(request.args.get('cp1'))
    cp2 = float(request.args.get('cp2'))
    cp3 = float(request.args.get('cp3'))
    cp4 = float(request.args.get('cp4'))
    cp5 = float(request.args.get('cp5'))
    cp6 = float(request.args.get('cp6'))
    cp7 = float(request.args.get('cp7'))
    cp8 = float(request.args.get('cp8'))
    cp9 = float(request.args.get('cp9'))
    cp10 = float(request.args.get('cp10'))

    last_10_close = np.array([cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10])

    # call the LSTM model to predict the next closing price
    result_close = np.round(lstm_close_predict(last_10_close), 5) 
    
    print("Calling from server.py")
    print(result_close)
    
    # create a dictionary containing the arrays
    response = {
        "result": result_close.round(5).tolist()
    }
    
    # convert the dictionary to a JSON string and return it as the response
    return json.dumps(response)






@app.route("/predict")
def predict():
    last_20_close = df['close'].tail(10).values

   
