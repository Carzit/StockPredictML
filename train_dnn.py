import os
import logging

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from train_window import window_forward

import utils

from keras import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation

class LSTM_DNN:
    def __init__(self):
        model = Sequential()
        model.add(LSTM(units = 4, input_shape = (5, 1), return_sequences = True))
        model.add(Dropout(0.1))
        model.add(LSTM(32))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.add(Activation("linear"))
        model.compile(loss="mse", optimizer="adam")
        model.summary()
        self.model = model
        
    def fit(self, features, target):
        self.model.fit(features, target, verbose=0)
    def predict(self, features):
        return self.model.predict(features, verbose=0)



if __name__ == "__main__":
    log_dir = "log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_dir, "DATE.txt")), logging.StreamHandler()])
    
    logging.info("Loading data...")
    df_stocks = pd.read_csv("data/processed_data.csv")
    df_stocks["Date"] = pd.to_datetime(df_stocks.Date)


    scaler = MinMaxScaler()
    df_stocks_scaled = df_stocks.copy().set_index("Date")
    scaler.fit(df_stocks_scaled)
    df_stocks_scaled = pd.DataFrame(scaler.transform(df_stocks_scaled), columns=df_stocks_scaled.columns)
    df_stocks_scaled["Date"] = df_stocks.dropna()["Date"]

    # 线性回归
    logging.info("Doing LSTM")
    lstm_model = LSTM_DNN()
    df_lstm = window_forward(lstm_model, df_stocks)
    scaler.fit(df_stocks[["Close"]])
    df_lstm["Predict"] = scaler.inverse_transform(df_lstm[["Predict"]])
    df_lstm["Close"] = scaler.inverse_transform(df_lstm[["Close"]])
    print(df_lstm)
    mse_lstm = utils.window_mse(df_lstm)
    utils.window_forward_plot(df_lstm[df_lstm.Predict.abs() < 1000], "LSTM", save_path="plots/LSTM.png")

    # Report
    logging.info(f"LSTM MSE: {mse_lstm}")