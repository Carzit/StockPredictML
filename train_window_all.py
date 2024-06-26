import os
import logging
import argparse

import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

import utils

def window_forward(model, df:pd.DataFrame, window_days:int=40, test_days:int=2):
    """
    window_days: 总窗口天数，包括用于训练和用于预测的
    test_days: 用于预测的天数

    返回一个预测的Close Price的DataFrame序列
    """

    features = ["Open", "High_s1", "Low_s1", "Close_s1", "delta_s1"]
    target = "Close"
    
    res_ls = []
    
    for i in tqdm(range(0, len(df), test_days)):        
        df_train = df.iloc[:i+window_days-test_days]
        df_test = df.iloc[i+window_days-test_days:i+window_days].copy()

        

        if df_train.empty or df_test.empty:
            break

        scaler = StandardScaler()
        X_train = df_train[features].values
        X_test = df_test[features].values
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, df_train[target].values)
        df_test["Predict"] = model.predict(X_test_scaled)
        res_ls.append(df_test)
            
    return pd.concat(res_ls)

def parse_args():
    parser = argparse.ArgumentParser(description="Window Model Training and Prediction")
    parser.add_argument("--data", type=str, default="data\\processed_data.csv", help="Data file path")
    parser.add_argument("--log", type=str, default="log\\log.txt", help="Log file path")
    parser.add_argument("--window_len", type=int, default=40, help="Window day num. Default 40")
    parser.add_argument("--test_len", type=int, default=2, help="Test day num. Default 2")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    
    logging.debug("Loading data...")
    df_stocks = pd.read_csv(args.data)
    df_stocks["Date"] = pd.to_datetime(df_stocks.Date)

    logging.info(f"Cumulative Sliding Window")

    logging.info(f"Window Day Num: {args.window_len}, Test Day Num: {args.test_len}")

    # Linear Regression
    logging.debug("Doing Window Linear Regression")
    linear_regression_model = LinearRegression()
    df_lr = window_forward(linear_regression_model, df_stocks, window_days=args.window_len, test_days=args.test_len)
    mse_lr = utils.window_mse(df_lr)
    logging.info(f"Linear Regression MSE: {mse_lr}")
    utils.window_forward_plot(df_lr[df_lr.Predict.abs() < 1000], "Linear Regression", save_path="plots/LinearRegression.png")

    # Ridge
    logging.debug("Doing Window Ridge")
    ridge_model = Ridge()
    df_ridge = window_forward(ridge_model, df_stocks, window_days=args.window_len, test_days=args.test_len)
    mse_ridge = utils.window_mse(df_ridge)
    logging.info(f"Ridge MSE: {mse_ridge}")
    utils.window_forward_plot(df_ridge[df_ridge.Predict.abs() < 1000], "Ridge", save_path="plots/Ridge.png")

    # ElasticNet
    logging.debug("Doing Window Elastic Net")
    elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=1)
    df_en = window_forward(elastic_net_model, df_stocks, window_days=args.window_len, test_days=args.test_len)
    mse_en = utils.window_mse(df_en)
    logging.info(f"Elastic Net MSE: {mse_en}")
    utils.window_forward_plot(df_en[df_en.Predict.abs() < 1000], "Elastic Net", save_path="plots/ElasticNet.png")

    # SVR
    logging.debug("Doing Window SVR")
    svr_model = SVR(C=0.1, gamma=0.1, kernel="linear")
    df_svr = window_forward(svr_model, df_stocks, window_days=args.window_len, test_days=args.test_len)
    mse_svr = utils.window_mse(df_svr)
    logging.info(f"SVR MSE: {mse_svr}")
    utils.window_forward_plot(df_svr[df_svr.Predict.abs() < 1000], "SVR", save_path="plots/SVR.png")

    # Random Forest
    logging.debug("Doing Window Random Forest")
    rf_model = RandomForestRegressor(50, n_jobs=4)
    df_rf = window_forward(rf_model, df_stocks, window_days=args.window_len, test_days=args.test_len)
    mse_rf = utils.window_mse(df_rf)
    logging.info(f"Random Forest MSE: {mse_rf}")
    utils.window_forward_plot(df_rf[df_rf.Predict.abs() < 1000], "Random Forest", save_path="plots/Random Forest.png")
