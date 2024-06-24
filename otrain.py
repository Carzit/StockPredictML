import os
import logging

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

if __name__ == "__main__":
    log_dir = "log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_dir, "DATE.txt")), logging.StreamHandler()])
    
    logging.info("Loading data...")
    df_stocks = pd.read_csv("data/processed_data.csv")


    # 数据准备
    X = df_stocks[['Open', 'High', 'Low', 'Volume']].values
    y = df_stocks['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 线性回归
    logging.info("Doing Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)

    # 支持向量机
    logging.info("Doing SVR (GridSearch)")
    svr = SVR()
    svr_params = {
        "kernel": ['linear', 'poly', 'rbf', 'sigmoid'],
        "C": [0.1, 1, 5, 10, 50, 100, 1000, 10000],
        "gamma": [0.1, 1, 5, 10]
    }
    svr = GridSearchCV(svr, svr_params, cv=10, verbose=4*8*4)
    svr.fit(X_train_scaled, y_train)
    y_pred_svr = svr.predict(X_test_scaled)


    # 随机森林
    logging.info("Doing Random Forest")
    rf = RandomForestRegressor()
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)


    # 性能比较
    logging.info("Report")
    print('Linear Regression RMSE:', mean_squared_error(y_test, y_pred_lr, squared=False))
    print('SVR RMSE:', mean_squared_error(y_test, y_pred_svr, squared=False))
    print('Random Forest RMSE:', mean_squared_error(y_test, y_pred_rf, squared=False))