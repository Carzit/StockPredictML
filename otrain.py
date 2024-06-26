import os
import logging

import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

import utils

if __name__ == "__main__":
    log_dir = "log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_dir, "DATE.txt")), logging.StreamHandler()])
    
    logging.info("Loading data...")
    df_stocks = pd.read_csv("data/processed_data.csv")

    logging.info(f"Fixed Data Partitioning")

    # 数据准备
    X = df_stocks[["Open", "High_s1", "Low_s1", "Close_s1", "delta_s1"]].values
    y = df_stocks['Close'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # 数据标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    # 线性回归
    logging.info("Doing Linear Regression")
    linear_regression_model = LinearRegression()
    linear_regression_model.fit(X_train_scaled, y_train)
    y_pred_linear_regression = linear_regression_model.predict(X_test_scaled)
    

    # 弹性网回归
    logging.info("Doing Elastic Net Regression")
    elastic_params = {'alpha': [0.1, 1.0, 10.0, 50.0], 
                      'l1_ratio': [0.1, 0.5, 0.7, 1.0]}
    elastic_net = ElasticNet()
    elastic_net_gridsearch= GridSearchCV(estimator=elastic_net, param_grid=elastic_params, cv=5, verbose=4*4, n_jobs=4)
    elastic_net_gridsearch.fit(X_train_scaled, y_train)
    logging.debug(f"Best parameters found: {elastic_net_gridsearch.best_params_}")
    logging.debug(f"Best cross-validation score: {elastic_net_gridsearch.best_score_}")
    elastic_net  = elastic_net_gridsearch.best_estimator_
    y_pred_elastic_net = elastic_net.predict(X_test_scaled)

    # 支持向量机
    logging.info("Doing SVR (GridSearch)")
    svr = SVR()
    svr_params = {
        "kernel": ['linear', 'rbf'],
        "C": [0.1, 1.0, 5.0, 10.0, 50.0],
        "gamma": [0.1, 1.0, 5.0, 10.0]
    }
    svr_gridsearch = GridSearchCV(svr, svr_params, cv=5, verbose=2*5*4, n_jobs=4)
    svr_gridsearch.fit(X_train_scaled, y_train)
    logging.debug(f"Best parameters found: {svr_gridsearch.best_params_}")
    logging.debug(f"Best cross-validation score: {svr_gridsearch.best_score_}")
    svr = svr_gridsearch.best_estimator_
    y_pred_svr = svr.predict(X_test_scaled)

    # 随机森林
    logging.info("Doing Random Forest")
    random_forest = RandomForestRegressor()
    random_forest.fit(X_train_scaled, y_train)
    y_pred_rf = random_forest.predict(X_test_scaled)

    # Adaboost
    logging.info("Doing AdaBoost")
    adaboost = AdaBoostRegressor()
    adaboost.fit(X_train_scaled, y_train)
    y_pred_adaboost = adaboost.predict(X_test_scaled)

    # 性能比较
    logging.info("Report")
    logging.info(f"Linear Regression RMSE: {mean_squared_error(y_test, y_pred_linear_regression)}")
    logging.info(f"Elastic Net RMSE: {mean_squared_error(y_test, y_pred_elastic_net)}")
    logging.info(f"SVR RMSE: {mean_squared_error(y_test, y_pred_svr)}")
    logging.info(f"Random Forest RMSE: {mean_squared_error(y_test, y_pred_rf)}")
    logging.info(f"AdaBoost RMSE: {mean_squared_error(y_test, y_pred_adaboost)}")

    # 保存模型
    models = {
        "LinearRegression": linear_regression_model,
        "ElasticNet": elastic_net,
        "SVR": svr,
        "RandomForest": random_forest,
        "Adaboost": adaboost
    }
    utils.serialize(models, "best_models.pkl")