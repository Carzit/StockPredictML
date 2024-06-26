import os
import pickle
import logging
from typing import List, Callable

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import utils


def copy_reset(df:pd.DataFrame):
    result = df.copy()
    result.reset_index(drop=True, inplace=True)
    return result

def copy_reset_decorator(func: Callable):
    def wrapper(*args, **kwargs):
        df = func(*args, **kwargs)
        return copy_reset(df=df)
    return wrapper

@copy_reset_decorator
def remove_missing_values(df:pd.DataFrame):
    """
    打印存在缺失值的行index和缺失值所在列名，返回去除所有缺失值所在行后的df。

    Parameters:
    df (pd.DataFrame): 输入的DataFrame

    Returns:
    pd.DataFrame: 去除所有缺失值所在行后的DataFrame
    """
    # 遍历每一行，查找缺失值
    for index, row in df.iterrows():
        missing_columns = row[row.isnull()].index.tolist()
        if missing_columns:
            logging.info(f'Index {index} has missing values in columns: {missing_columns}')
    
    # 去除所有缺失值所在的行
    cleaned_df = df.dropna()
    
    return cleaned_df



if __name__ == "__main__":
    log_dir = "log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(os.path.join(log_dir, "DATE.txt")), logging.StreamHandler()])
    
    logging.info("Loading data...")
    df_stocks = pd.read_csv("data/stocks/ALOT.csv")

    logging.info("Preprosessing")
    df_stocks["Date"] = pd.to_datetime(df_stocks.Date)
    df_stocks.sort_values("Date", inplace=True)

    logging.debug("plotting...")
    utils.numeric_plots(df_stocks, col_names=["Date","Open","High","Low","Close","Volume"], save_path="save.png")

    df_stocks["High_s1"] = df_stocks["High"].shift(1)
    df_stocks["Low_s1"] = df_stocks["Low"].shift(1)
    df_stocks["Close_s1"] = df_stocks["Close"].shift(1)
    df_stocks["Volume_s1"] = df_stocks["Volume"].shift(1)
    df_stocks["delta_s1"] = (df_stocks["High"] - df_stocks["Low"]).shift(1)
    df_stocks = remove_missing_values(df_stocks)
    df_stocks["Volume_s1"] = np.log(df_stocks['Volume_s1'])
    df_stocks.replace(-np.inf, 0, inplace=True)
    print(df_stocks)

    
    

    logging.debug("Saved...")
    df_stocks.to_csv("data\\preprocessed_data.csv")



