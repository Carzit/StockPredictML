import os
import logging
import argparse
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
    for index, row in df.iterrows():
        missing_columns = row[row.isnull()].index.tolist()
        if missing_columns:
            logging.info(f'Index {index} has missing values in columns: {missing_columns}')
    cleaned_df = df.dropna()
    
    return cleaned_df

def parse_args():
    parser = argparse.ArgumentParser(description="Data Preprocessor")
    parser.add_argument("--data", type=str, required=True, help="Path of raw Data csv file")
    parser.add_argument("--save", type=str, required=True, help="Path of processed csv file to be saved")
    parser.add_argument("--log", type=str, default=os.path.join("log", "data.txt"), help="Path of log file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - [%(levelname)s] : %(message)s',
        handlers=[logging.FileHandler(args.log), logging.StreamHandler()])
    
    logging.debug("Loading data...")
    df_stocks = pd.read_csv(args.data)

    logging.debug("Preprosessing")
    df_stocks["Date"] = pd.to_datetime(df_stocks.Date)
    df_stocks.sort_values("Date", inplace=True)

    logging.debug("Plotting...")
    utils.numeric_plots(df_stocks, col_names=["Date","Open","High","Low","Close","Volume"], save_path="eda.png")

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
    df_stocks.to_csv(args.save)



