import os
import pickle
import logging
from typing import List, Callable


import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns




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

def numeric_plots(df:pd.DataFrame, col_names:List[str], save_path=None):
    colors = sns.color_palette("husl", len(col_names))
    fig, axes = plt.subplots(len(col_names), 3, figsize=(15, len(col_names)*5))
    for i, col_name in enumerate(col_names):
        sns.scatterplot(df[col_name], ax=axes[i, 0], color=colors[i])
        axes[i, 0].set_title(f'{col_name}_Scatterplot')
        axes[i, 0].set_xlabel(col_name)

        sns.boxplot(df[col_name], ax=axes[i, 1], orient='h', color=colors[i])
        axes[i, 1].set_title(f'{col_name  }_Boxplot')
        axes[i, 1].set_xlabel(col_name)

        sns.histplot(df[col_name], ax=axes[i, 2], color=colors[i])
        axes[i, 2].set_title(f'{col_name}_Histogram')
        axes[i, 2].set_xlabel(col_name)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()

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
    df_stocks["High_s1"] = df_stocks["High"].shift(1)
    df_stocks["Low_s1"] = df_stocks["Low"].shift(1)
    df_stocks["Close_s1"] = df_stocks["Close"].shift(1)
    df_stocks["delta_s1"] = (df_stocks["High"] - df_stocks["Low"]).shift(1)
    df_stocks = remove_missing_values(df_stocks)

    logging.info("Saved...")
    df_stocks.to_csv("data\\processed_data.csv")



