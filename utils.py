import os
import pickle
from typing import List

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def numeric_plots(df:pd.DataFrame, col_names:List[str], save_path=None):
    colors = sns.color_palette("husl", len(col_names))
    fig, axes = plt.subplots(len(col_names), 3, figsize=(15, len(col_names)*5))
    for i, col_name in enumerate(col_names):
        sns.scatterplot(df[col_name], ax=axes[i, 0], color=colors[i])
        axes[i, 0].set_title(f'{col_name}_Scatterplot')
        axes[i, 0].set_ylabel(col_name)

        sns.boxplot(df[col_name], ax=axes[i, 1], orient='h', color=colors[i])
        axes[i, 1].set_title(f'{col_name  }_Boxplot')
        axes[i, 1].set_ylabel(col_name)

        sns.histplot(df[col_name], ax=axes[i, 2], color=colors[i])
        axes[i, 2].set_title(f'{col_name}_Histogram')
        axes[i, 2].set_ylabel(col_name)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def window_forward_plot(df, model_name="Linear Regression", save_path=None):
    forecast = df[['Date']].copy()
    forecast['yhat'] = df['Predict']
    forecast['yhat_lower'] = forecast['yhat'] - 10
    forecast['yhat_upper'] = forecast['yhat'] + 10

    plt.figure(figsize=(10, 6))

    # Plot actual values
    plt.scatter(df['Date'], df['Close'], color='blue', label='Actual')

    # Plot predicted values
    plt.plot(forecast['Date'], forecast['yhat'], color='orange', label='Predicted')

    # Plot confidence interval
    plt.fill_between(forecast['Date'], forecast['yhat_lower'], forecast['yhat_upper'], 
                     color='gray', alpha=0.2, label='Confidence Interval')

    # Title and labels
    plt.title(f"{model_name} Forecast with Confidence Interval")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

def mse(series1:pd.Series, series2:pd.Series):
    squared_errors = (series1 - series2) ** 2
    mse = squared_errors.mean()
    return mse

def window_mse(df:pd.DataFrame):
    # Ensure 'predict' and 'Close' columns exist
    if 'Predict' not in df.columns or 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Predict' and 'Close' columns")
    
    return mse(df["Predict"], df['Close'])

def serialize(obj, path:str, binary:bool=True):
    if not path.endswith(".pkl"):
        path = path + ".pkl"
    mode = "wb" if binary else "w"
    
    with open(path, mode=mode) as file:
        pickle.dump(obj, file)