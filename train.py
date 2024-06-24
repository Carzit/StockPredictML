import pandas as pd
import numpy as np
from tqdm import tqdm

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor





def short_time_train_predict(model, df:pd.DataFrame, days:int=10):
    features = ["Open", "High_s1", "Low_s1", "Close_s1", "delta_s1"]
    target = "Close"
    min_date = df.Date.min()
    max_date = df.Date.max()
    
    n_splits = (max_date - min_date).days // days
    
    res_ls = []
    start_date = min_date
    
    for i in tqdm(range(0, len(df), 2)):        
        df_train = df.iloc[i:i+days-2]
        df_test = df.iloc[i+days-2:i+days]

        if df_train.empty or df_test.empty:
            break
        
        model.fit(df_train[features], df_train[target])
        df_test["predict"] = model.predict(df_test[features])
        res_ls.append(df_test)
            
    return pd.concat(res_ls)