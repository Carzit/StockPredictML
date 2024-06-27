# Sliding Window Stock Prediction with Traditional ML Models

>Stock market prediction has always been a critical issue in financial analysis and investment decision-making. Accurate prediction models can provide valuable insights into market trends, enabling investors to make informed decisions. However, due to the high volatility and nonlinear characteristics of the stock market, building a stable and efficient prediction model has always been a huge challenge. This paper aims to explore the use of multiple machine learning models to predict Nasdaq stock market data and compare the performance of the models under different models and sliding window strategies in order to find a more effective stock price prediction method.

## 1. Batch Processing
Simple execution of my report experiments
```
cmd.bat
```

## 2. Commandline Use 
### Data Preprocessing
```
python data.py --data RAW_DATA_PATH --save PROCESSED_DATA_PATH --log LOG_PATH
```

### Fixed Partition
To find best params
```
python fixed_partition.py --data PROCESSED_DATA_PATH --log LOG_PATH
```

### Fixed Sliding Window
Use a sliding window method with a window size of `5m`. Use the first `5m - n` data for training each time, and the last `n` data for prediction. Move the window fforward by `n` data and repeat the above steps until the entire data set is covered. This method simulates the scenario of real-time prediction, and the model updates the training data before each prediction.
```
python fixed_window.py --data PROCESSED_DATA_PATH --log LOG_PATH --window_len WINDOW_LEN --test_len TEST_LEN
```

### Cumulative Sliding Window
Each time, the first `5m - n` data in the current window and all previous data are used for training, and the last `n` data are used for prediction. The window is moved back by `n` data and the above steps are repeated until the entire data set is covered. This method aims to utilize all available historical data and may improve the predictive ability of the model.
```
python cumulative_window.py --data PROCESSED_DATA_PATH --log LOG_PATH --window_len WINDOW_LEN --test_len TEST_LEN
```


