import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

sc = MinMaxScaler(feature_range=(0, 1))

def get_and_preprocess_dataset(dataset_name):
    stock_info = yf.Ticker(dataset_name).history(period='max')
    stock_info.reset_index(inplace=True)
    stock_info = delete_columns(stock_info)
    stock_info = normalize_dataset(stock_info)  
    X_train, X_test, y_train, y_test = split_dataset_LSTM(stock_info)
    y_train = normalize_dataset(y_train.reshape(-1, 1))
    print(y_test)
    
    return X_train, X_test, y_train, y_test


def get_stock_info(dataset_name):
    stock_info = yf.Ticker(dataset_name)
    info = stock_info.info
    return f"""Stock Info:
    Website: {info["website"]}
    Industry: {info["industry"]}
    Long Business Summary: {info["longBusinessSummary"]}"""

import pandas as pd

def delete_columns(dataset, columns=['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits']):
    dataset = pd.DataFrame(dataset)
    dataset = dataset.drop(columns=[col for col in columns if col in dataset.columns], errors='ignore')

    if 'Close' in dataset.columns:
        dataset = dataset[['Close']]
    return dataset.to_numpy()

def normalize_dataset(dataset):
    global sc
    dataset = sc.fit_transform(dataset)
    return dataset

def denormalize_dataset(previsao):
    global sc
    previsao = sc.inverse_transform(previsao)
    return previsao

def split_dataset_LSTM(dataset):
    step = 60
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    for i in range(step, int(0.7 * len(dataset))):
        X_train.append(dataset[i-step:i])
        y_train.append(dataset[i, 0])
    
    for i in range(int(0.7 * len(dataset)), len(dataset)):
        X_test.append(dataset[i-step:i])
        y_test.append(dataset[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test, y_test = np.array(X_test), np.array(y_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    return X_train, X_test, y_train, y_test
    
def metrics(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

def get_date(dataset):
    return dataset.index[-1]