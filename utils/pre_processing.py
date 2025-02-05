import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

def get_dataset(dataset_name):
    stock_info = yf.Ticker(dataset_name)
    return pd.DataFrame(stock_info.history(period="max"))

def get_stock_info(dataset_name):
    stock_info = yf.Ticker(dataset_name)
    info = stock_info.info
    return f'Stock Info: \n 
             Website: {info["website"]}\n Industry: {info["industry"]}\n 
             longBusinessSummary: {info["longBusinessSummary"]}'

def delete_columns(dataset, columns=['Dividends', 'Stock Splits', 'Date']):
    return dataset.drop(columns=columns)

def normalize_dataset(dataset):
    y = dataset['Close']
    X = dataset.drop(['Close'], axis=1)
    sc = MinMaxScaler(feature_range=(0, 1))
    dataset = sc.fit_transform(dataset)
    return dataset

def split_dataset_train_test(dataset):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)
    return X_train, X_test, y_train, y_test

def metrics(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae