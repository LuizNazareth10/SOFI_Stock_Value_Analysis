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
    # print(stock_info)
    datas = stock_info['Date']
    # stock_info['Date'] = pd.to_datetime(stock_info['Date'], unit='s')
    # datas = pd.to_datetime(stock_info['Date'], unit='s').dt.strftime('%Y-%m-%d')
    # datas = stock_info['Date']
    stock_info = normalize_dataset(stock_info)  
    X_train, X_test, y_train, y_test = split_dataset_LSTM(stock_info)
    datas_print = datas.iloc[-len(y_test):]
    print(f'/n/n/n/n PREPROCESSING{datas_print}/n/n/n/n')
    # Garantir que y_test tenha o formato correto antes de calcular datas
    y_test = denormalize_dataset(y_test.reshape(-1, 1))  # Normaliza y_test aqui para evitar problemas

    # datas = stock_info['Date'].tail(len(y_test))  # Pega as últimas datas

    return X_train, X_test, y_train, y_test, datas_print



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
        dataset = dataset[['Date', 'Close']]  # Mantém a coluna 'Date'
        
    dataset['Date'] = pd.to_datetime(dataset['Date']).astype(int) / 10**9

    return dataset

def normalize_dataset(dataset):
    global sc
    dataset = sc.fit_transform(dataset)
    return dataset

def denormalize_dataset(previsao):
    global sc
    # Pega apenas a escala da segunda coluna (preço de fechamento)
    previsao = sc.inverse_transform(np.hstack([np.zeros((previsao.shape[0], 1)), previsao]))[:, 1]
    return previsao.reshape(-1, 1)  # Garante o formato correto


def split_dataset_LSTM(dataset):
    step = 60
    X_train, y_train = [], []
    X_test, y_test = [], []

    # Separando apenas a coluna de preços (Close) para normalizar
    prices = dataset[:, 1].reshape(-1, 1)  # Pegando apenas os valores sem a data

    # Criando as sequências para treinamento
    for i in range(step, int(0.7 * len(prices))):
        X_train.append(prices[i - step:i])
        y_train.append(prices[i, 0])

    for i in range(int(0.7 * len(prices)), len(prices)):
        X_test.append(prices[i - step:i])
        y_test.append(prices[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, X_test, y_train, y_test

    
def metrics(y_test, y_pred):
    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return rmse, mae

def get_date(dataset):
    return dataset.index[-1]