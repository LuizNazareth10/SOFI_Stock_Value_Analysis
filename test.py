import yfinance as yf
from utils.pre_processing import get_and_preprocess_dataset, get_stock_info, delete_columns, normalize_dataset, denormalize_dataset, split_dataset_LSTM
import pandas as pd
data = yf.Ticker('PLTR').history(period='max')
data.reset_index(inplace=True)
data = delete_columns(data)
data['Date'] = pd.to_datetime(data['Date'], unit='s')
data['Date'] = pd.to_datetime(data['Date'], unit='s').dt.strftime('%Y-%m-%d')

print(data)
