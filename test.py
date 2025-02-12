import yfinance as yf
from utils.pre_processing import get_and_preprocess_dataset, get_stock_info, delete_columns, normalize_dataset, denormalize_dataset, split_dataset_LSTM

data = yf.Ticker('PLTR').history(period='max')
data.reset_index(inplace=True)
# data = delete_columns(data)
print(data['Date', 'Close'])
# print(data)
