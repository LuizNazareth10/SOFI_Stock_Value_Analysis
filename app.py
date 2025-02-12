from models.longShortTermMemory import LSTM_model
from utils.pre_processing import get_and_preprocess_dataset, denormalize_dataset
import pandas as pd

lstm = LSTM_model()
X_train, X_test, y_train, y_test, datas = get_and_preprocess_dataset('PLTR') 
datas = pd.to_datetime(datas, unit='s').dt.strftime('%Y-%m-%d')
print(f'/n/n/n/nMAIN{datas}/n/n/n/n')
previsao = lstm.generate_prediction('PLTR')

# Garantir que y_test e previsao tenham formato correto
y_test = y_test.flatten()  # Remove dimensões extras
previsao = previsao.flatten()

# Verifica o tamanho correto de datas
# datas = datas.iloc[-len(y_test):]

lstm.visualizar_previsao(previsao, y_test, datas)

