from models.longShortTermMemory import LSTM

lstm = LSTM()

lstm.generate_prediction('SOFI')
lstm.visualizar_previsao()