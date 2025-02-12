from models.longShortTermMemory import LSTM_model
from utils.pre_processing import get_and_preprocess_dataset, denormalize_dataset

lstm = LSTM_model()
X_train, X_test, y_train, y_test = get_and_preprocess_dataset('PLTR') 
lstm.visualizar_previsao(lstm.generate_prediction('PLTR'), y_test)