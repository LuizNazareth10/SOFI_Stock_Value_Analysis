from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import load_model
import os
from utils import pre_processing as pp
import matplotlib.pyplot as plt
class LSTM_model:
    def __init__(self, model_dir='./models'):
        self.model = Sequential()
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'models_saved/long_short_term_memory.keras')

    def compile_and_fit(self, X_train, y_train):
        if len(self.model.layers) == 0:
            self.create_layers(X_train)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        self.model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[es])

    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self):
        self.model.save(self.model_path)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise Exception('Model not found')
        self.model = load_model(self.model_path)

    
    def create_layers(self, X_train):
        # self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        # LayerNormalization()
        self.model.add(Dense(units=1))
    
    def generate_prediction(self, stock_name):
        X_train, X_test, y_train, y_test, datas = pp.get_and_preprocess_dataset(stock_name)
        if not os.path.exists(self.model_path):
            raise Exception('Model not found')
            
            self.compile_and_fit(X_train, y_train)
            self.save_model()
            
        else:
            self.load_model()
            
        previsao = self.predict(X_test)
        previsao = pp.denormalize_dataset(previsao)
        # print(f'PREVISAO\n\n\n{previsao}\n\n\n')
        return previsao
        

    def visualizar_previsao(self, previsao, y_test, datas):
        plt.figure(figsize=(12, 6))
        print(f'/n/n/n/nLSTM{datas}/n/n/n/n')
        # Convertendo as datas para um formato adequado
        datas = pd.to_datetime(datas)

        plt.plot(datas, y_test, label='Real', color='blue', linestyle='-', linewidth=2, alpha=0.8)
        plt.plot(datas, previsao, label='Previsão', color='red', linestyle='--', linewidth=2)
        
        plt.title('Previsão vs Real', fontsize=16, fontweight='bold')
        plt.xlabel('Data', fontsize=14)
        plt.ylabel('Preço', fontsize=14)
        plt.legend(loc='best', fontsize=12, frameon=True, shadow=True)
        
        # Melhorando a formatação das datas no eixo X
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)
        
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

