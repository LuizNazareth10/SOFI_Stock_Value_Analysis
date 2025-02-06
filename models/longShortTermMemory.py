from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import os
from utils import pre_processing as pp
import matplotlib.pyplot as plt
class LSTM:
    def __init__(self, model_dir='./models'):
        self.model = Sequential()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'long_short_term_memory.joblib')

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

    from tensorflow.keras.models import load_model

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise Exception('Model not found')
        self.model = load_model(self.model_path)

    
    def create_layers(self, X_train):
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=50))
        self.model.add(Dropout(0.2))
        # LayerNormalization()
        self.model.add(Dense(units=1))
    
    def generate_prediction(self, stock_name):
        X_train, X_test, y_train, y_test = pp.get_and_preprocess_dataset(stock_name)
        self.compile_and_fit(X_train, y_train)
        self.save_model()
        previsao = self.predict(X_test)
        previsao = pp.denormalize_dataset(previsao)
        return self.predict(X_test)
    
    def visualizar_previsao(self, previsao):
        plt.plot(previsao, color='red', label='Previsão')
        plt.plot(y_test, color='blue', label='Real')
        plt.title('Previsão vs Real')
        plt.xlabel('Tempo')
        plt.ylabel('Preco')
        plt.legend()
        plt.show()