from sklearn.ensemble import RandomForestRegressor
import os
from utils import pre_processing as pp
class LSTM:
    def __init__(self, model_dir='./models', n_estimators=100, max_depth=10, random_state=42, verbose=0):
        self.model = LSTM()
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, 'random_forest_regressor.joblib')
        self.model.set_params(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, verbose=verbose)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
    def save_model(self):
        import joblib
        joblib.dump(self.model, self.model_path)

    def load_model(self):
        if not os.path.exists(self.model_path):
            raise Exception('Model not found')
        import joblib
        self.model = joblib.load(self.model_path)
        
    def generate_prediction(self, stock_name):
        data = pp.get_dataset(stock_name)
        data = pp.delete_columns(data)
        data = pp.normalize_dataset(data)
        X_train, X_test, y_train, y_test = pp.split_dataset_train_test(data)
        return self.predict(X_test)        