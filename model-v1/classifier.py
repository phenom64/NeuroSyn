import numpy as np
import pandas as pd
import pickle

from pymyo.types import EmgValue
from tensorflow.keras.models import load_model

# from constants import CLASSES

CLASSES = {
    0: "rest", #saftey
    1: "closed_fist", #go up
    2: "open_palm", #go down
    3: "ok", #go forwards
    4: "pointer_finger", #go backwards
    5: "peace", #roatate left
    6: "shaaa", #rotate right
    7: "peace_among_worlds", #do a flip
}


class Classifier:
    def __init__(self, model_path: str, metadata_path: str):
        self.model = load_model(model_path)
        
        with open(metadata_path, 'rb') as f:
            self.scaler, self.columns = pickle.load(f)
    
    def classify(self, emg_data: tuple[EmgValue, EmgValue]) -> str:
        sensor1_data, sensor2_data = emg_data
        
        emg_features = np.concatenate((sensor1_data, sensor2_data))
        emg_features_df = pd.DataFrame([emg_features], columns=self.columns)

        emg_features_scaled = self.scaler.transform(emg_features_df)
        emg_features_reshaped = emg_features_scaled.reshape(1, -1)

        prediction = self.model.predict(emg_features_reshaped)
        predicted_class = np.argmax(prediction)
        
        print(f"Predicted probabilities: {prediction}")
        
        return CLASSES[predicted_class]