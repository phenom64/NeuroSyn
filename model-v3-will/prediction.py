import time
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from collections import deque
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode

from constants import MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH

class GesturePredictor:
    def __init__(self, window_size=10, prediction_interval=1.0):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.prediction_interval = prediction_interval
        self.last_prediction_time = 0
        
        # Load the model and metadata
        self.model = tf.keras.models.load_model(f"model/{MODEL_PATH}")
        with open(f"model/{METADATA_PATH}", 'rb') as f:
            self.scaler, self.columns = pickle.load(f)
            
    def process_window(self):
        """Process the current window of data and return a prediction."""
        if len(self.data_buffer) < self.window_size:
            return None
            
        # Convert buffer to a numpy array
        window_data = np.array(list(self.data_buffer))
        
        # Calculate RMS features
        rms_features = np.sqrt(np.mean(np.square(window_data), axis=0))
        
        # Create a DataFrame with the correct column names
        features_df = pd.DataFrame([rms_features], columns=self.columns)
        
        # Scale the features
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        return predicted_class, CLASSES[predicted_class]
    
    def add_data(self, emg_data):
        """
        Add new EMG data to the buffer and return prediction if enough time has passed.
        Returns prediction only once per prediction_interval seconds.
        """
        self.data_buffer.append(emg_data)
        
        current_time = time.time()
        time_since_last_prediction = current_time - self.last_prediction_time
        
        if (len(self.data_buffer) == self.window_size and 
            time_since_last_prediction >= self.prediction_interval):
            self.last_prediction_time = current_time
            return self.process_window()
        return None


async def main():    
    # Find and connect to Myo device
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    
    print("Starting gesture prediction session...")
    print("Make gestures to see predictions (updating every second)...")
    
    async with Myo(myo_device) as myo:
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)
        
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data):
            # Process the EMG data
            prediction = predictor.add_data(emg_data)
            
            if prediction:
                class_id, gesture_name = prediction
                print(f"\rPredicted gesture: {gesture_name} (class {class_id})")
    
        # Set EMG mode and collect data
        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        
        try:
            while True:
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping gesture prediction...")
            await myo.set_mode(emg_mode=None)


if __name__ == "__main__":
    predictor = GesturePredictor(prediction_interval=1.0)
    asyncio.run(main())