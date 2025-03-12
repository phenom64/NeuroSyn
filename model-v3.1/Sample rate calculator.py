import time
import keyboard
from pyparrot.Bebop import Bebop
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from collections import deque
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode

from constants import METADATA_PATH, MODEL_PATH, CLASSES, MYO_ADDRESS

class GesturePredictor:
    def __init__(self, window_size=10, prediction_interval=1.0):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.prediction_interval = prediction_interval
        self.last_prediction_time = 0
        
        # Timing measurements
        self.sample_times = deque(maxlen=1000)  # Store last 1000 sample timestamps
        self.prediction_durations = deque(maxlen=100)  # Store last 100 prediction durations
        self.last_sample_time = None
        
        # Load the model and metadata
        self.model = tf.keras.models.load_model(f"model/{MODEL_PATH}")
        with open(f"model/{METADATA_PATH}", 'rb') as f:
            self.scaler, self.columns = pickle.load(f)
    
    def calculate_sampling_rate(self):
        """Calculate the current sampling rate based on stored timestamps."""
        if len(self.sample_times) < 2:
            return None
        
        # Calculate time differences between consecutive samples
        time_diffs = np.diff(list(self.sample_times))
        avg_interval = np.mean(time_diffs)
        
        # Return sampling rate in Hz
        return 1.0 / avg_interval if avg_interval > 0 else None
            
    def process_window(self):
        """Process the current window of data and return a prediction."""
        if len(self.data_buffer) < self.window_size:
            return None
            
        prediction_start = time.perf_counter()
        
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
        
        # Store prediction duration
        prediction_duration = time.perf_counter() - prediction_start
        self.prediction_durations.append(prediction_duration)
        
        return predicted_class, CLASSES[predicted_class]
    
    def add_data(self, emg_data):
        """Add new EMG data to the buffer and return prediction if enough time has passed."""
        current_time = time.perf_counter()
        
        # Track sample timing
        if self.last_sample_time is not None:
            self.sample_times.append(current_time - self.last_sample_time)
        self.last_sample_time = current_time
        
        self.data_buffer.append(emg_data)
        
        time_since_last_prediction = current_time - self.last_prediction_time
        
        if (len(self.data_buffer) == self.window_size and 
            time_since_last_prediction >= self.prediction_interval):
            self.last_prediction_time = current_time
            return self.process_window()
        return None
    
    def get_timing_stats(self):
        """Return current timing statistics."""
        sampling_rate = self.calculate_sampling_rate()
        avg_prediction_time = np.mean(list(self.prediction_durations)) if self.prediction_durations else None
        
        return {
            'sampling_rate_hz': sampling_rate if sampling_rate else 0,
            'avg_prediction_time_ms': avg_prediction_time * 1000 if avg_prediction_time else 0,
            'min_prediction_time_ms': min(self.prediction_durations) * 1000 if self.prediction_durations else 0,
            'max_prediction_time_ms': max(self.prediction_durations) * 1000 if self.prediction_durations else 0
        }
    
async def main():    
    # Find and connect to Myo device
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    
    print("Starting gesture control session...")
    
    async with Myo(myo_device) as myo:
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)
        
        last_stats_time = time.time()
        stats_interval = 5  # Print stats every 5 seconds
        
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data):
            nonlocal last_stats_time
            current_time = time.time()
            
            # Process the EMG data
            prediction = predictor.add_data(emg_data)
            
            # Print timing stats periodically
            if current_time - last_stats_time >= stats_interval:
                stats = predictor.get_timing_stats()
                print("\nTiming Statistics:")
                print(f"EMG Sampling Rate: {stats['sampling_rate_hz']:.2f} Hz")
                print(f"Average Prediction Time: {stats['avg_prediction_time_ms']:.2f} ms")
                print(f"Min Prediction Time: {stats['min_prediction_time_ms']:.2f} ms")
                print(f"Max Prediction Time: {stats['max_prediction_time_ms']:.2f} ms")
                last_stats_time = current_time
            
            if prediction:
                class_id, gesture_name = prediction
                print(f"\rPredicted gesture: {gesture_name} (class {class_id})", end="")

        # Set EMG mode and collect data
        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        print("Myo band activated successfully, taking off in 5 seconds")

        try:
            while True:
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping gesture prediction...")
            await myo.set_mode(emg_mode=None)

if __name__ == "__main__":
    predictor = GesturePredictor()
    asyncio.run(main()) 