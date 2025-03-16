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

from constants import MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH, COLLECTION_TIME

class GesturePredictor:
    def __init__(self, window_size=10, prediction_interval=1.0, threshold=0.9, num_repetitions=1):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.prediction_interval = prediction_interval
        self.last_prediction_time = 0
        self.threshold = threshold
        self.num_repetitions = num_repetitions
        self.calibration_mean = None
        self.calibration_std = None
        self.is_calibrating = False  # Flag to control calibration data collection
        self.calibration_complete = False  # Flag to indicate calibration is done
        
        # Load the model and metadata
        self.model = tf.keras.models.load_model(f"model/{MODEL_PATH}")
        with open(f"model/{METADATA_PATH}", 'rb') as f:
            self.scaler, self.columns = pickle.load(f)
        
        # Calibration buffer to store data during calibration
        self.calibration_buffer = []

    def calibrate(self, emg_data):
        """Add EMG data to the calibration buffer if calibration is active."""
        if self.is_calibrating:
            self.calibration_buffer.append(emg_data)

    def finalize_calibration(self):
        """Calculate mean and SD from calibration data."""
        if not self.calibration_buffer:
            raise ValueError("No calibration data collected.")
        
        calibration_data = np.array(self.calibration_buffer)
        self.calibration_mean = np.mean(calibration_data)
        self.calibration_std = np.std(calibration_data)
        
        if self.calibration_std == 0:
            self.calibration_std = 1.0  # Avoid division by zero
        
        print(f"Calibration completed: Mean={self.calibration_mean:.4f}, SD={self.calibration_std:.4f}")
        self.calibration_buffer = []  # Clear buffer after calibration
        self.calibration_complete = True  # Mark calibration as complete

    def process_window(self):
        """Process the current window of z-score normalized data and return a prediction."""
        if len(self.data_buffer) < self.window_size or not self.calibration_complete:
            return None
        
        # Convert buffer to a numpy array and normalize to z-scores
        window_data = np.array(list(self.data_buffer))
        z_score_data = (window_data - self.calibration_mean) / self.calibration_std
        
        # Calculate RMS features
        rms_features = np.sqrt(np.mean(np.square(z_score_data), axis=0))
        
        # Scale the RMS values to match training preprocessing (e.g., *100)
        scaling_factor = 100  # Must match the training script
        rms_features_scaled = rms_features * scaling_factor
        
        # Create a DataFrame with the correct column names
        features_df = pd.DataFrame([rms_features_scaled], columns=self.columns)
        
        # Scale the features using the loaded scaler (if used in training)
        features_scaled = self.scaler.transform(features_df)
        
        # Make prediction
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        if predicted_class != 0 and confidence < self.threshold:
            return 0, CLASSES[0]
        
        return predicted_class, CLASSES[predicted_class]

    def add_data(self, emg_data):
        """Add new EMG data to the buffer and return prediction if enough time has passed."""
        self.data_buffer.append(emg_data)
        
        if not self.calibration_complete:
            return None
        
        current_time = time.time()
        time_since_last_prediction = current_time - self.last_prediction_time
        
        if (len(self.data_buffer) == self.window_size and 
            time_since_last_prediction >= self.prediction_interval):
            self.last_prediction_time = current_time
            return self.process_window()
        return None

async def calibrate_gestures(myo, predictor):
    """Run the calibration process for all gestures with controlled data collection."""
    print("Starting calibration process...")
    print(f"Please perform each gesture {predictor.num_repetitions} times for {COLLECTION_TIME} seconds each.")
    
    for class_id, gesture_name in CLASSES.items():
        for rep in range(predictor.num_repetitions):
            print(f"\nCalibration for '{gesture_name}' (class {class_id}), repetition {rep + 1}/{predictor.num_repetitions}")
            print(f"Get ready... (2 seconds)")
            await asyncio.sleep(2)
            
            print(f"Perform '{gesture_name}' now for {COLLECTION_TIME} seconds...")
            predictor.is_calibrating = True  # Start collecting data
            start_time = time.time()
            
            while time.time() - start_time < COLLECTION_TIME:
                await asyncio.sleep(0.01)  # Small delay to avoid blocking
            
            predictor.is_calibrating = False  # Stop collecting data
            print(f"Finished collecting '{gesture_name}' repetition {rep + 1}")

    # Finalize calibration after collecting all data
    predictor.finalize_calibration()

async def main():
    try:
        # Find and connect to Myo device
        myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
        if not myo_device:
            raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
        
        predictor = GesturePredictor(prediction_interval=1.0, threshold=0.9, num_repetitions=1)
        
        print("Connecting to Myo device...")
        async with Myo(myo_device) as myo:
            await asyncio.sleep(0.5)
            await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
            await asyncio.sleep(0.25)
            # Defines the IMU stuff
            
            @myo.on_imu
            def on_imu(orientation, accelerometer, gyroscope):
                print("IMU READING:")
                print("Orientation:", orientation)
                print("Acceleration:",accelerometer)
                print("Gyroscope:",gyroscope)
            
            # Define EMG callback for both calibration and prediction
            @myo.on_emg_smooth
            def on_emg_smooth(emg_data):
                if predictor.is_calibrating:
                    predictor.calibrate(emg_data)
                else:
                    prediction = predictor.add_data(emg_data)
                    if prediction:
                        class_id, gesture_name = prediction
                        print(f"Predicted gesture: {gesture_name} (class {class_id})")
            
            # Run calibration
            await myo.set_mode(emg_mode=EmgMode.SMOOTH)
            await calibrate_gestures(myo, predictor)
            
            print("\nStarting gesture prediction session...")
            print("Make gestures to see predictions (updating every second)...")
            
            try:
                while True:
                    await asyncio.sleep(0.01)
            except KeyboardInterrupt:
                print("\nStopping gesture prediction...")
                await myo.set_mode(emg_mode=None)
    
    except RuntimeError as e:
        print(f"Runtime error: {e}")
        exit()
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit()

if __name__ == "__main__":
    asyncio.run(main())