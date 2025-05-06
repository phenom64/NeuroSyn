#!/usr/bin/env python3
import os
import sys
import asyncio
import struct
import types
import time
from collections import deque
from bleak import BleakScanner

import numpy as np
import pandas as pd # Keep pandas import if scaler expects DataFrame, though numpy array might work
import pickle
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler # Ensure scaler type is available

# Import constants and types
from constants import (
    MYO_ADDRESS,
    MODEL_PATH,
    METADATA_PATH,
    CLASSES,
    WINDOW_SIZE,
    COLLECTION_TIME, # Used for calibration prompts
    REPETITIONS_PER_EXERCISE, # Used for calibration prompts (though not strictly needed now)
    NUM_EMG_SENSORS,
    NUM_IMU_VALUES,
    DATA_INPUT_PATH # For potential future saving/logging if needed
)
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode, ImuMode

# --- Patch pymyo’s buggy classifier handler (Keep if necessary) ---
def _safe_on_classifier(self, sender, value):
    if len(value) < 3: return
    try: struct.unpack("<B2s", value)
    except struct.error: return
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# --------------------------------------------------------------------

class NeuroSynPredictor:
    # <<< CHANGED prediction_interval default back to 1.0 >>>
    def __init__(self, window_size=WINDOW_SIZE, prediction_interval=1.0, threshold=0.7):
        self.window_size = window_size
        self.prediction_interval = prediction_interval # Controls prediction rate
        self.threshold = threshold

        self.emg_buf = deque(maxlen=window_size)
        self.imu_buf = deque(maxlen=window_size)
        self.last_imu = np.zeros(NUM_IMU_VALUES, dtype=float)
        self.last_prediction = None
        self.last_prediction_time = 0.0

        # Flags for controlling flow
        self.is_calibrating = False # Used by run_calibration_flow to pause prediction
        self.calibration_complete = False # Indicates the initial guidance phase is done

        # --- Load Model, Scaler, Features, AND Calibration Stats ---
        print("Loading model and metadata (scaler, features, calib_stats)...")
        self.model = load_model(MODEL_PATH)
        try:
            with open(METADATA_PATH, "rb") as f:
                # Unpack all four items saved during training
                self.scaler, self.feature_names, self.calibration_mean, self.calibration_std = pickle.load(f)

            print(f"Metadata loaded. Expecting features: {self.feature_names}")
            print(f"Using EMG Calibration Mean (loaded): {np.round(self.calibration_mean, 2)}")
            print(f"Using EMG Calibration StdDev (loaded): {np.round(self.calibration_std, 2)}")

            # Verify scaler type and calibration stats shapes
            if not isinstance(self.scaler, StandardScaler):
                print(f"Warning: Loaded scaler might not be StandardScaler (Type: {type(self.scaler)})")
            if self.calibration_mean.shape != (NUM_EMG_SENSORS,) or self.calibration_std.shape != (NUM_EMG_SENSORS,):
                 print(f"ERROR: Loaded calibration stats have unexpected shape! Mean: {self.calibration_mean.shape}, Std: {self.calibration_std.shape}. Check metadata file.")
                 sys.exit(1) # Exit if essential stats are malformed

        except FileNotFoundError:
             print(f"FATAL ERROR: Metadata file not found at {METADATA_PATH}.")
             sys.exit(1)
        except ValueError:
             print(f"FATAL ERROR: Could not unpack expected items from metadata file {METADATA_PATH}.")
             print("Ensure it contains (scaler, feature_names, calibration_mean, calibration_std).")
             sys.exit(1)
        except Exception as e:
            print(f"FATAL ERROR: Could not load metadata from {METADATA_PATH}. {e}")
            sys.exit(1)

        print("Predictor initialized successfully.")

    # --- Calibration Flow Control Methods (Simplified) ---
    def start_calibration_flow(self):
        """Indicates the user guidance sequence should begin."""
        print("\n--- Starting Calibration Guidance ---")
        self.is_calibrating = True # Prevent predictions during this phase
        self.calibration_complete = False

    def finalize_calibration_flow(self):
        """Marks the user guidance sequence as complete."""
        print("--- Calibration Guidance Complete ---")
        print("Prediction will now use loaded EMG calibration stats.")
        self.is_calibrating = False # Allow predictions now
        self.calibration_complete = True

    # --- Data Handling Callbacks ---
    def on_imu(self, orientation, accel, gyro):
        """Callback for IMU data updates."""
        self.last_imu = np.array(orientation, dtype=float)

    def on_emg(self, emg_data):
        """Callback for EMG data updates and triggering prediction."""
        # Don't process if calibration guidance isn't finished or if currently in that phase
        if not self.calibration_complete or self.is_calibrating:
            return

        # If emg_data is actually passed from the callback
        if emg_data is not None:
            emg = np.array(emg_data, dtype=float)
            # Append current data to buffers
            self.emg_buf.append(emg)
            self.imu_buf.append(self.last_imu)

        # Check if it's time to predict (based on buffer size and interval)
        now = time.time()
        if len(self.emg_buf) == self.window_size and (now - self.last_prediction_time) >= self.prediction_interval:
            self.last_prediction_time = now
            prediction_result = self.process_window_and_predict()
            self.last_prediction = prediction_result # Store the result for the main loop to print
            return

        return

    # --- Prediction Logic (Applies loaded normalization) ---
    def process_window_and_predict(self):
        """Processes the current data window and returns prediction."""
        if len(self.emg_buf) < self.window_size:
            return None # Not enough data yet

        # Combine EMG and IMU data from buffers
        emg_arr = np.vstack(self.emg_buf) # Shape: (window_size, 8)
        imu_arr = np.vstack(self.imu_buf) # Shape: (window_size, 4)

        # <<< Apply Z-score normalization to EMG part using LOADED stats >>>
        try:
            normalized_emg_arr = (emg_arr - self.calibration_mean) / self.calibration_std
        except AttributeError:
             print("ERROR: Predictor missing calibration_mean/std attributes.")
             return None
        except Exception as e:
            print(f"ERROR during EMG normalization: {e}")
            return None


        # <<< Calculate RMS on NORMALIZED EMG and ORIGINAL IMU >>>
        rms_emg = np.sqrt(np.mean(normalized_emg_arr**2, axis=0))
        rms_imu = np.sqrt(np.mean(imu_arr**2, axis=0))

        # Concatenate features in the correct order
        features = np.concatenate([rms_emg, rms_imu]) # Shape: (12,)
        features_reshaped = features[None, :] # Shape: (1, 12)

        # --- Scaling (using loaded scaler) ---
        try:
            features_scaled = self.scaler.transform(features_reshaped) # Shape: (1, 12)
        except Exception as e:
            print(f"ERROR during scaling: {e}")
            print(f"Feature shape before scaling: {features_reshaped.shape}")
            return None

        # --- Prediction ---
        try:
            prediction = self.model.predict(features_scaled, verbose=0) # Get probabilities
            probs = prediction[0] # Probabilities for this window
            predicted_class_id = int(np.argmax(probs))
            confidence = float(probs[predicted_class_id])
        except Exception as e:
            print(f"ERROR during model prediction: {e}")
            return None

        # --- Apply Confidence Threshold ---
        final_class_id = predicted_class_id
        if predicted_class_id != 0 and confidence < self.threshold:
            final_class_id = 0 # Default to Rest

        final_gesture_name = CLASSES.get(final_class_id, "Unknown") # Get name from constants

        # Return prediction results
        return final_class_id, final_gesture_name, confidence

# --- Calibration Guidance Function (Simplified - Just Prompts) ---
async def run_calibration_flow(predictor, cue=print):
    """Guides the user through the initial 'Rest' prompt."""
    predictor.start_calibration_flow() # Signal predictor to pause predictions

    print(f"\nPlease perform the 'Rest' gesture for {COLLECTION_TIME} seconds.")
    print("This allows the system to stabilize before prediction starts.")
    cue(f"CUE|0|Rest|{COLLECTION_TIME}") # Cue for external interface if needed

    print("Get ready... (2 seconds)")
    await asyncio.sleep(2)
    print(f"Perform 'Rest' now for {COLLECTION_TIME} seconds...")
    start_time = time.time()
    while time.time() - start_time < COLLECTION_TIME:
        # No data collection needed here, just wait
        await asyncio.sleep(0.01)
    print("Finished 'Rest' guidance period.")

    predictor.finalize_calibration_flow() # Signal predictor to allow predictions
    cue("CAL_DONE") # Signal external interface

# --- Main Execution ---
async def main():
    try:
        # <<< Using prediction_interval=1.0 from class default >>>
        predictor = NeuroSynPredictor(threshold=0.8) # Adjust threshold if needed

        # Find and connect to Myo device
        print(f"Scanning for Myo device at address: {MYO_ADDRESS}...")
        device = await BleakScanner.find_device_by_address(MYO_ADDRESS, timeout=20.0)

        if not device:
            print(f"ERROR: Myo device with address {MYO_ADDRESS} not found.")
            print("Ensure Bluetooth is enabled and the Myo armband is charged and nearby.")
            return

        print("Myo device found. Connecting...")
        async with Myo(device) as myo:
            print("Connected to Myo.")
            await asyncio.sleep(0.5)

            # Configure Myo
            print("Configuring Myo modes...")
            await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
            await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=ImuMode.DATA)
            print("Myo configured.")
            await asyncio.sleep(0.25)

            # Setup callbacks AFTER configuring modes fully
            myo.on_emg_smooth(predictor.on_emg)
            myo.on_imu(predictor.on_imu)

            # Run the simplified calibration guidance flow
            await run_calibration_flow(predictor)

            # Start prediction loop
            print("\n--- Starting Real-time Gesture Prediction ---")
            print("Perform physiotherapy exercises. Press Ctrl-C to stop.")
            last_printed_prediction = None

            while True:
                # Check predictor's last prediction state (updated by callback)
                current_prediction = predictor.last_prediction
                if current_prediction and current_prediction != last_printed_prediction:
                    pred_id, pred_name, conf = current_prediction
                    # Only print if confidence is above a certain level OR it's 'Rest'
                    # This reduces console noise from low-confidence fluctuations
                    # <<< CHANGED print format to match old script >>>
                    if conf >= 0.3 or pred_id == 0:
                       print(f"Predicted gesture: {pred_name} (class {pred_id}, Conf: {conf:.2f})") # Adjusted wording
                    last_printed_prediction = current_prediction

                await asyncio.sleep(0.01) # Yield control briefly

    except KeyboardInterrupt:
        print("\nStopping prediction...")
    except RuntimeError as e:
        print(f"Runtime error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Exiting NeuroSyn Predictor.")


if __name__ == "__main__":
    print("--- Initializing NeuroSyn Prediction Script ---")
    asyncio.run(main())
