#!/usr/bin/env python3
"""
Unified Controller for Artemis Project

This module combines the functionality from:
  - prediction.py
  - minecraft controller.py
  - hybrid drone controller.py
It exposes a unified main function that, based on a control_mode variable,
executes one of three behaviors when a gesture is predicted:
  1. "print"     -> simply print the prediction.
  2. "drone"     -> send a command to a Bebop drone.
  3. "minecraft" -> send keyboard commands for Minecraft control.

The variable 'control_mode' can later be set via the GUI.
 
Prerequisites:
  - Your constants (MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH) in constants.py
  - A trained model and metadata stored in a relative "model" folder.
  - pyparrot, keyboard (for drone and minecraft modes), pymyo, bleak, tensorflow, etc.
"""

import os
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

# Import any additional modules needed for drone and Minecraft control
# For drone mode:
try:
    from pyparrot.Bebop import Bebop
except ImportError:
    Bebop = None
# For Minecraft mode:
try:
    import keyboard
except ImportError:
    keyboard = None

from constants import MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH

# Define base directory and model paths (relative to this file)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_PATH, "model")
MODEL_FILE = os.path.join(MODEL_DIR, MODEL_PATH)
METADATA_FILE = os.path.join(MODEL_DIR, METADATA_PATH)

# --- Gesture Predictor (same as in prediction.py) ---
class GesturePredictor:
    def __init__(self, window_size=10, prediction_interval=1.0):
        self.window_size = window_size
        self.data_buffer = deque(maxlen=window_size)
        self.prediction_interval = prediction_interval
        self.last_prediction_time = 0
        
        # Load model and metadata using relative paths
        self.model = tf.keras.models.load_model(MODEL_FILE)
        with open(METADATA_FILE, 'rb') as f:
            self.scaler, self.columns = pickle.load(f)
            
    def process_window(self):
        if len(self.data_buffer) < self.window_size:
            return None
        window_data = np.array(list(self.data_buffer))
        rms_features = np.sqrt(np.mean(np.square(window_data), axis=0))
        features_df = pd.DataFrame([rms_features], columns=self.columns)
        features_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        return int(predicted_class), CLASSES[predicted_class]
    
    def add_data(self, emg_data):
        self.data_buffer.append(emg_data)
        current_time = time.time()
        if len(self.data_buffer) == self.window_size and (current_time - self.last_prediction_time) >= self.prediction_interval:
            self.last_prediction_time = current_time
            return self.process_window()
        return None

# --- Executor Functions for Different Modes ---

def execute_print(prediction):
    class_id, gesture_name = prediction
    print(f"Predicted: {gesture_name} (class {class_id})")

def execute_drone(prediction, bebop, movement_time=0.01):
    # Drone commands as defined in hybrid drone controller.py
    commands = {
        0: {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},
        1: {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 50},
        2: {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': -50},
        3: {'pitch': 50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},
        4: {'pitch': -50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},
        5: {'pitch': 0, 'roll': 0, 'yaw': -50, 'vertical_movement': 0},
        6: {'pitch': 0, 'roll': 0, 'yaw': 50, 'vertical_movement': -50},
    }
    class_id, gesture_name = prediction
    command = commands.get(class_id, commands[0])
    bebop.fly_direct(
        roll=command['roll'],
        pitch=command['pitch'],
        yaw=command['yaw'],
        vertical_movement=command['vertical_movement'],
        duration=movement_time
    )
    print(f"Drone Command: {gesture_name} (class {class_id})")

class MinecraftController:
    """A minimal Minecraft controller to convert gestures to key commands."""
    def __init__(self):
        self.current_keys = set()
        
    def gesture_to_minecraft_command(self, class_id):
        # Map gesture class to keyboard commands
        commands = {
            '0': [],  # resting
            '1': ['space'],  # jump
            '2': ['shift'],  # sneak
            '3': ['w'],      # move forward
            '4': ['s'],      # move backward
            '5': ['a'],      # move left
            '6': ['d'],      # move right
            '7': ['left_click']  # break block
        }
        return commands.get(str(class_id), [])

def execute_minecraft(prediction, minecraft_controller):
    class_id, gesture_name = prediction
    keys = minecraft_controller.gesture_to_minecraft_command(class_id)
    # Release any previously pressed keys
    for key in minecraft_controller.current_keys:
        keyboard.release(key)
    minecraft_controller.current_keys.clear()
    for key in keys:
        if key == 'left_click':
            keyboard.press_and_release('left')
        else:
            keyboard.press(key)
            minecraft_controller.current_keys.add(key)
    print(f"MC Command: {gesture_name} (class {class_id}) - Keys: {', '.join(keys) if keys else 'none'}")

# --- Unified Async Main Function ---
async def main_unified(control_mode, bebop=None, minecraft_controller=None):
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    print("Starting unified gesture control session...")
    async with Myo(myo_device) as myo:
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)
        
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data):
            prediction = predictor.add_data(emg_data)
            if prediction:
                if control_mode == "print":
                    execute_print(prediction)
                elif control_mode == "drone" and bebop is not None:
                    execute_drone(prediction, bebop)
                elif control_mode == "minecraft" and minecraft_controller is not None:
                    execute_minecraft(prediction, minecraft_controller)
        
        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        try:
            while True:
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping unified gesture control session...")
            await myo.set_mode(emg_mode=None)

# --- Main Block ---
if __name__ == "__main__":
    # Set control_mode here or later from the UI. Options: "print", "drone", "minecraft"
    control_mode = "print"  # default mode
    
    # Initialize the gesture predictor
    predictor = GesturePredictor(prediction_interval=1.0)
    
    # Prepare additional controllers if needed
    bebop = None
    if control_mode == "drone":
        if Bebop is None:
            raise RuntimeError("pyparrot.Bebop not available. Please install pyparrot.")
        bebop = Bebop()
        # Connect to drone, etc. (drone initialization code goes here)
    
    minecraft_controller = None
    if control_mode == "minecraft":
        if keyboard is None:
            raise RuntimeError("keyboard module not available. Please install the keyboard module.")
        minecraft_controller = MinecraftController()
    
    # Run the unified main function
    asyncio.run(main_unified(control_mode, bebop=bebop, minecraft_controller=minecraft_controller))
