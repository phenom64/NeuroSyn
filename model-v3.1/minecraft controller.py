import time
import keyboard
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
import asyncio
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
from collections import deque

from constants import MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH

# Global
window_size = 50
prediction_interval = 1  # Reduced from 1 to make controls more responsive
movement_time = 0.1

class GesturePredictor:
    def __init__(self):
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
            
        window_data = np.array(list(self.data_buffer))
        rms_features = np.sqrt(np.mean(np.square(window_data), axis=0))
        features_df = pd.DataFrame([rms_features], columns=self.columns)
        features_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        return predicted_class, CLASSES[predicted_class]
    
    def add_data(self, emg_data):
        """Add new EMG data and return prediction if enough time has passed."""
        self.data_buffer.append(emg_data)
        
        current_time = time.time()
        time_since_last_prediction = current_time - self.last_prediction_time
        
        if (len(self.data_buffer) == self.window_size and 
            time_since_last_prediction >= self.prediction_interval):
            self.last_prediction_time = current_time
            return self.process_window()
        return None

class MinecraftController:
    def __init__(self):
        self.current_keys = set()  # Track currently pressed keys
        
    def gesture_to_minecraft_command(self, class_id):
        """Convert gesture to Minecraft controls."""
        # Release all currently pressed keys
        for key in self.current_keys:
            keyboard.release(key)
        self.current_keys.clear()
        
        commands = {
            '0': [],  # resting - release all keys
            '1': ['space'],  # open palm - jump
            '2': ['shift'],  # closed fist - sneak
            '3': ['w'],  # ok gesture - move forward
            '4': ['s'],  # pointer finger - move backward
            '5': ['a'],  # peace sign - move left
            '6': ['d'],  # sha gesture - move right
            '7': ['left_click'],  # trigger gesture - break block
        }
        
        # Get the keys to press for this gesture
        keys_to_press = commands.get(str(class_id), [])
        
        # Press the new keys
        for key in keys_to_press:
            if key == 'left_click':
                keyboard.press_and_release('left')
            else:
                keyboard.press(key)
                self.current_keys.add(key)
        
        return keys_to_press

async def main():
    # Initialize the controller
    minecraft_controller = MinecraftController()
    
    # Find and connect to Myo device
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    
    print("Starting Minecraft gesture control session...")
    print("Make gestures to control your character:")
    print("- Open palm: Jump")
    print("- Closed fist: Sneak")
    print("- OK gesture: Move forward")
    print("- Pointer finger: Move backward")
    print("- Peace sign: Move left")
    print("- Sha gesture: Move right")
    print("- Special gesture: Break block")
    print("\nPress Ctrl+C to exit")
    
    async with Myo(myo_device) as myo:
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)
        
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data):
            prediction = predictor.add_data(emg_data)
            
            if prediction:
                class_id, gesture_name = prediction
                keys = minecraft_controller.gesture_to_minecraft_command(class_id)
                print(f"\rPredicted gesture: {gesture_name} - Keys: {', '.join(keys) if keys else 'none'}")

        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        print("Myo band activated successfully!")

        try:
            while True:
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping gesture control...")
            # Release any held keys
            for key in minecraft_controller.current_keys:
                keyboard.release(key)
            await myo.set_mode(emg_mode=None)

if __name__ == "__main__":
    predictor = GesturePredictor()
    asyncio.run(main())