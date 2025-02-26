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

from constants import MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH

# Global
window_size = 10
prediction_interval = 1
movement_time = 0.01

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
        
        return int(predicted_class), CLASSES[predicted_class]
    
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

def drone_command(class_id):
    """Convert gesture to drone command parameters."""
    
    commands = {
        0: {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},  # resting
        1: {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 50},  # open palm, go up
        2: {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': -50},  # closed fist, go down
        3: {'pitch': 50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},  # ok, go forward
        4: {'pitch': -50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},  # pointer finger, go back
        5: {'pitch': 0, 'roll': 0, 'yaw': -50, 'vertical_movement': 0},  # peace, rotate left
        6: {'pitch': 0, 'roll': 0, 'yaw': 50, 'vertical_movement': -50},  # sha, rotate right
    }
    return commands.get(class_id, commands[0])

async def main():   
    # Find and connect to Myo device
    
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device: 
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")

    print("Starting gesture control session...")
    print(f"Make gestures to control drone (updating every {prediction_interval})...")
    
    async with Myo(myo_device) as myo:
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)

        KEYBOARD_COMMANDS = {
            'space': 1,  
            'shift': 2,  
            'w': 3,      
            's': 4,      
            'e': 5,      
            'q': 6       
        }
        
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data):
            # Process the EMG data
            prediction = predictor.add_data(emg_data)
            
            if prediction:
                for key, gesture_id in KEYBOARD_COMMANDS.items():
                    if keyboard.is_pressed(key):
                        return (gesture_id, f"keyboard_(key)")
                    else:
                        class_id, gesture_name = prediction
            
                if class_id == 7:
                    return
                    # bebop.flip("back")
                else:
                    command = drone_command(class_id)

                    bebop.fly_direct(
                        roll=command['roll'],
                        pitch=command['pitch'],
                        yaw=command['yaw'],
                        vertical_movement=command['vertical_movement'],
                        duration=movement_time
                        )
                    
                    print(f"\rPredicted gesture: {gesture_name} (class {class_id})")

        # Set EMG mode and collect data
        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        print("Myo band activated succesfully, taking off in 5 seconds")

        try:
            while True:
                await asyncio.sleep(0.01)
        except KeyboardInterrupt:
            print("\nStopping gesture prediction...")
            await myo.set_mode(emg_mode=None)

if __name__ == "__main__":
    predictor = GesturePredictor()

    bebop = Bebop()

    keyboard.block_key('l')
    keyboard.block_key('w')
    keyboard.block_key('e')
    keyboard.block_key('s')
    keyboard.block_key('q')
    keyboard.block_key('SPACE')
    keyboard.block_key('SHIFT')

    try: 
        # Press L for emergency land
        if keyboard.is_pressed('l'):
            bebop.safe_land(10)
            exit()

        # # Initialize drone    
        print('Connecting to drone...')
        if bebop.connect(10):
            print('Connected')

        # Drone activation 
        bebop.ask_for_state_update()
        bebop.safe_takeoff(10)
        #bebop.set_max_altitude(1)

        asyncio.run(main())
    
    except:
        print('broke')
        bebop.safe_land(10)
        bebop.disconnect()