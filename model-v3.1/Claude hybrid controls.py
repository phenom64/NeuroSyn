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

# Global configuration
WINDOW_SIZE = 10
PREDICTION_INTERVAL = 1
MAX_ALTITUDE = 1
MOVEMENT_DURATION = 0.1

class GesturePredictor:
    def __init__(self):
        self.window_size = WINDOW_SIZE
        self.data_buffer = deque(maxlen=WINDOW_SIZE)
        self.prediction_interval = PREDICTION_INTERVAL
        self.last_prediction_time = 0
        
        # Load the model and metadata
        try:
            self.model = tf.keras.models.load_model(f"model/{MODEL_PATH}")
            with open(f"model/{METADATA_PATH}", 'rb') as f:
                self.scaler, self.columns = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or metadata: {str(e)}")
            
    def process_window(self):
        """Process the current window of data and return a prediction."""
        if len(self.data_buffer) < self.window_size:
            return None
            
        try:
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
        except Exception as e:
            print(f"Error processing window: {str(e)}")
            return None
    
    def add_data(self, emg_data):
        """
        Add new EMG data to the buffer and return prediction if enough time has passed.
        Returns prediction only once per prediction_interval seconds.
        """
        try:
            self.data_buffer.append(emg_data)
            
            current_time = time.time()
            time_since_last_prediction = current_time - self.last_prediction_time
            
            if (len(self.data_buffer) == self.window_size and 
                time_since_last_prediction >= self.prediction_interval):
                self.last_prediction_time = current_time
                return self.process_window()
            return None
        except Exception as e:
            print(f"Error adding data: {str(e)}")
            return None

class DroneController:
    def __init__(self):
        self.bebop = Bebop()
        self.is_connected = False
        self.is_flying = False
        
    async def connect(self):
        """Connect to the drone and initialize it."""
        print('Connecting to drone...')
        success = self.bebop.connect(10)
        
        if not success:
            raise RuntimeError("Failed to connect to Bebop")
            
        self.is_connected = True
        print("Drone connected successfully")
        
        # Initialize drone
        #self.bebop.smart_sleep(2)
        self.bebop.ask_for_state_update()
        self.bebop.set_max_altitude(MAX_ALTITUDE)
        
    async def takeoff(self):
        """Safely take off."""
        if not self.is_connected:
            raise RuntimeError("Drone not connected")
            
        print("Taking off...")
        success = self.bebop.safe_takeoff(10)
        self.is_flying = success
        return success
        
    async def land(self):
        """Safely land the drone."""
        if self.is_flying:
            print("Landing drone...")
            await asyncio.to_thread(self.bebop.safe_land, 10)
            self.is_flying = False
            
    def disconnect(self):
        """Disconnect from the drone."""
        if self.is_connected and not self.is_flying:
            self.bebop.disconnect()
            self.is_connected = False
        elif self.is_flying:
            print('Cannot disconect while drone is in flight')
        else:
            print('Could not discoect')    

    async def execute_command(self, command):
        """Execute a drone movement command."""
        if not self.is_flying:
            return
            
        try:
            await asyncio.to_thread(
                self.bebop.fly_direct,
                roll=command['roll'],
                pitch=command['pitch'],
                yaw=command['yaw'],
                vertical_movement=command['vertical_movement'],
                duration=MOVEMENT_DURATION
            )
        except Exception as e:
            print(f"Error executing drone command: {str(e)}")
            
    async def execute_flip(self):
        """Execute a flip maneuver."""
        if not self.is_flying:
            return
            
        try:
            await asyncio.to_thread(self.bebop.flip, "back")
        except Exception as e:
            print(f"Error executing flip: {str(e)}")

def get_drone_command(class_id):
    """Convert gesture to drone command parameters."""
    commands = {
        '0': {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},    # resting
        '1': {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 50},   # open palm, go up
        '2': {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': -50},  # closed fist, go down
        '3': {'pitch': 50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},   # ok, go forward
        '4': {'pitch': -50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},  # pointer finger, go back
        '5': {'pitch': 0, 'roll': 0, 'yaw': -50, 'vertical_movement': 0},  # peace, rotate left
        '6': {'pitch': 0, 'roll': 0, 'yaw': 50, 'vertical_movement': 0},   # sha, rotate right
        '8' : {'pitch': 0, 'roll': -50, 'yaw': 0, 'vertical_movement': 0},  # currently for the keyboard to move left
        '8' : {'pitch': 0, 'roll': 50, 'yaw': 0, 'vertical_movement': 0}, # currently for the keyboard to move right
    }
    return commands.get(str(class_id), commands['0'])

KEYBOARD_COMMANDS = {
    'space': 1,  # up
    'shift': 2,  # down
    'w': 3,      # forward
    's': 4,      # back
    'e': 5,      # rotate left
    'q': 6,      # rotate right
    'a' : 8,     # move left 
    'd' : 9,     # move right
}

class ControlSystem:
    def __init__(self):
        self.predictor = GesturePredictor()
        self.drone = DroneController()
        self.running = False
        
    async def process_emg(self, emg_data):
        """Process EMG data and execute corresponding drone commands."""
        prediction = self.predictor.add_data(emg_data)
        
        if prediction:
            class_id, gesture_name = prediction
            
            if class_id == 7:
                await self.drone.execute_flip()
            else:
                command = get_drone_command(class_id)
                await self.drone.execute_command(command)
                print(f"\rPredicted gesture: {gesture_name} (class {class_id})")
                
    async def process_keyboard(self):
        """Process keyboard inputs and execute corresponding drone commands."""
        while self.running:
            for key, gesture_id in KEYBOARD_COMMANDS.items():
                if keyboard.is_pressed(key):
                    command = get_drone_command(gesture_id)
                    await self.drone.execute_command(command)
            
            if keyboard.is_pressed('l'):  # Emergency land
                await self.drone.land()
                self.running = False
                break
                
            await asyncio.sleep(0.01)
            
    async def run(self):
        """Main control loop."""
        # Find and connect to Myo device
        myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
        if not myo_device:
            raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
            
        # Connect to drone
        await self.drone.connect()
        await self.drone.takeoff()
        
        print("Starting gesture control session...")
        print(f"Make gestures to control drone (updating every {PREDICTION_INTERVAL} seconds)...")
        
        self.running = True
        
        try:
            async with Myo(myo_device) as myo:
                await asyncio.sleep(0.5)
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
                await asyncio.sleep(0.25)
                
                # Set up EMG callback
                @myo.on_emg_smooth
                def on_emg_smooth(emg_data):
                    if self.running:
                        asyncio.create_task(self.process_emg(emg_data))
                
                # Start keyboard processing
                keyboard_task = asyncio.create_task(self.process_keyboard())
                
                # Set EMG mode
                await myo.set_mode(emg_mode=EmgMode.SMOOTH)
                print("Myo band activated successfully")
                
                # Wait for keyboard task to complete (emergency land)
                await keyboard_task
                
        except Exception as e:
            print(f"Error in control system: {str(e)}")
        finally:
            self.running = False
            await self.drone.land()
            self.drone.disconnect()

async def main():
    control_system = ControlSystem()
    await control_system.run()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram terminated by user")
    except Exception as e:
        print(f"Program terminated due to error: {str(e)}")