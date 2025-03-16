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

from constants import MYO_ADDRESS, CLASSES, MODEL_PATH, METADATA_PATH, COLLECTION_TIME

# Global configuration
WINDOW_SIZE = 10
PREDICTION_INTERVAL = 1
MAX_ALTITUDE = 1
MOVEMENT_DURATION = 0.1
THRESHOLD = 0.9
NUM_REPETITIONS = 1

class GesturePredictor:
    def __init__(self):
        self.window_size = WINDOW_SIZE
        self.data_buffer = deque(maxlen=WINDOW_SIZE)
        self.prediction_interval = PREDICTION_INTERVAL
        self.last_prediction_time = 0
        self.threshold = THRESHOLD
        self.num_repetitions = NUM_REPETITIONS
        self.calibration_mean = None
        self.calibration_std = None
        self.is_calibrating = False
        self.calibration_complete = False
        
        # Load the model and metadata
        try:
            self.model = tf.keras.models.load_model(f"model/{MODEL_PATH}")
            with open(f"model/{METADATA_PATH}", 'rb') as f:
                self.scaler, self.columns = pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load model or metadata: {str(e)}")
        
        # Calibration buffer
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
        self.calibration_buffer = []  # Clear buffer
        self.calibration_complete = True

    def process_window(self):
        """Process the current window of z-score normalized data and return a prediction."""
        if len(self.data_buffer) < self.window_size or not self.calibration_complete:
            return None
        
        try:
            # Convert buffer to a numpy array and normalize to z-scores
            window_data = np.array(list(self.data_buffer))
            z_score_data = (window_data - self.calibration_mean) / self.calibration_std
            
            # Calculate RMS features
            rms_features = np.sqrt(np.mean(np.square(z_score_data), axis=0))
            
            # Scale the RMS values to match training preprocessing
            scaling_factor = 100  # Must match training script
            rms_features_scaled = rms_features * scaling_factor
            
            # Create a DataFrame with the correct column names
            features_df = pd.DataFrame([rms_features_scaled], columns=self.columns)
            
            # Scale the features using the loaded scaler
            features_scaled = self.scaler.transform(features_df)
            
            # Make prediction
            prediction = self.model.predict(features_scaled, verbose=0)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class]
            
            if predicted_class != 0 and confidence < self.threshold:
                return 0, CLASSES[0]
            
            return predicted_class, CLASSES[predicted_class]
        except Exception as e:
            print(f"Error processing window: {str(e)}")
            return None

    def add_data(self, emg_data):
        """Add new EMG data to the buffer and return prediction if enough time has passed."""
        try:
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
            print('Cannot disconnect while drone is in flight')
        else:
            print('Could not disconnect')    

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
        '1': {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': 50},   # palm up, go up
        '2': {'pitch': 0, 'roll': 0, 'yaw': 0, 'vertical_movement': -50},  # closed fist, go down
        '3': {'pitch': 50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},   # ok, go forward
        '4': {'pitch': -50, 'roll': 0, 'yaw': 0, 'vertical_movement': 0},  # pointer finger, go back
        '5': {'pitch': 0, 'roll': 0, 'yaw': -50, 'vertical_movement': 0},  # peace, rotate left
        '6': {'pitch': 0, 'roll': 0, 'yaw': 50, 'vertical_movement': 0},   # sha, rotate right
        '8': {'pitch': 0, 'roll': -50, 'yaw': 0, 'vertical_movement': 0},  # move left (keyboard)
        '9': {'pitch': 0, 'roll': 50, 'yaw': 0, 'vertical_movement': 0},   # move right (keyboard)
    }
    return commands.get(str(class_id), commands['0'])

KEYBOARD_COMMANDS = {
    'space': 1,  # up
    'shift': 2,  # down
    'w': 3,      # forward
    's': 4,      # back
    'e': 5,      # rotate left
    'q': 6,      # rotate right
    'a': 8,      # move left 
    'd': 9,      # move right
}

class ControlSystem:
    def __init__(self):
        self.predictor = GesturePredictor()
        self.drone = DroneController()
        self.running = False
        
    async def calibrate_gestures(self, myo):
        """Run the calibration process for all gestures."""
        print("Starting calibration process...")
        print(f"Please perform each gesture {self.predictor.num_repetitions} times for {COLLECTION_TIME} seconds each.")
        
        for class_id, gesture_name in CLASSES.items():
            for rep in range(self.predictor.num_repetitions):
                print(f"\nCalibration for '{gesture_name}' (class {class_id}), repetition {rep + 1}/{self.predictor.num_repetitions}")
                print(f"Get ready... (2 seconds)")
                await asyncio.sleep(2)
                
                print(f"Perform '{gesture_name}' now for {COLLECTION_TIME} seconds...")
                self.predictor.is_calibrating = True
                start_time = time.time()
                
                while time.time() - start_time < COLLECTION_TIME:
                    await asyncio.sleep(0.01)
                
                self.predictor.is_calibrating = False
                print(f"Finished collecting '{gesture_name}' repetition {rep + 1}")

        self.predictor.finalize_calibration()

    async def process_emg(self, emg_data):
        """Process EMG data and execute corresponding drone commands."""
        prediction = self.predictor.add_data(emg_data)
        
        if prediction:
            class_id, gesture_name = prediction
            
            if class_id == 7:  # peace among worlds
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
            
        print("Connecting to Myo device...")
        async with Myo(myo_device) as myo:
            await asyncio.sleep(0.5)
            await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
            await asyncio.sleep(0.25)
            
            # Define EMG callback for both calibration and prediction
            @myo.on_emg_smooth
            def on_emg_smooth(emg_data):
                if self.predictor.is_calibrating:
                    self.predictor.calibrate(emg_data)
                elif self.running:
                    asyncio.create_task(self.process_emg(emg_data))
            
            # Run calibration before drone takeoff
            await myo.set_mode(emg_mode=EmgMode.SMOOTH)
            await self.calibrate_gestures(myo)
            
            print("Starting gesture control session...")
            print(f"Make gestures to control drone (updating every {PREDICTION_INTERVAL} seconds)...")
            
            self.running = True
            keyboard_task = asyncio.create_task(self.process_keyboard())
            
            try:
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