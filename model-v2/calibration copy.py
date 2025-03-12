import asyncio
import csv
import os
import uuid
from bleak import BleakScanner
from pathlib import Path
from pymyo import Myo
from pymyo.types import EmgMode, EmgValue, UnsupportedFeatureError, SleepMode
from constants import (
    CLASSES,
    MYO_ADDRESS
)

# Global variables
current_gesture = -1 #set to -1 for now as all other values could be 0 or higher
gesture_labels = [key for key, _ in CLASSES.items()]
columns = [f"Sensor{i}" for i in range(1, 17)] + ["Label"]  # Myo has 8 EMG sensors
collection_time = 5
data_collection = []  # Renamed from emg_data to avoid confusion with callback parameter
profile_means = None

async def collect_profile_data(myo, collection_time = 5) -> list:
    """Collect EMG data for profile creation."""
    global data_collection
    
    print("\nPreparing to collect profile data. Wait for the vibration and hold 'outward palm' gesture...")
    await asyncio.sleep(3)  # Preparation time
    
    print("Recording profile...")
    await asyncio.sleep(collection_time)
    
    profile_data = data_collection.copy()
    data_collection = []
    
    await myo.vibrate2((100, 200), (50, 255))
    return profile_data

def calculate_profile(profile_data: list) -> list:
    """Calculate average EMG values for each sensor."""
    # Exclude the label (last column) when calculating means
    sensor_values = list(zip(*[data[:-1] for data in profile_data]))
    return [sum(sensor)/len(sensor) for sensor in sensor_values]

def load_profile(first_name: str) -> list:
    """Load user's EMG profile."""
    profile_path = Path(f"profiles/{first_name}_profile.csv")
    if not profile_path.exists():
        raise FileNotFoundError(f"Profile not found for {first_name}")
    
    with open(profile_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        return [float(val) for val in next(reader)]

def normalize_data(data: list, profile_means: list) -> list:
    """Normalize EMG values using profile means."""
    return [val/mean if mean != 0 else val for val, mean in zip(data[:-1], profile_means)] + [data[-1]]

async def main() -> None:
    global data_collection, current_gesture, profile_means

        #Define EMG callback
        @myo.on_emg
        def on_emg(emg_data: EmgValue):
            sensor1_data, sensor2_data = emg_data
            raw_data = (*sensor1_data, *sensor2_data, current_gesture)
            normalized_data = normalize_data(raw_data, profile_means)
            data_collection.append(normalized_data)
        
        @myo.on_emg
        def on_emg(emg_data: EmgValue):
            if profile_means == None:
                sensor1_data, sensor2_data = emg_data
                data_collection.append((*sensor1_data, *sensor2_data, current_gesture))
            
            else:
                sensor1_data, sensor2_data = emg_data
                raw_data = (*sensor1_data, *sensor2_data, current_gesture)
                normalized_data = normalize_data(raw_data, profile_means)
                data_collection.append(normalized_data)
            
        # Try to enable battery notifications
        try:
            await myo.enable_battery_notifications()
        except UnsupportedFeatureError as e:
            print(f"Battery notifications not supported: {e}")
        
        # Set EMG mode
        await asyncio.sleep(1)
        await myo.set_mode(emg_mode=EmgMode.EMG)
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await asyncio.sleep(0.25)
        
        # Collect profile data first
        print("\nFirst, let's create your EMG profile...")
        await asyncio.sleep(2)
        profile_data = await collect_profile_data(myo)
        await asyncio.sleep(2)
        profile_means = calculate_profile(profile_data)
        
        # Save profile
        profile_filename = Path(f"profiles/{first_name}_profile.csv")
        with open(profile_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Sensor{i}" for i in range(1, 17)])
            writer.writerow(profile_means)
        
        print(f"Profile saved to {profile_filename}")
        await asyncio.sleep(1)

        print("Collecting EMG data. Please perform the gestures when prompted.")
        await asyncio.sleep(2)
        
        all_data = []
        
        # Collect data for each gesture
        for idx, gesture_label in enumerate(gesture_labels):
            current_gesture = idx
            
            for repetition in range(1):  # 10 repetitions per gesture
                print(f"\nPerform '{CLASSES[gesture_label]}' gesture (repetition {repetition + 1}/10)")
        
                # Give time to prepare
                print("Preparing...")
                await asyncio.sleep(3)  # 3 seconds to react
                
                #
                print("Recording...")
                await asyncio.sleep(collection_time)
                #
                
                for data in data_collection:
                    all_data.append(data)
                
                print(f"Collected {len(data_collection)} data points")
                data_collection = []
                
                await myo.vibrate2((100, 200), (50, 255))
                await asyncio.sleep(0.5)
        
        print("\nFinished collecting data.")
        
        # Save to CSV
        filename = Path(f"data/{gesture_label}_{repetition}_{first_name}.csv")
        while os.path.exists(str(filename)):
            filename = filename.stem + uuid.uuid4().hex[:4] + filename.suffix
        
        with open(filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            
            for row in all_data:
                writer.writerow(row)
        
        print(f"Data saved to {filename}")
        print(profile_means)

if __name__ == "__main__":
    first_name = input("Enter your first name in all lowercase, no spaces or special characters: ")
    asyncio.run(main())