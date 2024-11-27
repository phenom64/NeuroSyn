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
current_gesture = None
gesture_labels = [key for key, _ in CLASSES.items()]
columns = [f"Sensor{i}" for i in range(1, 17)] + ["Label"]  # Myo has 8 EMG sensors
collection_time = 5
data_collection = []  # Renamed from emg_data to avoid confusion with callback parameter

async def main() -> None:
    global data_collection
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Find Myo device
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    
    async with Myo(myo_device) as myo:
        # Print device info
        print("Device name:", await myo.name)
        print("Battery level:", await myo.battery)
        print("Firmware version:", await myo.firmware_version)
        print("Firmware info:", await myo.info)
        
        # Define EMG callback
        @myo.on_emg
        def on_emg(emg_data: EmgValue):
            sensor1_data, sensor2_data = emg_data
            data_collection.append((*sensor1_data, *sensor2_data, current_gesture))
        
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
        
        print("Collecting EMG data. Please perform the gestures when prompted.")
        
        all_data = []
        
        # Collect data for each gesture
        for idx, gesture_label in enumerate(gesture_labels):
            current_gesture = idx
            
            for repetition in range(10):  # 10 repetitions per gesture
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

if __name__ == "__main__":
    first_name = input("Enter your first name in all lowercase, no spaces or special characters: ")
    asyncio.run(main())