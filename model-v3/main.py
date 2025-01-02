import asyncio
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, EmgValue
from profile_manager import ProfileManager
from gesture_data_collector import GestureDataCollector
from constants import MYO_ADDRESS, COLLECTION_TIME, CLASSES

# Global Variables
current_gesture = None
columns = [f"Sensor{i}" for i in range(1, 17)] + ["Label"]  # Myo has 8 EMG sensors
gesture_labels = [key for key, _ in CLASSES.items()]

async def main():
    profile_name = input("Enter your profile name: ")
    profile_manager = ProfileManager()
    data_collector = GestureDataCollector()
    emg_data = []
    collector = []

    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    
    async with Myo(myo_device) as myo:

        # Profile creation - Collect profile data, calculate profile_means, and save
        print("Creating profile...")
        print("\nPreparing to collect profile data. Hold the 'outward palm' gesture...")
        await asyncio.sleep(3)

        @myo.on_emg
        def on_emg(emg_data: EmgValue):
            sensor1_data, sensor2_data = emg_data 
            emg_data = (*sensor1_data, *sensor2_data)
            print(emg_data)
            collector.append(emg_data)

        await asyncio.sleep(1)
        await myo.set_mode(emg_mode=EmgMode.EMG)
        await asyncio.sleep(0.5)

        # Wait for the specified collection time
        await asyncio.sleep(COLLECTION_TIME) # seconds
        if not collector:
            raise RuntimeError("No data was collected for the profile")

        @myo.on_emg
        def on_emg(emg_data: EmgValue):
            pass

        # determine average EMG value for each sensor
        sensor_values = list(zip(*collector))
        emg_profile = [sum(sensor) / len(sensor) for sensor in sensor_values]
        collector = []

        # Save the profile
        profile_manager.save_profile(profile_name, emg_profile)
        print(f"Profile for {profile_name} saved successfully.")
        
        # Data collection -  Collect and save gesture data
        print("Collecting gesture data...")
        rpt = int(input('How many repetitions of each gesture would you like to do? (maximum of 10)'))
        rpt = rpt if rpt <= 10 else 10
        
        print(f'\nYou will be required to complete {len(CLASSES)} gestures, {rpt} times')
        await asyncio.sleep(3)

        for idx, gesture_label in enumerate(gesture_labels):
            current_gesture = idx

            for repetition in range(rpt):  # 10 repetitions per gesture
                print(f"\nPerform '{CLASSES[gesture_label]}' gesture (repetition {repetition + 1}/10)")
        
                # Give time to prepare
                print("Preparing...")
                await asyncio.sleep(3)  # 3 seconds to react

                # Collect data
                @myo.on_emg
                def on_emg(emg_data: EmgValue):
                    sensor1_data, sensor2_data = emg_data 
                    emg_data = (*sensor1_data, *sensor2_data)
                    collector.append(emg_data)
        
                #
                print("Recording...")
                await asyncio.sleep(COLLECTION_TIME)
                #

                @myo.on_emg
                def on_emg(emg_data: EmgValue):
                    pass
                
                print(f"Collected {len(collector)} data points")
        
        all_data = collector
        filename = f"{profile_name}_gestures.csv"
        data_collector.save_data(filename, columns, all_data)

if __name__ == "__main__":
    asyncio.run(main())
