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

# async def create_profile(myo: Myo, data_collector, profile_manager, profile_name):
#     # @myo.on_emg
#     # def emg_callback(emg_data):
#     #     print("Collecting data")
#     #     data_collector.add_data((*emg_data[0], *emg_data[1], current_gesture))

    
#     @myo.on_emg
#     def on_emg(emg_data: EmgValue):
#         sensor1_data, sensor2_data = emg_data
#         data_collection.append((*sensor1_data, *sensor2_data, current_gesture))

#     def calculate_profile(profile_data: list) -> list:
#         """Calculate average EMG values for each sensor."""
#         sensor_values = list(zip(*[data[:-1] for data in profile_data]))  # Exclude gesture label
#         return [sum(sensor) / len(sensor) for sensor in sensor_values]

#     current_gesture = -1

#     data_collection = data_collector.data_collection

#     # initialize
#     myo.on_emg()

#     # Wait for the specified collection time
#     await asyncio.sleep(COLLECTION_TIME) # seconds

#     # Stop collecting EMG data
#     myo.on_emg = None
    
#     # Retrieve collected data
#     profile_data = data_collector.data_collection
    
#     if not profile_data:
#         raise RuntimeError("No data was collected for the profile")
    
#     # Calculate the profile (average sensor values)
#     profile_means = calculate_profile(profile_data)
#     print("Profile data collected and processed.")

#     # Save the profile
#     profile_manager.save_profile(profile_name, profile_means)
#     print(f"Profile for {profile_name} saved successfully.")

#     # # # # # # # # ## ## # # # # # # # # # # # #
#     #data_collection = []
#     data_collector.clear_data()
#     # # # # # # # # ## ## # # # # # # # # # # # #

#     return profile_means

async def gesture_data_collection(myo_manager, data_collector, profile_manager, emg_data, current_gesture, rpt):
    profile_means = profile_manager.load_profile()

    def data_normalizer(data: list, profile_means: list) -> list:
        """Normalize EMG values using profile means."""
        return [val/mean if mean != 0 else val for val, mean in zip(data[:-1], profile_means)] + [data[-1]]

    def profile_emg_callback():
        data = ((*emg_data[0], *emg_data[1]))
        data_collector.add_data(data_normalizer(data, profile_means))
    
    data_collector.clear_data()
    gesture_labels = [key for key, _ in CLASSES.items()]

    for idx, gesture_label in enumerate(gesture_labels):
        current_gesture = idx

        for repetition in range(rpt):  # 10 repetitions per gesture
            print(f"\nPerform '{CLASSES[gesture_label]}' gesture (repetition {repetition + 1}/10)")
    
            # Give time to prepare
            print("Preparing...")
            await asyncio.sleep(3)  # 3 seconds to react

            # Collect data
            await profile_emg_callback()
    
            #
            print("Recording...")
            await asyncio.sleep(COLLECTION_TIME)
            #

            await myo_manager.disable_emg_callback()
            
            for data in data_collection:
                data_collector.add_data(data)
            
            print(f"Collected {len(data_collection)} data points")
               
    # # # # # # # # ## ## # # # # # # # # # # # #
            #data_collection = []
    # # # # # # # # ## ## # # # # # # # # # # # #

            await myo_manager.vibrate()
            await asyncio.sleep(0.5)

async def main():
    profile_name = input("Enter your profile name: ")
    #myo_manager = MyoDeviceManager(MYO_ADDRESS)
    profile_manager = ProfileManager()
    data_collector = GestureDataCollector()
    emg_data = []
    collector = []

    COLLECTION_TIME = 5

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
        
        await myo.set_mode(emg_mode=None)

        # determine average EMG value for each sensor
        sensor_values = list(zip(*collector))
        emg_profile = [sum(sensor) / len(sensor) for sensor in sensor_values]

        # Save the profile
        profile_manager.save_profile(profile_name, emg_profile)
        print(f"Profile for {profile_name} saved successfully.")
        
        # Data collection -  Collect and save gesture data
        print("Collecting gesture data...")
        rpt = int(input('How many repetitions of each gesture would you like to do? (maximum of 10)'))
        rpt = rpt if rpt <= 10 else 10
        
        print(f'\nYou will be required to complete {len(CLASSES)} gestures, {rpt} times')

        await asyncio.sleep(3)
                
        all_data = data_collector.data_collection
        filename = f"{profile_name}_gestures.csv"
        data_collector.save_data(filename, columns, all_data)

if __name__ == "__main__":
    asyncio.run(main())
