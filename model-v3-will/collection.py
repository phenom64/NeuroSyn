import asyncio
import pandas as pd
import uuid
from datetime import datetime
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, EmgValue, SleepMode

from constants import MYO_ADDRESS, CLASSES, COLLECTION_TIME

async def collect_gesture_data(myo, gesture_name, gesture_id, collection_time):
    gesture_data = []
    
    print(f"\nPreparing to collect data for '{gesture_name}' gesture...")
    print(f"Get ready in 2 seconds...")
    await asyncio.sleep(2)
    print(f"START - Hold the '{gesture_name}' gesture")
    
    try:
        # Define the callback before setting the mode
        @myo.on_emg_smooth
        def on_emg_smooth(emg_data: EmgValue):        
            current_time = datetime.now()
            timestamp = current_time.timestamp()
            gesture_data.append({
                'id': uuid.uuid4(),
                'time': timestamp,
                'gesture_id': gesture_id,
                'gesture_name': gesture_name,
                **{f's{i+1}': value for i, value in enumerate(emg_data)}
            })

        # Set EMG mode and collect data
        await myo.set_mode(emg_mode=EmgMode.SMOOTH)
        await asyncio.sleep(collection_time)
        
        print(f"DONE - '{gesture_name}' gesture data collected")        
        return gesture_data
        
    except Exception as e:
        print(f"Error during data collection for {gesture_name}: {str(e)}")
        return []

async def main():
    all_data = []
    start_time = datetime.now()

    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")
    
    async with Myo(myo_device) as myo:
        print("Starting gesture data collection session...")
        
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        # await myo.set_sleep_mode(SleepMode.NORMAL)
        await asyncio.sleep(0.25)
        
        # Collect data for each gesture
        for gesture_id, gesture_name in CLASSES.items():

            for repetition in range(10):
                print(f"\nPerform '{gesture_name}' gesture, (repetition {repetition + 1}/10)")

                try:
                    # Ensure EMG mode is reset before collecting new gesture
                    await myo.set_mode(emg_mode=None)
                    await asyncio.sleep(0.5)  # Small delay to ensure mode is reset
                    
                    gesture_data = await collect_gesture_data(
                        myo, 
                        gesture_name, 
                        gesture_id, 
                        COLLECTION_TIME
                    )
                    all_data.extend(gesture_data)
                    
                    await myo.vibrate2((100, 200), (50, 255))
                    
                    # Reset EMG mode after collection
                    await myo.set_mode(emg_mode=None)
                    
                    # # Small break between gestures
                    # if gesture_id != list(CLASSES.keys())[-1]:  # If not the last gesture
                    #     print("\nTake a small break - 5 seconds until next gesture")
                    #     await asyncio.sleep(5)
                        
                except Exception as e:
                    print(f"Error processing gesture {gesture_name}: {str(e)}")
                    continue

            print("\nTake a small break - 5 seconds until next gesture")
            await asyncio.sleep(5)

        
        if not all_data:
            raise RuntimeError("No data was collected")

        # Create DataFrame with all collected data
        df = pd.DataFrame(all_data)
        
        # Ensure correct column order
        columns = ['id', 'time', 'gesture_id', 'gesture_name'] + [f's{i}' for i in range(1, 9)]
        df = df[columns]
        
        # Add sequential IDs
        df['id'] = range(len(df))
        
        # Generate filename with timestamp
        timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
        filename = f'data/emg_gestures_data_{timestamp_str}.csv'
        
        # Save to CSV
        df.to_csv(filename, index=False)
        print(f"\nAll gesture data saved to {filename}")
        print(f"Total samples collected: {len(df)}")
        
        # Print summary
        print("\nSummary of collected data:")
        summary = df.groupby('gesture_name').size()
        for gesture, count in summary.items():
            print(f"{gesture}: {count} samples")

if __name__ == "__main__":
    asyncio.run(main())