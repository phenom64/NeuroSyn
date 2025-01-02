from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, EmgValue
from constants import MYO_ADDRESS, COLLECTION_TIME, CLASSES
import asyncio

async def main():
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")

    async with Myo(myo_device) as myo:
        
         # Print device info
        print("Device name:", await myo.name)
        print("Battery level:", await myo.battery)
        print("Firmware version:", await myo.firmware_version)
        print("Firmware info:", await myo.info)

        @myo.on_emg
        def on_emg(emg_data: EmgValue):
            sensor1_data, sensor2_data = emg_data 
            emg_data = (*sensor1_data, *sensor2_data)
            print(emg_data)
            collector.append(emg_data)    

if __name__ == "__main__":
    asyncio.run(main())
