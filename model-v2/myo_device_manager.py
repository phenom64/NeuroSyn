from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, UnsupportedFeatureError, EmgValue, SleepMode

class MyoDeviceManager:
    def __init__(self, address):
        self.address = address

    async def connect(self):
        self.myo_device = await BleakScanner.find_device_by_address(self.address)

        if not self.myo_device:
            raise RuntimeError(f"Could not find Myo device with address {self.address}")

        async with Myo(self.myo_device) as self.myo:
            if not self.myo:
                raise RuntimeError(f"Could not find Myo device with address {self.address}")
            
            await self.myo.set_mode(emg_mode=EmgMode.EMG)
            await self.myo.set_sleep_mode(SleepMode.NEVER_SLEEP)

    async def on_emg(self, callback):
        #higher function that recieves other functions, callbacks
        @self.myo.on_emg
        def wrapper(emg_data: EmgValue):
            callback(emg_data)
        

    def disable_emg_callback(self):
        self.myo.on_emg = None

    def vibrate(myo):
        myo.vibrate2((100, 200), (50, 255))

    
