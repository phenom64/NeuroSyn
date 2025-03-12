import asyncio
from collections import deque
from typing import Deque, Tuple
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, EmgValue, SleepMode

import numpy as np

from classifier import Classifier
from constants import (
    MYO_ADDRESS,
    MODEL_PATH,
    METADATA_PATH,
    CLASSES,
)

class EMGProcessor:
    def __init__(self, classifier: Classifier, window_size: int = 60):
        """
        Initialize EMG processor with a classifier and window size.
        window_size: Number of samples to keep in buffer (assuming 200Hz sampling rate,
                    20 samples = 0.1s of data)
        """
        self.classifier = classifier
        self.window_size = window_size
        self.emg_buffer: Deque[Tuple[EmgValue, EmgValue]] = deque(maxlen=window_size)
        self.last_prediction_time = 0
        self.prediction_interval = 0.2  # seconds

    def add_sample(self, emg_data: Tuple[EmgValue, EmgValue]) -> None:
        """Add a new EMG sample to the buffer"""
        self.emg_buffer.append(emg_data)

    def should_predict(self, current_time: float) -> bool:
        """Check if enough time has passed for a new prediction"""
        return (current_time - self.last_prediction_time) >= self.prediction_interval

    def make_prediction(self, current_time: float) -> str:
        """Make a prediction using the current buffer of EMG data"""
        if len(self.emg_buffer) < self.window_size:
            return "Insufficient data"

        # Use the most recent sample for prediction
        latest_data = self.emg_buffer
        latest_data = np.sqrt(np.mean(np.square(latest_data), axis=0))
            
        #print(latest_data.shape)

        predicted_class = self.classifier.classify(latest_data)
        
        # Print the EMG data and prediction
        print(f"\nTime: {current_time:.1f}s")
        print(f"EMG Data (mV) - Sensors 1-8: {[f'{mv:.5f}' for mv in latest_data[0]]}")
        print(f"EMG Data (mV) - Sensors 9-16: {[f'{mv:.5f}' for mv in latest_data[1]]}")
        print(f"Predicted class: {predicted_class}")

        self.last_prediction_time = current_time
        return predicted_class

async def main() -> None:
    # Find Myo device
    myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
    if not myo_device:
        raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")

    # Initialize EMG processor
    emg_processor = EMGProcessor(classifier)
    start_time = asyncio.get_event_loop().time()

    async with Myo(myo_device) as myo:
        print("Device name:", await myo.name)
        print("Battery level:", await myo.battery)
        print("Firmware version:", await myo.firmware_version)
        print("Firmware info:", await myo.info)

        @myo.on_emg
        def on_emg(emg_data: Tuple[EmgValue, EmgValue]):
            # Just add the data to the buffer, don't process yet
            emg_processor.add_sample(emg_data)

        await asyncio.sleep(1)
        await myo.set_mode(emg_mode=EmgMode.EMG)
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)

        print("\nCollecting data in millivolts... Press Ctrl+C to stop")
        
        try:
            while True:
                current_time = asyncio.get_event_loop().time() - start_time
                
                # Check if it's time to make a prediction
                if emg_processor.should_predict(current_time):
                    emg_processor.make_prediction(current_time)
                
                await asyncio.sleep(0.01)  # Small sleep to prevent CPU overuse
                
        except KeyboardInterrupt:
            print("\nStopping data collection...")

if __name__ == "__main__":
    classifier = Classifier(MODEL_PATH, METADATA_PATH)
    asyncio.run(main())