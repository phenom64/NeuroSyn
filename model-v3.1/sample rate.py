from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
import time
import asyncio
import numpy as np
from bleak import BleakScanner
from collections import deque


from constants import MYO_ADDRESS 

stats_interval = 5
sample_times = deque(maxlen=1000)

def calculate_sampling_rate():
    interval=np.diff(list(sample_times))
    average_interval=np.mean(interval)
    sampling_rate=1/average_interval
    sample_times=()
    return sampling_rate

def get_timing_stats():
    """Return current timing statistics."""
    sampling_rate = calculate_sampling_rate()
    
    return {
        'sampling_rate_hz': sampling_rate if sampling_rate else 0,
    }

myo_device = BleakScanner.find_device_by_address(MYO_ADDRESS)
if not myo_device:
            raise RuntimeError(f"Could not find Myo device with address {MYO_ADDRESS}")

async def main():

    @myo_device.on_emg_smooth
    def on_emg_smooth(emg_data):
        current_time = time.time()
        sample_times.append(current_time)

        if current_time - last_stats_time >= stats_interval:
                stats = get_timing_stats()
                print("\nTiming Statistics:")
                print(f"EMG Sampling Rate: {stats['sampling_rate_hz']:.2f} Hz ({stats['n_samples']} samples)")
                last_stats_time = current_time

    async with Myo(myo_device) as myo:
        await asyncio.sleep(0.5)
        await myo.set_sleep_mode(SleepMode.NORMAL)
        await asyncio.sleep(0.25)

    myo.set_mode(emg_mode=EmgMode.SMOOTH)

if __name__ == "__main__":
       asyncio.run(main())