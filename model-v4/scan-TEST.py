# quick_emg_test.py
import asyncio, math
from bleak import BleakScanner
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
from constants import MYO_ADDRESS

async def main():
    print("Scanning…")
    dev = await BleakScanner.find_device_by_address(MYO_ADDRESS, timeout=10)
    if not dev:
        print("Myo not found"); return
    print("Found, connecting…")

    async with Myo(dev) as m:
        await m.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await m.set_mode(emg_mode=EmgMode.SMOOTH)

        # —— this is *exactly* what NSE-interfaceFX.py registers ——
        @m.on_emg_smooth
        def _(data):
            strength = math.sqrt(sum(e*e for e in data))
            print(f"EMG {strength:.1f}")

        print("Listening for 5 s… move the armband now.")
        await asyncio.sleep(5)

asyncio.run(main())
