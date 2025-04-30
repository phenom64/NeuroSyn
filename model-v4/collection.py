#!/usr/bin/env python3
"""
collection.py for NeuroSyn Physio Project (model-v6)
  • Patches pymyo’s classifier handler to ignore too-short packets.
  • Correctly registers IMU (orientation) + EMG callbacks.
  • Records synchronized EMG + quaternion.
"""

import asyncio
import os
import uuid
import struct
import types
import pandas as pd
from datetime import datetime
from bleak import BleakScanner, BleakError
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode

from constants import (
    MYO_ADDRESS,
    CLASSES,
    COLLECTION_TIME,
    REPETITIONS_PER_EXERCISE,
    PAUSE_DURATION,
    DATA_INPUT_PATH,
    RAW_DATA_FILENAME_PREFIX,
    NUM_EMG_SENSORS,
    NUM_IMU_VALUES,
)

# ─── Patch pymyo’s buggy classifier handler ─────────────────────────────
def _safe_on_classifier(self, sender, value: bytearray):
    # Ignore packets shorter than 3 bytes
    if len(value) < 3:
        return
    try:
        struct.unpack("<B2s", value)
    except struct.error:
        return
# Monkey-patch onto Myo
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# ────────────────────────────────────────────────────────────────────────

async def collect_one_repetition(myo: Myo, gesture_id: int, gesture_name: str):
    """
    Records EMG + IMU orientation for one repetition of one gesture.
    Returns a list of sample dicts.
    """
    samples = []
    imu_q = None

    # IMU callback signature: (orientation, accelerometer, gyroscope)
    def on_imu(orientation, accel, gyro):
        nonlocal imu_q
        imu_q = orientation  # quaternion tuple

    # EMG callback signature: (emg_data)
    def on_emg(emg_data):
        nonlocal imu_q, samples
        if (
            len(emg_data) == NUM_EMG_SENSORS
            and imu_q is not None
            and len(imu_q) >= NUM_IMU_VALUES
        ):
            ts = datetime.now().timestamp()
            entry = {
                "id": uuid.uuid4().hex,
                "time": ts,
                "gesture_id": gesture_id,
                "gesture_name": gesture_name,
            }
            # EMG channels
            for i, val in enumerate(emg_data, start=1):
                entry[f"s{i}"] = val
            # quaternion fields
            entry.update({
                "quat_w": imu_q[0],
                "quat_x": imu_q[1],
                "quat_y": imu_q[2],
                "quat_z": imu_q[3],
            })
            samples.append(entry)

    # Prep message
    print(f"\n--- Prepare '{gesture_name}' (class {gesture_id}) in {PAUSE_DURATION}s …")
    await asyncio.sleep(PAUSE_DURATION)
    print(f"--- Recording '{gesture_name}' for {COLLECTION_TIME}s …")

    # Register callbacks
    myo.on_imu(on_imu)
    myo.on_emg_smooth(on_emg)

    # Wake + enable EMG+IMU
    await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
    await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=True)

    # Record for the duration
    await asyncio.sleep(COLLECTION_TIME)

    # Teardown
    print(f"--- Stopping recording for '{gesture_name}' …")
    await myo.set_mode(emg_mode=None, imu_mode=False)
    await myo.set_sleep_mode(SleepMode.NORMAL)

    print(f"→ Collected {len(samples)} samples.")
    return samples


async def main():
    os.makedirs(DATA_INPUT_PATH, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DATA_INPUT_PATH, f"{RAW_DATA_FILENAME_PREFIX}{ts_str}.csv")

    print(f"--- NeuroSyn Physio Data Collection (v6) ---")
    print(f"Saving to: {out_file}")
    print("Exercises to record:")
    for gid, name in CLASSES.items():
        print(f"  {gid}: {name}")
    print(f"\nEach: {REPETITIONS_PER_EXERCISE} reps, {COLLECTION_TIME}s each, pause {PAUSE_DURATION}s")
    print("-" * 50)

    try:
        print(f"Scanning for Myo at {MYO_ADDRESS} …")
        device = await BleakScanner.find_device_by_address(MYO_ADDRESS, timeout=10.0)
    except BleakError as e:
        print("Bluetooth scan error:", e)
        return

    if not device:
        print(f"ERROR: Could not find Myo at {MYO_ADDRESS}.")
        return

    print(f"Found device: {device.name} ({device.address})")

    all_samples = []
    try:
        async with Myo(device) as myo:
            print("Connected to Myo. Initializing …")
            await asyncio.sleep(1.0)

            for gid, gname in CLASSES.items():
                for rep in range(1, REPETITIONS_PER_EXERCISE + 1):
                    print(f"\nStarting: [{gname}] rep {rep}/{REPETITIONS_PER_EXERCISE}")
                    repsamp = await collect_one_repetition(myo, gid, gname)
                    all_samples.extend(repsamp)
                    await asyncio.sleep(0.5)

            if not all_samples:
                print("WARNING: No data collected. Exiting.")
                return

            # Build DataFrame
            df = pd.DataFrame(all_samples)
            cols = (
                ["id", "time", "gesture_id", "gesture_name"] +
                [f"s{i}" for i in range(1, NUM_EMG_SENSORS+1)] +
                ["quat_w","quat_x","quat_y","quat_z"]
            )
            # Ensure all columns exist
            for c in cols:
                if c not in df.columns:
                    df[c] = pd.NA
            df = df[cols]

            print(f"\nSaving {len(df)} samples to CSV: {out_file}")
            df.to_csv(out_file, index=False)
            print("Save complete.")

    except Exception as e:
        print("ERROR during collection:", e)

    finally:
        print("\nCollection script finished.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt—exiting.")
