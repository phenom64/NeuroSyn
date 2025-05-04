import asyncio
import numpy as np
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode

#constants
try:
    from constants import (
        NUM_EMG_SENSORS,
        CALIBRATION_POSES,
        PAUSE_DURATION,
        CALIBRATION_HOLD_TIME
    )
except ImportError as e:
    print(f"ERROR (calibration.py): Failed to import constants: {e}")
    raise # Or exit()

async def run_calibration_phase(myo: Myo):
    """Guides user through calibration poses and calculates EMG mean/std dev."""
    print("\n" + "="*20 + " EMG Calibration Phase " + "="*20)
    all_calibration_emg = []; calibration_active = True
    def on_calib_emg(emg_data):
        nonlocal all_calibration_emg, calibration_active
        if calibration_active and emg_data is not None and len(emg_data) == NUM_EMG_SENSORS: all_calibration_emg.append(emg_data)
    print("Preparing Myo for calibration...");
    try:
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP);
        await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=False);
        myo.on_emg_smooth(on_calib_emg);
        print("Allowing Myo to stabilize...")
        await asyncio.sleep(1.0)
        print("Myo ready.")
    except Exception as e:
        print(f"ERROR: Failed to set Myo mode for calibration: {e}");
        return None, None
    for pose_name, pose_id in CALIBRATION_POSES.items():
        print(f"\n--- Prepare Calibration Pose: '{pose_name}' in {PAUSE_DURATION}s …"); await asyncio.sleep(PAUSE_DURATION)
        print(f"--- HOLD '{pose_name}' for {CALIBRATION_HOLD_TIME}s …"); calibration_active = True; await asyncio.sleep(CALIBRATION_HOLD_TIME); calibration_active = False
        print(f"--- Relax... ({PAUSE_DURATION}s)"); await asyncio.sleep(PAUSE_DURATION)
    try: await myo.set_mode(emg_mode=None, imu_mode=False); await myo.set_sleep_mode(SleepMode.NORMAL)
    except Exception as e: print(f"Warning: Failed to reset Myo mode after calibration: {e}")
    print("\nCalibration data collection complete.")
    if not all_calibration_emg: print("ERROR: No EMG data collected during calibration!"); return None, None
    try:
        emg_array = np.array(all_calibration_emg); emg_mean = np.mean(emg_array, axis=0); emg_std = np.std(emg_array, axis=0)
        print(f"Calculated EMG Mean: {np.round(emg_mean, 2)}"); print(f"Calculated EMG Std Dev: {np.round(emg_std, 2)}")
        print("="*25 + " Calibration Complete " + "="*25 + "\n"); return emg_mean, emg_std
    except Exception as e: print(f"ERROR: Failed to calculate EMG statistics: {e}"); return None, None