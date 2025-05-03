#!/usr/bin/env python3
"""
combined_collection.py for NeuroSyn Physio Project
Combines EMG, IMU, and real-time landmark data collection for gesture recognition.
Uses multiprocessing for OpenCV/MediaPipe isolation.
Uses separate MediaPipe Pose and Hands(max_num_hands=1) models.
Extracts LEFT Arm Pose (11,13,15) and RIGHT Hand landmarks.
NORMALIZATION: All extracted landmarks normalized relative to Pose LEFT Shoulder (index 11).
*** FIX: Cleaned up console output, added professional splash screen. ***
"""

import asyncio
import os
import uuid
import struct
import types
import pandas as pd
# cv2 and mediapipe are imported inside the worker process
import numpy as np
import multiprocessing as mp # Use multiprocessing
import traceback # For detailed error reporting
from datetime import datetime
from bleak import BleakScanner, BleakError
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
import time # For potential sleeps

# --- Constants Loading ---
# <<< Constants are imported here, but the print block is removed >>>
try:
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
        # Add other constants if needed by the main script/splash screen
    )
    # <<< Removed the old header print block from here >>>
except ImportError:
    print("ERROR: Failed to import constants.py. Critical configuration missing.")
    # Define minimal defaults ONLY if absolutely necessary for basic script structure,
    # but ideally, the script should fail if constants are missing.
    MYO_ADDRESS = "FF:FF:FF:FF:FF:FF"; CLASSES = {0: 'Rest'}; COLLECTION_TIME = 1
    REPETITIONS_PER_EXERCISE = 1; PAUSE_DURATION = 1; DATA_INPUT_PATH = "data"
    RAW_DATA_FILENAME_PREFIX = "error_data_"; NUM_EMG_SENSORS = 8; NUM_IMU_VALUES = 4
    print("WARNING: Running with minimal default values due to import error.")
except Exception as e: print(f"ERROR loading constants: {e}"); exit()
# --- End Constants Loading ---

# --- Define Landmarks to Extract ---
# (Landmark definitions remain the same)
POSE_LANDMARKS_TO_EXTRACT = {
    "Pose_L_Shoulder": 11, "Pose_L_Elbow": 13, "Pose_L_Wrist": 15,
}
POSE_SHOULDER_INDEX_FOR_NORM = 11
HAND_LANDMARK_BASE_NAMES = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip", "Index_MCP", "Index_PIP",
    "Index_DIP", "Index_Tip", "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip", "Pinky_MCP", "Pinky_PIP",
    "Pinky_DIP", "Pinky_Tip"
]
POSE_LANDMARK_NAMES_NORM = [ f"{name}_{coord}_shldr_norm" for name in POSE_LANDMARKS_TO_EXTRACT.keys() for coord in ["x", "y", "z", "vis"] ]
HAND_LANDMARK_NAMES_NORM = [ f"Hand_R_{name_part}_{coord}_shldr_norm" for name_part in HAND_LANDMARK_BASE_NAMES for coord in ["x", "y", "z"] ]
ALL_LANDMARK_NAMES = POSE_LANDMARK_NAMES_NORM + HAND_LANDMARK_NAMES_NORM
# --- End Landmark Definitions ---


# ─── Patch pymyo's buggy classifier handler ─────────────────────────────
def _safe_on_classifier(self, sender, value: bytearray):
    if len(value) < 3: return
    try: struct.unpack("<B2s", value)
    except struct.error: return
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# ────────────────────────────────────────────────────────────────────────

# <<< NEW: Function to print the professional splash screen >>>
def print_splash_screen():
    """Prints the application header/splash screen."""
    SCRIPT_VERSION = "5.5" # Define script version
    COPYRIGHT_YEAR = 2025 # Or use datetime.now().year
    DESIGNER = "Kavish Krishnakumar"
    LOCATION = "Manchester"

    print("=" * 60)
    print("--- NeuroSyn Physio Collection Assistant ---")
    print(f"Version {SCRIPT_VERSION}")
    print(f"TM & (C) {COPYRIGHT_YEAR} Syndromatic Inc. All rights reserved.")
    print(f"Designed by {DESIGNER} in {LOCATION}.")
    print("-" * 60)
    # Optionally print key config loaded from constants, but keep it brief
    print(f"Target Myo Address: {MYO_ADDRESS}")
    print(f"Data Output Path: ./{DATA_INPUT_PATH}/")
    print(f"Gesture Classes: {len(CLASSES)}")
    print("=" * 60)
    print("\n") # Add a newline for spacing

# (get_landmarks function remains the same as the working version 5.4)
def get_landmarks_pose_hands_shoulder_norm(frame, pose_results, hands_results, cv2_module, mp_drawing_module, mp_pose_module, mp_hands_module):
    """ Extracts LEFT Pose (arm) and RIGHT Hand landmarks, normalizes ALL relative to the Pose LEFT shoulder landmark. """
    if cv2_module is None or mp_drawing_module is None or mp_pose_module is None or mp_hands_module is None: return None, None, None
    normalized_landmarks = {}; original_pose_lm_list = None; original_right_hand_lm_list = None; shoulder_coords = None
    if pose_results and pose_results.pose_landmarks:
        original_pose_lm_list = pose_results.pose_landmarks
        try:
            shoulder_landmark = original_pose_lm_list.landmark[POSE_SHOULDER_INDEX_FOR_NORM]
            if shoulder_landmark.visibility > 0.5: shoulder_coords = (shoulder_landmark.x, shoulder_landmark.y, shoulder_landmark.z)
        except IndexError: print(f"ERROR: Pose shoulder index ({POSE_SHOULDER_INDEX_FOR_NORM}) OOB.")
        except Exception as pose_e: print(f"ERROR: Finding shoulder: {pose_e}"); traceback.print_exc()
    if shoulder_coords:
        shoulder_x, shoulder_y, shoulder_z = shoulder_coords
        if original_pose_lm_list:
            try:
                pose_landmark_idx_norm = 0
                for name, index_orig in POSE_LANDMARKS_TO_EXTRACT.items():
                    landmark = original_pose_lm_list.landmark[index_orig]
                    if pose_landmark_idx_norm < len(POSE_LANDMARK_NAMES_NORM):
                        key_x, key_y, key_z, key_vis = POSE_LANDMARK_NAMES_NORM[pose_landmark_idx_norm : pose_landmark_idx_norm+4]
                        normalized_landmarks[key_x] = landmark.x - shoulder_x; normalized_landmarks[key_y] = landmark.y - shoulder_y
                        normalized_landmarks[key_z] = landmark.z - shoulder_z; normalized_landmarks[key_vis] = landmark.visibility
                        pose_landmark_idx_norm += 4
                    else: break
            except IndexError: print(f"ERROR: Pose index OOB during norm.")
            except Exception as pose_norm_e: print(f"ERROR: Normalizing pose: {pose_norm_e}"); traceback.print_exc()
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    original_right_hand_lm_list = hand_landmarks
                    try:
                        hand_landmark_idx_norm = 0
                        for i, landmark in enumerate(original_right_hand_lm_list.landmark):
                            if hand_landmark_idx_norm < len(HAND_LANDMARK_NAMES_NORM):
                                key_x, key_y, key_z = HAND_LANDMARK_NAMES_NORM[hand_landmark_idx_norm : hand_landmark_idx_norm+3]
                                normalized_landmarks[key_x] = landmark.x - shoulder_x; normalized_landmarks[key_y] = landmark.y - shoulder_y
                                normalized_landmarks[key_z] = landmark.z - shoulder_z
                                hand_landmark_idx_norm += 3
                            else: break
                    except Exception as hand_norm_e: print(f"ERROR: Normalizing hand: {hand_norm_e}"); traceback.print_exc(); original_right_hand_lm_list = None; [normalized_landmarks.pop(key, None) for key in HAND_LANDMARK_NAMES_NORM]
                    break
    return normalized_landmarks, original_pose_lm_list, original_right_hand_lm_list


# (opencv_worker function remains the same as the working version 5.4)
def opencv_worker(shared_landmarks_proxy, lock, stop_event):
    """Worker PROCESS using Pose+Hands, draws only Left Arm + Right Hand."""
    cv2 = None; mp = None; mp_drawing = None; mp_pose = None; mp_hands = None
    try:
        import cv2; import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils; mp_pose = mp.solutions.pose; mp_hands = mp.solutions.hands
        print("[Worker Process] OpenCV and MediaPipe imported.")
    except ImportError as import_err: print(f"ERROR: [Worker Process] Failed imports: {import_err}"); traceback.print_exc(); return
    except Exception as general_import_err: print(f"ERROR: [Worker Process] Unexpected import error: {general_import_err}"); traceback.print_exc(); return

    pose_instance = None; hands_instance = None; cap = None
    try:
        print("[Worker Process] Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened(): print("ERROR: [Worker Process] Could not open camera!"); return
        print("[Worker Process] Camera opened.")
        print("[Worker Process] Initializing MediaPipe Pose...")
        try:
            pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("[Worker Process] MediaPipe Pose initialized.")
        except Exception as mp_init_e:
            print(f"ERROR: Init Pose failed: {mp_init_e}")
            traceback.print_exc()
            if cap:
                cap.release()
            return

        print("[Worker Process] Initializing MediaPipe Hands...")
        try:
            hands_instance = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
            print("[Worker Process] MediaPipe Hands initialized (max_num_hands=1).")
        except Exception as mp_init_e:
            print(f"ERROR: Init Hands failed: {mp_init_e}")
            traceback.print_exc()
            if pose_instance:
                pose_instance.close()
            if cap:
                cap.release()
            return

        left_arm_connections_drawing = [(POSE_LANDMARKS_TO_EXTRACT["Pose_L_Shoulder"], POSE_LANDMARKS_TO_EXTRACT["Pose_L_Elbow"]), (POSE_LANDMARKS_TO_EXTRACT["Pose_L_Elbow"], POSE_LANDMARKS_TO_EXTRACT["Pose_L_Wrist"])]
        left_arm_indices_to_draw = list(POSE_LANDMARKS_TO_EXTRACT.values())

        while not stop_event.is_set():
            ret, frame = cap.read();
            if not ret: time.sleep(0.01); continue
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results_pose = pose_instance.process(rgb_frame); results_hands = hands_instance.process(rgb_frame)
            rgb_frame.flags.writeable = True
            normalized_landmarks_dict, original_pose_lm_list, original_right_hand_lm_list = get_landmarks_pose_hands_shoulder_norm(frame, results_pose, results_hands, cv2, mp_drawing, mp_pose, mp_hands)
            if lock is None: print("ERROR: [Worker Process] Lock object is None!"); break
            try:
                with lock: shared_landmarks_proxy.clear();
                if normalized_landmarks_dict: shared_landmarks_proxy.update(normalized_landmarks_dict)
            except Exception as lock_e: print(f"ERROR: [Worker Process] Lock exception: {lock_e}"); traceback.print_exc(); break
            if original_pose_lm_list:
                pose_landmarks_list = original_pose_lm_list.landmark; image_height, image_width, _ = frame.shape
                for connection in left_arm_connections_drawing:
                    start_idx, end_idx = connection[0], connection[1]
                    try:
                        if (pose_landmarks_list[start_idx].visibility > 0.5 and pose_landmarks_list[end_idx].visibility > 0.5):
                            start_point = (int(pose_landmarks_list[start_idx].x * image_width), int(pose_landmarks_list[start_idx].y * image_height))
                            end_point = (int(pose_landmarks_list[end_idx].x * image_width), int(pose_landmarks_list[end_idx].y * image_height))
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                    except IndexError: continue
                for idx in left_arm_indices_to_draw:
                     try:
                         if pose_landmarks_list[idx].visibility > 0.5:
                             point = (int(pose_landmarks_list[idx].x * image_width), int(pose_landmarks_list[idx].y * image_height))
                             cv2.circle(frame, point, 5, (0, 0, 255), -1)
                     except IndexError: continue
            if original_right_hand_lm_list:
                try: mp_drawing.draw_landmarks(frame, original_right_hand_lm_list, mp_hands.HAND_CONNECTIONS)
                except Exception as draw_e: print(f"ERROR: [Worker Process] Hands drawing exception: {draw_e}")
            try:
                cv2.imshow('MediaPipe Pose(L Arm)+Hands(R Hand)', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'): print("[Worker Process] 'q' key pressed, exiting worker."); break
            except Exception as cv_e: print(f"ERROR: [Worker Process] imshow/waitKey exception: {cv_e}"); break
    except Exception as e: print(f"ERROR: [Worker Process] Unhandled Exception: {e}"); traceback.print_exc()
    finally:
        cv2_local = locals().get('cv2'); print("[Worker Process] Exiting...")
        if cap and cap.isOpened():
            print("[Worker Process] Releasing camera...")
            cap.release()
        if pose_instance:
            print("[Worker Process] Closing MediaPipe Pose...")
            try:
                pose_instance.close()
            except Exception as e:
                print(f"Error closing pose: {e}")
        if hands_instance:
            print("[Worker Process] Closing MediaPipe Hands...")
            try:
                hands_instance.close()
            except Exception as e:
                print(f"Error closing hands: {e}")
        if cv2_local:
            print("[Worker Process] Destroying OpenCV windows...")
            try:
                cv2_local.destroyAllWindows()
            except Exception as e:
                print(f"Error destroying windows: {e}")
        print("[Worker Process] Finished.")


# (collect_one_repetition function remains the same)
async def collect_one_repetition(myo: Myo, gesture_id: int, gesture_name: str, shared_landmarks_proxy, lock):
    """ Records EMG + IMU + shoulder-normalized L_Pose/R_Hand landmarks """
    samples = []
    imu_q = None
    emg_callback_active = True
    def on_imu(orientation, accel, gyro): nonlocal imu_q; imu_q = orientation
    def on_emg(emg_data):
        nonlocal imu_q, samples, emg_callback_active
        if not emg_callback_active: return
        if (emg_data is not None and len(emg_data) == NUM_EMG_SENSORS and
            imu_q is not None and len(imu_q) >= NUM_IMU_VALUES):
            ts = datetime.now().timestamp()
            entry = {"id": uuid.uuid4().hex, "time": ts, "gesture_id": gesture_id, "gesture_name": gesture_name}
            for i, val in enumerate(emg_data, start=1): entry[f"s{i}"] = val
            entry.update({"quat_w": imu_q[0], "quat_x": imu_q[1], "quat_y": imu_q[2], "quat_z": imu_q[3]})
            if lock is None: current_landmarks = {}
            else:
                try:
                    with lock: current_landmarks = dict(shared_landmarks_proxy)
                except Exception as lock_e: print(f"ERROR: [on_emg] Lock exception: {lock_e}"); current_landmarks = {}
            for name in ALL_LANDMARK_NAMES: entry[name] = current_landmarks.get(name, pd.NA)
            samples.append(entry)

    print(f"\n--- Prepare '{gesture_name}' (class {gesture_id}) in {PAUSE_DURATION}s …")
    await asyncio.sleep(PAUSE_DURATION)
    print(f"--- Recording '{gesture_name}' for {COLLECTION_TIME}s …")
    myo.on_imu(on_imu); myo.on_emg_smooth(on_emg)
    try:
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
        await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=True)
    except Exception as mode_e: print(f"ERROR: Failed to set Myo modes: {mode_e}"); traceback.print_exc(); return []
    await asyncio.sleep(COLLECTION_TIME)
    print(f"--- Stopping recording for '{gesture_name}' …")
    emg_callback_active = False
    try:
        await myo.set_mode(emg_mode=None, imu_mode=False)
        await myo.set_sleep_mode(SleepMode.NORMAL)
    except Exception as teardown_e: print(f"ERROR: Failed during Myo teardown: {teardown_e}"); traceback.print_exc()
    print(f"→ Collected {len(samples)} samples.")
    return samples


# (main function remains the same)
async def main():
    os.makedirs(DATA_INPUT_PATH, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DATA_INPUT_PATH, f"{RAW_DATA_FILENAME_PREFIX}{ts_str}.csv")
    # <<< Title print moved to splash screen >>>
    # print(f"--- NeuroSyn Physio Data Collection (Specific Pose+Hands, Shoulder Normalized) ---")
    print(f"Saving to: {out_file}")
    print("Exercises to record:"); [print(f"  {gid}: {name}") for gid, name in CLASSES.items()]
    print(f"\nEach: {REPETITIONS_PER_EXERCISE} reps, {COLLECTION_TIME}s each, pause {PAUSE_DURATION}s")
    print("-" * 50)

    device = None
    print("Scanning for Myo device...")
    try:
        scanner = BleakScanner(); devices = await scanner.discover(timeout=10.0)
        device = next((d for d in devices if d.address == MYO_ADDRESS), None)
    except Exception as e: print("ERROR: Bluetooth scan failed:", e); traceback.print_exc(); return
    if not device: print(f"ERROR: Could not find Myo at {MYO_ADDRESS}."); return
    print(f"Found device: {device.name} ({device.address})")

    with mp.Manager() as manager:
        latest_landmarks_proxy = manager.dict(); landmarks_lock = mp.Lock(); stop_event = mp.Event(); opencv_process = None
        opencv_process = mp.Process(target=opencv_worker, args=(latest_landmarks_proxy, landmarks_lock, stop_event))
        opencv_process.start()
        print("OpenCV process started. Allowing time for initialization...")
        await asyncio.sleep(6.0)

        all_samples = []; myo_connection = None
        print("Connecting to Myo...")
        try:
            myo_connection = Myo(device)
            async with myo_connection as myo:
                print("Connected to Myo.")
                await asyncio.sleep(1.0)
                for gid, gname in CLASSES.items():
                    for rep in range(1, REPETITIONS_PER_EXERCISE + 1):
                        print(f"\nStarting rep {rep}/{REPETITIONS_PER_EXERCISE} for gesture {gid} ('{gname}')...")
                        try:
                            repsamp = await collect_one_repetition(myo, gid, gname, latest_landmarks_proxy, landmarks_lock)
                            all_samples.extend(repsamp)
                        except Exception as collect_e: print(f"ERROR: Exception during collect_one_repetition: {collect_e}"); traceback.print_exc(); print("WARNING: Continuing collection despite error.")
                        await asyncio.sleep(0.1)
                print("\nFinished all repetitions.")
                if not all_samples: print("WARNING: No data collected. Exiting without saving.")
                else:
                    print("Building DataFrame..."); df = pd.DataFrame(all_samples)
                    cols = (["id", "time", "gesture_id", "gesture_name"] + [f"s{i}" for i in range(1, NUM_EMG_SENSORS+1)] + ["quat_w", "quat_x", "quat_y", "quat_z"] + ALL_LANDMARK_NAMES)
                    print("Ensuring all columns exist...");
                    for c in cols:
                        if c not in df.columns: df[c] = pd.NA
                    df = df[cols]
                    print(f"\nSaving {len(df)} samples to CSV: {out_file}")
                    try: df.to_csv(out_file, index=False); print("Save complete.")
                    except Exception as save_e: print(f"ERROR: Failed to save DataFrame to CSV: {save_e}"); traceback.print_exc()
        except Exception as e: print(f"ERROR during collection: {e}"); traceback.print_exc()
        finally:
            print("Cleaning up OpenCV process...")
            if opencv_process and opencv_process.is_alive():
                stop_event.set(); opencv_process.join(timeout=10.0)
                if opencv_process.is_alive(): print("WARNING: OpenCV process did not exit gracefully. Terminating..."); opencv_process.terminate(); opencv_process.join()
                else: print("OpenCV process finished.")
            else: print("OpenCV process was not running or already finished.")
            print("\nCollection script finished.")

# Protect main execution for multiprocessing
if __name__ == "__main__":
    # mp.freeze_support() # Might be needed on Windows if creating executables

    # <<< Call the splash screen function ONCE here >>>
    print_splash_screen()

    try:
        if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt: print("\nKeyboard interrupt—exiting.")
    except Exception as e: print(f"ERROR: Top-level exception: {e}"); traceback.print_exc()
    finally: print("Script execution ended.") # Keep a final message

