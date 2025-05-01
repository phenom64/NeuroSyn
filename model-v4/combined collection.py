#!/usr/bin/env python3
"""
combined_collection.py for NeuroSyn Physio Project
Combines EMG, IMU, and real-time landmark data collection for gesture recognition.
Uses multiprocessing for OpenCV/MediaPipe isolation.
Uses separate MediaPipe Pose and Hands(max_num_hands=1) models.
Extracts LEFT Arm Pose (11,13,15) and RIGHT Hand landmarks.
NORMALIZATION: All extracted landmarks normalized relative to Pose LEFT Shoulder (index 11).
*** FIX: Corrected AttributeError in custom drawing logic within opencv_worker. ***
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
    )
    print("--- NeuroSyn Physio Collection ---")
    print(f"Version 5.4 (Pose+Hands Drawing Fix)") # Updated version marker
    print(f"TM & (C) 2025 Syndromatic Inc. All rights reserved.")
    print(f"Myo Address: {MYO_ADDRESS}")
    print(f"Data Input Path: {DATA_INPUT_PATH}")
    print(f"Collection Time per Rep (s): {COLLECTION_TIME}")
    print(f"Repetitions per Exercise: {REPETITIONS_PER_EXERCISE}")
    print(f"Number of Classes: {len(CLASSES)}")
    print(f"Classes Map: {CLASSES}")
    print("-----------------------------------")
except ImportError:
    print("WARNING: Failed to import constants.py. Using default values.")
    MYO_ADDRESS = "FF:39:C8:DC:AC:BA"; CLASSES = {0: 'Rest', 1: 'Gesture1'}; COLLECTION_TIME = 5
    REPETITIONS_PER_EXERCISE = 5; PAUSE_DURATION = 2; DATA_INPUT_PATH = "data"
    RAW_DATA_FILENAME_PREFIX = "physio_data_"; NUM_EMG_SENSORS = 8; NUM_IMU_VALUES = 4
except Exception as e: print(f"ERROR loading constants: {e}"); exit()
# --- End Constants Loading ---

# --- Define Landmarks to Extract ---
# Using LEFT arm indices based on user testing for stability (Indices 11, 13, 15)
POSE_LANDMARKS_TO_EXTRACT = {
    "Pose_L_Shoulder": 11,
    "Pose_L_Elbow": 13,
    "Pose_L_Wrist": 15,
}
POSE_SHOULDER_INDEX_FOR_NORM = 11 # Left Shoulder for normalization origin

# Define Hand Landmark Names (base names for generating normalized columns) - 21 landmarks
HAND_LANDMARK_BASE_NAMES = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip"
]

# Define Pose Landmark Names (for normalized data relative to shoulder)
POSE_LANDMARK_NAMES_NORM = [
    f"{name}_{coord}_shldr_norm"
    for name in POSE_LANDMARKS_TO_EXTRACT.keys()
    for coord in ["x", "y", "z", "vis"]
]

# Define RIGHT Hand Landmark Names (for normalized data relative to shoulder)
HAND_LANDMARK_NAMES_NORM = [
    f"Hand_R_{name_part}_{coord}_shldr_norm"
    for name_part in HAND_LANDMARK_BASE_NAMES
    for coord in ["x", "y", "z"]
]

# Combine all landmark names for the DataFrame header
ALL_LANDMARK_NAMES = POSE_LANDMARK_NAMES_NORM + HAND_LANDMARK_NAMES_NORM
# --- End Landmark Definitions ---


# ─── Patch pymyo's buggy classifier handler ─────────────────────────────
def _safe_on_classifier(self, sender, value: bytearray):
    if len(value) < 3: return
    try: struct.unpack("<B2s", value)
    except struct.error: return
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# ────────────────────────────────────────────────────────────────────────

def get_landmarks_pose_hands_shoulder_norm(frame, pose_results, hands_results, cv2_module, mp_drawing_module, mp_pose_module, mp_hands_module):
    """
    Extracts LEFT Pose (arm) and RIGHT Hand landmarks, normalizes ALL relative to the Pose LEFT shoulder landmark.
    Returns a combined dictionary of normalized landmarks.
    Also returns original landmarks needed for drawing.
    """
    if cv2_module is None or mp_drawing_module is None or mp_pose_module is None or mp_hands_module is None:
        print("ERROR: [get_landmarks] Required modules not provided.")
        return None, None, None

    normalized_landmarks = {}
    original_pose_landmarks_for_drawing = None
    original_right_hand_landmarks_for_drawing = None
    shoulder_coords = None

    # 1. Process Pose Landmarks to find Shoulder
    if pose_results and pose_results.pose_landmarks:
        # <<< Store the landmark list directly for drawing >>>
        original_pose_landmarks_for_drawing = pose_results.pose_landmarks
        try:
            shoulder_landmark = original_pose_landmarks_for_drawing.landmark[POSE_SHOULDER_INDEX_FOR_NORM]
            if shoulder_landmark.visibility > 0.5:
                 shoulder_coords = (shoulder_landmark.x, shoulder_landmark.y, shoulder_landmark.z)
        except IndexError: print(f"ERROR: [get_landmarks] Pose shoulder index ({POSE_SHOULDER_INDEX_FOR_NORM}) out of bounds.")
        except Exception as pose_e: print(f"ERROR: [get_landmarks] Finding shoulder: {pose_e}"); traceback.print_exc()

    # 2. If Shoulder found, proceed with normalization
    if shoulder_coords:
        shoulder_x, shoulder_y, shoulder_z = shoulder_coords

        # 2a. Normalize selected POSE landmarks
        if original_pose_landmarks_for_drawing: # Check if landmark list exists
            try:
                pose_landmark_idx_norm = 0
                for name, index_orig in POSE_LANDMARKS_TO_EXTRACT.items():
                    landmark = original_pose_landmarks_for_drawing.landmark[index_orig] # Access landmark list
                    if pose_landmark_idx_norm < len(POSE_LANDMARK_NAMES_NORM):
                        key_x, key_y, key_z, key_vis = POSE_LANDMARK_NAMES_NORM[pose_landmark_idx_norm : pose_landmark_idx_norm+4]
                        normalized_landmarks[key_x] = landmark.x - shoulder_x
                        normalized_landmarks[key_y] = landmark.y - shoulder_y
                        normalized_landmarks[key_z] = landmark.z - shoulder_z
                        normalized_landmarks[key_vis] = landmark.visibility
                        pose_landmark_idx_norm += 4
                    else: break
            except IndexError: print(f"ERROR: [get_landmarks] Pose index out of bounds during norm.")
            except Exception as pose_norm_e: print(f"ERROR: [get_landmarks] Normalizing pose: {pose_norm_e}"); traceback.print_exc()

        # 2b. Normalize RIGHT HAND landmarks
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                if handedness.classification[0].label == 'Right':
                    # <<< Store the landmark list directly for drawing >>>
                    original_right_hand_landmarks_for_drawing = hand_landmarks
                    try:
                        hand_landmark_idx_norm = 0
                        for i, landmark in enumerate(original_right_hand_landmarks_for_drawing.landmark): # Access landmark list
                            if hand_landmark_idx_norm < len(HAND_LANDMARK_NAMES_NORM):
                                key_x, key_y, key_z = HAND_LANDMARK_NAMES_NORM[hand_landmark_idx_norm : hand_landmark_idx_norm+3]
                                normalized_landmarks[key_x] = landmark.x - shoulder_x
                                normalized_landmarks[key_y] = landmark.y - shoulder_y
                                normalized_landmarks[key_z] = landmark.z - shoulder_z
                                hand_landmark_idx_norm += 3
                            else: break
                    except Exception as hand_norm_e:
                        print(f"ERROR: [get_landmarks] Normalizing hand: {hand_norm_e}"); traceback.print_exc()
                        for key in HAND_LANDMARK_NAMES_NORM: normalized_landmarks.pop(key, None)
                        original_right_hand_landmarks_for_drawing = None
                    break

    # Return normalized dict, original pose landmark list, original RIGHT hand landmark list
    return normalized_landmarks, original_pose_landmarks_for_drawing, original_right_hand_landmarks_for_drawing


# NOTE: This function now runs in a separate PROCESS
def opencv_worker(shared_landmarks_proxy, lock, stop_event):
    """Worker PROCESS using Pose+Hands, draws only Left Arm + Right Hand."""
    # Import heavy libraries INSIDE the process
    cv2 = None; mp = None; mp_drawing = None; mp_pose = None; mp_hands = None
    try:
        import cv2; import mediapipe as mp
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        print("[Worker Process] OpenCV and MediaPipe imported.")
    except ImportError as import_err: print(f"ERROR: [Worker Process] Failed imports: {import_err}"); traceback.print_exc(); return
    except Exception as general_import_err: print(f"ERROR: [Worker Process] Unexpected import error: {general_import_err}"); traceback.print_exc(); return

    pose_instance = None; hands_instance = None; cap = None
    try:
        # STEP 1: Initialize Camera FIRST
        print("[Worker Process] Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened(): print("ERROR: [Worker Process] Could not open camera!"); return
        print("[Worker Process] Camera opened.")

        # STEP 2: Initialize MediaPipe Models AFTER Camera
        print("[Worker Process] Initializing MediaPipe Pose...")
        try:
            pose_instance = mp_pose.Pose(
                model_complexity=0, # Add this
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("[Worker Process] MediaPipe Pose initialized.")
        except Exception as mp_init_e:
            print(f"ERROR: Init Pose failed: {mp_init_e}")
            traceback.print_exc()
            if cap:
                cap.release()
            return

        print("[Worker Process] Initializing MediaPipe Hands...")
        try:
            hands_instance = mp_hands.Hands(
                model_complexity=0, # Add this
                max_num_hands=1,
                min_detection_confidence=0.7, # Keep this for now, or try 0.5 later
                min_tracking_confidence=0.7  # Keep this for now, or try 0.5 later
            )
            print("[Worker Process] MediaPipe Hands initialized (max_num_hands=1).")
        except Exception as mp_init_e:
            print(f"ERROR: Init Hands failed: {mp_init_e}")
            traceback.print_exc()
            if pose_instance:
                pose_instance.close()
            if cap:
                cap.release()
            return

        # --- Define LEFT Arm Connections for Drawing (Indices from POSE_LANDMARKS_TO_EXTRACT) ---
        left_arm_connections_drawing = [
             (POSE_LANDMARKS_TO_EXTRACT["Pose_L_Shoulder"], POSE_LANDMARKS_TO_EXTRACT["Pose_L_Elbow"]),
             (POSE_LANDMARKS_TO_EXTRACT["Pose_L_Elbow"], POSE_LANDMARKS_TO_EXTRACT["Pose_L_Wrist"]),
        ]
        left_arm_indices_to_draw = list(POSE_LANDMARKS_TO_EXTRACT.values())

        # --- Main Loop ---
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: time.sleep(0.01); continue

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results_pose = pose_instance.process(rgb_frame)
            results_hands = hands_instance.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Get combined & shoulder-normalized landmarks (for saving)
            # Also get original landmark lists (for drawing)
            normalized_landmarks_dict, original_pose_lm_list, original_right_hand_lm_list = get_landmarks_pose_hands_shoulder_norm(
                frame, results_pose, results_hands, cv2, mp_drawing, mp_pose, mp_hands
            )

            # Update shared proxy with NORMALIZED data
            if lock is None: print("ERROR: [Worker Process] Lock object is None!"); break
            try:
                with lock:
                    shared_landmarks_proxy.clear()
                    if normalized_landmarks_dict:
                        shared_landmarks_proxy.update(normalized_landmarks_dict)
            except Exception as lock_e: print(f"ERROR: [Worker Process] Lock exception: {lock_e}"); traceback.print_exc(); break

            # --- Drawing --- (Uses original landmark lists)
            # <<< START CUSTOM POSE DRAWING >>>
            # <<< FIX: Check if original_pose_lm_list exists >>>
            if original_pose_lm_list:
                # pose_landmarks_list = original_pose_lm_list.landmark # This was the error source if original_pose_lm_list is the list itself
                pose_landmarks_list = original_pose_lm_list.landmark # Access the .landmark attribute correctly
                image_height, image_width, _ = frame.shape

                # Draw specific LEFT arm connections using cv2.line
                for connection in left_arm_connections_drawing:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    try: # Add try-except for potential index errors if visibility check fails early
                        if (pose_landmarks_list[start_idx].visibility > 0.5 and
                            pose_landmarks_list[end_idx].visibility > 0.5):
                            start_point = (int(pose_landmarks_list[start_idx].x * image_width),
                                           int(pose_landmarks_list[start_idx].y * image_height))
                            end_point = (int(pose_landmarks_list[end_idx].x * image_width),
                                         int(pose_landmarks_list[end_idx].y * image_height))
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2) # Green lines
                    except IndexError: continue # Skip drawing if index is out of bounds

                # Draw specific LEFT arm landmarks using cv2.circle
                for idx in left_arm_indices_to_draw:
                     try: # Add try-except for potential index errors
                         if pose_landmarks_list[idx].visibility > 0.5:
                             point = (int(pose_landmarks_list[idx].x * image_width),
                                      int(pose_landmarks_list[idx].y * image_height))
                             cv2.circle(frame, point, 5, (0, 0, 255), -1) # Red circles
                     except IndexError: continue # Skip drawing if index is out of bounds
            # <<< END CUSTOM POSE DRAWING >>>

            # Draw RIGHT hand landmarks (using standard utility)
            # <<< FIX: Check if original_right_hand_lm_list exists >>>
            if original_right_hand_lm_list:
                try: mp_drawing.draw_landmarks(frame, original_right_hand_lm_list, mp_hands.HAND_CONNECTIONS)
                except Exception as draw_e: print(f"ERROR: [Worker Process] Hands drawing exception: {draw_e}")

            # Display window
            try:
                cv2.imshow('MediaPipe Pose(L Arm)+Hands(R Hand)', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'): print("[Worker Process] 'q' key pressed, exiting worker."); break
            except Exception as cv_e: print(f"ERROR: [Worker Process] imshow/waitKey exception: {cv_e}"); break

    except Exception as e: print(f"ERROR: [Worker Process] Unhandled Exception: {e}"); traceback.print_exc()
    finally:
        # (Cleanup remains the same)
        cv2_local = locals().get('cv2')
        print("[Worker Process] Exiting...")
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


async def collect_one_repetition(myo: Myo, gesture_id: int, gesture_name: str, shared_landmarks_proxy, lock):
    """ Records EMG + IMU + shoulder-normalized L_Pose/R_Hand landmarks """
    # (This function remains the same)
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


async def main():
    # (Main function remains the same)
    os.makedirs(DATA_INPUT_PATH, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DATA_INPUT_PATH, f"{RAW_DATA_FILENAME_PREFIX}{ts_str}.csv")
    print(f"--- NeuroSyn Physio Data Collection (Specific Pose+Hands, Shoulder Normalized) ---")
    print(f"Saving to: {out_file}")
    print("Exercises to record:"); [print(f"  {gid}: {name}") for gid, name in CLASSES.items()]
    print(f"\nEach: {REPETITIONS_PER_EXERCISE} reps, {COLLECTION_TIME}s each, pause {PAUSE_DURATION}s")
    print("-" * 50)

    # STEP 1: Perform Bleak Scan FIRST
    device = None
    print("Scanning for Myo device...")
    try:
        scanner = BleakScanner()
        devices = await scanner.discover(timeout=10.0)
        device = next((d for d in devices if d.address == MYO_ADDRESS), None)
    except Exception as e: print("ERROR: Bluetooth scan failed:", e); traceback.print_exc(); return
    if not device: print(f"ERROR: Could not find Myo at {MYO_ADDRESS}."); return
    print(f"Found device: {device.name} ({device.address})")

    # STEP 2: Use multiprocessing Manager for shared state
    with mp.Manager() as manager:
        latest_landmarks_proxy = manager.dict()
        landmarks_lock = mp.Lock()
        stop_event = mp.Event()
        opencv_process = None

        # STEP 3: Start OpenCV worker PROCESS AFTER scan
        opencv_process = mp.Process(target=opencv_worker, args=(latest_landmarks_proxy, landmarks_lock, stop_event))
        opencv_process.start()
        print("OpenCV process started. Allowing time for initialization...")
        await asyncio.sleep(6.0)

        # STEP 4: Connect to Myo and start collection
        all_samples = []; myo_connection = None
        print("Connecting to Myo...")
        try:
            myo_connection = Myo(device)
            async with myo_connection as myo:
                print("Connected to Myo.")
                await asyncio.sleep(1.0)

                # --- Main Collection Loop ---
                for gid, gname in CLASSES.items():
                    for rep in range(1, REPETITIONS_PER_EXERCISE + 1):
                        print(f"\nStarting rep {rep}/{REPETITIONS_PER_EXERCISE} for gesture {gid} ('{gname}')...")
                        try:
                            repsamp = await collect_one_repetition(myo, gid, gname, latest_landmarks_proxy, landmarks_lock)
                            all_samples.extend(repsamp)
                        except Exception as collect_e: print(f"ERROR: Exception during collect_one_repetition: {collect_e}"); traceback.print_exc(); print("WARNING: Continuing collection despite error.")
                        await asyncio.sleep(0.1)
                # --- End Main Collection Loop ---
                print("\nFinished all repetitions.")

                if not all_samples: print("WARNING: No data collected. Exiting without saving.")
                else:
                    # Build and save DataFrame
                    print("Building DataFrame...")
                    df = pd.DataFrame(all_samples)
                    cols = (["id", "time", "gesture_id", "gesture_name"] +
                            [f"s{i}" for i in range(1, NUM_EMG_SENSORS+1)] +
                            ["quat_w", "quat_x", "quat_y", "quat_z"] +
                            ALL_LANDMARK_NAMES)
                    print("Ensuring all columns exist...")
                    for c in cols:
                        if c not in df.columns: df[c] = pd.NA
                    df = df[cols]
                    print(f"\nSaving {len(df)} samples to CSV: {out_file}")
                    try: df.to_csv(out_file, index=False); print("Save complete.")
                    except Exception as save_e: print(f"ERROR: Failed to save DataFrame to CSV: {save_e}"); traceback.print_exc()
        except Exception as e: print(f"ERROR during collection: {e}"); traceback.print_exc()
        finally:
            print("Cleaning up OpenCV process...")
            # STEP 5: Signal and Join the worker PROCESS
            if opencv_process and opencv_process.is_alive():
                stop_event.set()
                opencv_process.join(timeout=10.0)
                if opencv_process.is_alive(): print("WARNING: OpenCV process did not exit gracefully. Terminating..."); opencv_process.terminate(); opencv_process.join()
                else: print("OpenCV process finished.")
            else: print("OpenCV process was not running or already finished.")
            print("\nCollection script finished.")

# Protect main execution for multiprocessing
if __name__ == "__main__":
    # mp.freeze_support() # Might be needed on Windows if creating executables

    print("Script execution started.")
    try:
        if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(main())
    except KeyboardInterrupt: print("\nKeyboard interrupt—exiting.")
    except Exception as e: print(f"ERROR: Top-level exception: {e}"); traceback.print_exc()
    finally: print("Script execution ended.")

