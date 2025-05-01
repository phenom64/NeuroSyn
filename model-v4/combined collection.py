#!/usr/bin/env python3
"""
combined_collection.py for NeuroSyn Physio Project
Combines EMG, IMU, and real-time landmark data collection for gesture recognition.
Uses multiprocessing for OpenCV/MediaPipe isolation.
*** ADDED: 3D Hand Landmark normalization relative to the wrist (landmark 0). ***
Includes DEBUG prints for normalization verification.
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
        # <<< ADDED: Define wrist index constant >>>
        WRIST_LANDMARK_INDEX # Expecting this in constants.py, default to 0 if not found
    )
    print("--- NeuroSyn Physio Collection ---")
    print(f"Version 4.5") # Example version
    print(f"TM & (C) 2025 Syndromatic Inc. All rights reserved.")
    print(f"Myo Address: {MYO_ADDRESS}")
    print(f"Data Input Path: {DATA_INPUT_PATH}")
    print(f"Collection Time per Rep (s): {COLLECTION_TIME}")
    print(f"Repetitions per Exercise: {REPETITIONS_PER_EXERCISE}")
    print(f"Number of Classes: {len(CLASSES)}")
    print(f"Classes Map: {CLASSES}")
    print(f"Wrist Landmark Index: {WRIST_LANDMARK_INDEX}") # Print the index being used
    print("-----------------------------------")
except ImportError:
    print("WARNING: Failed to import constants.py or WRIST_LANDMARK_INDEX not defined. Defaulting wrist index to 0.")
    # Define defaults if constants.py is missing or incomplete for this part
    MYO_ADDRESS = "FF:39:C8:DC:AC:BA" # Example default
    CLASSES = {0: 'Rest', 1: 'Gesture1'} # Example default
    COLLECTION_TIME = 5 # Example default
    REPETITIONS_PER_EXERCISE = 5 # Example default
    PAUSE_DURATION = 2 # Example default
    DATA_INPUT_PATH = "data" # Example default
    RAW_DATA_FILENAME_PREFIX = "physio_data_" # Example default
    NUM_EMG_SENSORS = 8 # Example default
    NUM_IMU_VALUES = 4 # Example default
    WRIST_LANDMARK_INDEX = 0 # Default wrist index
except Exception as e: print(f"ERROR loading constants: {e}"); exit()
# --- End Constants Loading ---


# Define landmark column names
landmark_names = [
    "Wrist_x", "Wrist_y", "Wrist_z", "Thumb_CMC_x", "Thumb_CMC_y", "Thumb_CMC_z",
    "Thumb_MCP_x", "Thumb_MCP_y", "Thumb_MCP_z", "Thumb_IP_x", "Thumb_IP_y", "Thumb_IP_z",
    "Thumb_Tip_x", "Thumb_Tip_y", "Thumb_Tip_z", "Index_MCP_x", "Index_MCP_y", "Index_MCP_z",
    "Index_PIP_x", "Index_PIP_y", "Index_PIP_z", "Index_DIP_x", "Index_DIP_y", "Index_DIP_z",
    "Index_Tip_x", "Index_Tip_y", "Index_Tip_z", "Middle_MCP_x", "Middle_MCP_y", "Middle_MCP_z",
    "Middle_PIP_x", "Middle_PIP_y", "Middle_PIP_z", "Middle_DIP_x", "Middle_DIP_y", "Middle_DIP_z",
    "Middle_Tip_x", "Middle_Tip_y", "Middle_Tip_z", "Ring_MCP_x", "Ring_MCP_y", "Ring_MCP_z",
    "Ring_PIP_x", "Ring_PIP_y", "Ring_PIP_z", "Ring_DIP_x", "Ring_DIP_y", "Ring_DIP_z",
    "Ring_Tip_x", "Ring_Tip_y", "Ring_Tip_z", "Pinky_MCP_x", "Pinky_MCP_y", "Pinky_MCP_z",
    "Pinky_PIP_x", "Pinky_PIP_y", "Pinky_PIP_z", "Pinky_DIP_x", "Pinky_DIP_y", "Pinky_DIP_z",
    "Pinky_Tip_x", "Pinky_Tip_y", "Pinky_Tip_z"
]

# ─── Patch pymyo's buggy classifier handler ─────────────────────────────
def _safe_on_classifier(self, sender, value: bytearray):
    if len(value) < 3: return
    try: struct.unpack("<B2s", value)
    except struct.error: return
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# ────────────────────────────────────────────────────────────────────────

def get_landmarks(frame, hands_instance, cv2_module, mp_drawing_module, mp_hands_module):
    """
    Extracts and normalizes hand landmarks from a frame using MediaPipe.
    Normalization is done relative to the wrist landmark (index WRIST_LANDMARK_INDEX).
    """
    if hands_instance is None: return None, None
    if cv2_module is None: return None, None
    if mp_drawing_module is None: return None, None
    if mp_hands_module is None: return None, None

    rgb_frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
    try:
        results = hands_instance.process(rgb_frame)
    except Exception as e: print(f"ERROR: [get_landmarks] Exception during hands.process: {e}"); traceback.print_exc(); return None, None

    if results.multi_hand_landmarks:
        # Assuming the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]

        # --- Normalization Steps ---
        try:
            # 1. Extract Wrist Coordinates (using the defined index)
            wrist_landmark = hand_landmarks.landmark[WRIST_LANDMARK_INDEX]
            wrist_x = wrist_landmark.x
            wrist_y = wrist_landmark.y
            wrist_z = wrist_landmark.z

            # DEBUG: Print original wrist coordinates (optional)
            # print(f"DEBUG: Original Wrist Coords: x={wrist_x:.4f}, y={wrist_y:.4f}, z={wrist_z:.4f}")

            landmarks = {}
            # 2. Normalize all landmarks relative to the wrist
            for i, landmark in enumerate(hand_landmarks.landmark):
                if i * 3 < len(landmark_names): # Check bounds using landmark_names list
                    base_name = landmark_names[i * 3].rsplit('_', 1)[0]
                    normalized_x = landmark.x - wrist_x
                    normalized_y = landmark.y - wrist_y
                    normalized_z = landmark.z - wrist_z # Keep Z for 3D info

                    landmarks[f"{base_name}_x"] = normalized_x
                    landmarks[f"{base_name}_y"] = normalized_y
                    landmarks[f"{base_name}_z"] = normalized_z
                else:
                    # This case should ideally not happen if landmark_names is correct
                    print(f"Warning: [get_landmarks] Index {i} out of bounds for landmark_names list during normalization.")
                    break

            # DEBUG: Verify normalized wrist coordinates are close to zero (optional)
            # if 'Wrist_x' in landmarks: # Check if wrist key exists (it should if index is 0)
            #    print(f"DEBUG: Normalized Wrist Coords: x={landmarks['Wrist_x']:.4f}, y={landmarks['Wrist_y']:.4f}, z={landmarks['Wrist_z']:.4f}")

            return landmarks, hand_landmarks # Return normalized dict and original landmarks for drawing

        except IndexError:
            print(f"ERROR: [get_landmarks] Wrist landmark index {WRIST_LANDMARK_INDEX} out of bounds for detected landmarks.")
            return None, None # Return None if wrist index is invalid
        except Exception as norm_e:
            print(f"ERROR: [get_landmarks] Exception during normalization: {norm_e}")
            traceback.print_exc()
            return None, None # Return None on other normalization errors
        # --- End Normalization Steps ---

    return None, None # Return None if no hand landmarks detected

# NOTE: This function now runs in a separate PROCESS
def opencv_worker(shared_landmarks_proxy, lock, stop_event):
    """Worker PROCESS for handling OpenCV camera and landmark detection."""
    # Import heavy libraries INSIDE the process
    cv2 = None
    mp = None
    mp_drawing = None
    mp_hands = None
    try:
        # print("DEBUG: [Worker Process] Importing cv2...") # Removed for cleaner output
        import cv2
        # print("DEBUG: [Worker Process] Importing mediapipe...") # Removed for cleaner output
        import mediapipe as mp
        # print("DEBUG: [Worker Process] Imports successful.") # Removed for cleaner output
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
    except ImportError as import_err: print(f"ERROR: [Worker Process] Failed to import cv2 or mediapipe: {import_err}"); traceback.print_exc(); return
    except Exception as general_import_err: print(f"ERROR: [Worker Process] Unexpected error during import: {general_import_err}"); traceback.print_exc(); return

    # print("DEBUG: [Worker Process] Started execution after imports.") # Removed for cleaner output
    hands_instance = None
    cap = None

    try:
        # STEP 1 (Worker): Initialize Camera FIRST
        print("[Worker Process] Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
            print("ERROR: [Worker Process] Could not open camera!")
            return
        print("[Worker Process] Camera opened.")

        # STEP 2 (Worker): Initialize MediaPipe Hands AFTER Camera
        print("[Worker Process] Initializing MediaPipe Hands...")
        try:
            hands_instance = mp_hands.Hands(
                static_image_mode=False, max_num_hands=1,
                min_detection_confidence=0.7, min_tracking_confidence=0.7
            )
            print("[Worker Process] MediaPipe Hands initialized.")
        except Exception as mp_init_e:
            print(f"ERROR: [Worker Process] Failed to initialize MediaPipe Hands: {mp_init_e}")
            traceback.print_exc()
            if cap and cap.isOpened(): cap.release()
            return

        # --- Main Loop ---
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret: time.sleep(0.01); continue # Avoid busy-waiting

            # Get NORMALIZED landmarks
            normalized_landmarks_dict, original_hand_landmarks_for_drawing = get_landmarks(frame, hands_instance, cv2, mp_drawing, mp_hands)

            # Update shared proxy with NORMALIZED data
            if lock is None: print("ERROR: [Worker Process] Lock object is None!"); break
            try:
                with lock:
                    shared_landmarks_proxy.clear()
                    if normalized_landmarks_dict: # Check if dict is not None
                        shared_landmarks_proxy.update(normalized_landmarks_dict)
            except Exception as lock_e: print(f"ERROR: [Worker Process] Exception acquiring/using lock: {lock_e}"); traceback.print_exc(); break

            # Draw using ORIGINAL landmarks
            if original_hand_landmarks_for_drawing:
                try: mp_drawing.draw_landmarks(frame, original_hand_landmarks_for_drawing, mp_hands.HAND_CONNECTIONS)
                except Exception as draw_e: print(f"ERROR: [Worker Process] Exception during mp_drawing.draw_landmarks: {draw_e}")

            # Display window
            try:
                cv2.imshow('Hand Tracking', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'): print("[Worker Process] 'q' key pressed, exiting worker."); break
            except Exception as cv_e: print(f"ERROR: [Worker Process] Exception during cv2.imshow/waitKey: {cv_e}"); break

            # time.sleep(0.001) # Optional small sleep

    except Exception as e:
        print(f"ERROR: [Worker Process] Unhandled Exception in OpenCV loop: {e}")
        traceback.print_exc()
    finally:
        cv2_local = locals().get('cv2')
        print("[Worker Process] Exiting...")
        if cap and cap.isOpened(): print("[Worker Process] Releasing camera..."); cap.release()
        if hands_instance:
            print("[Worker Process] Closing MediaPipe Hands...")
            try: hands_instance.close()
            except Exception as mp_close_e: print(f"ERROR: [Worker Process] Exception closing MediaPipe Hands: {mp_close_e}")
        if cv2_local:
            print("[Worker Process] Destroying OpenCV windows...")
            try: cv2_local.destroyAllWindows()
            except Exception as destroy_e: print(f"ERROR: [Worker Process] Exception destroying OpenCV windows: {destroy_e}")
        print("[Worker Process] Finished.")


async def collect_one_repetition(myo: Myo, gesture_id: int, gesture_name: str, shared_landmarks_proxy, lock):
    """ Records EMG + IMU + NORMALIZED landmarks (from proxy) for one repetition. """
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
                    with lock: current_landmarks = dict(shared_landmarks_proxy) # Read NORMALIZED landmarks
                except Exception as lock_e: print(f"ERROR: [on_emg] Exception acquiring/using lock: {lock_e}"); current_landmarks = {}
            # Add normalized landmarks (or NA if missing) to the sample
            for name in landmark_names: entry[name] = current_landmarks.get(name, pd.NA)
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
    # print("DEBUG: Starting main function.") # Removed debug print
    os.makedirs(DATA_INPUT_PATH, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DATA_INPUT_PATH, f"{RAW_DATA_FILENAME_PREFIX}{ts_str}.csv")
    print(f"--- NeuroSyn Physio Data Collection (Combined + Normalized) ---") # Updated title
    print(f"Saving to: {out_file}")
    print("Exercises to record:"); [print(f"  {gid}: {name}") for gid, name in CLASSES.items()]
    print(f"\nEach: {REPETITIONS_PER_EXERCISE} reps, {COLLECTION_TIME}s each, pause {PAUSE_DURATION}s")
    print("-" * 50)

    # STEP 1: Perform Bleak Scan FIRST
    device = None
    print("Scanning for Myo device...") # Keep user-facing message
    try:
        scanner = BleakScanner()
        devices = await scanner.discover(timeout=10.0)
        device = next((d for d in devices if d.address == MYO_ADDRESS), None)
        # print(f"DEBUG: Bleak scan finished. Device found: {device is not None}") # Removed debug print
    except Exception as e: print("ERROR: Bluetooth scan failed:", e); traceback.print_exc(); return
    if not device: print(f"ERROR: Could not find Myo at {MYO_ADDRESS}."); return
    print(f"Found device: {device.name} ({device.address})")

    # STEP 2: Use multiprocessing Manager for shared state
    with mp.Manager() as manager:
        # print("DEBUG: Multiprocessing Manager started.") # Removed debug print
        latest_landmarks_proxy = manager.dict()
        landmarks_lock = mp.Lock()
        stop_event = mp.Event()
        opencv_process = None

        # print("DEBUG: Initializing shared resources for process...") # Removed debug print

        # STEP 3: Start OpenCV worker PROCESS AFTER scan
        # print("DEBUG: Creating OpenCV process...") # Removed debug print
        opencv_process = mp.Process(target=opencv_worker, args=(latest_landmarks_proxy, landmarks_lock, stop_event))
        # print("DEBUG: Starting OpenCV process...") # Removed debug print
        opencv_process.start()
        print("OpenCV process started. Allowing time for initialization...") # Keep user-facing message
        await asyncio.sleep(5.0)
        # print("DEBUG: Post-process-start delay finished.") # Removed debug print

        # STEP 4: Connect to Myo and start collection
        all_samples = []; myo_connection = None
        print("Connecting to Myo...") # Keep user-facing message
        try:
            myo_connection = Myo(device)
            async with myo_connection as myo:
                print("Connected to Myo.") # Keep user-facing message
                await asyncio.sleep(1.0)

                # --- Main Collection Loop ---
                for gid, gname in CLASSES.items():
                    for rep in range(1, REPETITIONS_PER_EXERCISE + 1):
                        # print(f"\nDEBUG: Starting rep {rep} for gesture {gid} ('{gname}')...") # Removed debug print
                        try:
                            repsamp = await collect_one_repetition(myo, gid, gname, latest_landmarks_proxy, landmarks_lock)
                            # print(f"DEBUG: Finished rep {rep} for gesture {gid}. Samples: {len(repsamp)}") # Removed debug print
                            all_samples.extend(repsamp)
                        except Exception as collect_e: print(f"ERROR: Exception during collect_one_repetition for gesture {gid}, rep {rep}: {collect_e}"); traceback.print_exc(); print("WARNING: Continuing collection despite error.")
                        # print(f"DEBUG: Pausing for {PAUSE_DURATION}s after rep {rep}...") # Removed debug print
                        await asyncio.sleep(0.1) # Keep tiny pause
                # --- End Main Collection Loop ---
                print("\nFinished all repetitions.") # Keep user-facing message

                if not all_samples: print("WARNING: No data collected. Exiting without saving.")
                else:
                    # Build and save DataFrame
                    print("Building DataFrame...") # Keep user-facing message
                    df = pd.DataFrame(all_samples)
                    cols = (["id", "time", "gesture_id", "gesture_name"] + [f"s{i}" for i in range(1, NUM_EMG_SENSORS+1)] + ["quat_w", "quat_x", "quat_y", "quat_z"] + landmark_names)
                    # print("DEBUG: Ensuring all columns exist...") # Removed debug print
                    for c in cols:
                        if c not in df.columns: df[c] = pd.NA
                    df = df[cols]
                    print(f"\nSaving {len(df)} samples to CSV: {out_file}") # Keep user-facing message
                    try: df.to_csv(out_file, index=False); print("Save complete.") # Keep user-facing message
                    except Exception as save_e: print(f"ERROR: Failed to save DataFrame to CSV: {save_e}"); traceback.print_exc()
        except Exception as e: print(f"ERROR during collection: {e}"); traceback.print_exc()
        finally:
            # print("DEBUG: Entering finally block (main process).") # Removed debug print
            print("Cleaning up OpenCV process...") # Keep user-facing message
            # STEP 5: Signal and Join the worker PROCESS
            if opencv_process and opencv_process.is_alive():
                # print("DEBUG: Stop event set for OpenCV process.") # Removed debug print
                stop_event.set()
                # print("DEBUG: Waiting for OpenCV process to join...") # Removed debug print
                opencv_process.join(timeout=10.0)
                if opencv_process.is_alive():
                    print("WARNING: OpenCV process did not exit gracefully. Terminating...")
                    opencv_process.terminate(); opencv_process.join()
                # else: print("DEBUG: OpenCV process join finished.") # Removed debug print
            # else: print("DEBUG: OpenCV process was not running or already finished.") # Removed debug print
            print("\nCollection script finished.") # Keep user-facing message

# Protect main execution for multiprocessing
if __name__ == "__main__":
    # mp.freeze_support() # Might be needed on Windows if creating executables

    # print("DEBUG: Script execution started.") # Removed debug print
    try:
        if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # print("DEBUG: Running asyncio.run(main)...") # Removed debug print
        asyncio.run(main())
        # print("DEBUG: asyncio.run(main) finished.") # Removed debug print
    except KeyboardInterrupt: print("\nKeyboard interrupt—exiting.")
    except Exception as e: print(f"ERROR: Top-level exception: {e}"); traceback.print_exc()
    # finally: print("DEBUG: Script execution ended.") # Removed debug print

