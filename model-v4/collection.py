"""
New collection.py for NeuroSyn Physio Project
Combines EMG, IMU, and real-time landmark data collection for gesture recognition.

Based on the Artemis Project, Copyright (c) The Regents of the Manchester Universities 2025
Uses multiprocessing for OpenCV/MediaPipe isolation.
Uses separate MediaPipe Pose and Hands(max_num_hands=1) models.
Extracts LEFT Arm Pose (11,13,15) and RIGHT Hand landmarks.
NORMALIZATION:
    - Landmarks: All extracted landmarks normalized relative to Pose LEFT Shoulder (index 11).
    - EMG: Normalized using mean/std dev from an initial calibration phase (imported).
PROTOCOL: Includes imported MVC Calibration Phase and Randomized Dynamic Hold Collection Phase.
*** FIX: Implemented multiprocessing.Event for robust OpenCV worker readiness signaling. ***
*** FIX: Imports calibration function from calibration.py ***
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
from datetime import datetime, timedelta
# Make sure bleak is installed: pip install bleak
try:
    from bleak import BleakScanner, BleakError
except ImportError:
    print("ERROR: The 'bleak' library is not installed. Please install it using 'pip install bleak'")
    exit()
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
import time # For potential sleeps
import random # For randomized protocol

# <<< Import the calibration function >>>
try:
    from calibration import run_calibration_phase #importing the calibration func
    print("Successfully imported run_calibration_phase from calibration.py")
except ImportError:
    print("ERROR: Failed to import 'run_calibration_phase' from 'calibration.py'.")
    print("Please ensure 'calibration.py' exists in the same directory and contains the function.")
    exit()
except Exception as e:
    print(f"ERROR: An unexpected error occurred importing from calibration.py: {e}")
    traceback.print_exc()
    exit()


# --- Constants Loading ---
try:
    # <<< Ensure all necessary constants are imported for the splash screen too >>>
    from constants import (
        MYO_ADDRESS,
        CLASSES, # Used by splash screen for count
        PAUSE_DURATION, # Used in main loop
        DATA_INPUT_PATH, # Used by splash screen
        RAW_DATA_FILENAME_PREFIX, # Used in main loop
        NUM_EMG_SENSORS, # Used indirectly
        NUM_IMU_VALUES, # Used indirectly
        CALIBRATION_POSES, # Still needed for splash screen info
        CALIBRATION_HOLD_TIME, # Still needed for splash screen info
        RANDOM_POSES, # If different from CLASSES, otherwise use CLASSES
        SESSION_DURATION_MINUTES,
        MIN_HOLD_TIME,
        MAX_HOLD_TIME,
        EMG_NORMALIZATION_EPSILON
    )
    # Use RANDOM_POSES if defined, otherwise fallback to CLASSES for splash screen info
    DISPLAY_CLASSES_INFO = RANDOM_POSES if 'RANDOM_POSES' in locals() and isinstance(RANDOM_POSES, dict) else CLASSES

except ImportError as e:
    print(f"ERROR: Failed to import constants.py or required constants missing: {e}")
    print("Please ensure necessary constants are defined in constants.py")
    exit()
except Exception as e: print(f"ERROR loading constants: {e}"); exit()
# --- End Constants Loading ---

# --- Define Landmarks to Extract ---
# (Landmark definitions remain the same)
POSE_LANDMARKS_TO_EXTRACT = { "Pose_L_Shoulder": 11, "Pose_L_Elbow": 13, "Pose_L_Wrist": 15 }
POSE_SHOULDER_INDEX_FOR_NORM = 11
HAND_LANDMARK_BASE_NAMES = [
    "Wrist", "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip", "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip", "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip" ]
POSE_LANDMARK_NAMES_NORM = [ f"{name}_{coord}_shldr_norm" for name in POSE_LANDMARKS_TO_EXTRACT.keys() for coord in ["x", "y", "z", "vis"] ]
HAND_LANDMARK_NAMES_NORM = [ f"Hand_R_{name_part}_{coord}_shldr_norm" for name_part in HAND_LANDMARK_BASE_NAMES for coord in ["x", "y", "z"] ]
EMG_RAW_NAMES = [f"s{i}" for i in range(1, NUM_EMG_SENSORS + 1)]
EMG_NORM_NAMES = [f"s{i}_norm" for i in range(1, NUM_EMG_SENSORS + 1)]
ALL_LANDMARK_NAMES = POSE_LANDMARK_NAMES_NORM + HAND_LANDMARK_NAMES_NORM
ALL_DATA_COLUMNS = (["id", "time", "gesture_id", "gesture_name"] + EMG_NORM_NAMES +
                    ["quat_w", "quatx", "quaty", "quatz"] + ALL_LANDMARK_NAMES)
# --- End Landmark Definitions ---


# ─── Patch pymyo's buggy classifier handler ─────────────────────────────
def _safe_on_classifier(self, sender, value: bytearray):
    if len(value) < 3: return
    try: struct.unpack("<B2s", value)
    except struct.error: return
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# ────────────────────────────────────────────────────────────────────────

# <<< RESTORED: Function to print the professional splash screen >>>
def print_splash_screen():
    """Prints the application header/splash screen."""
    SCRIPT_VERSION = "6.3" # Updated version for sync + external calib
    COPYRIGHT_YEAR = datetime.now().year # Use current year
    DESIGNER = "Kavish Krishnakumar"
    LOCATION = "Manchester"

    print("=" * 60)
    print("--- NeuroSyn Physio Collection Assistant ---")
    print(f"Version {SCRIPT_VERSION}")
    print(f"TM & (C) {COPYRIGHT_YEAR} Syndromatic Inc. All rights reserved.")
    print(f"Designed by {DESIGNER} in {LOCATION}.")
    print("-" * 60)
    # Print key config loaded from constants
    print(f"Target Myo Address: {MYO_ADDRESS}")
    print(f"Data Output Path: ./{DATA_INPUT_PATH}/")
    # Use the determined pose dictionary for the count
    print(f"Collection Poses: {len(RANDOM_POSES)} ({list(RANDOM_POSES.values())})")
    # Display calibration info from constants, even though function is external
    print(f"Calibration Poses Defined: {len(CALIBRATION_POSES)} ({list(CALIBRATION_POSES.keys())})")
    print(f"Session Duration: {SESSION_DURATION_MINUTES} min")
    print("=" * 60)

# (get_landmarks function remains the same)
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


# <<< OpenCV/MediaPipe worker process function >>>
# <<< Modified to accept and set a readiness event >>>
def opencv_worker(shared_landmarks_proxy, lock, stop_event, ready_event): # Added ready_event parameter
    """Worker PROCESS using Pose+Hands, draws only Left Arm + Right Hand. Signals when ready."""
    cv2 = None; mp = None; mp_drawing = None; mp_pose = None; mp_hands = None
    cap = None; pose_instance = None; hands_instance = None # Define earlier for broader scope in try/except/finally
    initialization_successful = False # Flag to track if init completed

    try:
        # --- Import Libraries ---
        try:
            # These imports happen *inside* the spawned process
            import cv2
            import mediapipe as mp
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            mp_hands = mp.solutions.hands
            print("[Worker Process] OpenCV and MediaPipe imported.")
        except ImportError as import_err:
            print(f"ERROR: [Worker Process] Failed imports: {import_err}")
            print("Ensure OpenCV (cv2) and MediaPipe are installed in the Python environment.")
            traceback.print_exc()
            # Signal failure before returning
            ready_event.set() # Let main process know *something* happened, even if it's failure
            return

        # --- Initialize Camera ---
        print("[Worker Process] Initializing camera...")
        cap = cv2.VideoCapture(0)
        if not cap or not cap.isOpened():
             print("ERROR: [Worker Process] Could not open camera!")
             ready_event.set() # Signal failure
             return

        print("[Worker Process] Camera opened.")

        # --- Initialize MediaPipe Pose ---
        print("[Worker Process] Initializing MediaPipe Pose...")
        try:
            pose_instance = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
            print("[Worker Process] MediaPipe Pose initialized.")
        except Exception as mp_init_e:
            print(f"ERROR: Init Pose failed: {mp_init_e}"); traceback.print_exc()
            ready_event.set() # Signal failure
            return # Exit on failure

        # --- Initialize MediaPipe Hands ---
        print("[Worker Process] Initializing MediaPipe Hands...")
        try:
            hands_instance = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
            print("[Worker Process] MediaPipe Hands initialized (max_num_hands=1).")
        except Exception as mp_init_e:
            print(f"ERROR: Init Hands failed: {mp_init_e}"); traceback.print_exc()
            ready_event.set() # Signal failure
            return # Exit on failure

        # --- Initialization Complete - Signal Readiness ---
        print("[Worker Process] Initialization complete. Signaling ready.")
        ready_event.set() # <<< Signal main process that initialization is done
        initialization_successful = True # Mark successful init

        # --- Define Drawing Specifics ---
        left_arm_connections_drawing = [(POSE_LANDMARKS_TO_EXTRACT["Pose_L_Shoulder"], POSE_LANDMARKS_TO_EXTRACT["Pose_L_Elbow"]), (POSE_LANDMARKS_TO_EXTRACT["Pose_L_Elbow"], POSE_LANDMARKS_TO_EXTRACT["Pose_L_Wrist"])]
        left_arm_indices_to_draw = list(POSE_LANDMARKS_TO_EXTRACT.values())

        # --- Main Processing Loop ---
        while not stop_event.is_set():
            ret, frame = cap.read();
            if not ret: time.sleep(0.01); continue # Skip if frame read fails

            # Process frame
            frame = cv2.flip(frame, 1); rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results_pose = pose_instance.process(rgb_frame); results_hands = hands_instance.process(rgb_frame)
            rgb_frame.flags.writeable = True

            # Extract landmarks
            normalized_landmarks_dict, original_pose_lm_list, original_right_hand_lm_list = get_landmarks_pose_hands_shoulder_norm(frame, results_pose, results_hands, cv2, mp_drawing, mp_pose, mp_hands)

            # Update shared memory
            if lock is None: print("ERROR: [Worker Process] Lock object is None!"); break
            try:
                with lock:
                    shared_landmarks_proxy.clear()
                    if normalized_landmarks_dict: shared_landmarks_proxy.update(normalized_landmarks_dict)
            except Exception as lock_e: print(f"ERROR: [Worker Process] Lock exception: {lock_e}"); traceback.print_exc(); break

            # Draw landmarks
            # Draw Left Arm Pose
            if original_pose_lm_list:
                pose_landmarks_list = original_pose_lm_list.landmark; image_height, image_width, _ = frame.shape
                # Draw connections
                for connection in left_arm_connections_drawing:
                    start_idx, end_idx = connection[0], connection[1]
                    try:
                        if (pose_landmarks_list[start_idx].visibility > 0.5 and pose_landmarks_list[end_idx].visibility > 0.5):
                            start_point = (int(pose_landmarks_list[start_idx].x * image_width), int(pose_landmarks_list[start_idx].y * image_height))
                            end_point = (int(pose_landmarks_list[end_idx].x * image_width), int(pose_landmarks_list[end_idx].y * image_height))
                            cv2.line(frame, start_point, end_point, (0, 255, 0), 2)
                    except IndexError: continue # Skip if index out of bounds
                # Draw points
                for idx in left_arm_indices_to_draw:
                     try:
                         if pose_landmarks_list[idx].visibility > 0.5:
                             point = (int(pose_landmarks_list[idx].x * image_width), int(pose_landmarks_list[idx].y * image_height))
                             cv2.circle(frame, point, 5, (0, 0, 255), -1)
                     except IndexError: continue # Skip if index out of bounds

            # Draw Right Hand
            if original_right_hand_lm_list:
                try: mp_drawing.draw_landmarks(frame, original_right_hand_lm_list, mp_hands.HAND_CONNECTIONS)
                except Exception as draw_e: print(f"ERROR: [Worker Process] Hands drawing exception: {draw_e}")

            # Display frame
            try:
                cv2.imshow('MediaPipe Pose(L Arm)+Hands(R Hand)', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    print("[Worker Process] 'q' key pressed, setting stop event.")
                    stop_event.set() # Signal main process to stop too
                    break
            except Exception as cv_e:
                print(f"ERROR: [Worker Process] imshow/waitKey exception: {cv_e}")
                stop_event.set() # Signal main process on critical error
                break

    except Exception as e:
        print(f"ERROR: [Worker Process] Unhandled Exception during setup or loop: {e}")
        traceback.print_exc()
        if not stop_event.is_set(): stop_event.set() # Signal main process on major error
        if not ready_event.is_set(): ready_event.set() # Ensure main doesn't hang if error before init complete
    finally:
        print("[Worker Process] Exiting...")
        # Ensure event is set if initialization failed AFTER imports but before the set() call
        # to prevent the main process hanging indefinitely.
        if not initialization_successful and ready_event is not None and not ready_event.is_set():
             print("[Worker Process] Signaling readiness during cleanup due to potential init error.")
             ready_event.set()

        # --- Resource Cleanup ---
        if cap and cap.isOpened():
            print("[Worker Process] Releasing camera...")
            cap.release()
        if pose_instance:
            print("[Worker Process] Closing MediaPipe Pose...")
            try: pose_instance.close()
            except Exception as e: print(f"Error closing pose: {e}")
        if hands_instance:
            print("[Worker Process] Closing MediaPipe Hands...")
            try: hands_instance.close()
            except Exception as e: print(f"Error closing hands: {e}")
        # Use the imported cv2 module for cleanup
        cv2_module = locals().get('cv2')
        if cv2_module is not None:
            print("[Worker Process] Destroying OpenCV windows...")
            try: cv2_module.destroyAllWindows()
            except Exception as e: print(f"Error destroying windows: {e}")

        print("[Worker Process] Finished.")


# <<< Dynamic Hold Data Collection Function >>>
# <<< Modified to accept and check stop_event >>>
async def collect_dynamic_hold(myo: Myo, gesture_id: int, gesture_name: str, hold_duration: float, shared_landmarks_proxy, lock, emg_mean, emg_std, stop_event):
    """ Records EMG (raw+norm) + IMU + normalized landmarks for one dynamic hold. Returns samples or None if stopped early. """
    samples = []; imu_q = None; emg_callback_active = True
    emg_mean_np = np.array(emg_mean); emg_std_np = np.array(emg_std) + EMG_NORMALIZATION_EPSILON

    def on_imu(orientation, accel, gyro): nonlocal imu_q; imu_q = orientation
    def on_emg(emg_data_raw):
        nonlocal imu_q, samples, emg_callback_active, emg_mean_np, emg_std_np
        if not emg_callback_active: return
        if (emg_data_raw is not None and len(emg_data_raw) == NUM_EMG_SENSORS and
            imu_q is not None and len(imu_q) >= NUM_IMU_VALUES): # Check IMU data validity

            ts = datetime.now().timestamp(); entry = {"id": uuid.uuid4().hex, "time": ts, "gesture_id": gesture_id, "gesture_name": gesture_name}
            emg_raw_np = np.array(emg_data_raw);

            # Normalized EMG only
            emg_norm_np = (emg_raw_np - emg_mean_np) / emg_std_np
            for i, val_norm in enumerate(emg_norm_np, start=1): entry[f"s{i}_norm"] = val_norm

            # IMU Quaternion
            entry.update({"quat_w": imu_q[0], "quatx": imu_q[1], "quaty": imu_q[2], "quatz": imu_q[3]})

            # Landmarks (from shared proxy)
            current_landmarks = {} # Default to empty
            if lock is not None:
                try:
                    with lock: current_landmarks = dict(shared_landmarks_proxy)
                except Exception as lock_e: print(f"ERROR: [on_emg] Lock exception: {lock_e}")
            # Fill landmark columns, using pd.NA for missing ones
            for name in ALL_LANDMARK_NAMES: entry[name] = current_landmarks.get(name, pd.NA)

            samples.append(entry)

    print(f"\n--- Hold '{gesture_name}' (class {gesture_id}) for {hold_duration:.1f}s …")
    # Setup Myo for this hold
    myo.on_imu(on_imu); myo.on_emg_smooth(on_emg);
    print("Allowing Myo to stabilize..."); await asyncio.sleep(0.5) # Shorter stabilization within loop
    try:
        await myo.set_sleep_mode(SleepMode.NEVER_SLEEP);
        await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=True)
        print("Myo ready for hold.")
    except Exception as mode_e:
        print(f"ERROR: Failed to set Myo modes for hold: {mode_e}"); traceback.print_exc(); return [] # Return empty list on error

    # Wait for hold duration or stop signal
    start_hold_time = time.time()
    while (time.time() - start_hold_time) < hold_duration:
        if stop_event.is_set():
             print(f"--- STOP SIGNAL received during hold '{gesture_name}'. Aborting hold early.")
             emg_callback_active = False # Stop collecting samples
             # Attempt to clean up Myo state quickly
             try: await myo.set_mode(emg_mode=None, imu_mode=False); await myo.set_sleep_mode(SleepMode.NORMAL)
             except Exception as teardown_e: print(f"ERROR: Failed during Myo early teardown: {teardown_e}")
             return None # Indicate stop signal was received

        await asyncio.sleep(0.05) # Check stop event frequently

    print(f"--- Relax...")
    emg_callback_active = False # Stop collecting samples after hold duration

    # Teardown Myo after hold
    try:
        await myo.set_mode(emg_mode=None, imu_mode=False);
        await myo.set_sleep_mode(SleepMode.NORMAL)
    except Exception as teardown_e:
        print(f"ERROR: Failed during Myo teardown after hold: {teardown_e}"); traceback.print_exc()

    print(f"→ Collected {len(samples)} samples for this hold.")
    return samples


# <<< Main Asynchronous Function >>>
# <<< Modified to use mp.Event for synchronization >>>
async def main():
    # --- File Setup ---
    os.makedirs(DATA_INPUT_PATH, exist_ok=True)
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(DATA_INPUT_PATH, f"{RAW_DATA_FILENAME_PREFIX}{ts_str}.csv")
    print(f"Saving to: {out_file}")
    print("-" * 50)

    # --- Bluetooth Scan ---
    device = None; print("Scanning for Myo device...")
    try:
        scanner = BleakScanner(); devices = await scanner.discover(timeout=10.0);
        device = next((d for d in devices if d.address == MYO_ADDRESS), None)
    except BleakError as be: print(f"ERROR: Bluetooth error during scan: {be}"); return
    except Exception as e: print(f"ERROR: Bluetooth scan failed: {e}"); traceback.print_exc(); return

    if not device: print(f"ERROR: Could not find Myo at {MYO_ADDRESS}."); return
    print(f"Found device: {device.name} ({device.address})")

    # --- Main Execution Block ---
    myo_connection = None; emg_mean = None; emg_std = None
    opencv_process = None; stop_event = None; ready_event = None # Define for broader scope

    try:
        myo_connection = Myo(device)
        print("Connecting to Myo...")
        async with myo_connection as myo:
            print("Connected to Myo.")
            await asyncio.sleep(1.0) # Allow connection to settle

            # --- Run EMG Calibration (Imported) ---
            # The imported function needs the 'myo' object and constants
            print("Running imported calibration phase...")
            try:
                 # Pass necessary arguments if the imported function requires them
                 # Assuming it needs 'myo', 'PAUSE_DURATION', 'CALIBRATION_HOLD_TIME',
                 # 'CALIBRATION_POSES', 'NUM_EMG_SENSORS'. Adjust if needed.
                 emg_mean, emg_std = await run_calibration_phase(myo=myo)
            except Exception as calib_e:
                 print(f"ERROR: Exception during imported calibration phase: {calib_e}")
                 traceback.print_exc()
                 return # Exit if calibration fails

            if emg_mean is None or emg_std is None:
                print("ERROR: Imported EMG Calibration failed (returned None). Exiting."); return

            # --- Setup Multiprocessing for OpenCV ---
            with mp.Manager() as manager:
                latest_landmarks_proxy = manager.dict()
                landmarks_lock = mp.Lock()
                stop_event = mp.Event() # For signaling stop to both processes
                ready_event = mp.Event() # For worker to signal readiness <<< NEW
                opencv_process = None

                print("Starting OpenCV worker process...")
                opencv_process = mp.Process(
                    target=opencv_worker,
                    args=(latest_landmarks_proxy, landmarks_lock, stop_event, ready_event) # Pass the ready event <<< MODIFIED
                )
                opencv_process.start()

                # --- Wait for OpenCV Worker to be Ready ---
                print("Waiting for OpenCV process to initialize and signal ready...")
                wait_success = ready_event.wait(timeout=30.0) # Wait for signal with a timeout <<< MODIFIED

                if not wait_success:
                     print("ERROR: Timeout waiting for OpenCV process to become ready. Exiting.")
                     if opencv_process.is_alive():
                          print("Attempting to stop OpenCV process...")
                          stop_event.set()
                          opencv_process.join(timeout=5.0)
                          if opencv_process.is_alive(): opencv_process.terminate()
                     return # Exit if worker didn't signal readiness

                # After waiting, check if the stop event was set (indicating an error during worker init)
                if stop_event.is_set():
                     print("ERROR: OpenCV worker signaled stop during initialization or failed. Exiting.")
                     if opencv_process.is_alive(): opencv_process.join(timeout=5.0) # Wait briefly for cleanup
                     return

                print("OpenCV process signaled ready. Proceeding with collection.")

                # --- Randomized Data Collection Phase ---
                all_samples = []; start_time = time.time(); session_duration_seconds = SESSION_DURATION_MINUTES * 60
                print(f"\n--- Starting Randomized Collection Phase ({SESSION_DURATION_MINUTES} min) ---")

                while (time.time() - start_time) < session_duration_seconds:
                    if stop_event.is_set(): # Check if OpenCV process requested stop (e.g., 'q' key or error)
                        print("\nStop event detected. Ending collection phase early.")
                        break

                    elapsed_time = time.time() - start_time; remaining_time = session_duration_seconds - elapsed_time
                    print(f"\nSession time remaining: {timedelta(seconds=int(remaining_time))}")

                    # Select next pose and duration
                    pose_id, pose_name = random.choice(list(RANDOM_POSES.items()))
                    hold_duration = random.uniform(MIN_HOLD_TIME, MAX_HOLD_TIME)

                    # Check if enough time remains for this hold + pause
                    if elapsed_time + hold_duration + PAUSE_DURATION > session_duration_seconds + 2: # Add buffer
                        print("Nearing end of session, skipping last hold to allow for cleanup.")
                        break

                    try:
                        # Collect data for the hold, passing the stop_event
                        repsamp = await collect_dynamic_hold(myo, pose_id, pose_name, hold_duration, latest_landmarks_proxy, landmarks_lock, emg_mean, emg_std, stop_event)

                        if repsamp is None: # Indicates collection was stopped early by signal
                             print("Collection stopped during hold by signal. Ending session.")
                             break # Exit the main collection loop

                        all_samples.extend(repsamp)
                    except Exception as collect_e:
                        print(f"ERROR: Exception during dynamic hold for '{pose_name}': {collect_e}"); traceback.print_exc()
                        print("WARNING: Continuing collection despite error.")

                    # Check stop event again *after* collection attempt and pause
                    if stop_event.is_set(): break

                    await asyncio.sleep(PAUSE_DURATION / 2.0) # Shorter pause

                print("\n--- Randomized Collection Phase Finished ---")

                # --- Data Saving ---
                if not all_samples: print("WARNING: No data collected during randomized phase.")
                else:
                    print("Building DataFrame..."); df = pd.DataFrame(all_samples); cols = ALL_DATA_COLUMNS
                    print("Ensuring all columns exist...");
                    for c in cols:
                        if c not in df.columns: df[c] = pd.NA # Add missing columns with NA
                    df = df[cols] # Ensure correct column order
                    print(f"\nSaving {len(df)} samples to CSV: {out_file}")
                    try: df.to_csv(out_file, index=False); print("Save complete.")
                    except Exception as save_e: print(f"ERROR: Failed to save DataFrame to CSV: {save_e}"); traceback.print_exc()

    except Exception as e:
        print(f"ERROR during main execution: {e}"); traceback.print_exc()
        # Ensure stop event is set if main execution fails unexpectedly
        if stop_event is not None and not stop_event.is_set():
             stop_event.set()
    finally:
        print("\nInitiating cleanup...")
        # --- Cleanup OpenCV Process ---
        if stop_event is not None and not stop_event.is_set():
             print("Setting stop event for worker process...")
             stop_event.set() # Ensure worker process knows to stop

        if opencv_process and opencv_process.is_alive():
            print("Joining OpenCV process...")
            opencv_process.join(timeout=10.0) # Wait for graceful exit
            if opencv_process.is_alive():
                print("WARNING: OpenCV process did not exit gracefully. Terminating...")
                opencv_process.terminate()
                opencv_process.join(timeout=2.0) # Wait for termination
            else: print("OpenCV process finished.")
        elif opencv_process: print("OpenCV process was not running or already finished.")
        else: print("OpenCV process was not created.")

        # Myo connection is automatically closed by 'async with'
        print("Collection script finished.")


# --- Script Entry Point ---
if __name__ == "__main__":
    # mp.freeze_support() # Uncomment if needed for Windows executables

    # Print splash screen once
    print_splash_screen()

    try:
        # Required for Windows asyncio with multiprocessing/Bleak
        if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        # Run the main async function
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Exiting...")
    except Exception as e:
        print(f"ERROR: Top-level exception caught: {e}"); traceback.print_exc()
    finally:
        print("Script execution ended.") # Final message