# constants.py for NeuroSyn Physio Project (model-v4)
# v7: Updated ICON_PATHS based on user request, other paths untouched.

import os

# --- Essential Configuration ---

# Myo Armband Bluetooth MAC Address
# IMPORTANT: Verify this is the correct address for YOUR Myo Armband.
#MYO_ADDRESS = "FF:39:C8:DC:AC:BA" # As provided from your previous constants
MYO_ADDRESS = "DD:31:D8:40:BC:22"

# Directory for storing collected raw data and potentially processed data
# Using the original variable name for compatibility.
MAIN_PATH = "model-v4"
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes constants.py is in the main folder
DATA_INPUT_PATH = "data"
MODEL_INPUT_PATH = "model"
#if not os.path.exists(DATA_INPUT_PATH):
#    os.makedirs(DATA_INPUT_PATH)
#    print(f"Created data directory: {DATA_INPUT_PATH}")

# --- Model & Metadata Paths ---
# Using original variable names but pointing to new physio-specific filenames
# to avoid overwriting old models and maintain compatibility.
MODEL_FILENAME = "physio_model.h5"
METADATA_FILENAME = "physio_metadata.pkl"

MODEL_PATH = os.path.join(MODEL_INPUT_PATH, MODEL_FILENAME)
METADATA_PATH = os.path.join(MODEL_INPUT_PATH, METADATA_FILENAME)
DATA_PATH = os.path.join(MAIN_PATH, DATA_INPUT_PATH)
MEDIA_DIR = os.path.join(BASE_DIR, "NSEmedia") # Used by GUI


# --- Physiotherapy Exercise Definitions ---
# Using the original 'CLASSES' dictionary name for compatibility,
# but populated with the new physiotherapy exercises.
CLASSES = {
    0: "Rest",
    1: "Wrist Flexion",
    2: "Wrist Extension",
    3: "Elbow Flexion",    # Bicep Curl action
    4: "Elbow Extension",    # Tricep Extension action
    5: "Hand Close",         # Fist Clench
    6: "Hand Open",
    7: "Forearm Pronation",  # Rotating palm downwards
    8: "Forearm Supination"   # Rotating palm upwards
}

# Total number of distinct exercises (classes) including 'Rest'
# Can be accessed via len(CLASSES) in other scripts if needed.
NUM_CLASSES = len(CLASSES) # Kept for convenience if used elsewhere

# --- Icon Filenames (Relative to NSEmedia/gestures/) ---
# The GUI script (NSE-interfaceFX.py) constructs the full path:
# os.path.join(MEDIA_PATH, "gestures", ICON_PATHS[class_id])
# Therefore, these should just be the filenames.
# <<< UPDATED based on user request >>>
ICON_PATHS = {
    0: "gesture_resting.png",          # Use existing file
    1: "physio_wrist_flexion.png",     # Use specific names
    2: "physio_wrist_extension.png",
    3: "physio_elbow_flexion.png",
    4: "physio_elbow_extension.png",
    5: "physio_hand_close.png",
    6: "physio_hand_open.png",
    7: "physio_pronation.png",
    8: "physio_supination.png",
    # IMPORTANT: Ensure these files actually exist in NSEmedia/gestures/
}

# --- Data Collection Parameters ---
# Using the original variable name 'COLLECTION_TIME' to represent
# the duration (in seconds) to record data for each exercise repetition.
COLLECTION_TIME = 5 # seconds per repetition

# Poses for EMG Calibration Phase (e.g., MVCs or strong static holds)
CALIBRATION_POSES = {
    #"Fist": 10,       # Strong fist clench
    #"Spread": 11,     # Fingers spread wide
    "Rest": 0         # Relaxed hand state
}

# Duration (seconds) to hold each calibration pose
CALIBRATION_HOLD_TIME = 5 # seconds

# Minimum duration (seconds) to hold a randomly selected pose
MIN_HOLD_TIME = 4 # seconds

# Maximum duration (seconds) to hold a randomly selected pose
MAX_HOLD_TIME = 8 # seconds

# Poses for the randomized collection phase (can be same as CLASSES)
RANDOM_POSES = CLASSES

# Epsilon value to prevent division by zero in EMG normalization
EMG_NORMALIZATION_EPSILON = 1e-6

# Total duration (minutes) for the randomized data collection session
SESSION_DURATION_MINUTES = 5 # minutes

# Number of repetitions to collect for each exercise (New parameter)
REPETITIONS_PER_EXERCISE = 5

# Pause duration (in seconds) between exercises or repetitions (New parameter)
PAUSE_DURATION = 2

# Base filename prefix for saved raw data files (New parameter)
# Timestamp will be appended during collection.
RAW_DATA_FILENAME_PREFIX = "physio_emg_imu_data_"

# --- Myo Armband Configuration ---
# Expected number of EMG sensors on the Myo Armband
NUM_EMG_SENSORS = 8

# Expected number of IMU orientation values (quaternion: w, x, y, z)
NUM_IMU_VALUES = 4 # Using Quaternions (w, x, y, z)

# --- Data Processing / Model Training Parameters ---
# (These might be adjusted later based on experimentation in train.ipynb)

# Window size for creating sequences from time-series data
WINDOW_SIZE = 30 # Example: 100 time steps (~0.5s at 200Hz) - NEEDS TUNING

# Step size (overlap) between consecutive windows
WINDOW_STEP = 25 # Example: 50 time steps overlap - NEEDS TUNING

# Index of the wrist landmark in MediaPipe Hand landmarks list
WRIST_LANDMARK_INDEX = 0

# --- Print Confirmation ---
# Using print format from user's uploaded file
"""
print("--- NeuroSyn Physio Constants Loaded---")
print("Version 3.2")
print("TM & (C) 2025 Syndromatic Inc. All rights reserved.")
print(f"Myo Address: {MYO_ADDRESS if MYO_ADDRESS else 'Not Set!'}")
print(f"Data Input Path: {DATA_INPUT_PATH}") # User's file prints this relative path
print(f"Model Path: {MODEL_PATH}")
print(f"Metadata Path: {METADATA_PATH}")
print(f"Collection Time per Rep (s): {COLLECTION_TIME}")
print(f"Repetitions per Exercise: {REPETITIONS_PER_EXERCISE}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Classes Map: {CLASSES}")
# print(f"Icon Paths: {ICON_PATHS}") # Keep commented unless debugging icons
print("-" * 35)
"""