# constants.py for NeuroSyn Physio Project (model-v4)
# v3: Restored original variable names for compatibility, updated values for physio.

import os

# --- Essential Configuration ---

# Myo Armband Bluetooth MAC Address
# IMPORTANT: Verify this is the correct address for YOUR Myo Armband.
MYO_ADDRESS = "DD:31:D8:40:BC:22" # As provided from your previous constants

# Directory for storing collected raw data and potentially processed data
# Using the original variable name for compatibility.
MAIN_PATH = "model-v4"
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

# --- Icon Paths (Placeholder) ---
# Using the original variable name. Paths need to be updated
# if/when icons are created for the physiotherapy GUI.
ICON_PATHS = {
    0: "icons/physio_rest.png", # Example placeholder path
    1: "icons/physio_wrist_flexion.png",
    2: "icons/physio_wrist_extension.png",
    3: "icons/physio_elbow_flexion.png",
    4: "icons/physio_elbow_extension.png",
    5: "icons/physio_hand_close.png",
    6: "icons/physio_hand_open.png",
    7: "icons/physio_pronation.png",
    8: "icons/physio_supination.png",
    # Add more placeholders if needed, ensure keys match CLASSES
}
# Ensure the base 'icons' directory exists if these paths are used directly
ICON_DIR = "icons"
if not os.path.exists(ICON_DIR):
     os.makedirs(ICON_DIR)
     print(f"Created icon directory placeholder: {ICON_DIR}")


# --- Data Collection Parameters ---
# Using the original variable name 'COLLECTION_TIME' to represent
# the duration (in seconds) to record data for each exercise repetition.
COLLECTION_TIME = 5 # seconds per repetition

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
WINDOW_SIZE = 100 # Example: 100 time steps (~0.5s at 200Hz) - NEEDS TUNING

# Step size (overlap) between consecutive windows
WINDOW_STEP = 50 # Example: 50 time steps overlap - NEEDS TUNING


# --- Print Confirmation ---
print("--- NeuroSyn Physio Constants Loaded (v3 - Compatibility Names) ---")
print(f"Myo Address: {MYO_ADDRESS if MYO_ADDRESS else 'Not Set!'}")
print(f"Data Input Path: {DATA_PATH}")
print(f"Model Path: {MODEL_PATH}")
print(f"Metadata Path: {METADATA_PATH}")
print(f"Collection Time per Rep (s): {COLLECTION_TIME}")
print(f"Repetitions per Exercise: {REPETITIONS_PER_EXERCISE}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Classes Map: {CLASSES}")
# print(f"Icon Paths: {ICON_PATHS}") # Keep commented unless debugging icons
print("-" * 35)

