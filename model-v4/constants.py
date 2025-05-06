# constants.py for NeuroSyn Physio Project (model-v4)
# v9: Added CALIBRATION_POSES/HOLD_TIME, updated PREDICTED_LANDMARK_NAMES, using .keras model format.

import os
import numpy as np # Needed for EMG_NORMALIZATION_EPSILON if used

# --- Essential Configuration ---
MYO_ADDRESS = "FF:39:C8:DC:AC:BA" # Keep your verified address

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PATH = BASE_DIR # Assuming constants.py is in the root of model-v4
DATA_INPUT_PATH = os.path.join(MAIN_PATH, "data")
MODEL_INPUT_PATH = os.path.join(MAIN_PATH, "model")
MEDIA_DIR = os.path.join(MAIN_PATH, "NSEmedia") # Used by GUI
GESTURES_MEDIA_DIR = os.path.join(MEDIA_DIR, "gestures") # Specific dir for icons

# Ensure data directories exist
os.makedirs(DATA_INPUT_PATH, exist_ok=True)
os.makedirs(MODEL_INPUT_PATH, exist_ok=True)
os.makedirs(GESTURES_MEDIA_DIR, exist_ok=True)

# --- Model & Metadata Paths ---
MODEL_FILENAME = "physio_model.keras"     # USE .keras format
METADATA_FILENAME = "physio_metadata.pkl" # Metadata for scaler, names etc.

MODEL_PATH = os.path.join(MODEL_INPUT_PATH, MODEL_FILENAME)
METADATA_PATH = os.path.join(MODEL_INPUT_PATH, METADATA_FILENAME)

# --- Data Collection Parameters ---
RAW_DATA_FILENAME_PREFIX = "physio_emg_imu_data_"
SESSION_DURATION_MINUTES = 2 # Duration of the randomized collection phase
MIN_HOLD_TIME = 3.0 # Minimum hold duration for a pose in seconds
MAX_HOLD_TIME = 5.0 # Maximum hold duration for a pose in seconds
PAUSE_DURATION = 2.0 # Pause duration between poses in seconds

# --- Myo Data Specs ---
NUM_EMG_SENSORS = 8
NUM_IMU_VALUES = 4 # For Quaternions (w, x, y, z)
EMG_NORMALIZATION_EPSILON = 1e-6 # Small value added to std dev to prevent division by zero

# --- Gesture / Pose Definitions ---
# Used by collection script for prompts and saving gesture_id/name
# Ensure 'Rest' has ID 0 if calibration relies on it specifically
CLASSES = {
    0: "Rest",
    1: "Wrist Flexion",
    2: "Wrist Extension",
    3: "Elbow Flexion",
    4: "Elbow Extension",
    5: "Hand Close",
    6: "Hand Open",
    7: "Forearm Pronation",
    8: "Forearm Supination",
    # Add any other poses you want to collect data for
}

# Define poses used specifically for the randomized collection phase
# Can be the same as CLASSES or a subset/different set
RANDOM_POSES = CLASSES # Use the same set for now, adjust if needed

# --- Calibration Definitions (Required by calibration.py) ---
# Define the poses the user will be guided through during calibration.
# IMPORTANT: calibration.py currently calculates mean/std from ALL these poses.
# If you only want 'Rest' data used, you'll need to modify calibration.py.
CALIBRATION_POSES = {
    "Rest": 0,
    # Add other poses IF your calibration script guides through them
    # e.g., "Max Contraction": 99 # Example if you add an MVC step
}
CALIBRATION_HOLD_TIME = 5.0 # Duration to hold each calibration pose in seconds

# --- Landmark Prediction Definitions (For Training Target) ---
# List of the EXACT landmark coordinate column names the regression model should predict.
# Based on May 3rd CSV data structure. Excludes visibility columns.
PREDICTED_LANDMARK_NAMES = [
    # Pose Landmarks (Right Arm, using Left Arm as reference for MediaPipe eccentricities)
    "Pose_L_Shoulder_x_shldr_norm", "Pose_L_Shoulder_y_shldr_norm", "Pose_L_Shoulder_z_shldr_norm",
    "Pose_L_Elbow_x_shldr_norm",    "Pose_L_Elbow_y_shldr_norm",    "Pose_L_Elbow_z_shldr_norm",
    "Pose_L_Wrist_x_shldr_norm",    "Pose_L_Wrist_y_shldr_norm",    "Pose_L_Wrist_z_shldr_norm",
    # Hand Landmarks (Right Hand)
    "Hand_R_Wrist_x_shldr_norm",      "Hand_R_Wrist_y_shldr_norm",      "Hand_R_Wrist_z_shldr_norm",
    "Hand_R_Thumb_CMC_x_shldr_norm",  "Hand_R_Thumb_CMC_y_shldr_norm",  "Hand_R_Thumb_CMC_z_shldr_norm",
    "Hand_R_Thumb_MCP_x_shldr_norm",  "Hand_R_Thumb_MCP_y_shldr_norm",  "Hand_R_Thumb_MCP_z_shldr_norm",
    "Hand_R_Thumb_IP_x_shldr_norm",   "Hand_R_Thumb_IP_y_shldr_norm",   "Hand_R_Thumb_IP_z_shldr_norm",
    "Hand_R_Thumb_Tip_x_shldr_norm",  "Hand_R_Thumb_Tip_y_shldr_norm",  "Hand_R_Thumb_Tip_z_shldr_norm",
    "Hand_R_Index_MCP_x_shldr_norm",  "Hand_R_Index_MCP_y_shldr_norm",  "Hand_R_Index_MCP_z_shldr_norm",
    "Hand_R_Index_PIP_x_shldr_norm",  "Hand_R_Index_PIP_y_shldr_norm",  "Hand_R_Index_PIP_z_shldr_norm",
    "Hand_R_Index_DIP_x_shldr_norm",  "Hand_R_Index_DIP_y_shldr_norm",  "Hand_R_Index_DIP_z_shldr_norm",
    "Hand_R_Index_Tip_x_shldr_norm",  "Hand_R_Index_Tip_y_shldr_norm",  "Hand_R_Index_Tip_z_shldr_norm",
    "Hand_R_Middle_MCP_x_shldr_norm", "Hand_R_Middle_MCP_y_shldr_norm", "Hand_R_Middle_MCP_z_shldr_norm",
    "Hand_R_Middle_PIP_x_shldr_norm", "Hand_R_Middle_PIP_y_shldr_norm", "Hand_R_Middle_PIP_z_shldr_norm",
    "Hand_R_Middle_DIP_x_shldr_norm", "Hand_R_Middle_DIP_y_shldr_norm", "Hand_R_Middle_DIP_z_shldr_norm",
    "Hand_R_Middle_Tip_x_shldr_norm", "Hand_R_Middle_Tip_y_shldr_norm", "Hand_R_Middle_Tip_z_shldr_norm",
    "Hand_R_Ring_MCP_x_shldr_norm",   "Hand_R_Ring_MCP_y_shldr_norm",   "Hand_R_Ring_MCP_z_shldr_norm",
    "Hand_R_Ring_PIP_x_shldr_norm",   "Hand_R_Ring_PIP_y_shldr_norm",   "Hand_R_Ring_PIP_z_shldr_norm",
    "Hand_R_Ring_DIP_x_shldr_norm",   "Hand_R_Ring_DIP_y_shldr_norm",   "Hand_R_Ring_DIP_z_shldr_norm",
    "Hand_R_Ring_Tip_x_shldr_norm",   "Hand_R_Ring_Tip_y_shldr_norm",   "Hand_R_Ring_Tip_z_shldr_norm",
    "Hand_R_Pinky_MCP_x_shldr_norm",  "Hand_R_Pinky_MCP_y_shldr_norm",  "Hand_R_Pinky_MCP_z_shldr_norm",
    "Hand_R_Pinky_PIP_x_shldr_norm",  "Hand_R_Pinky_PIP_y_shldr_norm",  "Hand_R_Pinky_PIP_z_shldr_norm",
    "Hand_R_Pinky_DIP_x_shldr_norm",  "Hand_R_Pinky_DIP_y_shldr_norm",  "Hand_R_Pinky_DIP_z_shldr_norm",
    "Hand_R_Pinky_Tip_x_shldr_norm",  "Hand_R_Pinky_Tip_y_shldr_norm",  "Hand_R_Pinky_Tip_z_shldr_norm",
]

# Optional: Calculate number of landmarks if needed elsewhere
NUM_TARGET_VALUES = len(PREDICTED_LANDMARK_NAMES)
# NUM_PREDICTED_LANDMARKS = NUM_TARGET_VALUES // 3 # If needed

# --- GUI Related Constants (Example) ---
EXERCISE_PAIRS = {
    "Wrist Mobility": (1, 2), # Maps to Wrist Flexion (1), Wrist Extension (2)
    "Elbow Mobility": (3, 4), # Maps to Elbow Flexion (3), Elbow Extension (4)
    # Add other exercises mapping to gesture IDs from CLASSES
}

# Maps exercise name to a list of icon filenames (relative to GESTURES_MEDIA_DIR)
# Replace with your actual icon filenames when available
EXERCISE_ICONS = {
    "Wrist Mobility": ["icon_wrist_flex_1.png", "icon_wrist_flex_2.png", "icon_wrist_ext_1.png", "icon_wrist_ext_2.png"],
    "Elbow Mobility": ["icon_elbow_flex_1.png", "icon_elbow_flex_2.png", "icon_elbow_ext_1.png", "icon_elbow_ext_2.png"],
    "Default": ["default_icon_1.png", "default_icon_2.png"]
}

DEFAULT_REPETITIONS = 3 # Default number of reps per exercise set
SESSION_HISTORY_FILE = os.path.join(MAIN_PATH, "session_history.json") # For GUI state


# --- Data Processing / Model Training Parameters (Keep for reference/training) ---
WINDOW_SIZE = 30 # Input sequence length for the model (e.g., 30 time steps)
WINDOW_STEP = 25 # Step size for creating overlapping windows (adjust as needed)

# Print confirmation
print(f"Constants loaded. Model path: {MODEL_PATH}, Metadata path: {METADATA_PATH}")
if CALIBRATION_POSES: print(f"Calibration poses defined: {list(CALIBRATION_POSES.keys())}")