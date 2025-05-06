# constants.py for NeuroSyn Physio Project (model-v4)
# v8: Added EXERCISE_PAIRS, EXERCISE_ICONS, DEFAULT_REPETITIONS for Physio GUI

import os

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
# <<< NOTE: Update these filenames if your landmark regression model
#     and its metadata have different names >>>
MODEL_FILENAME = "physio_model.h5" # Example name for the new model
METADATA_FILENAME = "physio_metadata.pkl" # Example name for its metadata

MODEL_PATH = os.path.join(MODEL_INPUT_PATH, MODEL_FILENAME)
METADATA_PATH = os.path.join(MODEL_INPUT_PATH, METADATA_FILENAME)

# --- Base Physiotherapy Classes (Keep for reference/training labels if needed) ---
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
NUM_CLASSES = len(CLASSES)

# --- NEW: Physiotherapy Exercise Definitions ---
# Maps user-facing exercise names to the base class IDs involved
EXERCISE_PAIRS = {
    "Wrist Mobility": (1, 2),      # Flexion & Extension
    "Elbow Mobility": (3, 4),      # Flexion & Extension
    "Hand Dexterity": (5, 6),      # Close & Open
    "Forearm Rotation": (7, 8),    # Pronation & Supination
    # Add more complex exercises if needed, potentially involving more classes
}

# --- NEW: Animated Icon Filenames (Relative to NSEmedia/gestures/) ---
# List the PNG filenames for each frame of the animation for each exercise.
# **IMPORTANT**: Replace these with your actual filenames!
EXERCISE_ICONS = {
    "Wrist Mobility": [
        "wrist_mobility_frame1.png",
        "wrist_mobility_frame2.png",
        "wrist_mobility_frame3.png",
        "wrist_mobility_frame2.png", # Example loop back
    ],
    "Elbow Mobility": [
        "elbow_mobility_frame1.png",
        "elbow_mobility_frame2.png",
        "elbow_mobility_frame3.png",
        "elbow_mobility_frame2.png",
    ],
    "Hand Dexterity": [
        "hand_dexterity_frame1.png",
        "hand_dexterity_frame2.png",
    ],
    "Forearm Rotation": [
        "forearm_rotation_frame1.png",
        "forearm_rotation_frame2.png",
        "forearm_rotation_frame3.png",
        "forearm_rotation_frame2.png",
    ],
    # Add entries for any other exercises defined in EXERCISE_PAIRS
}

# --- Icon path for Calibration 'Rest' (Keep if CalibrationDialog uses it) ---
ICON_PATHS = {
    0: "gesture_resting.png", # Used by CalibrationDialog for the 'Rest' prompt
    # Other entries might not be needed if EXERCISE_ICONS is used for main GUI
}

# --- Data Collection Parameters (Keep relevant ones) ---
COLLECTION_TIME = 5 # Used by CalibrationDialog for 'Rest' duration
REPETITIONS_PER_EXERCISE = 5 # From old collection script, might not be needed directly here
PAUSE_DURATION = 2 # Might be useful for timing between reps/exercises in GUI

# --- NEW: Exercise Execution Parameters ---
DEFAULT_REPETITIONS = 3 # Default number of reps per exercise set

# --- Myo Armband Configuration ---
NUM_EMG_SENSORS = 8
NUM_IMU_VALUES = 4 # Quaternions (w, x, y, z)

# --- Landmark Prediction Model Configuration (Examples) ---
# Define the names of the landmarks your regression model will predict.
# The order should match the output layer of your model.
# Example using names similar to MediaPipe Pose + Hands:
PREDICTED_LANDMARK_NAMES = [
    # Left Arm (Pose)
    "Pose_L_Shoulder_x", "Pose_L_Shoulder_y", "Pose_L_Shoulder_z",
    "Pose_L_Elbow_x", "Pose_L_Elbow_y", "Pose_L_Elbow_z",
    "Pose_L_Wrist_x", "Pose_L_Wrist_y", "Pose_L_Wrist_z",
    # Right Hand (Example - adjust if Myo is on other arm or predict different hand)
    "Hand_R_Wrist_x", "Hand_R_Wrist_y", "Hand_R_Wrist_z", # May overlap/be same as Pose_L_Wrist if arm is same
    "Hand_R_Index_MCP_x", "Hand_R_Index_MCP_y", "Hand_R_Index_MCP_z",
    "Hand_R_Pinky_MCP_x", "Hand_R_Pinky_MCP_y", "Hand_R_Pinky_MCP_z",
    # Add ALL other landmarks your model predicts (Thumb, other fingers, etc.)
]
NUM_PREDICTED_LANDMARKS = len(PREDICTED_LANDMARK_NAMES) // 3 # Assuming x,y,z per landmark

# --- Data Processing / Model Training Parameters (Keep for reference/training) ---
WINDOW_SIZE = 30 # Input sequence length for the model
WINDOW_STEP = 25 # Step for creating sequences during training

# --- Print Confirmation (Optional) ---
# print("--- NeuroSyn Physio Constants Loaded (v8) ---")
# print(f"Myo Address: {MYO_ADDRESS}")
# print(f"Model Path: {MODEL_PATH}")
# print(f"Metadata Path: {METADATA_PATH}")
# print(f"Exercises: {list(EXERCISE_PAIRS.keys())}")
# print(f"Default Reps: {DEFAULT_REPETITIONS}")
# print(f"Predicted Landmarks Count: {NUM_PREDICTED_LANDMARKS}")
# print("-" * 35)

