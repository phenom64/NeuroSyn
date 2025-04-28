# collection.py for NeuroSyn Physio Project (Adapted for model-v4)
# v2: Corrected to use variable names from constants.py (v3 - Compatibility Names)
# Collects EMG and IMU data for defined physiotherapy exercises using Myo Armband.

import time
import csv
import os
from datetime import datetime
import sys
import threading # Used for pausing input during recording
import platform # To potentially adjust Myo connection if needed

# Attempt to import pymyo, provide guidance if missing
try:
    from pymyo import Myo, emg_mode
except ImportError:
    print("Error: pymyo library not found.")
    print("Please install it: pip install pymyo")
    sys.exit(1)

# Import constants from the constants file in the same directory
try:
    # Assuming constants.py is in the same directory
    import constants as const
except ImportError:
    print("Error: constants.py not found in the current directory.")
    print("Make sure constants.py is in the same folder as collection.py.")
    sys.exit(1)

# --- Global Variables ---
script_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
current_exercise_label = -1  # Label for the exercise being recorded (-1 indicates not recording)
collected_data = []          # List to store (timestamp, emg, imu, label) tuples
is_recording = False         # Flag to control data collection in the handler
recording_stop_event = threading.Event() # Event to signal stopping recording

# --- Myo Data Handler ---
# This function will be called by pymyo when new data is available
def handle_data(emg, imu, timestamp):
    """
    Callback function to handle incoming EMG and IMU data from the Myo armband.

    Args:
        emg (tuple): A tuple containing EMG readings from the 8 sensors.
        imu (tuple): A tuple containing IMU data (quaternion: w, x, y, z).
                     Adjust indexing if pymyo provides more/different IMU data.
        timestamp (int): The timestamp associated with the data packet.
    """
    global collected_data, is_recording, current_exercise_label

    if is_recording and current_exercise_label != -1:
        # Ensure we capture the correct number of EMG and IMU values
        if len(emg) == const.NUM_EMG_SENSORS and len(imu) >= const.NUM_IMU_VALUES:
            orientation_quat = imu[0:const.NUM_IMU_VALUES] # Extract quaternion (w, x, y, z)
            # Append data as a tuple: (timestamp, emg_tuple, imu_tuple, label)
            collected_data.append((timestamp, emg, orientation_quat, current_exercise_label))
        # Optional: Add logging here if data format issues occur

# --- Data Saving Function ---
def save_data_to_csv():
    """Saves the collected data to a CSV file."""
    global collected_data, script_start_time

    if not collected_data:
        print("No data collected to save.")
        return

    # Ensure the data directory exists (using the compatible variable name)
    if not os.path.exists(const.DATA_INPUT_PATH):
        print(f"Creating data directory: {const.DATA_INPUT_PATH}")
        os.makedirs(const.DATA_INPUT_PATH)

    # Construct filename using the compatible prefix and path
    filename = os.path.join(const.DATA_INPUT_PATH, f"{const.RAW_DATA_FILENAME_PREFIX}{script_start_time}.csv")

    print(f"\nSaving collected data to {filename}...")

    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # --- Write Header Row ---
            header = ['timestamp']
            header.extend([f'emg{i+1}' for i in range(const.NUM_EMG_SENSORS)])
            header.extend(['quat_w', 'quat_x', 'quat_y', 'quat_z']) # IMU columns
            header.append('label')
            writer.writerow(header)

            # --- Write Data Rows ---
            for timestamp, emg_data, imu_data, label in collected_data:
                row = [timestamp]
                row.extend(emg_data)
                row.extend(imu_data)
                row.append(label)
                writer.writerow(row)

        print(f"Successfully saved {len(collected_data)} data points.")
        # Clear collected data after saving
        collected_data = []

    except IOError as e:
        print(f"Error saving data to CSV: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during saving: {e}")


# --- Main Data Collection Logic ---
def main():
    """Main function to manage Myo connection and data collection loop."""
    global is_recording, current_exercise_label, collected_data, recording_stop_event

    print("--- NeuroSyn Physio Data Collection (v2 - Compatible) ---")
    # Using const.CLASSES for the exercise list
    print(f"Exercises to record ({const.REPETITIONS_PER_EXERCISE} reps each, {const.COLLECTION_TIME}s per rep):")
    for label, name in const.CLASSES.items():
        print(f"  {label}: {name}")
    print("-" * 30)

    # --- Initialize Myo ---
    print("Connecting to Myo Armband...")
    myo_connection_params = {'mode': emg_mode.FILTERED}

    # Explicitly use MYO_ADDRESS if provided in constants
    if hasattr(const, 'MYO_ADDRESS') and const.MYO_ADDRESS:
        print(f"Attempting connection to specific address: {const.MYO_ADDRESS}")
        # pymyo uses 'mac_address' parameter
        myo_connection_params['mac_address'] = const.MYO_ADDRESS
        # Note: Some older libraries/versions might use 'tty' for serial port on Linux/Mac
        # if platform.system() != "Windows":
        #     # You might need to find the correct device path, e.g., /dev/ttyACM0
        #     # myo_connection_params['tty'] = '/dev/ttyACM0' # Example
        #     pass
    else:
        print("No specific MYO_ADDRESS set in constants.py, attempting auto-detection.")
        # pymyo attempts auto-detection if mac_address is not provided

    try:
        m = Myo(**myo_connection_params)
        # Setup handlers: Use lambda to pass necessary data to our handle_data function
        # We get orientation directly from m.orientation within the EMG handler call
        m.add_emg_handler(lambda emg, moving, times: handle_data(emg, m.orientation, times[-1]))
        # IMU handler might not be strictly needed if orientation is accessed directly,
        # but add it just in case and do nothing.
        m.add_imu_handler(lambda imu, times: None)

        m.connect() # Establish the connection
        print("Myo connected successfully!")

    except Exception as e:
        print(f"\nError connecting to Myo: {e}")
        print("Troubleshooting:")
        print("- Is the Myo charged and nearby?")
        print("- Is Bluetooth enabled?")
        print("- Is the MYO_ADDRESS in constants.py correct?")
        print("- Do you have the necessary drivers/permissions (e.g., udev rules on Linux)?")
        print("- If auto-detection fails, ensure MYO_ADDRESS is set.")
        sys.exit(1)

    # --- Data Collection Loop ---
    try:
        # Start Myo data streaming in a background thread
        m.run_daemon() # Use run_daemon to run in background without blocking

        # Iterate through exercises using const.CLASSES
        for label, exercise_name in const.CLASSES.items():
            print(f"\n--- Preparing for Exercise: {label} - {exercise_name} ---")
            current_exercise_label = label

            for rep in range(const.REPETITIONS_PER_EXERCISE):
                print(f"\nRepetition {rep + 1}/{const.REPETITIONS_PER_EXERCISE}")
                print(f"Get ready to perform: '{exercise_name}'")
                # Using const.PAUSE_DURATION
                print(f"Recording will start in {const.PAUSE_DURATION} seconds...")
                time.sleep(const.PAUSE_DURATION)

                # Using const.COLLECTION_TIME for recording duration
                print(f"*** RECORDING '{exercise_name}' NOW for {const.COLLECTION_TIME} seconds... ***")
                is_recording = True
                recording_stop_event.clear() # Reset event

                # Start a timer thread to stop recording after the duration
                stop_timer = threading.Timer(const.COLLECTION_TIME, lambda: recording_stop_event.set())
                stop_timer.start()

                # Wait until the timer signals to stop
                recording_stop_event.wait()
                is_recording = False
                print("*** RECORDING STOPPED ***")

                # Brief pause before next rep or exercise
                time.sleep(0.5)

            # Reset label after all reps for an exercise are done
            current_exercise_label = -1

        print("\n--- Data Collection Complete ---")

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Stopping data collection...")
    except Exception as e:
        print(f"\nAn error occurred during collection: {e}")
    finally:
        # --- Cleanup ---
        print("Disconnecting Myo and saving data...")
        is_recording = False # Ensure recording stops
        m.disconnect()
        print("Myo disconnected.")
        # Use the correct path variable for saving
        save_data_to_csv()

    print("\nCollection script finished.")


if __name__ == "__main__":
    main()
