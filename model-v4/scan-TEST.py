#!/usr/bin/env python3
"""
Minimal script to test Bleak scanner functionality with asyncio.
*** Added 'import cv2' AND camera init/release before scan to test for conflicts. ***
"""

import asyncio
import os
import traceback
from bleak import BleakScanner, BleakError
import cv2 # <<< IMPORTED GLOBALLY >>>

# --- Define MYO_ADDRESS directly here for the test ---
# Replace with your actual Myo address if different
MYO_ADDRESS = "FF:39:C8:DC:AC:BA"
# -----------------------------------------------------

async def run_minimal_scan():
    print("DEBUG: Starting minimal scan function.")
    device = None
    try:
        # <<< ADDED: Initialize and release camera BEFORE scan >>>
        print("DEBUG: Attempting to initialize and release camera...")
        cap = None
        try:
            cap = cv2.VideoCapture(0)
            if cap and cap.isOpened():
                print("DEBUG: Camera opened successfully.")
                cap.release()
                print("DEBUG: Camera released successfully.")
            else:
                print("WARNING: Could not open camera during pre-scan check.")
        except Exception as cam_err:
            print(f"ERROR: Exception during pre-scan camera check: {cam_err}")
            # Continue to scan even if camera check fails, to isolate Bleak issue
        print("DEBUG: Pre-scan camera check finished.")
        # <<< END ADDED SECTION >>>

        print("DEBUG: Attempting Bleak scan using discover()...")
        # Using discover first, as it's slightly simpler than find_by_address
        devices = await BleakScanner.discover(timeout=10.0)
        print(f"DEBUG: Bleak scan finished. Found {len(devices)} devices.")

        # Check if Myo was found among discovered devices
        myo_found = False
        if devices:
            print("--- Discovered Devices ---")
            for d in devices:
                print(f"- {d.address} ({d.name})")
                if d.address == MYO_ADDRESS:
                    myo_found = True
                    device = d # Store the device object if found
            print("------------------------")
        else:
            print("No BLE devices found.")

        if myo_found:
             print(f"SUCCESS: Myo device ({MYO_ADDRESS}) found during scan!")
        else:
             print(f"INFO: Myo device ({MYO_ADDRESS}) not found during this scan.")

    except BleakError as be:
        # Catch BleakError specifically
        print(f"ERROR: BleakError during scan: {be}")
        traceback.print_exc()
    except Exception as e:
        print(f"ERROR: General exception during scan: {e}")
        traceback.print_exc()

    print("DEBUG: Minimal scan function finished.")
    return device # Return device if found, otherwise None

if __name__ == "__main__":
    print("DEBUG: Minimal script execution started.")
    try:
        # Set the event loop policy for Windows (Keep it for now)
        if os.name == 'nt':
            print("DEBUG: Setting asyncio policy to SelectorEventLoop")
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        print("DEBUG: Running asyncio.run(run_minimal_scan)...")
        found_device = asyncio.run(run_minimal_scan())
        print(f"DEBUG: asyncio.run finished. Device object: {found_device}")

    except KeyboardInterrupt:
        print("\nDEBUG: Keyboard interrupt detected.")
        print("\nKeyboard interrupt—exiting.")
    except Exception as e:
        print(f"ERROR: Top-level exception in minimal script: {e}")
        traceback.print_exc()
    finally:
        print("DEBUG: Minimal script execution ended.")
