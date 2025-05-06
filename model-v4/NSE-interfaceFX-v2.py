#!/usr/bin/env python3
"""
NeuroSyn ReTrain – Syndromatic Home Physio GUI
Created by Syndromatic Inc. / Kavish Krishnakumar (2025)
---------------------------------------------------------------------------
v10 (Draft): COMPLETE CODE. Fixes AttributeError on CalibrationDialog display.
             Restores full WelcomeWindow definition. Includes previous fixes:
             Calibration cancel logic, Connection window Z-order,
             Removed unwanted UI borders via QSS update.
             Enhanced ArmVisualizationWidget grid (smoother animation,
             better perspective attempt, horizon glow).
             **NOTE:** Requires constants update (v8). Landmark prediction
                      and analysis still needed. Arm drawing is placeholder.
"""

import os
import sys
import time
import asyncio
import math
import platform
import traceback
import json
from pathlib import Path
from collections import deque
import random # For placeholder score generation

import numpy as np # Needed for vector math in visualization

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QStyleFactory, QProgressBar, QTextEdit, QSizePolicy, QMessageBox, QGraphicsOpacityEffect,
    QGraphicsDropShadowEffect, QFrame, QGridLayout, QDialog # Added QDialog
)
from PyQt6.QtGui import (
    QPixmap, QAction, QIcon, QFont, QFontDatabase, QColor, QFontMetrics, QPalette,
    QPainter, QPen, QBrush, QPolygonF, QLinearGradient, QPainterPath, QRadialGradient # Added for drawing
)
from PyQt6.QtCore import (
    QThread, pyqtSignal, pyqtSlot, QObject, Qt, QTimer, QEasingCurve, QPropertyAnimation,
    QSequentialAnimationGroup, QParallelAnimationGroup, QPauseAnimation, QRect, QPointF, QPoint, QSize # Added for drawing
)

# Define base path and media path (relative to the script)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MEDIA_PATH = os.path.join(BASE_PATH, "NSEmedia")
DATA_PATH_BASE = os.path.join(BASE_PATH, "data") # Base for saving session history
SESSION_HISTORY_FILE = os.path.join(DATA_PATH_BASE, "session_history.json")

# --- Imports for Prediction Logic ---
try:
    # Assume prediction.py is updated later
    # <<< NOTE: NeuroSynPredictor needs update for landmark regression >>>
    from prediction import NeuroSynPredictor, run_calibration_flow
    # <<< NOTE: Assumes constants.py v8 with new definitions >>>
    from constants import (
        MYO_ADDRESS, ICON_PATHS, CLASSES, COLLECTION_TIME,
        NUM_EMG_SENSORS, NUM_IMU_VALUES,
        EXERCISE_PAIRS, EXERCISE_ICONS, DEFAULT_REPETITIONS,
        PREDICTED_LANDMARK_NAMES # Needed for arm drawing structure
    )
except ImportError as e:
    print(f"ERROR: Failed to import prediction/constants module: {e}")
    # Attempt to show error even if app hasn't fully started
    app = QApplication.instance()
    if not app: app = QApplication(sys.argv)
    QMessageBox.critical(None, "Import Error", f"Failed to import required files:\n{e}\nPlease check installation and file locations.")
    sys.exit(1)
except AttributeError as e:
     print(f"ERROR: Missing required constant in constants.py: {e}")
     app = QApplication.instance()
     if not app: app = QApplication(sys.argv)
     QMessageBox.critical(None, "Constant Error", f"Missing required constant in constants.py:\n{e}")
     sys.exit(1)


from pymyo import Myo
from pymyo.types import EmgMode, SleepMode, ImuMode
from bleak import BleakScanner, BleakError

# --- Patch pymyo (Keep if necessary) ---
# This patches a potentially buggy classifier handler in older pymyo versions.
# Keep it unless you are sure your version doesn't need it.
import types, struct
def _safe_on_classifier(self, sender, value):
    """Safely handle classifier events, ignoring malformed ones."""
    if len(value) < 3: return # Check minimum length
    try:
        # Attempt to unpack expected structure
        struct.unpack("<B2s", value)
    except struct.error:
        # Ignore if unpacking fails (malformed data)
        return
# Apply the patched method to the Myo class prototype
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# --------------------------------------------------------------------

# Global variables for fonts (Keep existing)
# Define font family names for consistent use throughout the application.
# Fallbacks are provided in case the specific fonts are not found.
GARAMOND_LIGHT_FAMILY = "ITCGaramondStd Light Condensed"
GARAMOND_LIGHT_ITALIC_FAMILY = "ITCGaramondStd Light Condensed Italic"
GARAMOND_BOOK_FAMILY = "ITCGaramondStd Book Condensed"
DOTMATRIX_FAMILY = "DotMatrix" # Used for potential log areas (can be removed if no logs)

# ==============================================================================
# == Worker for handling Myo data & Landmark Prediction ==
# ==============================================================================
class PredictionWorker(QObject):
    """
    Handles Myo communication, data buffering, and landmark prediction
    in a separate thread to keep the GUI responsive.
    """
    # --- Signals ---
    # Emits predicted landmark coordinates as a dictionary {base_name: np.array([x,y,z])}
    predicted_landmarks_signal = pyqtSignal(dict)
    # Emits performance scores (e.g., after a rep analysis)
    performance_update_signal = pyqtSignal(int, float) # (current_rep_score, avg_session_score)
    # Emits a value (0-100) representing EMG activity/strength
    progress_signal   = pyqtSignal(int)
    # Emits error messages
    error_signal      = pyqtSignal(str)
    # Emits status updates (connection, calibration, state changes)
    status_signal     = pyqtSignal(str)
    # Emitted when the worker's run() method finishes cleanly or due to error/stop
    finished_signal   = pyqtSignal()

    def __init__(self, predictor: NeuroSynPredictor):
        """
        Initializes the worker.
        Args:
            predictor: An instance of the (updated) NeuroSynPredictor class
                       responsible for landmark regression.
        """
        super().__init__()
        self.predictor    = predictor
        self._is_running  = True # Flag to control the main loop and callbacks
        # Asyncio events for synchronization with the GUI thread
        self._cal_start_event = asyncio.Event() # Signalled by GUI to start calibration flow
        self._stop_event      = asyncio.Event() # Signalled by GUI to stop the worker
        self._myo_task    = None # Holds the main asyncio task
        self.loop         = None # The asyncio event loop for this thread
        self._current_exercise_state = "idle" # Internal state tracking (optional)

    async def run_async(self):
        """The main asynchronous task run by the worker thread."""
        self.loop = asyncio.get_running_loop()
        try:
            # --- Myo Connection ---
            self.status_signal.emit(f"Scanning for Myo ({MYO_ADDRESS})...")
            # Use BleakScanner to find the Myo device by its specific address
            myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS, timeout=20.0)

            # Check if worker was stopped during scan or device not found
            if not self._is_running: return
            if not myo_device:
                self.error_signal.emit(f"Could not find Myo device at {MYO_ADDRESS}.")
                return

            self.status_signal.emit("Myo found. Connecting...")
            # Establish connection using pymyo context manager
            async with Myo(myo_device) as myo:
                if not self._is_running: return # Check if stopped during connection attempt
                self.status_signal.emit("Myo connected. Configuring...")
                # Configure Myo: prevent sleep, enable necessary data streams
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
                # Set modes required by the landmark prediction model
                await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=ImuMode.DATA)
                self.status_signal.emit("Myo configured.")

                # --- Setup Myo Callbacks ---
                # These functions will be called by pymyo when new data arrives.
                @myo.on_emg_smooth
                def on_emg_smooth_callback(emg_data):
                    """Handles incoming EMG data."""
                    # Ignore data if worker is stopping or data is invalid
                    if not self._is_running or emg_data is None: return
                    # Pass EMG data to the predictor instance.
                    # **NOTE:** The predictor's on_emg method needs to be updated
                    #           to handle data buffering and trigger landmark prediction.
                    predicted_landmarks = self.predictor.on_emg(emg_data)
                    if predicted_landmarks:
                        # Emit the predicted landmarks dictionary to the GUI thread
                        self.predicted_landmarks_signal.emit(predicted_landmarks)

                    # Calculate and emit raw EMG strength/activity for visualization
                    norm = math.sqrt(sum(e**2 for e in emg_data))
                    self.progress_signal.emit(int(min(100, (norm / 300) * 100))) # Example scaling

                @myo.on_imu
                def on_imu_callback(orientation, accel, gyro):
                    """Handles incoming IMU data."""
                    if not self._is_running: return
                    # Pass IMU data to the predictor instance.
                    # **NOTE:** The predictor's on_imu method needs to be updated.
                    self.predictor.on_imu(orientation, accel, gyro)
                # -----------------------

                # --- Initial Calibration Flow ---
                # Signal GUI that worker is ready for the calibration step
                self.status_signal.emit("READY_FOR_CAL")
                self.status_signal.emit("Waiting for user to start calibration guidance...")
                # Wait for either the GUI to signal start or a stop request
                start_cal_task = asyncio.create_task(self._cal_start_event.wait())
                stop_task_1    = asyncio.create_task(self._stop_event.wait())
                # Wait for the first task to complete
                done, pending = await asyncio.wait({start_cal_task, stop_task_1}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel() # Cancel the other pending task
                if stop_task_1 in done: # Check if stop was requested before starting
                    self.status_signal.emit("Stop requested before calibration guidance started.")
                    return

                # User clicked Start in GUI, run the calibration flow (e.g., 'Rest' pose)
                self.status_signal.emit("User started guidance. Running calibration flow...")
                # Call the imported calibration function (might need adjustments)
                cal_task = asyncio.create_task(run_calibration_flow(self.predictor, cue=self.status_signal.emit))
                stop_task_2 = asyncio.create_task(self._stop_event.wait())
                # Wait for calibration or stop signal
                done, pending = await asyncio.wait({cal_task, stop_task_2}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()
                if stop_task_2 in done: # Check if stop was requested during calibration
                    self.status_signal.emit("Stop requested during calibration guidance.")
                    return

                # Calibration finished normally
                self.status_signal.emit("Calibration flow finished normally.")
                self._current_exercise_state = "ready_for_exercise"
                # Signal GUI that exercises can now begin
                self.status_signal.emit("READY_FOR_EXERCISE")

                # --- Main Exercise Loop ---
                # Worker now primarily streams data via signals. Control logic (starting reps,
                # analyzing landmarks, calculating scores) resides in the main GUI thread.
                # The worker waits here until the GUI signals it to stop.
                if self._is_running:
                    self.status_signal.emit("Ready to start exercises.")
                    await self._stop_event.wait() # Wait indefinitely for the stop signal
                    self.status_signal.emit("Stop event received. Ending Myo loop.")

        except asyncio.CancelledError:
             # Handle case where the main task is cancelled externally
             self.status_signal.emit("Worker task explicitly cancelled.")
        except BleakError as e:
             print(f"Bluetooth Error in PredictionWorker: {e}")
             traceback.print_exc()
             if self._is_running: self.error_signal.emit(f"Bluetooth Error: {e}")
        except Exception as e:
            # Catch any other unexpected errors
            print(f"Error in PredictionWorker run_async: {e}")
            traceback.print_exc()
            if self._is_running: self.error_signal.emit(f"Worker Error: {e}")
        finally:
             # Cleanup code that runs whether the task finished normally or with an error
             self.status_signal.emit("Worker async run finished.")
             # Call predictor cleanup method if it exists (e.g., close TF session)
             if hasattr(self.predictor, 'cleanup'):
                 try:
                     self.predictor.cleanup()
                 except Exception as cleanup_e:
                     print(f"Error during predictor cleanup: {cleanup_e}")

    def run(self):
        """Entry point when QThread starts."""
        self.status_signal.emit("Prediction worker thread starting.")
        try:
            # Setup a new asyncio event loop for this thread
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            # Reset flags and events before starting
            self._is_running = True
            self._stop_event.clear()
            self._cal_start_event.clear()
            # Create and run the main async task within the loop
            self._myo_task = self.loop.create_task(self.run_async())
            self.loop.run_until_complete(self._myo_task)
        except Exception as e:
            # Catch errors during loop setup or execution
            print(f"Error during worker run execution: {e}")
            traceback.print_exc()
            # Emit error signal only if worker was intended to be running
            if self._is_running:
                self.error_signal.emit(f"Worker Run Error: {e}")
        finally:
            # --- Graceful asyncio loop cleanup ---
            print("Worker run() method finishing. Cleaning up loop...")
            try:
                if self.loop and self.loop.is_running():
                    print("Stopping running asyncio loop...")
                    # Schedule stop and run briefly to process
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    # Allow loop to process the stop command and pending tasks
                    # Using run_forever() after stop can sometimes hang, use short sleep
                    # self.loop.run_forever() # Avoid this if possible
                    time.sleep(0.2) # Short delay to allow tasks to finish/cancel

                if self.loop:
                    # Cancel any remaining tasks in the loop
                    tasks = asyncio.all_tasks(loop=self.loop)
                    if tasks:
                        print(f"Cancelling {len(tasks)} remaining worker tasks...")
                        for task in tasks:
                            if not task.done(): task.cancel()
                        # Wait for tasks to acknowledge cancellation
                        async def wait_cancelled(tasks_to_wait):
                            await asyncio.gather(*tasks_to_wait, return_exceptions=True)
                        try:
                            # Ensure the loop is running to process cancellations
                            if not self.loop.is_running():
                                self.loop.run_until_complete(asyncio.sleep(0.1)) # Run briefly if stopped
                            self.loop.run_until_complete(wait_cancelled(tasks))
                        except RuntimeError as run_err:
                             print(f"Runtime error during task cancellation wait: {run_err}")


            except Exception as e:
                print(f"Error during worker loop cleanup: {e}")
            finally:
                 # Close the loop if it's not already closed
                 if self.loop and not self.loop.is_closed():
                     self.loop.close()
                     print("Worker asyncio loop closed.")
                 self.status_signal.emit("Worker loop closed.")
                 # Signal the main GUI thread that the worker is completely finished
                 self.finished_signal.emit()

    def stop(self):
        """Signals the worker's asyncio loop to stop gracefully."""
        if not self._is_running:
            print("Worker stop already requested.")
            return

        self.status_signal.emit("Stop requested for worker.")
        self._is_running = False # Prevent further processing in callbacks

        # Signal the asyncio loop and tasks to stop
        if self.loop and self.loop.is_running():
            # Set events to unblock any asyncio.Event.wait() calls
            self.loop.call_soon_threadsafe(self._cal_start_event.set)
            self.loop.call_soon_threadsafe(self._stop_event.set)
            # Attempt to cancel the main task (_myo_task)
            if self._myo_task and not self._myo_task.done():
                 self.status_signal.emit("Attempting to cancel main worker task.")
                 # Schedule cancellation from the loop's thread
                 self.loop.call_soon_threadsafe(self._myo_task.cancel)
            # Optionally, schedule loop stop slightly later to allow cancellation to process
            # self.loop.call_soon_threadsafe(self.loop.stop)
        else:
            # If loop isn't running, just set events directly
            # This might happen if stop is called before run() fully starts the loop
            self._cal_start_event.set()
            self._stop_event.set()
            self.status_signal.emit("Worker loop not running, set stop events directly.")

    @pyqtSlot()
    def proceed_with_calibration(self):
        """Sets the event allowing the calibration flow to start (called from GUI)."""
        if self.loop and self.loop.is_running():
             self.status_signal.emit("GUI signalled to proceed with calibration.")
             # Safely set the event from the main thread to wake up the worker's wait
             self.loop.call_soon_threadsafe(self._cal_start_event.set)
        else:
             # This case should ideally not happen if GUI enables button correctly
             self.status_signal.emit("Warning: Tried to proceed calibration, but worker loop not running.")


# ==============================================================================
# == Calibration Assistant Dialog (Logic Fixed) ==
# ==============================================================================
class CalibrationDialog(QMainWindow): # Inherits QMainWindow, use show() not exec_()
    """
    Dialog window to guide the user through the initial 'Rest' pose
    stabilization period.
    """
    # Signal emitted when user clicks "Start"
    proceed_clicked = pyqtSignal()
    # Signal emitted ONLY when user confirms cancellation via the dialog
    cancel_clicked  = pyqtSignal()

    def __init__(self, parent=None): # Accept parent for proper ownership
        super().__init__(parent)
        self.setWindowTitle("Calibration Assistant")
        # Make it appear modal-like by staying on top
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(520, 340)

        # --- UI Elements Setup ---
        central = QWidget(self); self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central); self.vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo
        self.logo = QLabel()
        logo_pix = QPixmap(os.path.join(MEDIA_PATH, "SetupAssistant.png"))
        self.logo.setPixmap(logo_pix.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Header Text
        self.head = QLabel("Calibration Assistant")
        self.head.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold))
        self.head.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Body Text
        self.body = QLabel()
        self.body.setFont(QFont(GARAMOND_BOOK_FAMILY, 14))
        self.body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.setWordWrap(True)

        # Icon Placeholder (for gesture image)
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Timer Label (for countdown)
        self.timer_lbl = QLabel("")
        self.timer_lbl.setFont(QFont(GARAMOND_BOOK_FAMILY, 22))
        self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Button Layout
        self.btn_layout_widget = QWidget()
        self.btn_layout = QHBoxLayout(self.btn_layout_widget)
        self.btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # OK/Start Button
        self.ok_btn = QPushButton("Start")
        self.ok_btn.clicked.connect(self._on_ok)
        self.NSEWidgets_emboss(self.ok_btn)
        self.btn_layout.addWidget(self.ok_btn)
        # Cancel Button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel_request) # Connect to confirmation logic
        self.NSEWidgets_emboss(self.cancel_btn)
        self.btn_layout.addWidget(self.cancel_btn)

        # Add widgets to main layout
        self.vbox.addWidget(self.logo)
        self.vbox.addWidget(self.head)
        self.vbox.addWidget(self.body)
        self.vbox.addWidget(self.icon)
        self.vbox.addWidget(self.timer_lbl)
        self.vbox.addWidget(self.btn_layout_widget)

        # Initial state
        self.icon.hide()
        self.timer_lbl.hide()
        self.show_welcome_screen() # Set initial text and button states
        self.countdown = None # Timer object
        self._cancelled = False # Flag if cancellation was confirmed
        self._finished  = False # Flag if calibration step completed normally
        self._ok_proceeded = False # Flag if user clicked "Start"

    def show_welcome_screen(self):
        """Sets the initial view of the dialog."""
        self.logo.show()
        self.icon.hide()
        self.timer_lbl.hide()
        self.head.setText("Calibration Assistant")
        self.body.setText(
            "Before starting your exercises, we need to establish a baseline.\n"
            "Follow the on-screen instructions.\n"
            "Click 'Start' when ready."
        )
        self.ok_btn.setText("Start")
        self.ok_btn.show()
        self.ok_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel")
        self.cancel_btn.show()
        self._ok_proceeded = False # Reset flag

    @pyqtSlot()
    def _on_ok(self):
        """Handles clicks on the OK/Start button."""
        # If dialog is in cancelled or finished state, OK button just closes it
        if self._cancelled or self._finished:
            self.close()
            return

        # If it's the initial "Start" click
        if not self._ok_proceeded:
            self._ok_proceeded = True
            # Update UI for waiting state
            self.ok_btn.setEnabled(False)
            self.ok_btn.hide()
            self.cancel_btn.setText("Cancel Guidance") # Change cancel button text
            self.body.setText("Waiting for 'Rest' instruction from device...")
            self.logo.hide()
            # Emit signal to main window to tell worker to proceed
            self.proceed_clicked.emit()

    def show_gesture(self, title: str, seconds: int, icon_filename: str):
        """Displays the gesture prompt (e.g., 'Rest') with icon and timer."""
        # Only show if calibration is proceeding normally
        if self._cancelled or self._finished or not self._ok_proceeded: return

        # Update UI
        self.logo.hide()
        self.ok_btn.hide() # OK button remains hidden during prompt
        self.head.setText(title)
        self.body.setText(f"Hold for {seconds} s …")

        # Load and display icon
        gestures_dir = os.path.join(MEDIA_PATH, "gestures")
        full_icon_path = os.path.join(gestures_dir, icon_filename)
        if not os.path.exists(full_icon_path):
             print(f"Warning: Icon not found at {full_icon_path}")
             self.icon.setPixmap(QPixmap()) # Clear icon
             self.icon.hide()
        else:
             pix = QPixmap(full_icon_path)
             if pix.isNull():
                 print(f"Error: QPixmap is null for icon: {full_icon_path}")
                 self.icon.setPixmap(QPixmap())
                 self.icon.hide()
             else:
                 # Scale and set pixmap
                 self.icon.setPixmap(pix.scaled(140, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
                 self.icon.show()

        # Start countdown timer
        self.timer_lbl.show()
        if self.countdown: self.countdown.stop() # Stop previous timer if any
        self.time_left = seconds
        self.timer_lbl.setText(str(self.time_left))
        self.countdown = QTimer(self)
        self.countdown.timeout.connect(self._tick)
        self.countdown.start(1000) # Tick every second

    def _tick(self):
        """Updates the countdown timer label each second."""
        if self._cancelled or self._finished:
             if self.countdown: self.countdown.stop()
             return
        self.time_left -= 1
        # Update label or show checkmark when done
        self.timer_lbl.setText(str(self.time_left) if self.time_left > 0 else "✓")
        if self.time_left <= 0 and self.countdown:
            self.countdown.stop() # Stop timer when time runs out

    def mark_as_finished(self):
        """Updates UI when the calibration step is successfully completed."""
        if self._cancelled: return # Do nothing if already cancelled
        self._finished = True
        # Update UI to show completion message
        self.icon.hide()
        self.timer_lbl.hide()
        self.logo.show()
        self.head.setText("Setup complete!")
        self.body.setText("Ready to start exercises.")
        self.cancel_btn.hide()
        self.ok_btn.hide()
        # Automatically close the dialog after a short delay
        QTimer.singleShot(1500, self.close)

    def closeEvent(self, event):
        """Handles the window close event (e.g., clicking 'X')."""
        if self._finished or self._cancelled:
            # If finished normally or already cancelled, allow closing
            event.accept()
        else:
            # If user tries to close prematurely, trigger cancel confirmation
            self._on_cancel_request()
            event.ignore() # Prevent closing immediately

    @pyqtSlot()
    def _on_cancel_request(self):
        """Shows the confirmation dialog when cancel is requested."""
        if self._finished or self._cancelled: return # Don't show if already done

        reply = QMessageBox.question(
            self, "Cancel Setup?", # Dialog Title
            "Are you sure you want to cancel the initial setup?", # Message
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, # Buttons
            QMessageBox.StandardButton.No # Default button is No
        )
        if reply == QMessageBox.StandardButton.Yes:
            # User confirmed cancellation, update UI and signal main window
            self._show_cancelled_screen()
        # else: User clicked No, the confirmation dialog closes, this dialog remains open.

    def _show_cancelled_screen(self):
        """Updates UI to show cancellation message and allows closing."""
        if self._cancelled: return # Avoid running twice
        self._cancelled = True
        self._finished = False # Ensure finished flag is false
        if self.countdown: self.countdown.stop() # Stop timer

        # Update UI elements for cancelled state
        self.logo.show()
        self.icon.hide()
        self.timer_lbl.hide()
        self.cancel_btn.hide()
        self.head.setText("Setup cancelled")
        self.body.setText("Initial setup was interrupted.\nClick 'OK' to exit.")
        self.ok_btn.setText("OK")
        self.ok_btn.setEnabled(True)
        self.ok_btn.show()

        # Re-wire OK button to simply close the dialog now
        try: self.ok_btn.clicked.disconnect()
        except TypeError: pass # Ignore if no connection exists
        self.ok_btn.clicked.connect(self.close) # Connect OK to close

        # Emit signal to notify main window that cancellation was confirmed
        self.cancel_clicked.emit()

    # def accept(self): # Not needed if inheriting QMainWindow
    #     """Closes the dialog when OK is clicked on the cancelled screen."""
    #     super().accept() # Use QDialog's accept method if it's a QDialog, or close() for QMainWindow

    def NSEWidgets_emboss(self, btn: QPushButton):
        """Applies a subtle emboss effect to a button."""
        effect = QGraphicsDropShadowEffect(btn)
        effect.setBlurRadius(1)
        effect.setOffset(0, 1)
        effect.setColor(QColor(255, 255, 255, 80))
        btn.setGraphicsEffect(effect)

# ==============================================================================
# == Animated Connection Screen (Modal Dialog) ==
# ==============================================================================
class ConnectionWindow(QDialog): # Inherits QDialog for modality
    """
    A modal dialog showing an animation while connecting to the Myo device.
    Stays on top of the main window.
    """
    def __init__(self, parent=None): # Accept parent for proper centering/ownership
        super().__init__(parent)
        self.setWindowTitle("Connecting to Myo...")
        # Make it modal (blocks interaction with parent window)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        # Optional: Remove question mark button from title bar if desired
        # self.setWindowFlags(self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint)
        self.setFixedSize(600, 200) # Keep size fixed

        # --- Layout and Widgets (Identical to previous versions) ---
        layout = QHBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Computer Icon
        self.computer_label = QLabel()
        comp_pix = QPixmap(os.path.join(MEDIA_PATH, "computer_icon.png"))
        comp_scaled = comp_pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.computer_label.setPixmap(comp_scaled)
        layout.addWidget(self.computer_label)
        # Orbs Container
        self.orbs_container = QWidget()
        self.orbs_container.setFixedSize(300, 80)
        self.orbs_container.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.orbs_container)
        # Myo Icon
        self.myo_label = QLabel()
        myo_pix = QPixmap(os.path.join(MEDIA_PATH, "myo_icon.png"))
        myo_scaled = myo_pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.myo_label.setPixmap(myo_scaled)
        layout.addWidget(self.myo_label)

        # Create Orb Labels and Effects
        self.orb_labels = []
        self.orb_effects = []
        for _ in range(3):
            orb_label = QLabel(self.orbs_container)
            orb_pix = QPixmap(os.path.join(MEDIA_PATH, "orb.png")).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            orb_label.setPixmap(orb_pix)
            orb_label.move(0, 25) # Position within container
            effect = QGraphicsOpacityEffect(orb_label)
            effect.setOpacity(0.0) # Start invisible
            orb_label.setGraphicsEffect(effect)
            self.orb_labels.append(orb_label)
            self.orb_effects.append(effect)

        # Setup Animation Group
        self.anim_group = QParallelAnimationGroup(self)
        start_delays = [0, 500, 1000] # Stagger start times
        for i, orb_label in enumerate(self.orb_labels):
            seq_anim = self._create_orb_sequence(orb_label, self.orb_effects[i], start_delays[i])
            self.anim_group.addAnimation(seq_anim)
        self.anim_group.setLoopCount(-1) # Loop indefinitely
        self.anim_group.start() # Start animation immediately

    def _create_orb_sequence(self, orb_label: QLabel, opacity_effect: QGraphicsOpacityEffect, start_delay: int):
        """Creates the complex back-and-forth animation sequence for a single orb."""
        # This complex animation logic remains unchanged
        orb_seq = QSequentialAnimationGroup()
        if start_delay > 0: orb_seq.addAnimation(QPauseAnimation(start_delay))
        fade_time=300; travel_time=1000; left_rect=QRect(0,25,30,30); right_rect=QRect(self.orbs_container.width()-30,25,30,30)
        fade_in=QPropertyAnimation(opacity_effect,b"opacity");fade_in.setDuration(fade_time);fade_in.setStartValue(0.0);fade_in.setEndValue(1.0); move_in=QPropertyAnimation(orb_label,b"geometry");move_in.setDuration(fade_time);quarter_rect=QRect(int(right_rect.x()*0.25),25,30,30);move_in.setStartValue(left_rect);move_in.setEndValue(quarter_rect); forward_in_group=QParallelAnimationGroup();forward_in_group.addAnimation(fade_in);forward_in_group.addAnimation(move_in);orb_seq.addAnimation(forward_in_group)
        move_to_right=QPropertyAnimation(orb_label,b"geometry");move_to_right.setDuration(travel_time);move_to_right.setStartValue(quarter_rect);move_to_right.setEndValue(right_rect);orb_seq.addAnimation(move_to_right); fade_out_right=QPropertyAnimation(opacity_effect,b"opacity");fade_out_right.setDuration(fade_time);fade_out_right.setStartValue(1.0);fade_out_right.setEndValue(0.0);orb_seq.addAnimation(fade_out_right)
        fade_in2=QPropertyAnimation(opacity_effect,b"opacity");fade_in2.setDuration(fade_time);fade_in2.setStartValue(0.0);fade_in2.setEndValue(1.0); move_in2=QPropertyAnimation(orb_label,b"geometry");move_in2.setDuration(fade_time);three_quarter_rect=QRect(int(right_rect.x()*0.75),25,30,30);move_in2.setStartValue(right_rect);move_in2.setEndValue(three_quarter_rect); backward_in_group=QParallelAnimationGroup();backward_in_group.addAnimation(fade_in2);backward_in_group.addAnimation(move_in2);orb_seq.addAnimation(backward_in_group)
        move_to_left=QPropertyAnimation(orb_label,b"geometry");move_to_left.setDuration(travel_time);move_to_left.setStartValue(three_quarter_rect);move_to_left.setEndValue(left_rect);orb_seq.addAnimation(move_to_left); fade_out_left=QPropertyAnimation(opacity_effect,b"opacity");fade_out_left.setDuration(fade_time);fade_out_left.setStartValue(1.0);fade_out_left.setEndValue(0.0);orb_seq.addAnimation(fade_out_left)
        orb_seq.setLoopCount(-1); return orb_seq

    def stop_animation_and_close(self):
        """Stops animation and closes the dialog."""
        if self.anim_group and self.anim_group.state() == QPropertyAnimation.State.Running:
            self.anim_group.stop()
        # Use accept() for QDialog to close it and return control
        self.accept()

# ==============================================================================
# == REFINED: Arm Visualization Widget ==
# ==============================================================================
class ArmVisualizationWidget(QWidget):
    """Widget for drawing the arm/hand visualization with improved animated grid."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.background_color = QColor("#101015") # Very dark blue/grey
        # Set background color directly on the widget
        self.setAutoFillBackground(True)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, self.background_color)
        self.setPalette(pal)

        # --- Grid Animation ---
        self.grid_color = QColor(60, 60, 75, 160) # Dark grey-blue, semi-transparent
        self.horizon_color_center = QColor(80, 80, 110, 100) # Faint purple/blue glow center
        self.horizon_color_edge = QColor(25, 25, 35, 0) # Fades to transparent dark blue
        self.grid_offset_z = 0.0 # Controls the forward/backward scroll animation
        self.grid_timer = QTimer(self)
        self.grid_timer.timeout.connect(self._update_grid_animation)
        self.grid_timer.start(33) # Target ~30 FPS animation update rate

        # --- Arm Drawing ---
        self.landmarks = {} # Stores {base_name: np.array([x, y, z])}
        self.performance_color = QColor(128, 128, 128) # Start grey
        self.arm_line_width = 5
        self.joint_radius = 7

        # --- Perspective Projection Parameters ---
        self.fov = 75.0 # Wider field of view
        self.near_plane = 0.1
        self.far_plane = 40.0 # Reduce far plane for grid density
        self.camera_pos = np.array([0.0, 0.4, 2.0]) # Slightly higher, further back

    def _update_grid_animation(self):
        """Updates the grid offset for scrolling effect."""
        self.grid_offset_z = (self.grid_offset_z + 0.010) % 1.0 # Slower, smoother speed, loops using modulo
        self.update() # Trigger repaint for both grid and arm

    def project_3d_to_2d(self, point_3d):
        """Projects a 3D point (numpy array) to 2D screen coordinates."""
        width = self.width(); height = self.height()
        if width == 0 or height == 0: return None
        aspect_ratio = width / height; fov_rad = math.radians(self.fov / 2.0)
        # Avoid tan(0) or close to it
        if abs(fov_rad) < 1e-6 or abs(math.cos(fov_rad)) < 1e-6: return None
        f = 1.0 / math.tan(fov_rad)

        # Translate point relative to camera
        p_cam = point_3d - self.camera_pos

        # Check if point is behind or too close to the near plane
        # Z is negative away from camera in typical right-handed coordinate systems
        if p_cam[2] >= -self.near_plane: return None

        # Perspective divide (handle potential division by zero)
        z_cam_inv = -1.0 / p_cam[2] # Precompute inverse Z
        x_ndc = (f * p_cam[0] * z_cam_inv) / aspect_ratio
        y_ndc = f * p_cam[1] * z_cam_inv

        # Basic clipping check (-1 to 1 range for NDC)
        if not (-1.1 <= x_ndc <= 1.1 and -1.1 <= y_ndc <= 1.1):
            # Allow slightly outside bounds for line drawing continuity
            return None # Clip points far outside

        # Convert NDC to screen coordinates
        screen_x = int((x_ndc + 1.0) * 0.5 * width)
        screen_y = int((1.0 - y_ndc) * 0.5 * height) # Y is inverted

        return QPoint(screen_x, screen_y)

    def set_landmarks(self, landmark_data):
        """
        Update the landmarks to be drawn.
        Input: Dict where keys are full landmark names (e.g., "Pose_L_Shoulder_x")
               and values are coordinates. Converts to {base_name: np.array([x,y,z])}.
        """
        new_landmarks = {}
        base_names_processed = set()
        for full_name, coord in landmark_data.items():
            try:
                # Extract base name (e.g., "Pose_L_Shoulder")
                base_name = "_".join(full_name.split('_')[:-1])
                if base_name not in base_names_processed and base_name: # Ensure base_name is not empty
                    # Get all coords for this base name, default to 0 if missing
                    x = landmark_data.get(f"{base_name}_x", 0.0)
                    y = landmark_data.get(f"{base_name}_y", 0.0)
                    z = landmark_data.get(f"{base_name}_z", 0.0)
                    new_landmarks[base_name] = np.array([x, y, z])
                    base_names_processed.add(base_name)
            except Exception as e:
                # Log error but continue processing other landmarks
                print(f"Error processing landmark {full_name}: {e}")
                continue
        self.landmarks = new_landmarks
        # Repaint is triggered by the grid timer

    def set_performance_color(self, score_percent):
        """Update the arm/joint color based on performance score (0-100)."""
        score_percent = max(0, min(100, score_percent)) # Clamp value
        # Red -> Yellow -> Green gradient
        if score_percent < 50: # Transition Red to Yellow over first half
            ratio = score_percent / 50.0
            red = 255
            green = int(255 * ratio)
            blue = 0
        else: # Transition Yellow to Green over second half
             ratio = (score_percent - 50.0) / 50.0
             red = int(255 * (1.0 - ratio))
             green = 255
             blue = 0
        self.performance_color = QColor(red, green, blue)
        # Repaint is triggered by the grid timer

    def paintEvent(self, event):
        """Handles drawing the widget content: grid and arm skeleton."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Background is handled by widget palette/autofill

        # Draw the animated grid first (background)
        self._draw_perspective_grid(painter)

        # Draw the arm skeleton on top
        self._draw_arm_skeleton(painter)

        painter.end()

    def _draw_perspective_grid(self, painter):
        """Draws the improved animated perspective grid."""
        width = self.width(); height = self.height()
        horizon_y = height * 0.48 # Horizon line position

        # --- Draw Horizon Glow (Radial Gradient) ---
        horizon_glow_radius = width * 0.8
        glow_center = QPointF(width / 2.0, horizon_y) # Use float for center
        radial_grad = QRadialGradient(glow_center, horizon_glow_radius)
        radial_grad.setColorAt(0, self.horizon_color_center) # Brighter center
        radial_grad.setColorAt(1, self.horizon_color_edge) # Fades out
        # Draw the gradient over the entire widget area
        painter.fillRect(self.rect(), QBrush(radial_grad))

        # --- Draw Grid Lines ---
        num_lines_x = 20; num_lines_z = 50 # Grid density
        grid_range_x = 6.0; grid_range_z = 20.0 # Spatial extent of grid
        grid_step_z = grid_range_z / num_lines_z # Distance between horizontal lines

        pen = QPen(self.grid_color, 1.0) # Base pen for grid
        painter.setPen(pen)

        # Draw horizontal lines (Z-axis lines)
        for i in range(num_lines_z + 1):
            z_world = i * grid_step_z
            # Animate Z position using the offset, looping with modulo
            z_anim = (z_world + self.grid_offset_z * grid_step_z) % grid_range_z
            # Define line endpoints in 3D (Y=0 plane, Z negative away from camera)
            p1_3d = np.array([-grid_range_x, 0, -z_anim])
            p2_3d = np.array([grid_range_x, 0, -z_anim])
            # Project 3D points to 2D screen coordinates
            p1_2d = self.project_3d_to_2d(p1_3d)
            p2_2d = self.project_3d_to_2d(p2_3d)

            # Draw line only if both points are valid and below horizon
            if p1_2d and p2_2d and p1_2d.y() >= horizon_y:
                 # Calculate alpha based on distance below horizon (fade near horizon)
                 alpha_ratio = max(0, (p1_2d.y() - horizon_y) / (height - horizon_y))
                 current_alpha = int(self.grid_color.alpha() * alpha_ratio**1.5) # Faster fade
                 if current_alpha > 5: # Optimization: don't draw if nearly transparent
                     pen.setColor(QColor(self.grid_color.red(), self.grid_color.green(), self.grid_color.blue(), current_alpha))
                     painter.setPen(pen)
                     painter.drawLine(p1_2d, p2_2d)

        # Draw vertical lines (X-axis lines, converging)
        pen.setWidthF(1.0) # Reset pen width
        for i in range(num_lines_x + 1):
            # Calculate X coordinate in 3D space
            x = (i / num_lines_x - 0.5) * grid_range_x * 2 # Spread lines wider
            # Define line endpoints in 3D (from near plane to far plane)
            p1_3d = np.array([x, 0, -self.near_plane * 1.1]) # Start just in front
            p2_3d = np.array([x, 0, -grid_range_z])       # End far away
            # Project to 2D
            p1_2d = self.project_3d_to_2d(p1_3d)
            p2_2d = self.project_3d_to_2d(p2_3d)

            # Draw line only if both points are valid
            if p1_2d and p2_2d:
                draw_start_point = p1_2d
                # Clip line start point to the horizon if it starts above
                if p1_2d.y() < horizon_y:
                    # If end point is also above horizon, skip line
                    if p2_2d.y() <= horizon_y: continue
                    # Calculate intersection point with horizon line
                    try:
                        # Parametric line equation: P = P1 + t*(P2-P1)
                        # We want point where P.y = horizon_y
                        t = (horizon_y - p1_2d.y()) / (p2_2d.y() - p1_2d.y())
                        intersect_x = p1_2d.x() + t * (p2_2d.x() - p1_2d.x())
                        draw_start_point = QPointF(intersect_x, horizon_y)
                    except ZeroDivisionError: # Avoid division by zero if line is horizontal
                        continue # Skip horizontal lines exactly on horizon

                # Set pen color (no fading for vertical lines currently)
                pen.setColor(self.grid_color)
                painter.setPen(pen)
                # Draw line from calculated start point (at or below horizon) to end point
                painter.drawLine(QPoint(int(draw_start_point.x()), int(draw_start_point.y())), p2_2d)

    def _draw_arm_skeleton(self, painter):
        """Draws the arm skeleton based on predicted landmarks."""
        if not self.landmarks:
            # Display placeholder text if no landmark data is available
            painter.setPen(QColor(200, 200, 200))
            painter.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for landmark data...")
            return

        # --- Define connections (bones) based on landmark BASE names ---
        # **IMPORTANT**: Adjust these based on the actual landmark names defined
        # in constants.PREDICTED_LANDMARK_NAMES and output by your model.
        connections = [
            # Left Arm (Example: Assuming Myo on Left Arm)
            ("Pose_L_Shoulder", "Pose_L_Elbow"),
            ("Pose_L_Elbow", "Pose_L_Wrist"),
            # Right Hand (Example: If model predicts hand landmarks)
            # Use the appropriate wrist name based on your model output
            ("Hand_R_Wrist", "Hand_R_Index_MCP"),
            ("Hand_R_Wrist", "Hand_R_Pinky_MCP"),
            ("Hand_R_Index_MCP", "Hand_R_Pinky_MCP"), # Across knuckles
            # Add connections for fingers, thumb etc. based on available landmarks
            # e.g., ("Hand_R_Index_MCP", "Hand_R_Index_PIP"), etc.
        ]

        # --- Project available 3D landmarks to 2D screen coordinates ---
        projected_landmarks = {}
        for name, coords_3d in self.landmarks.items():
            # Use the projection function
            point_2d = self.project_3d_to_2d(coords_3d)
            if point_2d: # Store only if projection is valid
                projected_landmarks[name] = point_2d

        # --- Draw Bones (Lines between connected joints) ---
        pen = QPen(self.performance_color, self.arm_line_width) # Use performance color
        pen.setCapStyle(Qt.PenCapStyle.RoundCap) # Rounded line ends
        painter.setPen(pen)

        drawn_connections = set() # Keep track to avoid drawing lines twice
        for p1_name, p2_name in connections:
            # Check if both landmarks for the connection exist and were projected
            if p1_name in projected_landmarks and p2_name in projected_landmarks:
                # Create a unique key for the connection pair (order doesn't matter)
                conn_key = tuple(sorted((p1_name, p2_name)))
                if conn_key not in drawn_connections:
                    p1 = projected_landmarks[p1_name]
                    p2 = projected_landmarks[p2_name]
                    painter.drawLine(p1, p2) # Draw the bone
                    drawn_connections.add(conn_key) # Mark as drawn

        # --- Draw Joints (Circles at landmark positions) ---
        # Use a slightly darker shade of the performance color for joints
        joint_color = self.performance_color.darker(130)
        painter.setPen(Qt.PenStyle.NoPen) # No outline for joints
        painter.setBrush(QBrush(joint_color)) # Fill joints
        for name, point_2d in projected_landmarks.items():
            # Draw ellipse centered at the projected point
            painter.drawEllipse(point_2d, self.joint_radius, self.joint_radius)


# ==============================================================================
# == Main Physiotherapy Exercise Window ==
# ==============================================================================
class EMGControllerMainWindow(QMainWindow):
    """Main application window for physiotherapy exercises."""
    # Signals to control the worker thread
    start_worker_signal = pyqtSignal()
    stop_worker_signal = pyqtSignal()

    def __init__(self):
        """Initializes the main window UI and state."""
        super().__init__()
        self.setWindowTitle("NeuroSyn ReTrain - Physiotherapy")
        self.resize(850, 650) # Adjust size as needed
        self._create_menubar() # Setup File/Help menus

        # --- Initialize State Variables ---
        self.worker = None # Reference to the PredictionWorker instance
        self.worker_thread = QThread() # Thread object for the worker
        self.connect_window = None # Reference to the connection dialog
        self.cal_dlg = None # Reference to the calibration dialog
        self.session_history = self._load_session_history() # Load past performance

        # Exercise state tracking
        self.current_exercise_key = None # Name of the active exercise (e.g., "Wrist Mobility")
        self.current_rep = 0 # Current repetition number
        self.total_reps = DEFAULT_REPETITIONS # Target reps per set
        self.rep_scores = [] # List to store scores for each rep in the current set
        self.is_exercise_active = False # Flag indicating if an exercise set is in progress

        # --- Main Widget and Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        # Horizontal layout to split window into two panes
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10) # Add padding around the edges

        # --- Left Pane: Visualization ---
        left_pane_widget = QFrame()
        left_pane_widget.setObjectName("VisualizationFrame") # Name for specific styling
        # Use QVBoxLayout to hold the visualization widget
        left_layout = QVBoxLayout(left_pane_widget)
        left_layout.setContentsMargins(0,0,0,0) # No internal margins
        # Create and add the refined visualization widget
        self.vis_widget = ArmVisualizationWidget()
        left_layout.addWidget(self.vis_widget, 1) # Allow widget to expand
        # Add left pane to main layout, give it more horizontal space (ratio 2)
        main_layout.addWidget(left_pane_widget, 2)

        # --- Right Pane: Information & Controls ---
        right_pane_widget = QFrame()
        right_pane_widget.setObjectName("ControlFrame") # Name for specific styling
        right_layout = QVBoxLayout(right_pane_widget)
        right_layout.setSpacing(15) # Space between elements
        # Align content towards the top-center
        right_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignCenter)

        # --- UI Elements in Right Pane ---
        # Exercise Icon Display
        self.icon_label = QLabel("Icon") # Placeholder text
        self.icon_label.setFixedSize(150, 150) # Set fixed size for icon area
        self.icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Basic styling for the icon area
        self.icon_label.setStyleSheet("background-color: #282830; border-radius: 10px; color: white;")
        # Timer for animating the icon frames
        self.icon_anim_timer = QTimer(self)
        self.icon_anim_timer.timeout.connect(self._animate_icon)
        self.icon_frames = [] # List to hold loaded QPixmap frames
        self.current_icon_frame = 0 # Index for animation
        right_layout.addWidget(self.icon_label, 0, Qt.AlignmentFlag.AlignCenter)

        # Exercise Name Label
        self.exercise_name_label = QLabel("Exercise: ---")
        self.exercise_name_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 20, QFont.Weight.Bold))
        self.exercise_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.exercise_name_label)

        # Repetition Counter Label
        self.rep_label = QLabel(f"Rep: 0 / {self.total_reps}")
        self.rep_label.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 14))
        self.rep_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_layout.addWidget(self.rep_label)

        # Score Display Area (using QGridLayout for alignment)
        score_layout = QGridLayout()
        score_layout.setColumnStretch(1, 1) # Allow progress bar column to stretch
        # Current Session Score Elements
        self.current_score_label = QLabel("This Session:")
        self.current_score_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 16))
        self.current_score_progress = QProgressBar()
        self.current_score_progress.setRange(0, 100)
        self.current_score_progress.setValue(0)
        self.current_score_progress.setTextVisible(False) # Hide default percentage text
        self.current_score_value = QLabel("0%") # Custom percentage label
        self.current_score_value.setFont(QFont(GARAMOND_BOOK_FAMILY, 16))
        # Add to grid layout
        score_layout.addWidget(self.current_score_label, 0, 0) # Row 0, Col 0
        score_layout.addWidget(self.current_score_progress, 0, 1) # Row 0, Col 1
        score_layout.addWidget(self.current_score_value, 0, 2) # Row 0, Col 2
        # Last Session Score Elements
        self.last_score_label = QLabel("Last Session:")
        self.last_score_label.setFont(QFont(GARAMOND_LIGHT_FAMILY, 14))
        self.last_score_value = QLabel("--- %") # Placeholder
        self.last_score_value.setFont(QFont(GARAMOND_LIGHT_FAMILY, 14))
        # Add to grid layout
        score_layout.addWidget(self.last_score_label, 1, 0) # Row 1, Col 0
        score_layout.addWidget(self.last_score_value, 1, 2) # Row 1, Col 2 (align right)
        # Add the grid layout to the main right pane layout
        right_layout.addLayout(score_layout)

        # Add stretchable space to push controls to the bottom
        right_layout.addStretch(1)

        # --- Control Buttons ---
        # Record Repetition Button
        self.record_rep_button = QPushButton("Record Next Rep")
        self.record_rep_button.setFont(QApplication.font("QPushButton"))
        self.NSEWidgets_emboss(self.record_rep_button) # Apply emboss effect
        self.record_rep_button.clicked.connect(self.record_next_rep) # Connect click signal
        self.record_rep_button.setEnabled(False) # Disabled until an exercise starts
        right_layout.addWidget(self.record_rep_button)

        # Exit Exercise Button
        self.exit_exercise_button = QPushButton("Exit Exercise")
        self.exit_exercise_button.setFont(QApplication.font("QPushButton"))
        self.NSEWidgets_emboss(self.exit_exercise_button)
        self.exit_exercise_button.clicked.connect(self.exit_exercise) # Connect click signal
        self.exit_exercise_button.setEnabled(False) # Disabled until an exercise starts
        right_layout.addWidget(self.exit_exercise_button)

        # Add right pane to main layout, take up less space (ratio 1)
        main_layout.addWidget(right_pane_widget, 1)

        # --- Status Bar ---
        self.status_label = QLabel("Status: Initializing...")
        self.status_label.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 12))
        self.statusBar().addWidget(self.status_label) # Add label to status bar

        # --- Initialize Worker Thread and Connections ---
        self._setup_worker()
        # Trigger the connection process immediately on startup
        self.start_worker_signal.emit()

    def _setup_worker(self):
        """Initializes the predictor, worker, and thread connections."""
        try:
            # **NOTE:** Predictor initialization might need adjustment based on
            #           how the landmark regression model is loaded/configured.
            predictor = NeuroSynPredictor()
        except Exception as e:
            self.handle_error(f"Failed to initialize predictor: {e}")
            return # Stop if predictor fails

        # Create worker and move it to the dedicated thread
        self.worker_thread = QThread()
        self.worker = PredictionWorker(predictor)
        self.worker.moveToThread(self.worker_thread)

        # --- Connect Signals ---
        # Internal signals to start/stop worker from main thread
        self.start_worker_signal.connect(self._start_worker_thread)
        self.stop_worker_signal.connect(self._stop_worker_thread)

        # Connect worker signals to main thread slots
        self.worker_thread.started.connect(self.worker.run) # Start worker's run() method
        self.worker.predicted_landmarks_signal.connect(self.handle_predicted_landmarks)
        self.worker.performance_update_signal.connect(self.update_performance_display)
        self.worker.progress_signal.connect(self.update_emg_strength) # For EMG activity level
        self.worker.error_signal.connect(self.handle_error)
        self.worker.status_signal.connect(self.handle_status)
        self.worker.finished_signal.connect(self._on_worker_finished) # Worker cleanup done

        # Connect thread finished signal for final cleanup
        self.worker_thread.finished.connect(self._on_thread_finished)

    def _start_worker_thread(self):
        """Starts the worker thread if not already running."""
        if self.worker_thread and not self.worker_thread.isRunning():
            self.status_label.setText("Status: Connecting...")
            self.worker_thread.start()
            # Show connection window as a modal dialog
            if self.connect_window:
                # Ensure previous instance is closed before creating new one
                self.connect_window.stop_animation_and_close()
            # Pass 'self' as parent so it centers on main window
            self.connect_window = ConnectionWindow(self)
            # Show the dialog (it's modal, so it stays on top)
            self.connect_window.show()
        else:
            print("Warning: Worker thread already running or not initialized.")

    def _stop_worker_thread(self):
        """Requests the worker thread to stop gracefully."""
        if self.worker:
            self.worker.stop() # Signal worker's asyncio loop/task to stop
            self.status_label.setText("Status: Stopping...")
        elif self.worker_thread and self.worker_thread.isRunning():
            # Fallback if worker object somehow got deleted but thread is running
            self.worker_thread.quit() # Ask thread to exit event loop
            self.status_label.setText("Status: Quitting thread...")
        else:
            # If nothing was running, just reset UI
            self._reset_ui_after_stop()

    def _load_session_history(self):
        """Loads previous session scores from the JSON file."""
        os.makedirs(DATA_PATH_BASE, exist_ok=True) # Ensure data directory exists
        try:
            if os.path.exists(SESSION_HISTORY_FILE):
                with open(SESSION_HISTORY_FILE, 'r') as f:
                    return json.load(f)
            else:
                print("Session history file not found. Starting fresh.")
                return {} # Return empty dict if file doesn't exist yet
        except json.JSONDecodeError:
            # Handle case where file exists but is corrupted/empty
            print(f"Error: Could not decode JSON from {SESSION_HISTORY_FILE}. Starting fresh.")
            # Optionally backup corrupted file here
            return {}
        except Exception as e:
            # Catch other potential file reading errors
            print(f"Error loading session history: {e}")
            QMessageBox.warning(self, "History Error", f"Could not load session history:\n{e}")
            return {}

    def _save_session_history(self):
        """Saves the current session history dictionary to the JSON file."""
        try:
            with open(SESSION_HISTORY_FILE, 'w') as f:
                # Use indent for readability
                json.dump(self.session_history, f, indent=4)
        except Exception as e:
            print(f"Error saving session history: {e}")
            QMessageBox.warning(self, "History Error", f"Could not save session history:\n{e}")

    def start_exercise(self, exercise_key):
        """Initiates a specific exercise session."""
        # Check prerequisites
        if not self.worker or not self.worker_thread.isRunning():
             QMessageBox.warning(self, "Cannot Start", "Worker thread not running. Please restart.")
             return
        if self.is_exercise_active:
            QMessageBox.warning(self, "Cannot Start", "Please finish the current exercise first.")
            return
        if exercise_key not in EXERCISE_PAIRS:
             QMessageBox.warning(self, "Error", f"Exercise '{exercise_key}' not defined in constants.")
             return

        print(f"Starting exercise: {exercise_key}")
        # Set exercise state
        self.current_exercise_key = exercise_key
        self.current_rep = 0
        self.rep_scores = [] # Clear scores from previous exercise
        self.is_exercise_active = True

        # Update UI elements
        self.exercise_name_label.setText(f"Exercise: {exercise_key}")
        self.update_rep_label()
        self.record_rep_button.setEnabled(True)
        self.exit_exercise_button.setEnabled(True)
        # Reset current session progress display
        self.current_score_progress.setValue(0)
        self.current_score_value.setText("0%")

        # Load and display last session score from history
        last_score = self.session_history.get(exercise_key, {}).get("last_avg_score", "---")
        self.last_score_value.setText(f"{last_score} %" if isinstance(last_score, (int, float)) else "--- %")

        # Load and start icon animation for the selected exercise
        self._load_exercise_icons(exercise_key)
        if self.icon_frames:
            self.icon_anim_timer.start(200) # Animation speed in ms (e.g., 5 FPS)

        # TODO: Send signal to worker/predictor if it needs context about the active exercise
        #       (e.g., which landmarks are most important for this exercise)
        self.status_label.setText(f"Status: Started {exercise_key}. Press 'Record Next Rep'.")

    def record_next_rep(self):
        """Handles the 'Record Next Rep' button click: Analyzes rep, updates state."""
        if not self.is_exercise_active: return

        # --- TODO: Implement Actual Rep Analysis ---
        # 1. Get the relevant landmark data collected during this rep.
        #    (Requires buffering landmarks between 'Record Rep' clicks).
        # 2. Call analysis functions (e.g., calculate_rom, check_form) using buffered landmarks.
        # 3. Calculate the performance score (0-100) for this single repetition.
        # 4. Store the score.
        # 5. Update average session score.
        # 6. Update UI progress/color.
        # 7. Clear the landmark buffer for the next rep.

        # --- Placeholder Logic ---
        print(f"Recording Rep {self.current_rep + 1} for {self.current_exercise_key}")
        # Simulate score calculation for this rep (replace with actual analysis)
        current_rep_score = random.randint(50, 95)
        self.rep_scores.append(current_rep_score)
        # Calculate average score for the session so far
        avg_score = sum(self.rep_scores) / len(self.rep_scores) if self.rep_scores else 0

        # Update display using the dedicated slot
        self.update_performance_display(current_rep_score, avg_score)

        # Increment repetition counter
        self.current_rep += 1
        self.update_rep_label()

        # Check if all reps for this exercise set are completed
        if self.current_rep >= self.total_reps:
            self.finish_exercise_set() # Trigger end-of-set logic
        else:
             # TODO: Provide specific guidance for the next rep phase (e.g., "Now Flex Wrist")
             self.status_label.setText(f"Status: Rep {self.current_rep} done. Ready for Rep {self.current_rep + 1}.")
             # Optional: Briefly disable button to prevent accidental double clicks?
             # self.record_rep_button.setEnabled(False)
             # QTimer.singleShot(500, lambda: self.record_rep_button.setEnabled(True))

    def finish_exercise_set(self):
        """Called when all reps for the current exercise are done."""
        if not self.is_exercise_active: return # Avoid finishing twice

        self.is_exercise_active = False # Mark exercise as inactive
        self.record_rep_button.setEnabled(False) # Disable rep button
        self.icon_anim_timer.stop() # Stop icon animation

        # Calculate final average score for the set
        avg_score = round(sum(self.rep_scores) / len(self.rep_scores)) if self.rep_scores else 0
        self.status_label.setText(f"Status: {self.current_exercise_key} Complete! Avg Score: {avg_score}%")
        print(f"Finished exercise set: {self.current_exercise_key}, Avg Score: {avg_score}%")

        # Save score to history JSON
        if self.current_exercise_key: # Ensure we have a valid key
            if self.current_exercise_key not in self.session_history:
                self.session_history[self.current_exercise_key] = {}
            # Store average score and timestamp
            self.session_history[self.current_exercise_key]['last_avg_score'] = avg_score
            self.session_history[self.current_exercise_key]['last_session_date'] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_session_history() # Write updated history to file

        # TODO: Improve post-exercise flow (e.g., show summary, ask for next action)
        QMessageBox.information(self, "Exercise Complete",
                                f"You finished {self.current_exercise_key}!\nAverage Score: {avg_score}%")
        # For now, just reset the UI, ready for a potential next exercise selection
        self.reset_exercise_ui()
        self.status_label.setText("Status: Ready for next exercise.")


    def exit_exercise(self):
        """Handles the 'Exit Exercise' button click, confirming if progress is lost."""
        if not self.is_exercise_active:
            # If nothing is active, maybe this button should do something else?
            # For now, just ensure UI is reset.
            self.reset_exercise_ui()
            self.status_label.setText("Status: Ready.")
            return

        # Check if minimum work done (e.g., at least 1 rep completed)
        if self.current_rep < 1: # Or check time elapsed < threshold
            # Ask user to confirm cancellation without saving
            reply = QMessageBox.question(self, "Cancel Exercise?",
                                         "Are you sure you want to exit this exercise early?\nProgress for this set will not be saved.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No) # Default No
            if reply == QMessageBox.StandardButton.Yes:
                self.is_exercise_active = False
                self.icon_anim_timer.stop()
                self.reset_exercise_ui()
                self.status_label.setText("Status: Exercise cancelled.")
        else:
            # If some reps done, ask to finish and save progress
            reply = QMessageBox.question(self, "Finish Exercise?",
                                         "Do you want to finish this exercise session now?\nYour progress will be saved.",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.Yes) # Default Yes
            if reply == QMessageBox.StandardButton.Yes:
                 # Call the normal finish routine
                 self.finish_exercise_set()
            # else: User clicked No, do nothing, continue exercise

    def reset_exercise_ui(self):
        """Resets the UI elements related to the exercise to their default state."""
        self.exercise_name_label.setText("Exercise: ---")
        self.rep_label.setText(f"Rep: 0 / {self.total_reps}")
        self.current_score_progress.setValue(0)
        self.current_score_value.setText("0%")
        self.last_score_value.setText("--- %")
        self.icon_label.setText("Icon") # Reset icon placeholder text
        self.icon_label.setPixmap(QPixmap()) # Clear any existing pixmap
        self.record_rep_button.setEnabled(False)
        self.exit_exercise_button.setEnabled(False)
        # Reset visualization state
        self.vis_widget.set_performance_color(0) # Reset color to default (grey)
        self.vis_widget.set_landmarks({}) # Clear landmarks
        # Reset state flags
        self.is_exercise_active = False
        self.current_exercise_key = None
        self.rep_scores = [] # Clear score buffer
        # Don't reset status label here, let calling function set appropriate status

    def update_rep_label(self):
        """Updates the repetition counter label (e.g., "Rep: 1 / 3")."""
        self.rep_label.setText(f"Rep: {self.current_rep} / {self.total_reps}")

    # --- Signal Handlers ---
    @pyqtSlot(dict)
    def handle_predicted_landmarks(self, landmarks):
        """Receives predicted landmarks from the worker and updates visualization."""
        # Pass landmarks dictionary {base_name: np.array([x,y,z])} to the visualization widget
        self.vis_widget.set_landmarks(landmarks)

        # --- TODO: Trigger real-time kinematic analysis based on landmarks ---
        if self.is_exercise_active:
            # This is where the continuous analysis during a rep would happen
            # score_this_moment = self.calculate_instantaneous_performance(landmarks, self.current_exercise_key)
            # self.vis_widget.set_performance_color(score_this_moment) # Update color frequently?

            # Buffering for end-of-rep analysis would also happen here
            # self.buffer_landmarks_for_rep(landmarks)
            pass # Placeholder for analysis logic

    @pyqtSlot(int, float)
    def update_performance_display(self, current_rep_score, avg_session_score):
        """Updates the progress bar and labels for scores (typically called after a rep)."""
        # Clamp scores to 0-100 range for progress bar
        avg_session_score_clamped = max(0, min(100, avg_session_score))
        # Update the progress bar and text display for the average session score
        self.current_score_progress.setValue(int(avg_session_score_clamped))
        self.current_score_value.setText(f"{int(avg_session_score)}%") # Show actual average
        # Update visualization color based on the average score for the session
        self.vis_widget.set_performance_color(int(avg_session_score_clamped))

    @pyqtSlot(int)
    def update_emg_strength(self, value):
        """Placeholder slot for EMG strength signal."""
        # Could potentially modulate grid brightness or add another visual indicator
        # print(f"EMG Strength (placeholder): {value}%")
        pass

    @pyqtSlot(str)
    def handle_status(self, text: str):
        """Handles status messages from the worker and updates the status bar/dialogs."""
        self.status_label.setText(f"Status: {text}") # Update status bar text

        # Close connection window automatically when connected/configured
        if "Myo connected." in text or "Myo configured." in text:
            if self.connect_window and self.connect_window.isVisible():
                # Use the close method for the dialog
                self.connect_window.stop_animation_and_close()
                self.connect_window = None # Clear reference

        # Show calibration dialog when worker signals readiness
        if text == "READY_FOR_CAL":
            if self.connect_window and self.connect_window.isVisible(): # Ensure connection window is closed
                 self.connect_window.stop_animation_and_close(); self.connect_window = None
            if self.cal_dlg and self.cal_dlg.isVisible():
                 # This shouldn't happen if previous instance was closed properly
                 print("Warning: Calibration dialog already exists.")
                 self.cal_dlg.close()
            # Create and show the calibration dialog
            self.cal_dlg = CalibrationDialog(self) # Pass parent
            # Connect signals from the dialog back to the main window
            self.cal_dlg.proceed_clicked.connect(self._relay_start_cal_worker)
            self.cal_dlg.cancel_clicked.connect(self._relay_cancel_cal) # Connected to confirmation logic
            # Use show() for QMainWindow, not exec_()
            self.cal_dlg.show()
            # Optionally disable main window while cal dialog is shown
            # self.setEnabled(False) # Re-enable when dialog closes

        # Handle calibration cues (e.g., 'Rest' prompt) from worker
        if text.startswith("CUE|"):
            # Ensure calibration dialog exists and is ready
            if self.cal_dlg and self.cal_dlg.isVisible() and not self.cal_dlg._cancelled and not self.cal_dlg._finished and self.cal_dlg._ok_proceeded:
                try:
                    # Parse the cue message (assuming format CUE|ID|Name|Seconds)
                    _, cid_str, gname, secs_str = text.split("|")
                    cid = int(cid_str); secs = int(secs_str)
                    # Get the corresponding icon filename from constants
                    icon_filename = ICON_PATHS.get(cid) # Assumes ICON_PATHS maps class IDs
                    if icon_filename:
                        # Tell the dialog to show the gesture prompt
                        self.cal_dlg.show_gesture(gname, secs, icon_filename)
                    else:
                        print(f"Error: Icon filename not found for class ID {cid}")
                except Exception as e:
                    print(f"Error parsing CUE signal '{text}': {e}")

        # Handle calibration completion signal from worker
        if text == "CAL_DONE":
             if self.cal_dlg and self.cal_dlg.isVisible():
                 # Tell the dialog it's finished (updates UI, starts auto-close timer)
                 self.cal_dlg.mark_as_finished()
             # Re-enable main window if it was disabled
             # self.setEnabled(True)

        # Handle signal indicating readiness for exercises
        if text == "READY_FOR_EXERCISE":
             # This status might arrive *after* cal_dlg has closed
             # TODO: Implement exercise selection UI or logic
             # For now, automatically start the first defined exercise as a test
             if EXERCISE_PAIRS:
                 first_exercise = list(EXERCISE_PAIRS.keys())[0]
                 # Start exercise directly
                 self.start_exercise(first_exercise)
             else:
                 self.status_label.setText("Status: Ready, but no exercises defined!")
                 QMessageBox.warning(self, "No Exercises", "Calibration complete, but no exercises are defined in constants.py.")


    @pyqtSlot(str)
    def handle_error(self, error_text):
        """Handles errors signalled from the worker."""
        print(f"ERROR received: {error_text}")
        self.status_label.setText(f"Status: Error!")
        # Display critical error message to the user
        QMessageBox.critical(self, "Process Error", f"An error occurred:\n{error_text}\n\nPlease check connection/logs and restart.")
        # Attempt graceful stop of the worker thread
        self._stop_worker_thread()
        # Force close any helper windows that might be open
        if self.connect_window and self.connect_window.isVisible():
            self.connect_window.stop_animation_and_close()
            self.connect_window = None
        if self.cal_dlg and self.cal_dlg.isVisible():
            self.cal_dlg.close() # Close directly
            self.cal_dlg = None
        # Reset the main exercise UI to a safe state
        self.reset_exercise_ui()

    # --- Slots to relay signals from dialogs to worker ---
    @pyqtSlot()
    def _relay_start_cal_worker(self):
         """Relays the 'Start' click from CalibrationDialog to the worker."""
         if self.worker:
              print("Relaying start signal to worker...")
              # Call the worker's method to proceed
              self.worker.proceed_with_calibration()
         else:
              print("Error: Cannot start calibration, worker not available.")
              QMessageBox.critical(self, "Error", "Worker not available to start calibration.")

    @pyqtSlot()
    def _relay_cancel_cal(self):
        """Handles confirmed cancel signal from CalibrationDialog."""
        print("Calibration setup cancelled by user confirmation.")
        # Stop the worker thread as calibration is mandatory
        self._stop_worker_thread()
        # Optionally close the main window or go back to welcome screen?
        # For now, just stop the worker. User might need to restart.
        self.status_label.setText("Status: Setup Cancelled. Please restart.")
        # Re-enable main window if it was disabled
        # self.setEnabled(True)


    # --- Slots handling worker/thread finishing ---
    @pyqtSlot()
    def _on_worker_finished(self):
        """Slot called when the worker's run() method signals it has finished."""
        print("Worker finished signal received.")
        # Request the QThread to quit if it's still running
        if self.worker_thread and self.worker_thread.isRunning():
             self.worker_thread.quit()
        # Reset UI elements to stopped state
        self._reset_ui_after_stop()
        self.worker = None # Clear worker reference, worker object is likely deleted now

    @pyqtSlot()
    def _on_thread_finished(self):
        """Slot called when the QThread itself finishes execution."""
        print("Worker QThread finished.")
        # Final UI reset to ensure everything is in a safe state
        self._reset_ui_after_stop()

    def _reset_ui_after_stop(self):
        """Resets UI elements after worker stops or fails."""
        print("Resetting UI after stop.")
        self.reset_exercise_ui() # Reset exercise specific UI elements
        self.status_label.setText("Status: Stopped. Please restart.") # Indicate need for restart
        # Close helper windows safely
        if self.connect_window and self.connect_window.isVisible():
             self.connect_window.stop_animation_and_close()
             self.connect_window = None
        if self.cal_dlg and self.cal_dlg.isVisible():
             self.cal_dlg.close() # Close directly
             self.cal_dlg = None
        # Disable controls that require a running worker
        self.record_rep_button.setEnabled(False)
        self.exit_exercise_button.setEnabled(False)

    def closeEvent(self, event):
        """Ensures worker thread is stopped and history saved on main window close."""
        print("Main window closing...")
        self._stop_worker_thread() # Request worker stop
        # Wait for the thread to actually finish
        if self.worker_thread and self.worker_thread.isRunning():
             print("Waiting briefly for worker thread exit before closing window...")
             if not self.worker_thread.wait(2000): # Wait up to 2 seconds
                  print("Worker thread did not exit gracefully, terminating.")
                  self.worker_thread.terminate() # Force termination if needed
                  self.worker_thread.wait() # Wait after terminate
        self._save_session_history() # Save session data on exit
        super().closeEvent(event) # Proceed with closing the window

    # --- Icon Animation ---
    def _load_exercise_icons(self, exercise_key):
        """Loads the icon frames for the given exercise from constants."""
        self.icon_frames = []
        self.current_icon_frame = 0
        # Get the list of filenames for the exercise from constants
        icon_filenames = EXERCISE_ICONS.get(exercise_key, [])
        if not icon_filenames:
            print(f"Warning: No icon frames defined for exercise '{exercise_key}' in constants.EXERCISE_ICONS")
            self.icon_label.setText("No Icon") # Display placeholder text
            self.icon_label.setPixmap(QPixmap()) # Clear any previous image
            return

        # Construct full path assuming icons are in NSEmedia/gestures/
        gestures_dir = os.path.join(MEDIA_PATH, "gestures")
        for fname in icon_filenames:
            full_path = os.path.join(gestures_dir, fname)
            if os.path.exists(full_path):
                pixmap = QPixmap(full_path)
                if not pixmap.isNull():
                    # Scale icons to fit the label while maintaining aspect ratio
                    # Use self.icon_label.size() for target size
                    scaled_pixmap = pixmap.scaled(self.icon_label.size(),
                                                  Qt.AspectRatioMode.KeepAspectRatio,
                                                  Qt.TransformationMode.SmoothTransformation)
                    self.icon_frames.append(scaled_pixmap)
                else:
                    print(f"Warning: Could not load icon image (Pixmap is null): {full_path}")
            else:
                print(f"Warning: Icon file not found: {full_path}")

        # Set the initial frame or placeholder text if loading failed
        if not self.icon_frames:
             self.icon_label.setText("Load Error")
             self.icon_label.setPixmap(QPixmap())
        else:
            self.icon_label.setText("") # Clear placeholder text
            self.icon_label.setPixmap(self.icon_frames[0]) # Show the first frame

    def _animate_icon(self):
        """Cycles through the loaded icon frames."""
        if not self.icon_frames: return # Do nothing if no frames loaded
        # Increment frame index and loop back to 0
        self.current_icon_frame = (self.current_icon_frame + 1) % len(self.icon_frames)
        # Set the pixmap of the label to the current frame
        self.icon_label.setPixmap(self.icon_frames[self.current_icon_frame])

    # --- Utility and Menubar ---
    def NSEWidgets_emboss(self, btn: QPushButton):
        """Applies a subtle emboss effect to a button."""
        effect = QGraphicsDropShadowEffect(btn)
        effect.setBlurRadius(1)
        effect.setOffset(0, 1) # Offset shadow slightly down
        effect.setColor(QColor(255, 255, 255, 80)) # Semi-transparent white shadow
        btn.setGraphicsEffect(effect)

    def _create_menubar(self):
        """Creates the main menu bar."""
        menubar = self.menuBar()
        # --- File Menu ---
        file_menu = menubar.addMenu("File")
        # Example action - TODO: Implement exercise selection dialog
        select_action = QAction("Select Exercise...", self)
        # select_action.triggered.connect(self._show_exercise_selection)
        select_action.setEnabled(False) # Disable until implemented
        file_menu.addAction(select_action)
        file_menu.addSeparator()
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close) # Connect to window's close method
        file_menu.addAction(exit_action)

        # --- Help Menu ---
        help_menu = menubar.addMenu("Help")
        about_action = QAction("About NeuroSyn ReTrain", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole) # Standard role for About action
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    def show_about_dialog(self):
        """Displays the About dialog box."""
        QMessageBox.about(self, "About NeuroSyn ReTrain",
            "NeuroSyn ReTrain\n"
            "Version 1.1 (Physio - Landmark Prediction)\n\n" # Updated version note
            "Developed by Syndromatic Inc. / Kavish Krishnakumar (2025)\n\n"
            "AI-powered home physiotherapy assistant using surface EMG+IMU\n"
            "with real-time landmark prediction and feedback."
        )


# ==============================================================================
# == Welcome Window (Fully Included) ==
# ==============================================================================
class WelcomeWindow(QMainWindow):
    """Initial welcome screen for the application."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to NeuroSyn ReTrain")
        self.resize(600, 500) # Default size

        # --- Central Widget and Layout ---
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center content vertically

        # --- Application Icon ---
        icon_label = QLabel()
        # Load icon from media path
        pixmap = QPixmap(os.path.join(MEDIA_PATH, "syn app light.png"))
        # Scale icon nicely
        scaled_pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Center icon horizontally
        layout.addWidget(icon_label)

        # --- Welcome Title ---
        welcome_label = QLabel("Welcome")
        # Use specific Garamond font loaded earlier
        welcome_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold))
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)

        # Calculate width of title for button sizing
        fm = QFontMetrics(welcome_label.font())
        # Use horizontalAdvance for accurate width based on text and font
        title_width = fm.horizontalAdvance(welcome_label.text()) + 40 # Add padding

        # --- Introduction Text ---
        intro_text = QLabel(
            "Welcome to NeuroSyn ReTrain,\n"
            "Your AI-powered home physiotherapy assistant.\n"
            "We'll use your EMG armband to guide you through exercises.\n\n"
            "Please ensure your Myo armband is charged and worn correctly,\n"
            "then click 'Get Started' to connect."
        )
        intro_text.setFont(QFont(GARAMOND_BOOK_FAMILY, 14)) # Use Garamond Book
        intro_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        intro_text.setWordWrap(True) # Ensure text wraps nicely
        layout.addWidget(intro_text)

        # --- Get Started Button ---
        self.next_button = QPushButton("Get Started")
        self.next_button.setFont(QApplication.font("QPushButton")) # Use default button font
        # Set fixed width based on title width for visual balance
        self.next_button.setFixedWidth(title_width)
        # Ensure button doesn't stretch vertically if window resizes
        self.next_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        # Apply subtle emboss effect
        self.NSEWidgets_emboss(self.next_button)
        # Connect button click to proceed to the main window
        self.next_button.clicked.connect(self.gotoMainWindow)
        # Add button centered horizontally, with some space above/below
        layout.addWidget(self.next_button, 0, Qt.AlignmentFlag.AlignCenter) # Add with stretch factor 0
        layout.addStretch(1) # Add stretchable space below button

        # --- Copyright Footer ---
        copyright_label = QLabel("Copyright © 2025 Syndromatic Inc. / Kavish Krishnakumar")
        copyright_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 10)) # Smaller font for copyright
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)

    def NSEWidgets_emboss(self, btn: QPushButton):
        """Applies a subtle emboss effect to a button."""
        effect = QGraphicsDropShadowEffect(btn)
        effect.setBlurRadius(1)
        effect.setOffset(0, 1) # Offset shadow slightly down
        effect.setColor(QColor(255, 255, 255, 80)) # Semi-transparent white shadow
        btn.setGraphicsEffect(effect)

    def gotoMainWindow(self):
        """Closes the welcome window and opens the main application window."""
        # Create instance of the main window
        # Use self.main_window to keep a reference, preventing garbage collection
        self.main_window = EMGControllerMainWindow()
        # Show the main window
        self.main_window.show()
        # Close this welcome window
        self.close()

# ==============================================================================
# == Main Execution (Styling Updated) ==
# ==============================================================================
def main():
    """Main function to set up and run the PyQt application."""
    app = QApplication(sys.argv)

    # --- FONT LOADING ---
    global GARAMOND_LIGHT_FAMILY, GARAMOND_LIGHT_ITALIC_FAMILY, GARAMOND_BOOK_FAMILY, DOTMATRIX_FAMILY
    # Load fonts and store family names globally
    light_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-LtCond.ttf"))
    GARAMOND_LIGHT_FAMILY = QFontDatabase.applicationFontFamilies(light_id)[0] if light_id != -1 else "ITCGaramondStd Light Condensed"
    light_italic_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-LtCondIta.ttf"))
    GARAMOND_LIGHT_ITALIC_FAMILY = QFontDatabase.applicationFontFamilies(light_italic_id)[0] if light_italic_id != -1 else "ITCGaramondStd Light Condensed Italic"
    book_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-BkCond.ttf"))
    GARAMOND_BOOK_FAMILY = QFontDatabase.applicationFontFamilies(book_id)[0] if book_id != -1 else "ITCGaramondStd Book Condensed"
    dotmatrix_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "DOTMATRI.ttf"))
    DOTMATRIX_FAMILY = QFontDatabase.applicationFontFamilies(dotmatrix_id)[0] if dotmatrix_id != -1 else "DotMatrix"
    print("Loaded Fonts:", GARAMOND_LIGHT_FAMILY, "|", GARAMOND_LIGHT_ITALIC_FAMILY, "|", GARAMOND_BOOK_FAMILY, "|", DOTMATRIX_FAMILY)

    # --- STYLING (Fusion style with custom palette and QSS) ---
    # Determine if system theme is dark or light to apply appropriate palette/QSS
    wincol = app.palette().color(QPalette.ColorRole.Window)
    is_dark = wincol.lightness() < 128 # Simple check based on window background lightness
    app.setStyle(QStyleFactory.create("Fusion")) # Use Fusion style for consistency
    pal = QPalette()
    # Define light and dark palettes
    if not is_dark: # Light theme colors
        pal.setColor(QPalette.ColorRole.Window,QColor(240,240,240)); pal.setColor(QPalette.ColorRole.WindowText,QColor(33,33,33)); pal.setColor(QPalette.ColorRole.Base,QColor(255,255,255)); pal.setColor(QPalette.ColorRole.AlternateBase,QColor(240,240,240)); pal.setColor(QPalette.ColorRole.ToolTipBase,QColor(255,255,220)); pal.setColor(QPalette.ColorRole.ToolTipText,QColor(0,0,0)); pal.setColor(QPalette.ColorRole.Text,QColor(33,33,33)); pal.setColor(QPalette.ColorRole.Button,QColor(240,240,240)); pal.setColor(QPalette.ColorRole.ButtonText,QColor(33,33,33)); pal.setColor(QPalette.ColorRole.BrightText,QColor(255,0,0)); pal.setColor(QPalette.ColorRole.Highlight,QColor(76,163,224)); pal.setColor(QPalette.ColorRole.HighlightedText,QColor(255,255,255))
    else: # Dark theme colors
        pal.setColor(QPalette.ColorRole.Window,QColor(45,45,45)); pal.setColor(QPalette.ColorRole.WindowText,QColor(220,220,220)); pal.setColor(QPalette.ColorRole.Base,QColor(30,30,30)); pal.setColor(QPalette.ColorRole.AlternateBase,QColor(45,45,45)); pal.setColor(QPalette.ColorRole.ToolTipBase,QColor(255,255,220)); pal.setColor(QPalette.ColorRole.ToolTipText,QColor(0,0,0)); pal.setColor(QPalette.ColorRole.Text,QColor(220,220,220)); pal.setColor(QPalette.ColorRole.Button,QColor(60,60,60)); pal.setColor(QPalette.ColorRole.ButtonText,QColor(220,220,220)); pal.setColor(QPalette.ColorRole.BrightText,QColor(255,0,0)); pal.setColor(QPalette.ColorRole.Highlight,QColor(76,163,224)); pal.setColor(QPalette.ColorRole.HighlightedText,QColor(255,255,255))
    app.setPalette(pal) # Apply the chosen palette

    # --- Define QSS Stylesheet ---
    # Button styles (light/dark variants)
    light_btn_qss="""QPushButton{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(250,250,250,255), stop:1 rgba(230,230,230,255));color:#202020;border:1px solid #c0c0c0;border-radius:6px;padding:6px 14px;}QPushButton:hover{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(240,240,240,255), stop:1 rgba(215,215,215,255));}QPushButton:pressed{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(220,220,220,255), stop:1 rgba(195,195,195,255));}QPushButton:disabled{background:rgba(245,245,245,255);color:rgba(160,160,160,255);border:1px solid rgba(200,200,200,255);}"""
    dark_btn_qss="""QPushButton{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(85,85,85,255), stop:1 rgba(51,51,51,255));color:#eeeeee;border:1px solid #202020;border-radius:6px;padding:6px 14px;}QPushButton:hover{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(100,100,100,255), stop:1 rgba(80,80,80,255));}QPushButton:pressed{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(70,70,70,255), stop:1 rgba(40,40,40,255));}QPushButton:disabled{background:rgba(70,70,70,255);color:rgba(120,120,120,255);border:1px solid rgba(30,30,30,255);}"""
    # Shared styles (Progress bar, removing unwanted borders)
    shared_qss="""
        QProgressBar {
            border: 1px solid #707070;
            border-radius: 5px;
            text-align: center;
            background: #444; /* Dark background for the trough */
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 rgba(76,163,224,255), stop:1 rgba(33,123,200,255)); /* Blue gradient */
            border-radius: 5px; /* Match progress bar radius */
            margin: 1px; /* Small margin for visual separation */
        }
        QTextEdit { /* Style for log area if used */
            border: 1px solid #505050;
            border-radius: 4px;
        }
        /* Remove default borders from common container/display widgets */
        QFrame, QLabel, QWidget, QDialog, QMainWindow {
            border: none;
            background-color: transparent; /* Ensure transparency unless overridden */
        }
        /* Ensure main window and dialogs use palette background */
        QMainWindow, QDialog {
             background-color: palette(window);
        }
        /* Style specific frames if needed using object names */
        /*
        QFrame#VisualizationFrame { border: 1px solid #333; border-radius: 5px; }
        QFrame#ControlFrame { border: 1px solid #333; border-radius: 5px; }
        */
    """
    # Combine button style with shared styles
    final_qss = (dark_btn_qss if is_dark else light_btn_qss) + shared_qss
    app.setStyleSheet(final_qss) # Apply the stylesheet to the application

    # Set application info (used by OS for window management etc.)
    app.setOrganizationName("Syndromatic Inc.")
    app.setApplicationName("NeuroSyn Retrain")
    app.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "myo_icon.png")))

    # --- Start Application ---
    # Create and show the welcome window first
    welcome = WelcomeWindow()
    welcome.show()
    # Start the Qt event loop (this runs until the application exits)
    sys.exit(app.exec())

if __name__ == "__main__":
    # --- Startup Checks ---
    # Ensure constants file exists and contains necessary definitions before starting GUI
    try:
        import constants
        # Access essential constants to trigger AttributeError if missing
        _ = constants.MYO_ADDRESS
        _ = constants.EXERCISE_PAIRS
        _ = constants.EXERCISE_ICONS
        _ = constants.DEFAULT_REPETITIONS
        _ = constants.PREDICTED_LANDMARK_NAMES # Check if landmark names are defined
    except ImportError:
        print("ERROR: constants.py not found or contains errors.")
        # Show message box even if app hasn't fully started
        temp_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Startup Error", "constants.py not found.\nPlease ensure it exists in the same directory.")
        sys.exit(1)
    except AttributeError as e:
        # Catch missing constants specifically
        print(f"ERROR: Missing required constant in constants.py: {e}")
        temp_app = QApplication.instance() or QApplication(sys.argv)
        QMessageBox.critical(None, "Startup Error", f"Missing required constant in constants.py:\n{e}\nPlease check your constants file.")
        sys.exit(1)

    # Run the main application function
    main()
