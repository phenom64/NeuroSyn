#!/usr/bin/env python3
"""
NeuroSyn ReTrain – Syndromatic Home Physio GUI
Created by Syndromatic Inc. / Kavish Krishnakumar (2025)
---------------------------------------------------------------------------
v5: Fixes Path import, scanlines/icon paths, restores all welcome/about text,
    fixes calibration auto-start, restores cancelled screen, improves stop stability,
    restores connection animation visibility.
"""

import os
import sys
import time
import asyncio
import math
import platform
import traceback
from pathlib import Path # <<< FIXED: Added Import >>>

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton, QStyleFactory,
    QProgressBar, QTextEdit, QSizePolicy, QHBoxLayout, QMessageBox, QGraphicsOpacityEffect, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QPixmap, QAction, QIcon, QFont, QFontDatabase, QColor, QFontMetrics, QPalette
from PyQt6.QtCore import (
    QThread, pyqtSignal, pyqtSlot, QObject, Qt, QTimer, QEasingCurve, QPropertyAnimation,
    QSequentialAnimationGroup, QParallelAnimationGroup, QPauseAnimation, QRect
)

# Define base path and media path (relative to the script)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MEDIA_PATH = os.path.join(BASE_PATH, "NSEmedia")

# --- Imports for Prediction Logic ---
try:
    from prediction import NeuroSynPredictor, run_calibration_flow
    # <<< Import constants AFTER ensuring script dir is in path if needed >>>
    # sys.path.insert(0, BASE_PATH) # Add script's dir to path if needed
    from constants import MYO_ADDRESS, ICON_PATHS, CLASSES, COLLECTION_TIME
except ImportError as e:
    print(f"ERROR: Failed to import prediction/constants module: {e}")
    print("Ensure prediction.py and constants.py are in the same directory or Python path.")
    # Show a GUI error message if possible before exiting
    app = QApplication.instance() # Get existing instance or create one
    if not app: app = QApplication(sys.argv)
    QMessageBox.critical(None, "Import Error", f"Failed to import required files:\n{e}\n\nPlease check installation and file locations.")
    sys.exit(1)


from pymyo import Myo
from pymyo.types import EmgMode, SleepMode, ImuMode
from bleak import BleakScanner

# --- Patch pymyo (Keep if necessary) ---
import types, struct
def _safe_on_classifier(self, sender, value):
    if len(value) < 3: return
    try: struct.unpack("<B2s", value)
    except struct.error: return
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# --------------------------------------------------------------------

# Global variables for fonts
GARAMOND_LIGHT_FAMILY = "ITCGaramondStd Light Condensed" # Default fallback
GARAMOND_LIGHT_ITALIC_FAMILY = "ITCGaramondStd Light Condensed Italic"
GARAMOND_BOOK_FAMILY = "ITCGaramondStd Book Condensed"
DOTMATRIX_FAMILY = "DotMatrix"

###############################################################################
# Worker for handling Myo data & predictions
###############################################################################
class PredictionWorker(QObject):
    prediction_signal = pyqtSignal(str)
    progress_signal   = pyqtSignal(int)
    error_signal      = pyqtSignal(str)
    status_signal     = pyqtSignal(str)
    finished_signal   = pyqtSignal() # Crucial for cleanup

    def __init__(self, predictor: NeuroSynPredictor):
        super().__init__()
        self.predictor    = predictor
        self._is_running  = True
        self._cal_start_event = asyncio.Event() # GUI signals user clicked Start
        self._stop_event      = asyncio.Event() # GUI signals stop/cancel
        self._myo_task    = None
        self.loop         = None

    async def run_async(self):
        self.loop = asyncio.get_running_loop()
        try:
            self.status_signal.emit(f"Scanning for Myo ({MYO_ADDRESS})...")
            myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS, timeout=20.0)
            if not self._is_running: return
            if not myo_device:
                self.error_signal.emit(f"Could not find Myo device at {MYO_ADDRESS}.")
                return

            self.status_signal.emit("Myo found. Connecting...")
            async with Myo(myo_device) as myo:
                if not self._is_running: return
                self.status_signal.emit("Myo connected. Configuring...")
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
                await myo.set_mode(emg_mode=EmgMode.SMOOTH, imu_mode=ImuMode.DATA)
                self.status_signal.emit("Myo configured.")

                # --- Setup Callbacks ---
                @myo.on_emg_smooth
                def on_emg_smooth_callback(emg_data):
                    if not self._is_running: return
                    if self.predictor.calibration_complete: # Only process post-calibration
                        norm = math.sqrt(sum(e**2 for e in emg_data))
                        self.progress_signal.emit(int(min(100, (norm / 300) * 100)))
                        prediction_result = self.predictor.on_emg(emg_data)
                        if prediction_result is not None:
                            cid, gname, conf = prediction_result
                            pred_text = f"Predicted gesture: {gname} (class {cid}, Conf: {conf:.2f})"
                            self.prediction_signal.emit(pred_text)

                @myo.on_imu
                def on_imu_callback(orientation, accel, gyro):
                     if not self._is_running: return
                     self.predictor.on_imu(orientation, accel, gyro)
                # -----------------------

                # Signal GUI to show Calibration Welcome Screen
                self.status_signal.emit("READY_FOR_CAL")

                # Wait for user to click "Start" in the Calibration Dialog
                self.status_signal.emit("Waiting for user to start calibration guidance...")
                start_cal_task = asyncio.create_task(self._cal_start_event.wait())
                stop_task_1    = asyncio.create_task(self._stop_event.wait())
                done, pending = await asyncio.wait({start_cal_task, stop_task_1}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()

                if stop_task_1 in done:
                    self.status_signal.emit("Stop requested before calibration guidance started.")
                    return

                # User clicked Start, run the calibration flow
                self.status_signal.emit("User started guidance. Running calibration flow...")
                cal_task = asyncio.create_task(run_calibration_flow(self.predictor, cue=self.status_signal.emit))
                stop_task_2 = asyncio.create_task(self._stop_event.wait())
                done, pending = await asyncio.wait({cal_task, stop_task_2}, return_when=asyncio.FIRST_COMPLETED)
                for task in pending: task.cancel()

                if stop_task_2 in done:
                    self.status_signal.emit("Stop requested during calibration guidance.")
                    return

                # Calibration finished normally
                self.status_signal.emit("Calibration flow finished normally.")

                # Main prediction phase - wait until stopped externally
                if self._is_running:
                    self.status_signal.emit("Prediction active. Perform exercises.")
                    await self._stop_event.wait() # Wait for stop signal
                    self.status_signal.emit("Stop event received. Ending Myo loop.")

        except asyncio.CancelledError:
             self.status_signal.emit("Worker task explicitly cancelled.")
        except Exception as e:
            print(f"Error in PredictionWorker run_async: {e}")
            traceback.print_exc()
            if self._is_running: self.error_signal.emit(f"Worker Error: {e}")
        finally:
             self.status_signal.emit("Worker async run finished.")


    def run(self):
        """Entry point when QThread starts."""
        self.status_signal.emit("Prediction worker thread starting.")
        try:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self._is_running = True
            self._stop_event.clear()
            self._cal_start_event.clear()
            self._myo_task = self.loop.create_task(self.run_async())
            self.loop.run_until_complete(self._myo_task)
        except Exception as e:
            print(f"Error during worker run execution: {e}")
            traceback.print_exc()
            if self._is_running: self.error_signal.emit(f"Worker Run Error: {e}")
        finally:
            print("Worker run() method finishing. Cleaning up loop...")
            try:
                if self.loop and self.loop.is_running():
                    print("Stopping running asyncio loop...")
                    # Schedule stop and run briefly to process
                    self.loop.call_soon_threadsafe(self.loop.stop)
                    # self.loop.run_forever() # Let it process the stop
                    time.sleep(0.1) # Short delay might help

                if self.loop:
                    tasks = asyncio.all_tasks(loop=self.loop)
                    if tasks:
                        print(f"Cancelling {len(tasks)} remaining worker tasks...")
                        for task in tasks: task.cancel()
                        async def wait_cancelled(tasks): await asyncio.gather(*tasks, return_exceptions=True)
                        self.loop.run_until_complete(wait_cancelled(tasks))

            except Exception as e: print(f"Error during worker loop cleanup: {e}")
            finally:
                 if self.loop and not self.loop.is_closed():
                     self.loop.close()
                     print("Worker asyncio loop closed.")
                 self.status_signal.emit("Worker loop closed.")
                 self.finished_signal.emit() # Signal GUI thread


    def stop(self):
        """Signals the worker's asyncio loop to stop gracefully."""
        if not self._is_running: print("Worker stop already requested."); return

        self.status_signal.emit("Stop requested for worker.")
        self._is_running = False

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self._cal_start_event.set)
            self.loop.call_soon_threadsafe(self._stop_event.set)
            if self._myo_task and not self._myo_task.done():
                 self.status_signal.emit("Attempting to cancel main worker task.")
                 self.loop.call_soon_threadsafe(self._myo_task.cancel)
        else:
            self._cal_start_event.set()
            self._stop_event.set()
            self.status_signal.emit("Worker loop not running, set stop events directly.")

    @pyqtSlot()
    def proceed_with_calibration(self):
        """Sets the event allowing the calibration flow to start."""
        if self.loop and self.loop.is_running():
             self.status_signal.emit("GUI signalled to proceed with calibration.")
             self.loop.call_soon_threadsafe(self._cal_start_event.set)
        else:
             self.status_signal.emit("Warning: Tried to proceed calibration, but worker loop not running.")


###############################################################################
# Calibration Assistant Dialog (Restores Welcome/Cancel Screens)
###############################################################################
class CalibrationDialog(QMainWindow):
    proceed_clicked = pyqtSignal() # User clicked Start
    cancel_clicked  = pyqtSignal() # User clicked Cancel

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Assistant")
        self.resize(520, 340)
        central = QWidget(self); self.setCentralWidget(central)
        self.vbox = QVBoxLayout(central); self.vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # --- UI Elements ---
        self.logo = QLabel(); logo_pix = QPixmap(os.path.join(MEDIA_PATH, "SetupAssistant.png")); self.logo.setPixmap(logo_pix.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)); self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.head = QLabel("Calibration Assistant"); self.head.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold)); self.head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body = QLabel(); self.body.setFont(QFont(GARAMOND_BOOK_FAMILY, 14)); self.body.setAlignment(Qt.AlignmentFlag.AlignCenter); self.body.setWordWrap(True)
        self.icon = QLabel(); self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_lbl = QLabel(""); self.timer_lbl.setFont(QFont(GARAMOND_BOOK_FAMILY, 22)); self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.btn_layout_widget = QWidget(); self.btn_layout = QHBoxLayout(self.btn_layout_widget); self.btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.ok_btn = QPushButton("Start"); self.ok_btn.clicked.connect(self._on_ok); self.NSEWidgets_emboss(self.ok_btn); self.btn_layout.addWidget(self.ok_btn)
        self.cancel_btn = QPushButton("Cancel"); self.cancel_btn.clicked.connect(self._on_cancel); self.NSEWidgets_emboss(self.cancel_btn); self.btn_layout.addWidget(self.cancel_btn)

        self.vbox.addWidget(self.logo); self.vbox.addWidget(self.head); self.vbox.addWidget(self.body); self.vbox.addWidget(self.icon); self.vbox.addWidget(self.timer_lbl); self.vbox.addWidget(self.btn_layout_widget)
        self.icon.hide(); self.timer_lbl.hide()
        self.show_welcome_screen() # Start in welcome state

        self.countdown = None; self._cancelled = False; self._finished  = False; self._ok_proceeded = False

    def show_welcome_screen(self):
        """Sets up the initial welcome view."""
        self.logo.show(); self.icon.hide(); self.timer_lbl.hide()
        self.head.setText("Calibration Assistant")
        # <<< RESTORED Welcome Text (Matches old code) >>>
        self.body.setText(
            "You need to calibrate your EMG armband.\n"
            "Follow the on-screen instructions and\n"
            "we'll get you up-and-running in no time.\n"
        )
        self.ok_btn.setText("Start"); self.ok_btn.show(); self.ok_btn.setEnabled(True)
        self.cancel_btn.setText("Cancel"); self.cancel_btn.show()
        self._ok_proceeded = False

    def closeEvent(self, event):
        if self._finished: return super().closeEvent(event)
        if not self._cancelled:
            # <<< RESTORED Cancel Confirmation Text (Matches old code) >>>
            reply = QMessageBox.question(
                self, "Cancel Calibration?",
                "Are you sure you want to cancel Calibration?\n\n"
                "Quitting Calibration Assistant cancels the calibration process "
                "and prevents you from using NeuroSyn ReTrain.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes: event.ignore(); self._show_cancelled_screen()
            else: event.ignore()
        else: event.accept()

    @pyqtSlot()
    def _on_ok(self):
        """Handles Start/OK button clicks."""
        if self._cancelled or self._finished: self.close(); return

        if not self._ok_proceeded: # First click (Start)
            self._ok_proceeded = True
            self.ok_btn.setEnabled(False); self.ok_btn.hide()
            self.cancel_btn.setText("Cancel Guidance") # Or just "Cancel"
            self.body.setText("Waiting for 'Rest' instruction from device...")
            self.logo.hide()
            self.proceed_clicked.emit() # Signal main window

    @pyqtSlot()
    def _on_cancel(self):
        # <<< RESTORED Cancel Confirmation Text (Matches old code) >>>
        reply = QMessageBox.question(
            self, "Cancel Calibration?",
            "Are you sure you want to cancel Calibration?\n\n"
            "Quitting Calibration Assistant cancels the calibration process "
            "and prevents you from using NeuroSyn ReTrain.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes: self._show_cancelled_screen()

    def _show_cancelled_screen(self):
        # <<< RESTORED Cancelled Screen Logic (Matches old code) >>>
        if self._cancelled: return
        self._cancelled = True; self._finished = False
        self.logo.show(); self.icon.hide(); self.timer_lbl.hide(); self.cancel_btn.hide()
        self.head.setText("Calibration cancelled")
        self.body.setText(
            "Calibration Assistant was interrupted and calibration has been cancelled.\n\n"
            "Click 'OK' to exit."
        )
        self.ok_btn.setText("OK"); self.ok_btn.setEnabled(True); self.ok_btn.show()
        # Re-wire OK to close
        try: self.ok_btn.clicked.disconnect()
        except TypeError: pass
        self.ok_btn.clicked.connect(self.close)

        if self.countdown: self.countdown.stop()
        self.cancel_clicked.emit() # Notify main window

    def show_gesture(self, title: str, seconds: int, icon_filename: str):
        """Displays the gesture prompt. Called when CUE is received."""
        if self._cancelled or self._finished or not self._ok_proceeded: return

        self.logo.hide(); self.ok_btn.hide()
        self.head.setText(title); self.body.setText(f"Hold for {seconds} s …")

        # <<< Corrected Icon Path Construction >>>
        gestures_dir = os.path.join(MEDIA_PATH, "gestures")
        full_icon_path = os.path.join(gestures_dir, icon_filename)

        if not os.path.exists(full_icon_path):
             print(f"Warning: Icon not found at {full_icon_path}")
             self.icon.hide()
        else:
             pix = QPixmap(full_icon_path)
             if pix.isNull(): print(f"Error: QPixmap is null for icon: {full_icon_path}"); self.icon.hide()
             else: self.icon.setPixmap(pix.scaled(140, 140, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)); self.icon.show()
        self.timer_lbl.show()

        if self.countdown: self.countdown.stop()
        self.time_left = seconds; self.timer_lbl.setText(str(self.time_left))
        self.countdown = QTimer(self); self.countdown.timeout.connect(self._tick); self.countdown.start(1000)

    def NSEWidgets_emboss(self, btn: QPushButton): effect = QGraphicsDropShadowEffect(btn); effect.setBlurRadius(1); effect.setOffset(0, 1); effect.setColor(QColor(255, 255, 255, 80)); btn.setGraphicsEffect(effect)

    def _tick(self):
        if self._cancelled or self._finished:
             if self.countdown: self.countdown.stop(); return
        self.time_left -= 1; self.timer_lbl.setText(str(self.time_left) if self.time_left > 0 else "✓")
        if self.time_left <= 0 and self.countdown: self.countdown.stop()

    def mark_as_finished(self):
        """Update UI when calibration guidance is fully complete."""
        # <<< RESTORED Finished Screen Logic (Matches old code) >>>
        if self._cancelled: return
        self._finished = True
        self.icon.hide(); self.timer_lbl.hide(); self.logo.show()
        self.head.setText("Calibration complete!")
        self.body.setText("You’re ready to go.")
        self.cancel_btn.hide()
        self.ok_btn.hide() # Hide OK button initially
        # Automatically close after a delay (as per old code behavior)
        QTimer.singleShot(1500, self.close)


###############################################################################
# Animated Connection Screen (Restored Animation Start)
###############################################################################
class ConnectionWindow(QMainWindow):
    # (Class code remains mostly the same, just ensure animation starts)
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connecting to Myo...")
        self.resize(600, 200)
        central_widget = QWidget(); self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.computer_label = QLabel(); comp_pix = QPixmap(os.path.join(MEDIA_PATH, "computer_icon.png")); comp_scaled = comp_pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation); self.computer_label.setPixmap(comp_scaled); layout.addWidget(self.computer_label)
        self.orbs_container = QWidget(); self.orbs_container.setFixedSize(300, 80); self.orbs_container.setStyleSheet("background-color: transparent;"); layout.addWidget(self.orbs_container)
        self.myo_label = QLabel(); myo_pix = QPixmap(os.path.join(MEDIA_PATH, "myo_icon.png")); myo_scaled = myo_pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation); self.myo_label.setPixmap(myo_scaled); layout.addWidget(self.myo_label)
        self.orb_labels = []; self.orb_effects = []
        for _ in range(3):
            orb_label = QLabel(self.orbs_container); orb_pix = QPixmap(os.path.join(MEDIA_PATH, "orb.png")).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation); orb_label.setPixmap(orb_pix); orb_label.move(0, 25); effect = QGraphicsOpacityEffect(orb_label); effect.setOpacity(0.0); orb_label.setGraphicsEffect(effect); self.orb_labels.append(orb_label); self.orb_effects.append(effect)
        self.anim_group = QParallelAnimationGroup(self); start_delays = [0, 500, 1000];
        for i, orb_label in enumerate(self.orb_labels):
            seq_anim = self._create_orb_sequence(orb_label, self.orb_effects[i], start_delays[i])
            self.anim_group.addAnimation(seq_anim)
        self.anim_group.setLoopCount(-1)
        # <<< Ensure animation starts >>>
        self.anim_group.start()

    def _create_orb_sequence(self, orb_label: QLabel, opacity_effect: QGraphicsOpacityEffect, start_delay: int):
        # (This complex animation logic remains unchanged)
        orb_seq = QSequentialAnimationGroup();
        if start_delay > 0: orb_seq.addAnimation(QPauseAnimation(start_delay))
        fade_time=300; travel_time=1000; left_rect=QRect(0,25,30,30); right_rect=QRect(self.orbs_container.width()-30,25,30,30)
        fade_in=QPropertyAnimation(opacity_effect,b"opacity");fade_in.setDuration(fade_time);fade_in.setStartValue(0.0);fade_in.setEndValue(1.0); move_in=QPropertyAnimation(orb_label,b"geometry");move_in.setDuration(fade_time);quarter_rect=QRect(int(right_rect.x()*0.25),25,30,30);move_in.setStartValue(left_rect);move_in.setEndValue(quarter_rect); forward_in_group=QParallelAnimationGroup();forward_in_group.addAnimation(fade_in);forward_in_group.addAnimation(move_in);orb_seq.addAnimation(forward_in_group)
        move_to_right=QPropertyAnimation(orb_label,b"geometry");move_to_right.setDuration(travel_time);move_to_right.setStartValue(quarter_rect);move_to_right.setEndValue(right_rect);orb_seq.addAnimation(move_to_right); fade_out_right=QPropertyAnimation(opacity_effect,b"opacity");fade_out_right.setDuration(fade_time);fade_out_right.setStartValue(1.0);fade_out_right.setEndValue(0.0);orb_seq.addAnimation(fade_out_right)
        fade_in2=QPropertyAnimation(opacity_effect,b"opacity");fade_in2.setDuration(fade_time);fade_in2.setStartValue(0.0);fade_in2.setEndValue(1.0); move_in2=QPropertyAnimation(orb_label,b"geometry");move_in2.setDuration(fade_time);three_quarter_rect=QRect(int(right_rect.x()*0.75),25,30,30);move_in2.setStartValue(right_rect);move_in2.setEndValue(three_quarter_rect); backward_in_group=QParallelAnimationGroup();backward_in_group.addAnimation(fade_in2);backward_in_group.addAnimation(move_in2);orb_seq.addAnimation(backward_in_group)
        move_to_left=QPropertyAnimation(orb_label,b"geometry");move_to_left.setDuration(travel_time);move_to_left.setStartValue(three_quarter_rect);move_to_left.setEndValue(left_rect);orb_seq.addAnimation(move_to_left); fade_out_left=QPropertyAnimation(opacity_effect,b"opacity");fade_out_left.setDuration(fade_time);fade_out_left.setStartValue(1.0);fade_out_left.setEndValue(0.0);orb_seq.addAnimation(fade_out_left)
        orb_seq.setLoopCount(-1); return orb_seq

    def stop_animation(self):
         if self.anim_group and self.anim_group.state() == QPropertyAnimation.State.Running:
              self.anim_group.stop()


###############################################################################
# Main EMG Controller Window
###############################################################################
class EMGControllerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroSyn ReTrain")
        self.resize(600, 400)
        self._create_menubar()

        central_widget = QWidget(); self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.prediction_label = QLabel("Connecting..."); self.prediction_label.setFont(QFont(GARAMOND_LIGHT_FAMILY, 18)); layout.addWidget(self.prediction_label)
        strength_layout = QHBoxLayout(); signal_lbl = QLabel("Signal Strength:"); signal_lbl.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 14, QFont.Weight.Normal, True)); strength_layout.addWidget(signal_lbl)
        self.progress_bar = QProgressBar(); self.progress_bar.setMinimum(0); self.progress_bar.setMaximum(100); self.progress_bar.setValue(0); self.progress_bar.setTextVisible(False); self.progress_bar.setFixedHeight(20); strength_layout.addWidget(self.progress_bar, 1)
        self.progress_pct = QLabel("0%"); self.progress_pct.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 14, QFont.Weight.Normal, True)); strength_layout.addWidget(self.progress_pct); layout.addLayout(strength_layout)
        self.start_button = QPushButton("Start Exercise"); self.start_button.setFont(QApplication.font("QPushButton")); self.NSEWidgets_emboss(self.start_button); self.start_button.clicked.connect(self.start_prediction); layout.addWidget(self.start_button)
        self.stop_button = QPushButton("Stop Exercise"); self.stop_button.setFont(QApplication.font("QPushButton")); self.NSEWidgets_emboss(self.stop_button); self.stop_button.clicked.connect(self.stop_prediction); self.stop_button.setEnabled(False); layout.addWidget(self.stop_button)
        self.log_area = QTextEdit(); self.log_area.setReadOnly(True)
        dotmatrix_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "DOTMATRI.ttf")); dotmatrix_family = QFontDatabase.applicationFontFamilies(dotmatrix_id)[0] if dotmatrix_id != -1 else "DotMatrix"; self.log_area.setFont(QFont(dotmatrix_family, 14))

        # <<< CORRECTED Scanline Path for Stylesheet >>>
        scanline_path_raw = os.path.join(MEDIA_PATH, "scanlines.png")
        scanline_path_url = Path(scanline_path_raw).as_uri()

        self.log_area.setStyleSheet(f"""
            QTextEdit {{
            background-color: #1e1e1e;
            background-image: url("{scanline_path_url}"); /* Ensure quotes */
            background-repeat: repeat;
            color: #ffff00;
            border: none;
            }}
        """)
        shadow = QGraphicsDropShadowEffect(); shadow.setBlurRadius(10); shadow.setColor(QColor("#ffff00")); shadow.setOffset(0, 0); self.log_area.setGraphicsEffect(shadow)
        layout.addWidget(self.log_area)

        self.worker = None; self.worker_thread = QThread(); self.connect_window = None; self.cal_dlg = None

    def NSEWidgets_emboss(self, btn: QPushButton): effect = QGraphicsDropShadowEffect(btn); effect.setBlurRadius(1); effect.setOffset(0, 1); effect.setColor(QColor(255, 255, 255, 80)); btn.setGraphicsEffect(effect)
    def _create_menubar(self): menubar = self.menuBar(); about_menu = menubar.addMenu("NeuroSyn") if platform.system() == "Darwin" else menubar.addMenu("Help"); about_action = QAction("About NeuroSyn", self); about_action.setMenuRole(QAction.MenuRole.AboutRole); about_action.triggered.connect(self.show_about_dialog); about_menu.addAction(about_action)
    # <<< RESTORED About Dialog Text >>>
    def show_about_dialog(self):
        QMessageBox.about(self, "About NeuroSyn ReTrain",
            "NeuroSyn ReTrain\n"
            "Version 1.0 (Physio)\n\n"
            "Developed by Syndromatic Inc. / Kavish Krishnakumar (2025)\n\n"
            "A home-based physiotherapy assistant using surface EMG\n"
            "and real-time AI-powered feedback to guide your exercises."
        )

    def start_prediction(self):
        if self.worker_thread.isRunning(): self.log_area.append("Warning: Worker already running."); return
        self.log_area.clear(); self.log_area.append("Initializing..."); self.prediction_label.setText("Initializing..."); self.progress_bar.setValue(0); self.progress_pct.setText("0%")
        try:
            predictor = NeuroSynPredictor(threshold=0.8)
        except Exception as e: self.handle_error(f"Failed to initialize predictor: {e}"); return

        self.worker_thread = QThread() # Create a new thread instance
        self.worker = PredictionWorker(predictor)
        self.worker.moveToThread(self.worker_thread)

        # Connect signals BEFORE starting thread
        self.worker_thread.started.connect(self.worker.run)
        self.worker.prediction_signal.connect(self.update_prediction)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.error_signal.connect(self.handle_error)
        self.worker.status_signal.connect(self.handle_status)
        self.worker.finished_signal.connect(self._on_worker_finished)
        self.worker_thread.finished.connect(self._on_thread_finished)

        self.worker_thread.start()
        self.start_button.setEnabled(False); self.stop_button.setEnabled(True)
        self.log_area.append("Attempting to connect to Myo...")
        # Show connection window (ensure previous one is closed if exists)
        if self.connect_window: self.connect_window.close()
        self.connect_window = ConnectionWindow()
        self.connect_window.show()

    # <<< MODIFIED: Improved Stop Logic >>>
    def stop_prediction(self):
        """Stops the prediction worker thread more reliably."""
        if not self.worker_thread.isRunning() and not self.worker:
             self.log_area.append("Stop requested, but worker not running/initialized.")
             self._reset_ui_after_stop()
             return

        self.log_area.append("Stop requested.")
        self.stop_button.setEnabled(False) # Disable immediately
        if self.worker:
            self.worker.stop() # Signal worker's asyncio loop/task
            self.log_area.append("Stop signal sent to worker. Waiting for finish...")
        else:
             if self.worker_thread.isRunning():
                  self.log_area.append("Worker object missing, attempting to quit thread directly.")
                  self.worker_thread.quit()
             self._reset_ui_after_stop()

    @pyqtSlot()
    def _on_worker_finished(self):
        """Slot called when the worker's run() method signals it has finished."""
        self.log_area.append("Worker finished signal received.")
        if self.worker_thread.isRunning():
             self.worker_thread.quit()
             # self.worker_thread.wait(500) # Avoid wait if possible
        self._reset_ui_after_stop()
        self.worker = None

    @pyqtSlot()
    def _on_thread_finished(self):
        """Slot called when the QThread itself finishes."""
        self.log_area.append("Worker QThread finished.")
        # Ensure UI is reset finally
        self._reset_ui_after_stop()


    def _reset_ui_after_stop(self):
        """Resets UI elements after worker stops or fails."""
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if not (self.prediction_label.text().startswith("Error") or self.prediction_label.text() == "Initializing..."):
             self.prediction_label.setText("Stopped.")
        self.progress_bar.setValue(0); self.progress_pct.setText("0%")
        self.log_area.append("Prediction stopped.")
        # Close helper windows safely
        if self.connect_window and self.connect_window.isVisible():
             self.connect_window.stop_animation()
             self.connect_window.close()
             self.connect_window = None
        if self.cal_dlg and self.cal_dlg.isVisible():
             self.cal_dlg.close()
             self.cal_dlg = None


    def update_prediction(self, text):
        """Updates the prediction label and logs the prediction."""
        # Close connection window first time a prediction comes in
        if self.connect_window and self.connect_window.isVisible():
            self.connect_window.stop_animation(); self.connect_window.close(); self.connect_window = None
        # Close cal dialog only if it's marked as finished
        if self.cal_dlg and self.cal_dlg.isVisible() and self.cal_dlg._finished:
             self.cal_dlg.close(); self.cal_dlg = None
        self.prediction_label.setText(text)
        timestamp = time.strftime("%H:%M:%S"); self.log_area.append(f"{timestamp} - {text}")

    # <<< NEW Slot to relay Start signal to worker >>>
    @pyqtSlot()
    def _relay_start_cal_worker(self):
         """Relays the 'Start' click from dialog to the worker."""
         if self.worker:
              self.log_area.append("Relaying start signal to worker...")
              self.worker.proceed_with_calibration()
         else:
              self.log_area.append("Error: Cannot start calibration, worker not available.")


    @pyqtSlot()
    def _relay_cancel_cal(self):
        """Handles cancel signal from CalibrationDialog."""
        self.log_area.append("Calibration guidance cancelled by user.")
        self.stop_prediction() # Use the main stop logic

    def update_progress(self, value):
        self.progress_bar.setValue(value); self.progress_pct.setText(f"{value}%")

    # <<< MODIFIED: handle_status for restored welcome screen & flow >>>
    def handle_status(self, text: str):
        """Handles status messages and calibration cues."""
        # 1) Show the calibration wizard when ready (Initial Welcome State)
        if text == "READY_FOR_CAL":
            # <<< Keep Connection Window open until Cal Dialog is shown >>>
            if self.connect_window:
                 # Don't close immediately, wait for cal dialog to show
                 self.log_area.append("Myo connected. Displaying calibration assistant...")
            else: # Should normally exist, but handle case if already closed
                 self.log_area.append("Myo connected (conn window closed early?). Displaying calibration assistant...")

            if self.cal_dlg: self.cal_dlg.close() # Close previous if any
            self.cal_dlg = CalibrationDialog()
            self.cal_dlg.proceed_clicked.connect(self._relay_start_cal_worker) # Connect start signal
            self.cal_dlg.cancel_clicked.connect(self._relay_cancel_cal)
            self.cal_dlg.show()
            # NOW close the connection window
            if self.connect_window:
                 self.connect_window.stop_animation()
                 self.connect_window.close()
                 self.connect_window = None
            return

        # 2) Handle the 'Rest' cue -> Show the gesture screen ONLY IF DIALOG READY
        if text.startswith("CUE|"):
            if not self.cal_dlg or not self.cal_dlg.isVisible() or self.cal_dlg._cancelled or self.cal_dlg._finished or not self.cal_dlg._ok_proceeded:
                 self.log_area.append(f"Warning: Received CUE '{text}' but calibration dialog is not ready/waiting.")
                 return

            try:
                _, cid_str, gname, secs_str = text.split("|")
                cid = int(cid_str); secs = int(secs_str)
                if cid == 0 and gname == "Rest":
                     # <<< Use ICON_PATHS correctly >>>
                     icon_filename = ICON_PATHS.get(cid) # Get filename directly
                     if icon_filename:
                          self.cal_dlg.show_gesture(gname, secs, icon_filename)
                          self.log_area.append(f"Calibration Cue: Perform '{gname}' for {secs}s")
                     else:
                          self.log_area.append(f"Error: Icon filename not found in constants.ICON_PATHS for class ID {cid}")
                else:
                     self.log_area.append(f"Ignoring non-Rest cue: {text}")
            except Exception as e:
                 self.log_area.append(f"Error parsing CUE signal '{text}': {e}")
                 traceback.print_exc()
            return

        # 3) Handle calibration completion
        if text == "CAL_DONE":
            self.log_area.append("Calibration guidance finished.")
            if self.cal_dlg and self.cal_dlg.isVisible():
                self.cal_dlg.mark_as_finished() # Updates UI, starts close timer
            return

        # 4) Log any other status messages
        self.log_area.append(text)


    def handle_error(self, error_text):
        """Handles errors from the worker."""
        self.log_area.append(f"ERROR: {error_text}")
        is_stopping = not self.stop_button.isEnabled()
        if not is_stopping:
             self.prediction_label.setText("Error encountered!")
             QMessageBox.critical(self, "Process Error", f"An error occurred:\n{error_text}\n\nPlease check connection/logs and restart.")
             self.stop_prediction()
        else: self.log_area.append("(Stop already in progress)")
        # Force close helper windows
        if self.connect_window and self.connect_window.isVisible(): self.connect_window.stop_animation(); self.connect_window.close(); self.connect_window = None
        if self.cal_dlg and self.cal_dlg.isVisible(): self.cal_dlg.close(); self.cal_dlg = None

    def closeEvent(self, event):
        """Ensure worker thread is stopped on main window close."""
        self.log_area.append("Main window closing...")
        self.stop_prediction()
        if self.worker_thread.isRunning():
             self.log_area.append("Waiting briefly for worker thread exit before closing window...")
             if not self.worker_thread.wait(2000):
                  self.log_area.append("Worker thread did not exit gracefully, terminating.")
                  self.worker_thread.terminate()
                  self.worker_thread.wait()
        super().closeEvent(event)


###############################################################################
# Welcome Window (Restored Text)
###############################################################################
class WelcomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Welcome to NeuroSyn ReTrain"); self.resize(600, 500)
        central_widget = QWidget(); self.setCentralWidget(central_widget); layout = QVBoxLayout(central_widget); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon_label = QLabel(); pixmap = QPixmap(os.path.join(MEDIA_PATH, "syn app light.png")); scaled_pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation); icon_label.setPixmap(scaled_pixmap); icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(icon_label)
        welcome_label = QLabel("Welcome"); welcome_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold)); welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(welcome_label)
        fm = QFontMetrics(welcome_label.font()); title_width = fm.horizontalAdvance(welcome_label.text())

        # <<< RESTORED Welcome Text (Matches old code) >>>
        intro_text = QLabel(
            "Welcome to NeuroSyn ReTrain,\n"
            "A home-based physiotherapy assistant that uses your EMG armband\n"
            "to monitor muscle activity and guide you through exercises.\n\n\n"
            "Please connect your EMG device, then click Start to begin.\n"
        )
        intro_text.setFont(QFont(GARAMOND_BOOK_FAMILY, 14)); intro_text.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(intro_text)

        self.next_button = QPushButton("Get Started"); self.next_button.setFont(QApplication.font("QPushButton")); self.next_button.setFixedWidth(title_width); self.next_button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed); self.NSEWidgets_emboss(self.next_button); self.next_button.clicked.connect(self.gotoMainWindow); layout.addWidget(self.next_button, alignment=Qt.AlignmentFlag.AlignCenter); layout.addStretch(1)
        copyright_label = QLabel("Copyright © 2025 Syndromatic Inc. / Kavish Krishnakumar"); copyright_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 10)); copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(copyright_label)
    def NSEWidgets_emboss(self, btn: QPushButton): effect = QGraphicsDropShadowEffect(btn); effect.setBlurRadius(1); effect.setOffset(0, 1); effect.setColor(QColor(255, 255, 255, 80)); btn.setGraphicsEffect(effect)
    def gotoMainWindow(self): self.main_window = EMGControllerMainWindow(); self.main_window.show(); self.close()

###############################################################################
# Main Execution (Font loading and Stylesheets Unchanged)
###############################################################################
def main():
    # <<< Added import >>>
    from pathlib import Path
    app = QApplication(sys.argv)
    # (Keep font loading, palette setup, and stylesheet logic exactly as it was)
    # --- FONT LOADING ---
    global GARAMOND_LIGHT_FAMILY, GARAMOND_LIGHT_ITALIC_FAMILY, GARAMOND_BOOK_FAMILY, DOTMATRIX_FAMILY
    light_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-LtCond.ttf")); GARAMOND_LIGHT_FAMILY = QFontDatabase.applicationFontFamilies(light_id)[0] if light_id != -1 else "ITCGaramondStd Light Condensed"
    light_italic_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-LtCondIta.ttf")); GARAMOND_LIGHT_ITALIC_FAMILY = QFontDatabase.applicationFontFamilies(light_italic_id)[0] if light_italic_id != -1 else "ITCGaramondStd Light Condensed Italic"
    book_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-BkCond.ttf")); GARAMOND_BOOK_FAMILY = QFontDatabase.applicationFontFamilies(book_id)[0] if book_id != -1 else "ITCGaramondStd Book Condensed"
    dotmatrix_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "DOTMATRI.ttf")); DOTMATRIX_FAMILY = QFontDatabase.applicationFontFamilies(dotmatrix_id)[0] if dotmatrix_id != -1 else "DotMatrix"
    print("Loaded Fonts:", GARAMOND_LIGHT_FAMILY, "|", GARAMOND_LIGHT_ITALIC_FAMILY, "|", GARAMOND_BOOK_FAMILY, "|", DOTMATRIX_FAMILY)
    # --- STYLING ---
    wincol = app.palette().color(QPalette.ColorRole.Window); is_dark = wincol.lightness() < 128; app.setStyle(QStyleFactory.create("Fusion")); pal = QPalette()
    if not is_dark: pal.setColor(QPalette.ColorRole.Window,QColor(240,240,240)); pal.setColor(QPalette.ColorRole.WindowText,QColor(33,33,33)); pal.setColor(QPalette.ColorRole.Base,QColor(255,255,255)); pal.setColor(QPalette.ColorRole.AlternateBase,QColor(240,240,240)); pal.setColor(QPalette.ColorRole.ToolTipBase,QColor(255,255,220)); pal.setColor(QPalette.ColorRole.ToolTipText,QColor(0,0,0)); pal.setColor(QPalette.ColorRole.Text,QColor(33,33,33)); pal.setColor(QPalette.ColorRole.Button,QColor(240,240,240)); pal.setColor(QPalette.ColorRole.ButtonText,QColor(33,33,33)); pal.setColor(QPalette.ColorRole.BrightText,QColor(255,0,0)); pal.setColor(QPalette.ColorRole.Highlight,QColor(76,163,224)); pal.setColor(QPalette.ColorRole.HighlightedText,QColor(255,255,255))
    else: pal.setColor(QPalette.ColorRole.Window,QColor(45,45,45)); pal.setColor(QPalette.ColorRole.WindowText,QColor(220,220,220)); pal.setColor(QPalette.ColorRole.Base,QColor(30,30,30)); pal.setColor(QPalette.ColorRole.AlternateBase,QColor(45,45,45)); pal.setColor(QPalette.ColorRole.ToolTipBase,QColor(255,255,220)); pal.setColor(QPalette.ColorRole.ToolTipText,QColor(0,0,0)); pal.setColor(QPalette.ColorRole.Text,QColor(220,220,220)); pal.setColor(QPalette.ColorRole.Button,QColor(60,60,60)); pal.setColor(QPalette.ColorRole.ButtonText,QColor(220,220,220)); pal.setColor(QPalette.ColorRole.BrightText,QColor(255,0,0)); pal.setColor(QPalette.ColorRole.Highlight,QColor(76,163,224)); pal.setColor(QPalette.ColorRole.HighlightedText,QColor(255,255,255))
    app.setPalette(pal)
    light_btn_qss="""QPushButton{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(250,250,250,255), stop:1 rgba(230,230,230,255));color:#202020;border:1px solid #c0c0c0;border-radius:6px;padding:6px 14px;}QPushButton:hover{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(240,240,240,255), stop:1 rgba(215,215,215,255));}QPushButton:pressed{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(220,220,220,255), stop:1 rgba(195,195,195,255));}QPushButton:disabled{background:rgba(245,245,245,255);color:rgba(160,160,160,255);border:1px solid rgba(200,200,200,255);}"""
    dark_btn_qss="""QPushButton{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(85,85,85,255), stop:1 rgba(51,51,51,255));color:#eeeeee;border:1px solid #202020;border-radius:6px;padding:6px 14px;}QPushButton:hover{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(100,100,100,255), stop:1 rgba(80,80,80,255));}QPushButton:pressed{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(70,70,70,255), stop:1 rgba(40,40,40,255));}QPushButton:disabled{background:rgba(70,70,70,255);color:rgba(120,120,120,255);border:1px solid rgba(30,30,30,255);}"""
    shared_qss="""QProgressBar{border:1px solid #707070;border-radius:5px;text-align:right;padding-right:4px;background:transparent;}QProgressBar::chunk{background:qlineargradient(x1:0, y1:0, x2:0, y2:1,stop:0 rgba(76,163,224,255), stop:1 rgba(33,123,200,255));border-radius:5px;}QTextEdit{border:1px solid #505050;border-radius:4px;}"""
    final_qss = (dark_btn_qss if is_dark else light_btn_qss) + shared_qss; app.setStyleSheet(final_qss)
    app.setOrganizationName("Syndromatic Inc."); app.setApplicationName("NeuroSyn Retrain")
    app.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "myo_icon.png")))

    # --- Start App ---
    welcome = WelcomeWindow(); welcome.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # Ensure constants are loaded before GUI starts
    try: import constants
    except ImportError: print("ERROR: constants.py not found or contains errors."); sys.exit(1)
    main()
