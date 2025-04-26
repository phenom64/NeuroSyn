#!/usr/bin/env python3
"""
Artemis Project - EMGController GUI
Created by Syndromatic Inc. / Kavish Krishnakumar (2025)
---------------------------------------------------------------------------
1) Welcome window (scaled icon, title, instructions)
2) Animated "Connecting" screen with a computer icon on the left, Myo icon on the right,
   and three orbs (orb.png) that fade/move from left to right in sequence.
3) Main window for controlling and monitoring Myo EMG data with a progress bar, logs, etc.
4) SynOS/Mac-style About menu item (cross-platform fallback if not on SynOS or Mac OS X).
5) The log area now uses Dot Matrix font and a CRT scanline background.

Prerequisites:
  - PyQt6
  - bleak
  - pymyo
  - 'prediction.py' (GesturePredictor)
  - 'constants.py' (with MYO_ADDRESS)
"""

import os
import sys
import time
import asyncio
import math
import platform

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel, QPushButton,
    QProgressBar, QTextEdit, QHBoxLayout, QMenu, QGraphicsOpacityEffect, QGraphicsDropShadowEffect
)
from PyQt6.QtGui import QPixmap, QAction, QIcon, QFont, QFontDatabase, QColor
from PyQt6.QtCore import (
    QThread, pyqtSignal, pyqtSlot, QObject, Qt, QTimer, QEasingCurve, QPropertyAnimation,
    QSequentialAnimationGroup, QParallelAnimationGroup, QPauseAnimation, QRect
)

# Define base path and media path (relative to the script)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MEDIA_PATH = os.path.join(BASE_PATH, "NSEmedia")

# Your custom modules
from prediction import GesturePredictor, calibrate_gestures
from constants import MYO_ADDRESS, ICON_PATHS

from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
from bleak import BleakScanner

# ─── Patch pymyo’s buggy classifier handler ────────────────────────────────────
import types, struct
from pymyo import Myo          # already imported above

def _safe_on_classifier(self, sender, value):
    # Ignore short packets that crash the stock handler
    if len(value) < 3:
        return
    try:
        event_type, event_data = struct.unpack("<B2s", value)
    except struct.error:
        return  # just skip malformed packets

# Replace the method on the Myo class (runs once at import time)
Myo._on_classifier = types.MethodType(_safe_on_classifier, Myo)
# ───────────────────────────────────────────────────────────────────────────────

# Global variables to store font family names once loaded
GARAMOND_LIGHT_FAMILY = None            # Loaded from ITCGaramondStd-LtCond.ttf
GARAMOND_LIGHT_ITALIC_FAMILY = None     # Loaded from ITCGaramondStd-LtCondIta.ttf
GARAMOND_BOOK_FAMILY = None             # Loaded from ITCGaramondStd-BkCond.ttf
DOTMATRIX_FAMILY = None                 # Loaded from DOTMATRI.ttf

###############################################################################
# Worker for handling Myo data & predictions
###############################################################################
class PredictionWorker(QObject):
    prediction_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    error_signal = pyqtSignal(str)
    status_signal       = pyqtSignal(str)
    
    def __init__(self, predictor: GesturePredictor):
        super().__init__()
        self.predictor = predictor
        self._is_running = True
        self._cal_start = asyncio.Event()   # waits until user clicks OK

    async def run_async(self):
        self._cal_start.clear()      # reset before we wait again
        try:
        # 1 Find the armband
            myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
            if not myo_device:
                self.error_signal.emit("Could not find Myo device.")
                return                # ← INDENTED under the if-block // SonoAI

        # 2 Keep the connection alive
            async with Myo(myo_device) as myo:   # ← all the code below this line
            #     MUST be indented one level further → SonoAI
                if hasattr(myo, "wait_for_services"):
                    await myo.wait_for_services()
                else:
                    await asyncio.sleep(0.3)

            # Never let the armband sleep
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)

            # Explicitly enable EMG notifications
                await myo.set_mode(emg_mode=EmgMode.SMOOTH)
                #await myo.emg_notifications(True)

        # 3 EMG callback
                @myo.on_emg_smooth
                def on_emg_smooth(emg_data):
                    if self.predictor.is_calibrating:
                        self.predictor.calibrate(emg_data)
                        return
                    norm = math.sqrt(sum(e ** 2 for e in emg_data))
                    self.progress_signal.emit(int(min(100, (norm / 300) * 100)))
                    if (pred := self.predictor.add_data(emg_data)):
                        cid, gname = pred
                        self.prediction_signal.emit(f"Predicted: {gname} (class {cid})")

        # 4  Calibration
                self.status_signal.emit("READY_FOR_CAL")          # show wizard
                await self._cal_start.wait()                      # ← waits for OK
                print("DEBUG  ➜  Event released, starting calibrate_gestures")   # <-- add

                self.status_signal.emit("▼ Calibration starting …")
                await calibrate_gestures(
                        myo,
                        self.predictor,
                        cue=self.status_signal.emit,
                        log=self.status_signal.emit
                )
                self.status_signal.emit("✓ Calibration complete!")
                #if self.connect_window:            # auto-close spinner
                #    self.connect_window.close()
                #    self.connect_window = None

        # 5 Main loop
                while self._is_running:
                    await asyncio.sleep(0.01)

        except Exception as e:
            self.error_signal.emit(str(e))
        
    @pyqtSlot()
    def start_calibration(self):
        """Called by the wizard when the user clicks OK."""
        print("DEBUG  ➜  start_calibration() got the click")   # <-- add
        self._cal_start.set()
    
    def run(self):
        asyncio.run(self.run_async())
        
    # ------------------------------------------------------------------ Qt slot
    def stop(self):
        """Called from the GUI when the Stop button is pressed."""
        self._is_running = False

###############################################################################
# Calibration Wizard Dialog
###############################################################################
class CalibrationDialog(QMainWindow):
    proceed_clicked = pyqtSignal()      # emitted when user clicks OK

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Assistant")
        self.resize(520, 340)

        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
       # ─── Add the SetupAssistant logo above the title ────────────
        self.logo = QLabel()
        logo_pix = QPixmap(os.path.join(MEDIA_PATH, "SetupAssistant.png"))
        self.logo.setPixmap(
            logo_pix.scaled(100, 100,
                            Qt.AspectRatioMode.KeepAspectRatio,
                            Qt.TransformationMode.SmoothTransformation)
        )
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.logo)

        # headline
        self.head = QLabel("Calibration Assistant")
        self.head.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold))
        self.head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.head)

        # body
        self.body = QLabel(
            "You need to calibrate your EMG armband.\n"
            "Follow the on-screen instructions and\n"
            "we'll get you up-and-running in no time."
        )

        self.body.setFont(QFont(GARAMOND_BOOK_FAMILY, 16))
        self.body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.setWordWrap(True)
        vbox.addWidget(self.body)

        # icon
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.icon)

        # countdown
        self.timer_lbl = QLabel("")
        self.timer_lbl.setFont(QFont(GARAMOND_BOOK_FAMILY, 22))
        self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.timer_lbl)

        # OK button
        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self._on_ok)
        vbox.addWidget(self.ok_btn)

        self.countdown = None
        
    @pyqtSlot()                              # NEW
    def _on_ok(self):
        self.proceed_clicked.emit()
        self.ok_btn.setEnabled(False)

    # called for every cue
    def show_gesture(self, title: str, seconds: int, icon_file: str):
        self.logo.hide()
        self.head.setText(title)
        self.body.setText(f"Hold for {seconds} s …")
        pix = QPixmap(os.path.join(MEDIA_PATH, "gestures", icon_file))
        self.icon.setPixmap(pix.scaled(140, 140,
                                        Qt.AspectRatioMode.KeepAspectRatio,
                                        Qt.TransformationMode.SmoothTransformation))
        # live countdown
        if self.countdown:
            self.countdown.stop()
        self.time_left = seconds
        self.timer_lbl.setText(str(self.time_left))
        self.countdown = QTimer(self)
        self.countdown.timeout.connect(self._tick)
        self.countdown.start(1000)

    def _tick(self):
        self.time_left -= 1
        self.timer_lbl.setText(str(self.time_left) if self.time_left > 0 else "✓")
        if self.time_left <= 0 and self.countdown:
            self.countdown.stop()

###############################################################################
# Animated Connection Screen
###############################################################################
class ConnectionWindow(QMainWindow):
    """
    An animated window that appears while the app tries to connect to the Myo device.
    Displays:
      - A computer icon on the left
      - A Myo icon on the right
      - Three orbs (orb.png) that travel left->right->left, fading in/out
      - Orbs are staggered in time to create a partial conga-line effect.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Connecting to Myo...")
        self.resize(600, 200)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Left icon (computer)
        self.computer_label = QLabel()
        comp_pix = QPixmap(os.path.join(MEDIA_PATH, "computer_icon.png"))
        comp_scaled = comp_pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.computer_label.setPixmap(comp_scaled)
        layout.addWidget(self.computer_label)

        # Container for orbs (absolute positioning)
        self.orbs_container = QWidget()
        self.orbs_container.setFixedSize(300, 80)
        self.orbs_container.setStyleSheet("background-color: transparent;")
        layout.addWidget(self.orbs_container)

        # Right icon (Myo)
        self.myo_label = QLabel()
        myo_pix = QPixmap(os.path.join(MEDIA_PATH, "myo_icon.png"))
        myo_scaled = myo_pix.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.myo_label.setPixmap(myo_scaled)
        layout.addWidget(self.myo_label)

        # Create 3 orb labels in absolute positions
        self.orb_labels = []
        self.orb_effects = []
        for _ in range(3):
            orb_label = QLabel(self.orbs_container)
            orb_pix = QPixmap(os.path.join(MEDIA_PATH, "orb.png")).scaled(30, 30, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            orb_label.setPixmap(orb_pix)
            orb_label.move(0, 25)

            effect = QGraphicsOpacityEffect(orb_label)
            effect.setOpacity(0.0)
            orb_label.setGraphicsEffect(effect)

            self.orb_labels.append(orb_label)
            self.orb_effects.append(effect)

        self.anim_group = QParallelAnimationGroup(self)
        start_delays = [0, 500, 1000]
        for i, orb_label in enumerate(self.orb_labels):
            seq_anim = self._create_orb_sequence(orb_label, self.orb_effects[i], start_delays[i])
            self.anim_group.addAnimation(seq_anim)
        self.anim_group.setLoopCount(-1)
        self.anim_group.start()

    def _create_orb_sequence(self, orb_label: QLabel, opacity_effect: QGraphicsOpacityEffect, start_delay: int):
        orb_seq = QSequentialAnimationGroup()
        if start_delay > 0:
            pause = QPauseAnimation(start_delay)
            orb_seq.addAnimation(pause)

        fade_time = 300
        travel_time = 1000
        left_rect = QRect(0, 25, 30, 30)
        right_rect = QRect(self.orbs_container.width() - 30, 25, 30, 30)

        # Forward trip (Left -> Right)
        fade_in = QPropertyAnimation(opacity_effect, b"opacity")
        fade_in.setDuration(fade_time)
        fade_in.setStartValue(0.0)
        fade_in.setEndValue(1.0)

        move_in = QPropertyAnimation(orb_label, b"geometry")
        move_in.setDuration(fade_time)
        quarter_rect = QRect(int(right_rect.x() * 0.25), 25, 30, 30)
        move_in.setStartValue(left_rect)
        move_in.setEndValue(quarter_rect)

        forward_in_group = QParallelAnimationGroup()
        forward_in_group.addAnimation(fade_in)
        forward_in_group.addAnimation(move_in)
        orb_seq.addAnimation(forward_in_group)

        move_to_right = QPropertyAnimation(orb_label, b"geometry")
        move_to_right.setDuration(travel_time)
        move_to_right.setStartValue(quarter_rect)
        move_to_right.setEndValue(right_rect)
        orb_seq.addAnimation(move_to_right)

        fade_out_right = QPropertyAnimation(opacity_effect, b"opacity")
        fade_out_right.setDuration(fade_time)
        fade_out_right.setStartValue(1.0)
        fade_out_right.setEndValue(0.0)
        orb_seq.addAnimation(fade_out_right)

        # Return trip (Right -> Left)
        fade_in2 = QPropertyAnimation(opacity_effect, b"opacity")
        fade_in2.setDuration(fade_time)
        fade_in2.setStartValue(0.0)
        fade_in2.setEndValue(1.0)

        move_in2 = QPropertyAnimation(orb_label, b"geometry")
        move_in2.setDuration(fade_time)
        three_quarter_rect = QRect(int(right_rect.x() * 0.75), 25, 30, 30)
        move_in2.setStartValue(right_rect)
        move_in2.setEndValue(three_quarter_rect)

        backward_in_group = QParallelAnimationGroup()
        backward_in_group.addAnimation(fade_in2)
        backward_in_group.addAnimation(move_in2)
        orb_seq.addAnimation(backward_in_group)

        move_to_left = QPropertyAnimation(orb_label, b"geometry")
        move_to_left.setDuration(travel_time)
        move_to_left.setStartValue(three_quarter_rect)
        move_to_left.setEndValue(left_rect)
        orb_seq.addAnimation(move_to_left)

        fade_out_left = QPropertyAnimation(opacity_effect, b"opacity")
        fade_out_left.setDuration(fade_time)
        fade_out_left.setStartValue(1.0)
        fade_out_left.setEndValue(0.0)
        orb_seq.addAnimation(fade_out_left)

        orb_seq.setLoopCount(-1)
        return orb_seq

###############################################################################
# Main EMG Controller Window
###############################################################################
class EMGControllerMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artemis Project - EMGController")
        self.resize(600, 400)
        self._create_menubar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Prediction label uses ITC Garamond Light
        self.prediction_label = QLabel("No prediction yet.")
        self.prediction_label.setFont(QFont(GARAMOND_LIGHT_FAMILY, 18))
        layout.addWidget(self.prediction_label)
        
        progress_layout = QHBoxLayout()
        # "Signal Strength:" uses ITC Garamond Light Italic
        progress_label = QLabel("Signal Strength:")
        progress_label.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 14, QFont.Weight.Normal, True))
        progress_layout.addWidget(progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)
        
        layout.addLayout(progress_layout)
        
        self.start_button = QPushButton("Start Prediction")
        self.start_button.setFont(QApplication.font("QPushButton"))
        self.start_button.clicked.connect(self.start_prediction)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Prediction")
        self.stop_button.setFont(QApplication.font("QPushButton"))
        self.stop_button.clicked.connect(self.stop_prediction)
        self.stop_button.setEnabled(False)
        layout.addWidget(self.stop_button)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        
        # Load Dot Matrix font for the log area
        dotmatrix_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "DOTMATRI.ttf"))
        if dotmatrix_id != -1:
            dotmatrix_family = QFontDatabase.applicationFontFamilies(dotmatrix_id)[0]
        else:
            dotmatrix_family = "DotMatrix"

        self.log_area.setFont(QFont(dotmatrix_family, 14))
        # Add a CRT-like background with scanlines
        self.log_area.setStyleSheet(f"""
            QTextEdit {{
            background-color: #1e1e1e;
            background-image: url("file://{os.path.join(MEDIA_PATH, "scanlines.png")}");
            background-repeat: repeat;
            color: #ffff00;
            border: none;
            }}
        """)

        
        # Keep the drop shadow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor("#ffff00"))
        shadow.setOffset(0, 0)
        self.log_area.setGraphicsEffect(shadow)
        
        layout.addWidget(self.log_area)
        
        self.worker = None
        self.worker_thread = QThread()
        self.connect_window = None

    def _create_menubar(self):
        menubar = self.menuBar()
        if platform.system() == "Darwin":
            about_menu = menubar.addMenu("EMGController")
        else:
            about_menu = menubar.addMenu("Help")

        about_action = QAction("About EMGController", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

    def show_about_dialog(self):
        from PyQt6.QtWidgets import QMessageBox
        about_box = QMessageBox(self)
        about_box.setWindowTitle("About EMGController")
        about_box.setText(
            "Artemis Project - EMGController\n"
            "Interface created by Kavish Krishnakumar\n\n"
            "This application demonstrates real-time Myo EMG gesture prediction\n"
            "with a sleek SynOS-style UI, an animated connection screen, and more!"
        )
        about_box.exec()

    def start_prediction(self):
        predictor = GesturePredictor(
            prediction_interval=1.0,
            threshold=0.90,          # <-- tweak if you like
            num_repetitions=1        # <-- how many reps per gesture during calibration
        )
        self.worker = PredictionWorker(predictor)
        self.worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker.prediction_signal.connect(self.update_prediction)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.error_signal.connect(self.handle_error)
        # was: self.worker.status_signal.connect(self.log_area.append)
        self.worker.status_signal.connect(self.handle_status)
        
        self.worker_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.log_area.append("Started prediction...")
        self.connect_window = ConnectionWindow()
        self.connect_window.show()

    def stop_prediction(self):
        if self.worker:
            self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.log_area.append("Stopped prediction.")
        #if self.connect_window:
        #    self.connect_window.close()
        #    self.connect_window = None

    def update_prediction(self, text):
        # Update the prediction label and append the prediction with a timestamp to the log.
        self.prediction_label.setText(text)
        timestamp = time.strftime("%H:%M:%S")
        self.log_area.append(f"{timestamp} - {text}")
        if self.connect_window:
            self.connect_window.close()
            self.connect_window = None

    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    @pyqtSlot()                      # relay that runs in the GUI thread
    def _relay_start_cal(self):
        self.worker.start_calibration()     # safe: Qt auto-queues to worker thread

        # ---------------------------------------------------------------- status
    def handle_status(self, text: str):
        """Drive the calibration wizard and log everything else."""
        # 1) show the wizard intro
        if text == "READY_FOR_CAL":
            if self.connect_window:          # close spinner
                self.connect_window.close()
                self.connect_window = None

            self.cal_dlg = CalibrationDialog()
            #self.cal_dlg.proceed_clicked.connect(
            #    self.worker.start_calibration,
            #    Qt.ConnectionType.QueuedConnection
            #)
            self.cal_dlg.logo.show()
            self.cal_dlg.proceed_clicked.connect(self._relay_start_cal)
            self.cal_dlg.show()
            return

        # 2) per-gesture cue  (format:  CUE|<cid>|<name>|<seconds>)
        if text.startswith("CUE|"):
            self.cal_dlg.logo.hide()
            _, cid, gname, secs = text.split("|")
            icon_file = ICON_PATHS[int(cid)]
            self.cal_dlg.show_gesture(
                gname.title(),
                int(secs),
                icon_file
            )
            return

        # 3) calibration finished
        if text == "CAL_DONE":
            # hide the per‐gesture icon and countdown
            self.cal_dlg.icon.hide()
            self.cal_dlg.timer_lbl.hide()
            # show the SetupAssistant logo again
            self.cal_dlg.logo.show()
            self.cal_dlg.head.setText("Calibration complete!")
            self.cal_dlg.body.setText("You’re ready to go.")
            self.cal_dlg.ok_btn.hide()
            QTimer.singleShot(1500, self.cal_dlg.close)
            return

        # 4) anything else → plain log
        self.log_area.append(text)


    def handle_error(self, error_text):
        self.log_area.append("Error: " + error_text)
        self.prediction_label.setText("Error: " + error_text)
        if self.connect_window:
            self.connect_window.close()
            self.connect_window = None

###############################################################################
# Welcome Window
###############################################################################
class WelcomeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artemis Project - EMGController")
        self.resize(500, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        icon_label = QLabel()
        pixmap = QPixmap(os.path.join(MEDIA_PATH, "syn app light.png"))
        scaled_pixmap = pixmap.scaled(120, 120, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)
        
        # Use ITC Garamond Book Condensed for the Welcome text
        welcome_label = QLabel("Welcome")
        welcome_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold))
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)
        
        intro_text = QLabel(
            "Welcome to the Artemis Project's EMGController App.\n"
            "Ensure you have your MYO EMG Armband close to you,\n"
            "and click 'Next' to continue."
        )
        intro_text.setFont(QFont(GARAMOND_BOOK_FAMILY, 16))
        intro_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(intro_text)
        
        self.next_button = QPushButton("Next")
        self.next_button.setFont(QApplication.font("QPushButton"))
        self.next_button.clicked.connect(self.gotoMainWindow)
        layout.addWidget(self.next_button)
        
        layout.addStretch(1)
        
        copyright_label = QLabel("UI Copyright © 2025 Syndromatic Inc.")
        copyright_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 12))
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)

    def gotoMainWindow(self):
        self.main_window = EMGControllerMainWindow()
        self.main_window.show()
        self.close()

###############################################################################
# Main Execution
###############################################################################
def main():
    app = QApplication(sys.argv)
    
    # Set the organization & application name for proper macOS labeling
    app.setOrganizationName("Syndromatic Inc.")
    app.setApplicationName("EMGController")
    
    global GARAMOND_LIGHT_FAMILY, GARAMOND_LIGHT_ITALIC_FAMILY, GARAMOND_BOOK_FAMILY, DOTMATRIX_FAMILY
    
    # Load ITC Garamond fonts
    light_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-LtCond.ttf"))
    if light_id != -1:
        GARAMOND_LIGHT_FAMILY = QFontDatabase.applicationFontFamilies(light_id)[0]
    else:
        GARAMOND_LIGHT_FAMILY = "ITCGaramondStd Light Condensed"
    
    light_italic_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-LtCondIta.ttf"))
    if light_italic_id != -1:
        GARAMOND_LIGHT_ITALIC_FAMILY = QFontDatabase.applicationFontFamilies(light_italic_id)[0]
    else:
        GARAMOND_LIGHT_ITALIC_FAMILY = "ITCGaramondStd Light Condensed Italic"
    
    book_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "ITCGaramondStd-BkCond.ttf"))
    if book_id != -1:
        GARAMOND_BOOK_FAMILY = QFontDatabase.applicationFontFamilies(book_id)[0]
    else:
        GARAMOND_BOOK_FAMILY = "ITCGaramondStd Book Condensed"
    
    # Load Dot Matrix font as well
    dotmatrix_id = QFontDatabase.addApplicationFont(os.path.join(MEDIA_PATH, "DOTMATRI.ttf"))
    if dotmatrix_id != -1:
        DOTMATRIX_FAMILY = QFontDatabase.applicationFontFamilies(dotmatrix_id)[0]
    else:
        DOTMATRIX_FAMILY = "DotMatrix"
    
    print("Loaded ITC Garamond Light:", GARAMOND_LIGHT_FAMILY)
    print("Loaded ITC Garamond Light Italic:", GARAMOND_LIGHT_ITALIC_FAMILY)
    print("Loaded ITC Garamond Book:", GARAMOND_BOOK_FAMILY)
    print("Loaded Dot Matrix:", DOTMATRIX_FAMILY)
    
    # Do not set a global font here so buttons remain native.
    app.setWindowIcon(QIcon(os.path.join(MEDIA_PATH, "myo_icon.png")))
    
    welcome = WelcomeWindow()
    welcome.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
