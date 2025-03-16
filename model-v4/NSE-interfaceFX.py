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
    QThread, pyqtSignal, QObject, Qt, QTimer, QEasingCurve, QPropertyAnimation,
    QSequentialAnimationGroup, QParallelAnimationGroup, QPauseAnimation, QRect
)

# Define base path and media path (relative to the script)
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
MEDIA_PATH = os.path.join(BASE_PATH, "NSEmedia")

# Your custom modules
from prediction import GesturePredictor
from constants import MYO_ADDRESS

from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
from bleak import BleakScanner

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
    
    def __init__(self, predictor: GesturePredictor):
        super().__init__()
        self.predictor = predictor
        self._is_running = True

    async def run_async(self):
        try:
            myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
            if not myo_device:
                self.error_signal.emit("Could not find Myo device.")
                return

            async with Myo(myo_device) as myo:
                await asyncio.sleep(0.5)
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
                await asyncio.sleep(0.25)

                @myo.on_emg_smooth
                def on_emg_smooth(emg_data):
                    norm = math.sqrt(sum(e ** 2 for e in emg_data))
                    max_norm = 300  # Adjust based on your expected maximum norm
                    strength = min(100, (norm / max_norm) * 100)
                    self.progress_signal.emit(int(strength))
                    
                    prediction = self.predictor.add_data(emg_data)
                    if prediction:
                        class_id, gesture_name = prediction
                        self.prediction_signal.emit(f"Predicted: {gesture_name} (class {class_id})")

                await myo.set_mode(emg_mode=EmgMode.SMOOTH)
                while self._is_running:
                    await asyncio.sleep(0.01)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

    def run(self):
        asyncio.run(self.run_async())

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
        predictor = GesturePredictor(prediction_interval=1.0)
        self.worker = PredictionWorker(predictor)
        self.worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker.prediction_signal.connect(self.update_prediction)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.error_signal.connect(self.handle_error)
        
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
        if self.connect_window:
            self.connect_window.close()
            self.connect_window = None

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
