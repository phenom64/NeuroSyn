#!/usr/bin/env python3
"""
NeuroSyn ReTrain – Syndromatic Home Physio GUI
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
    progress_signal   = pyqtSignal(int)
    error_signal      = pyqtSignal(str)
    status_signal     = pyqtSignal(str)
    
    def __init__(self, predictor: GesturePredictor):
        super().__init__()
        self.predictor   = predictor
        self._is_running = True
        self._cal_start  = asyncio.Event()   # fires on OK
        self._cancel_cal = asyncio.Event()   # fires on Cancel

    async def run_async(self):
        # reset events each time we start
        self._cal_start.clear()
        self._cancel_cal.clear()

        try:
            # 1) Find the armband
            myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
            if not myo_device:
                self.error_signal.emit("Could not find Myo device.")
                return

            # 2) Connect & configure
            async with Myo(myo_device) as myo:
                if hasattr(myo, "wait_for_services"):
                    await myo.wait_for_services()
                else:
                    await asyncio.sleep(0.3)
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
                await myo.set_mode(emg_mode=EmgMode.SMOOTH)

                # EMG callback for both calibration & prediction
                @myo.on_emg_smooth
                def on_emg_smooth(emg_data):
                    if self.predictor.is_calibrating:
                        self.predictor.calibrate(emg_data)
                        return
                    norm = math.sqrt(sum(e**2 for e in emg_data))
                    self.progress_signal.emit(int(min(100, (norm / 300) * 100)))
                    if (pred := self.predictor.add_data(emg_data)):
                        cid, gname = pred
                        self.prediction_signal.emit(f"Predicted: {gname} (class {cid})")

                # 3) GUI → “ready to calibrate”
                self.status_signal.emit("READY_FOR_CAL")

                # wait for either Start or Cancel
                start_task  = asyncio.create_task(self._cal_start.wait())
                cancel_task = asyncio.create_task(self._cancel_cal.wait())
                done, pending = await asyncio.wait({start_task, cancel_task},
                                                   return_when=asyncio.FIRST_COMPLETED)
                for t in pending: t.cancel()

                # if canceled before starting, bail out
                if cancel_task in done:
                    return

                # 4) user clicked Start → run calibration
                self.status_signal.emit("▼ Calibration starting...")
                cal_task     = asyncio.create_task(
                    calibrate_gestures(myo,
                                       self.predictor,
                                       cue=self.status_signal.emit,
                                       log=self.status_signal.emit)
                )
                cancel_task2 = asyncio.create_task(self._cancel_cal.wait())
                done, pending = await asyncio.wait({cal_task, cancel_task2},
                                                   return_when=asyncio.FIRST_COMPLETED)
                for t in pending: t.cancel()

                # if canceled during calibration, abort
                if cancel_task2 in done:
                    cal_task.cancel()
                    return

                # calibration finished
                self.status_signal.emit("✓ Calibration complete!")
                self.status_signal.emit("CAL_DONE")

                # 5) prediction loop
                while self._is_running:
                    await asyncio.sleep(0.01)

        except Exception as e:
            self.error_signal.emit(str(e))

    @pyqtSlot()
    def start_calibration(self):
        """Called by the wizard when the user clicks Start/OK."""
        self._cal_start.set()

    @pyqtSlot()
    def cancel_calibration(self):
        """Called by the wizard when the user clicks Cancel."""
        self._cancel_cal.set()

    def run(self):
        asyncio.run(self.run_async())

    def stop(self):
        """Called from the GUI when Stop Exercise is pressed."""
        self._is_running = False
        # also abort any in-flight calibration
        self._cancel_cal.set()

###############################################################################
# Calibration Assistant Dialog
###############################################################################
class CalibrationDialog(QMainWindow):
    proceed_clicked = pyqtSignal()  # Start/OK
    cancel_clicked  = pyqtSignal()  # Cancel

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calibration Assistant")
        self.resize(520, 340)

        central = QWidget(self)
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Logo
        self.logo = QLabel()
        logo_pix = QPixmap(os.path.join(MEDIA_PATH, "SetupAssistant.png"))
        self.logo.setPixmap(logo_pix.scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))
        self.logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.logo)

        # Headline
        self.head = QLabel("Calibration Assistant")
        self.head.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold))
        self.head.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(self.head)

        # Body
        self.body = QLabel(
            "You need to calibrate your EMG armband.\n"
            "Follow the on-screen instructions and\n"
            "we'll get you up-and-running in no time.\n"
        )
        self.body.setFont(QFont(GARAMOND_BOOK_FAMILY, 14))
        self.body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.body.setWordWrap(True)
        vbox.addWidget(self.body)

        # Gesture icon
        self.icon = QLabel()
        self.icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon.hide()
        vbox.addWidget(self.icon)

        # Countdown
        self.timer_lbl = QLabel("")
        self.timer_lbl.setFont(QFont(GARAMOND_BOOK_FAMILY, 22))
        self.timer_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_lbl.hide()
        vbox.addWidget(self.timer_lbl)

        # Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.ok_btn = QPushButton("Start")
        self.ok_btn.clicked.connect(self._on_ok)
        self.NSEWidgets_emboss(self.ok_btn)
        btn_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.NSEWidgets_emboss(self.cancel_btn)
        btn_layout.addWidget(self.cancel_btn)

        vbox.addLayout(btn_layout)

        self.countdown = None
        self._cancelled = False
        self._finished   = False    # NEW: set when calibration really finishes

    def closeEvent(self, event):
        # Treat window-close (“X”) as Cancel
        if self._finished:
            return super().closeEvent(event)
        if not self._cancelled:
            reply = QMessageBox.question(
                self, "Cancel Calibration?",
                "Are you sure you want to cancel Calibration?\n\n"
                "Quitting Calibration Assistant cancels the calibration process "
                "and prevents you from using NeuroSyn ReTrain.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                event.ignore()
                self._show_cancelled_screen()
            else:
                event.ignore()
        else:
            event.accept()

    @pyqtSlot()
    def _on_ok(self):
        if not self._cancelled:
            self.proceed_clicked.emit()
            self.ok_btn.setEnabled(False)
        else:
            self.close()

    @pyqtSlot()
    def _on_cancel(self):
        reply = QMessageBox.question(
            self, "Cancel Calibration?",
            "Are you sure you want to cancel Calibration?\n\n"
            "Quitting Calibration Assistant cancels the calibration process "
            "and prevents you from using NeuroSyn ReTrain.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._show_cancelled_screen()

    def _show_cancelled_screen(self):
        self._cancelled = True

        # reset UI
        self.logo.show()
        self.icon.hide()
        self.timer_lbl.hide()
        self.cancel_btn.hide()

        # update text
        self.head.setText("Calibration cancelled")
        self.body.setText(
            "Calibration Assistant was interrupted and calibration has been cancelled.\n\n"
            "Click 'OK' to exit."
        )

        # re-wire OK to close
        self.ok_btn.setEnabled(True)
        try: self.ok_btn.clicked.disconnect()
        except: pass
        self.ok_btn.setText("OK")
        self.NSEWidgets_emboss(self.ok_btn)
        self.ok_btn.clicked.connect(self.close)

        # notify owner
        self.cancel_clicked.emit()

    def show_gesture(self, title: str, seconds: int, icon_file: str):
        self.logo.hide()
        self.head.setText(title)
        self.body.setText(f"Hold for {seconds} s …")

        pix = QPixmap(os.path.join(MEDIA_PATH, "gestures", icon_file))
        self.icon.setPixmap(pix.scaled(140, 140,
                                       Qt.AspectRatioMode.KeepAspectRatio,
                                       Qt.TransformationMode.SmoothTransformation))
        self.icon.show()
        self.timer_lbl.show()

        if self.countdown:
            self.countdown.stop()
        self.time_left = seconds
        self.timer_lbl.setText(str(self.time_left))
        self.countdown = QTimer(self)
        self.countdown.timeout.connect(self._tick)
        self.countdown.start(1000)

    def NSEWidgets_emboss(self, btn: QPushButton):
        """
        Apply a subtle white emboss effect under button text
        to mimic SynOS’s ‘Textured’ look.
        """
        effect = QGraphicsDropShadowEffect(btn)
        effect.setBlurRadius(1)
        effect.setOffset(0, 1)
        effect.setColor(QColor(255, 255, 255, 80))
        btn.setGraphicsEffect(effect)

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
        self.setWindowTitle("NeuroSyn ReTrain")
        self.resize(600, 400)
        self._create_menubar()

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Prediction label uses ITC Garamond Light
        self.prediction_label = QLabel("No prediction yet.")
        self.prediction_label.setFont(QFont(GARAMOND_LIGHT_FAMILY, 18))
        layout.addWidget(self.prediction_label)
        
        # ─── Signal Strength bar + percent label ────────────────────────
        strength_layout = QHBoxLayout()
        # “Signal Strength:” label
        signal_lbl = QLabel("Signal Strength:")
        signal_lbl.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 14, QFont.Weight.Normal, True))
        strength_layout.addWidget(signal_lbl)

        # ProgressBar itself (hide its internal text)
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)      # ← no overlay text
        self.progress_bar.setFixedHeight(20)
        strength_layout.addWidget(self.progress_bar, 1)  # stretch factor

        # New QLabel to show “0%” → “100%” to the right
        self.progress_pct = QLabel("0%")
        self.progress_pct.setFont(QFont(GARAMOND_LIGHT_ITALIC_FAMILY, 14, QFont.Weight.Normal, True))
        strength_layout.addWidget(self.progress_pct)

        layout.addLayout(strength_layout)
        
        self.start_button = QPushButton("Start Exercise")
        self.start_button.setFont(QApplication.font("QPushButton"))
        self.NSEWidgets_emboss(self.start_button)
        self.start_button.clicked.connect(self.start_prediction)
        layout.addWidget(self.start_button)
        
        self.stop_button = QPushButton("Stop Exercise")
        self.stop_button.setFont(QApplication.font("QPushButton"))
        self.NSEWidgets_emboss(self.stop_button)
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

    def NSEWidgets_emboss(self, btn: QPushButton):
        """
        Apply a subtle white emboss effect under button text
        to mimic SynOS’s ‘Textured’ look.
        """
        effect = QGraphicsDropShadowEffect(btn)
        effect.setBlurRadius(1)
        effect.setOffset(0, 1)
        effect.setColor(QColor(255, 255, 255, 80))
        btn.setGraphicsEffect(effect)

    def _create_menubar(self):
        menubar = self.menuBar()
        if platform.system() == "Darwin":
            about_menu = menubar.addMenu("NeuroSyn")
        else:
            about_menu = menubar.addMenu("Help")

        about_action = QAction("About NeuroSyn", self)
        about_action.setMenuRole(QAction.MenuRole.AboutRole)
        about_action.triggered.connect(self.show_about_dialog)
        about_menu.addAction(about_action)

    def show_about_dialog(self):
        from PyQt6.QtWidgets import QMessageBox
        about_box = QMessageBox(self)
        about_box.setWindowTitle("About NeuroSyn ReTrain")
        about_box.setText(
            "NeuroSyn ReTrain\n"
            "Developed by Syndromatic Inc. / Kavish Krishnakumar\n\n"
            "A home-based physiotherapy assistant using surface EMG\n"
            "and real-time AI-powered feedback to guide your exercises."
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
    
    @pyqtSlot()
    def _relay_cancel_cal(self):
        # tell worker to abort calibration/prediction
        if self.worker:
            self.worker.cancel_calibration()
        # show cancelled dialog if still visible
        if hasattr(self, 'cal_dlg') and self.cal_dlg.isVisible():
            self.cal_dlg._show_cancelled_screen()
        dlg = self.cal_dlg
        if not getattr(dlg, '_has_logged_cancel', False):
            self.log_area.append("Calibration cancelled by user.")
            dlg._has_logged_cancel = True

            try:
                # disconnect so we don’t log again
                self.cal_dlg.cancel_clicked.disconnect(self._relay_cancel_cal)
            except Exception:
                pass

    def update_progress(self, value):
        self.progress_bar.setValue(value)
        self.progress_pct.setText(f"{value}%")
    
    @pyqtSlot()                      # relay that runs in the GUI thread
    def _relay_start_cal(self):
        self.worker.start_calibration()     # safe: Qt auto-queues to worker thread

        # ---------------------------------------------------------------- status
    def handle_status(self, text: str):
        """Drive the calibration wizard and log everything else."""
        # 1) show the wizard intro
        if text == "READY_FOR_CAL":
            if self.connect_window:
                self.connect_window.close()
                self.connect_window = None

            self.cal_dlg = CalibrationDialog()
            self.cal_dlg.proceed_clicked.connect(
                self._relay_start_cal,
                Qt.ConnectionType.QueuedConnection
            )
            self.cal_dlg.cancel_clicked.connect(
                self._relay_cancel_cal,
                Qt.ConnectionType.QueuedConnection
            )
            self.cal_dlg.logo.show()
            self.cal_dlg.ok_btn.setText("Start")
            self.cal_dlg.ok_btn.show()
            self.cal_dlg.cancel_btn.show()
            self.cal_dlg.show()
            return

        # 2) per-gesture cue (CUE|cid|name|secs)
        if text.startswith("CUE|"):
            self.cal_dlg.logo.hide()
            _, cid, gname, secs = text.split("|")
            icon_file = ICON_PATHS[int(cid)]
            self.cal_dlg.show_gesture(gname.title(), int(secs), icon_file)
            return

        # 3) calibration finished
        if text == "CAL_DONE":
            self.cal_dlg.icon.hide()
            self.cal_dlg.timer_lbl.hide()
            self.cal_dlg.logo.show()
            self.cal_dlg.head.setText("Calibration complete!")
            self.cal_dlg.body.setText("You’re ready to go.")
            self.cal_dlg.ok_btn.hide()
            self.cal_dlg.cancel_btn.hide()
            self.cal_dlg._finished = True
            QTimer.singleShot(1500, self.cal_dlg.close)
            return

        # 4) any other status → log it
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
        self.setWindowTitle("Welcome to NeuroSyn ReTrain")
        self.resize(600, 500)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # App Icon
        icon_label = QLabel()
        pixmap = QPixmap(os.path.join(MEDIA_PATH, "syn app light.png"))
        scaled_pixmap = pixmap.scaled(120, 120,
                                      Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(scaled_pixmap)
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        # Welcome Title
        welcome_label = QLabel("Welcome")
        welcome_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 30, QFont.Weight.Bold))
        welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(welcome_label)

        # Compute its pixel width so our button matches
        fm = QFontMetrics(welcome_label.font())
        title_width = fm.horizontalAdvance(welcome_label.text())

        # Intro Text
        intro_text = QLabel(
            "Welcome to NeuroSyn ReTrain,\n"
            "A home-based physiotherapy assistant that uses your EMG armband\n"
            "to monitor muscle activity and guide you through exercises.\n\n\n"
            "Please connect your EMG device, then click Start to begin.\n"
        )
        intro_text.setFont(QFont(GARAMOND_BOOK_FAMILY, 14))
        intro_text.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(intro_text)

        # Get Started Button
        self.next_button = QPushButton("Get Started")
        self.next_button.setFont(QApplication.font("QPushButton"))
        self.next_button.setFixedWidth(title_width)
        self.next_button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed
        )
        self.NSEWidgets_emboss(self.next_button)
        self.next_button.clicked.connect(self.gotoMainWindow)
        layout.addWidget(self.next_button, alignment=Qt.AlignmentFlag.AlignCenter)

        layout.addStretch(1)

        # Footer
        copyright_label = QLabel(
            "Copyright © 2025 Syndromatic Inc. / Kavish Krishnakumar"
        )
        copyright_label.setFont(QFont(GARAMOND_BOOK_FAMILY, 10))
        copyright_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(copyright_label)

    def NSEWidgets_emboss(self, btn: QPushButton):
        """
        Apply a subtle white emboss effect under button text
        to mimic SynOS’s ‘Textured’ look.
        """
        effect = QGraphicsDropShadowEffect(btn)
        effect.setBlurRadius(1)
        effect.setOffset(0, 1)
        effect.setColor(QColor(255, 255, 255, 80))
        btn.setGraphicsEffect(effect)

    def gotoMainWindow(self):
        self.main_window = EMGControllerMainWindow()
        self.main_window.show()
        self.close()

###############################################################################
# Main Execution
###############################################################################
def main():
    app = QApplication(sys.argv)
        # ─── auto‐detect light/dark via window background lightness ───
    wincol = app.palette().color(QPalette.ColorRole.Window)
    is_dark = wincol.lightness() < 128

    app.setStyle(QStyleFactory.create("Fusion"))
    pal = QPalette()

    if not is_dark:
        # ─── Light mode ────────────────────────────────────────────
        pal.setColor(QPalette.ColorRole.Window,          QColor(240, 240, 240))
        pal.setColor(QPalette.ColorRole.WindowText,      QColor( 33,  33,  33))
        pal.setColor(QPalette.ColorRole.Base,            QColor(255, 255, 255))
        pal.setColor(QPalette.ColorRole.AlternateBase,   QColor(240, 240, 240))
        pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(255, 255, 220))
        pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(  0,   0,   0))
        pal.setColor(QPalette.ColorRole.Text,            QColor( 33,  33,  33))
        pal.setColor(QPalette.ColorRole.Button,          QColor(240, 240, 240))
        pal.setColor(QPalette.ColorRole.ButtonText,      QColor( 33,  33,  33))
        pal.setColor(QPalette.ColorRole.BrightText,      QColor(255,   0,   0))
        pal.setColor(QPalette.ColorRole.Highlight,       QColor( 76, 163, 224))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    else:
        # ─── Dark mode ─────────────────────────────────────────────
        pal.setColor(QPalette.ColorRole.Window,          QColor( 45,  45,  45))
        pal.setColor(QPalette.ColorRole.WindowText,      QColor(220, 220, 220))
        pal.setColor(QPalette.ColorRole.Base,            QColor( 30,  30,  30))
        pal.setColor(QPalette.ColorRole.AlternateBase,   QColor( 45,  45,  45))
        pal.setColor(QPalette.ColorRole.ToolTipBase,     QColor(255, 255, 220))
        pal.setColor(QPalette.ColorRole.ToolTipText,     QColor(  0,   0,   0))
        pal.setColor(QPalette.ColorRole.Text,            QColor(220, 220, 220))
        pal.setColor(QPalette.ColorRole.Button,          QColor( 60,  60,  60))
        pal.setColor(QPalette.ColorRole.ButtonText,      QColor(220, 220, 220))
        pal.setColor(QPalette.ColorRole.BrightText,      QColor(255,   0,   0))
        pal.setColor(QPalette.ColorRole.Highlight,       QColor( 76, 163, 224))
        pal.setColor(QPalette.ColorRole.HighlightedText, QColor(255, 255, 255))
    app.setPalette(pal)

    # ─── Light-mode buttons ─────────────────────────────────────────────
    light_btn_qss = """
    QPushButton {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba(250,250,250,255), stop:1 rgba(230,230,230,255));
      color: #202020;
      border: 1px solid #c0c0c0;
      border-radius: 6px;
      padding: 6px 14px;
    }
    QPushButton:hover {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba(240,240,240,255), stop:1 rgba(215,215,215,255));
    }
    QPushButton:pressed {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba(220,220,220,255), stop:1 rgba(195,195,195,255));
    }
    QPushButton:disabled {
      background: rgba(245,245,245,255);
      color: rgba(160,160,160,255);
      border: 1px solid rgba(200,200,200,255);
    }
    """

    # ─── Dark-mode buttons (slightly lighter) ─────────────────────────────
    dark_btn_qss = """
    QPushButton {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba(85,85,85,255), stop:1 rgba(51,51,51,255));
      color: #eeeeee;
      border: 1px solid #202020;
      border-radius: 6px;
      padding: 6px 14px;
    }
    QPushButton:hover {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba(100,100,100,255), stop:1 rgba(80,80,80,255));
    }
    QPushButton:pressed {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba(70,70,70,255), stop:1 rgba(40,40,40,255));
    }
    QPushButton:disabled {
      background: rgba(70,70,70,255);
      color: rgba(120,120,120,255);
      border: 1px solid rgba(30,30,30,255);
    }
    """

    # ─── Shared styling for progress bar & log area ────────────────────
    shared_qss = """
    /* Progress Bar */
    QProgressBar {
      border: 1px solid #707070;
      border-radius: 5px;
      text-align: right;       /* percentage on the right */
      padding-right: 4px;
      background: transparent;
    }
    QProgressBar::chunk {
      background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
          stop:0 rgba( 76,163,224,255), stop:1 rgba( 33,123,200,255));
      border-radius: 5px;
    }

    /* Log area */
    QTextEdit {
      border: 1px solid #505050;
      border-radius: 4px;
    }
    """

    # ─── apply the combined stylesheet ───────────────────────────────────
    final_qss = (dark_btn_qss if is_dark else light_btn_qss) + shared_qss
    app.setStyleSheet(final_qss)
    
    # Set the organization & application name for proper macOS labeling
    app.setOrganizationName("Syndromatic Inc.")
    app.setApplicationName("NeuroSyn Retrain")    
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
