import sys
import asyncio
import time
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QThread, pyqtSignal, QObject

# Import components from your prediction module and constants.
from prediction import GesturePredictor
from constants import MYO_ADDRESS
from pymyo import Myo
from pymyo.types import EmgMode, SleepMode
from bleak import BleakScanner

class PredictionWorker(QObject):
    # Signal to send prediction text to the GUI.
    prediction_signal = pyqtSignal(str)
    error_signal = pyqtSignal(str)
    
    def __init__(self, predictor: GesturePredictor):
        super().__init__()
        self.predictor = predictor
        self._is_running = True

    async def run_async(self):
        try:
            # Find and connect to the Myo device.
            myo_device = await BleakScanner.find_device_by_address(MYO_ADDRESS)
            if not myo_device:
                self.error_signal.emit("Could not find Myo device.")
                return

            # Use the Myo context manager.
            async with Myo(myo_device) as myo:
                # Set up the device.
                await asyncio.sleep(0.5)
                await myo.set_sleep_mode(SleepMode.NEVER_SLEEP)
                await asyncio.sleep(0.25)

                # Define the callback for incoming EMG data.
                @myo.on_emg_smooth
                def on_emg_smooth(emg_data):
                    prediction = self.predictor.add_data(emg_data)
                    if prediction:
                        class_id, gesture_name = prediction
                        # Emit the new prediction.
                        self.prediction_signal.emit(
                            f"Predicted: {gesture_name} (class {class_id})"
                        )

                await myo.set_mode(emg_mode=EmgMode.SMOOTH)
                # Run the loop until stopped.
                while self._is_running:
                    await asyncio.sleep(0.01)
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self._is_running = False

    def run(self):
        # Run the asynchronous loop in this thread.
        asyncio.run(self.run_async())

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Artemis Myo EMG Prediction")
        self.resize(400, 200)
        self.layout = QVBoxLayout(self)
        
        # Label to show the prediction.
        self.prediction_label = QLabel("No prediction yet.")
        self.layout.addWidget(self.prediction_label)
        
        # Button to start the prediction.
        self.start_button = QPushButton("Start Prediction")
        self.start_button.clicked.connect(self.start_prediction)
        self.layout.addWidget(self.start_button)
        
        # Button to stop the prediction.
        self.stop_button = QPushButton("Stop Prediction")
        self.stop_button.clicked.connect(self.stop_prediction)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)
        
        self.worker_thread = QThread()
        self.worker = None

    def start_prediction(self):
        # Create an instance of your predictor.
        predictor = GesturePredictor(prediction_interval=1.0)
        self.worker = PredictionWorker(predictor)
        self.worker.moveToThread(self.worker_thread)
        
        # Connect thread start signal to the worker's run method.
        self.worker_thread.started.connect(self.worker.run)
        self.worker.prediction_signal.connect(self.update_prediction)
        self.worker.error_signal.connect(self.handle_error)
        self.worker_thread.start()
        
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)

    def stop_prediction(self):
        if self.worker:
            self.worker.stop()
        self.worker_thread.quit()
        self.worker_thread.wait()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)

    def update_prediction(self, text):
        self.prediction_label.setText(text)

    def handle_error(self, error_text):
        self.prediction_label.setText("Error: " + error_text)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
