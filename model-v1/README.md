# MYO EMG Gesture Recognition

This project implements real-time gesture recognition using the Myo armband's EMG (Electromyography) sensors. It can detect different hand gestures by analyzing muscle electrical activity.

## Features

- Real-time EMG data collection
- Gesture classification using neural networks
- Support for multiple gesture classes
- Data collection for training custom models
- Real-time prediction visualization

## Prerequisites

- Python 3.11+
- Myo armbanda
- Required Python packages:
  ```
  bleak==0.22.3
  numpy==2.1.3
  pandas==2.2.3
  pymyo==2.2.3
  tensorflow==2.18.0
  tensorflow_intel==2.18.0
  ```

## Installation

1. Clone this repository
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Connect your Myo armband and note its Bluetooth address

## Getting Your Myo Armband Address

Before using the system, you need to get your Myo armband's Bluetooth address:

1. Put your Myo armband in pairing mode (hold the logo button until it flashes rapidly)
2. On Windows:
   - Open Settings → Bluetooth & other devices
   - Look for your Myo in the list of paired devices
   - The address will be in the format "XX:XX:XX:XX:XX:XX"
3. On macOS/Linux:
   - Open terminal
   - Run `bluetoothctl`
   - Run `scan on`
   - Look for a device named "Myo"
   - Note the address

Update the `MYO_ADDRESS` in `constants.py` with your device's address.

## Project Structure

- `collection.py`: Script for collecting EMG training data
- `predict.py`: Real-time gesture prediction
- `train.ipynb`: Jupyter notebook for training the model
- `classifier.py`: Implementation of the gesture classifier
- `constants.py`: Configuration and constants

## Usage

### 1. Data Collection

To collect training data:

```bash
python collection.py
```

This will:

- Connect to your Myo armband
- Guide you through performing gestures
- Save the collected data as CSV files

### 2. Training the Model

1. Open `train.ipynb` in Jupyter Notebook
2. Run all cells to:
   - Load the collected data
   - Preprocess and clean the data
   - Train the neural network
   - Save the trained model and preprocessing parameters

### 3. Real-time Prediction

To use the trained model for real-time prediction:

```bash
python predict.py
```

This will:

- Connect to your Myo armband
- Load the trained model
- Show real-time gesture predictions

## Gesture Classes

Currently supported gestures:

- Rest (0)
- Closed Fist (1)

You can add more gestures by:

1. Modifying the `CLASSES` dictionary in `constants.py`
2. Collecting data for new gestures
3. Retraining the model

## Troubleshooting

1. **Connection Issues**

   - Ensure the Myo armband is charged
   - Verify the Bluetooth address is correct
   - Try re-pairing the device

2. **Poor Recognition**
   - Collect more training data
   - Ensure proper armband placement
   - Try adjusting the model architecture in `train.ipynb`

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.
