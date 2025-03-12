import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the data
data = pd.read_csv('data/emg_gestures_data_20241209_142310.csv')

# Create figure with subplots
fig, axes = plt.subplots(4, 2, figsize=(20, 25))
fig.suptitle('EMG Sensor Comparisons Across Gestures', fontsize=16, y=0.92)

# Define gestures and their properties
gestures = {
    'thumbs-up': {'id': 0, 'color': 'blue'},
    'peace-sign': {'id': 2, 'color': 'green'},
    'gun-fingers': {'id': 3, 'color': 'red'}
}

# Plot each sensor
for sensor_idx in range(8):
    row = sensor_idx // 2
    col = sensor_idx % 2
    ax = axes[row, col]
    
    # Plot each gesture for this sensor
    for gesture_name, gesture_info in gestures.items():
        # Get data for this gesture
        gesture_data = data[data['gesture_id'] == gesture_info['id']]
        # Create sequential indices
        indices = range(len(gesture_data))
        # Plot the sensor data
        ax.plot(indices, 
                gesture_data[f's{sensor_idx + 1}'], 
                label=gesture_name,
                color=gesture_info['color'],
                alpha=0.8)
    
    # Customize the subplot
    ax.set_title(f'Sensor {sensor_idx + 1}')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('EMG Value')
    ax.grid(True, alpha=0.3)
    ax.legend()

# Adjust layout
plt.tight_layout()
plt.savefig('images/sensor_values_for_each_gesture.png')
plt.show()