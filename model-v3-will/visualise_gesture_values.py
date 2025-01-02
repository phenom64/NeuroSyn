import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read and process the data
def process_data(data):
    # Convert time to relative time starting from 0
    data['relative_time'] = data['time'] - data['time'].min()
    return data

# Create subplot for a specific gesture
def plot_gesture(ax, data, gesture_name, show_legend=True):
    sensor_columns = [f's{i}' for i in range(1, 9)]
    
    for sensor in sensor_columns:
        ax.plot(data['relative_time'], data[sensor], label=f'Sensor {sensor}')
    
    ax.set_title(f'EMG Signals - {gesture_name}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('EMG Value')
    ax.set_yticks(np.arange(0, 1250, 100))
    if show_legend:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

# Read the data
data = pd.read_csv('data/emg_gestures_data_20241209_142310.csv')

# Process data
data = process_data(data)

# Create figure with subplots
fig = plt.figure(figsize=(20, 15))

# Plot individual gestures
gestures = {
    'thumbs-up': 0,
    'peace-sign': 2,
    'gun-fingers': 3
}

# Create subplots for each gesture
for i, (gesture_name, gesture_id) in enumerate(gestures.items(), 1):
    gesture_data = data[data['gesture_id'] == gesture_id]
    ax = plt.subplot(4, 1, i)
    plot_gesture(ax, gesture_data, gesture_name, show_legend=(i==1))

# Create combined plot with sequential indices
ax = plt.subplot(4, 1, 4)
for gesture_name, gesture_id in gestures.items():
    gesture_data = data[data['gesture_id'] == gesture_id]
    # Create sequential indices for this gesture
    indices = range(len(gesture_data))
    # Plot using sequential indices instead of time
    ax.plot(indices, gesture_data['s1'], label=f'{gesture_name} (Sensor 1)')

ax.set_title('Combined EMG Signals (Sensor 1)')
ax.set_xlabel('Sample Index')
ax.set_ylabel('EMG Value')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

# Adjust layout
plt.tight_layout()
plt.savefig('images/emg_signals.png')
plt.show()

