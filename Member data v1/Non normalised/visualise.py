import pandas as pd
import matplotlib.pyplot as plt
import os
import math

def plot_emg_sensors(input_folder, selected_sensors=None):
    """
    Visualize EMG sensor data with voltage conversion.
    
    Voltage calculation: ADC_reading_value * (5/256) / 200
    """
    # Read CSV files from input folder
    all_data = []
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(input_folder, filename)
            df = pd.read_csv(file_path)
            
            # Convert ADC readings to voltage
            sensor_cols = [col for col in df.columns if col.startswith('Sensor')]
            for col in sensor_cols:
                df[col] = df[col] * (3.1/256) / 200
            
            all_data.append(df)

    # Combine all dataframes
    df = pd.concat(all_data, ignore_index=True)
    
    # Get all sensor columns if not specified
    if selected_sensors is None:
        selected_sensors = [col for col in df.columns if col.startswith('Sensor')]
    
    # Remove any specified sensors that don't exist in the dataframe
    selected_sensors = [sensor for sensor in selected_sensors if sensor in df.columns]
    
    # If no valid sensors, return
    if not selected_sensors:
        print("No valid sensors found to plot.")
        return
    
    # Calculate grid dimensions
    num_sensors = len(selected_sensors)
    cols = min(4, num_sensors)  # Max 4 columns
    rows = math.ceil(num_sensors / cols)
    
    # Create figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows), squeeze=False)
    fig.suptitle('EMG Sensor Voltage Data')
    
    # Flatten axes for easy iteration
    axes_flat = axes.flatten()
    
    # Plot each selected sensor
    for i, sensor in enumerate(selected_sensors):
        ax = axes_flat[i]
        ax.plot(df.index, df[sensor], label=sensor)
        ax.set_title(f'{sensor} Voltage')
        ax.legend()
        ax.grid(True)
        ax.set_xlabel('Time Point')
        ax.set_ylabel('Voltage (V)')
    
    # Remove any unused subplots
    for j in range(i+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])
    
    plt.tight_layout()
    plt.savefig('sensor_voltage_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

# Example usage
plot_emg_sensors('input', ['Sensor1', 'Sensor2'])  # Plot specific sensors
# plot_emg_sensors('input')  # Plot all sensors