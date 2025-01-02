import pandas as pd
import numpy as np

# Read the raw data from CSV
df = pd.read_csv('will_profile_raw_data_closed_fist.csv', header=None)

# The data appears to be in columns, where each column represents all values for one sensor
# We need to reshape it so each row contains values for all sensors at a single time point
data = []
num_sensors = 16
num_samples = len(df.columns)  # Each column is a separate sample

# Reshape the data
for i in range(num_samples):
    row_data = df[i].values[:num_sensors]  # Get first 16 values from each column
    data.append(row_data)

# Create DataFrame with proper structure
result_df = pd.DataFrame(data, columns=[f's{i+1}' for i in range(num_sensors)])

# Add id and time columns
result_df.insert(0, 'id', range(len(result_df)))
result_df.insert(1, 'time', [f"{(i * 0.01):.2f}" for i in range(len(result_df))])  # 10ms intervals

# Save to CSV
result_df.to_csv('formatted_will_profile_raw_data_closed_fist.csv', index=False)

print("Data conversion completed. First few rows of the result:")
print(result_df.head())
print(f"\nTotal samples converted: {len(result_df)}")