import csv
from pathlib import Path

class GestureDataCollector:
    def __init__(self, data_dir="data", collection_time=5):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.collection_time = collection_time
        self.data_collection = []

    def add_data(self, emg_data):
        self.data_collection.append(emg_data)

    def clear_data(self):
        self.data_collection = []

    def save_data(self, filename, columns, all_data):
        file_path = self.data_dir / filename

        with open(file_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(columns)
            writer.writerows(all_data)
            
        return file_path
