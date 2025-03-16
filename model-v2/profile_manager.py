import csv
from pathlib import Path

class ProfileManager:
    def __init__(self, profile_dir="profiles"):
        self.profile_dir = Path(profile_dir)
        self.profile_dir.mkdir(exist_ok=True)

    def save_profile(self, profile_name, profile_means):
        profile_path = self.profile_dir / f"{profile_name}_profile.csv"
        with open(profile_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"Sensor{i}" for i in range(1, 17)])
            writer.writerow(profile_means)

    def load_profile(self, profile_name) -> float:
        profile_path = self.profile_dir / f"{profile_name}_profile.csv"
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile {profile_name} not found")
        with open(profile_path, "r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            return [float(value) for value in next(reader)]
