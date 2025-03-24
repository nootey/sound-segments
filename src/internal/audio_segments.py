import re

import numpy as np
import pandas as pd


class AudioSegments:
    def generate_dataframe(self, metadata_path):
        import json

        def clean_key(key):
            return re.sub(r"\s*\[.*?\]", "", key).strip()

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        all_recordings = {}

        for entry in metadata["data"]:
            index = entry.get("Index")

            per_second_fields = {
                clean_key(key): list(map(float, entry[key].split(",")))
                if isinstance(entry[key], str) else [entry[key]]
                for key in entry
                if key.endswith("[s]") and clean_key(key) != "Duration"
            }

            df = pd.DataFrame(per_second_fields)
            df["Index"] = index

            exclude_list = ["Index", "Timestamp", "Duration", "Percentile Method"]

            for key in entry:
                if not key.endswith("[s]") and key not in exclude_list:
                    df[clean_key(key)] = entry[key]

            all_recordings[index] = df

        return all_recordings

    def calculate_dynamic_threshold(self, df):
        """
        Adapts the threshold based on:
        - The difference between peak loudness (LAF max) and background noise (LA90)
        - The variability of loudness over time (standard deviation of LAeq)

        A higher standard deviation reduces the threshold, ensuring sensitivity adapts
        to noisier environments without triggering too many false positives.
        """

        laeq_values = df["LAeq"].values

        # Measure how much loudness fluctuates over time
        std_dev = np.std(laeq_values)

        # Measure the difference between the loudest peak and the background noise level
        peak_to_background = df["LAF max"].iloc[0] - df["LA90"].iloc[0]

        initial_threshold_offset = peak_to_background / (1 + (0.5 * std_dev))

        # Constrain the threshold within a reasonable range
        initial_threshold_offset = max(2, min(initial_threshold_offset, 8))

        print(f"Dynamic threshold calculated: {initial_threshold_offset:.2f} dB "
              f"(σ={std_dev:.2f}, Peak-BG={peak_to_background:.2f})")

        return round(initial_threshold_offset, 2)