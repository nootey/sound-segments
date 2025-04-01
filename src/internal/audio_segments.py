import os
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

    def detect_events(self, sound_levels, baseline_levels, dynamic_threshold, sustained_offset=5):
        events = []
        in_event = False
        event_start = None

        frame_duration = 1  # 1 second per LAeq point
        frame_rate = 1 / frame_duration
        grace_period_sec = 0.5
        grace_frames = int(grace_period_sec * frame_rate)
        grace_counter = 0

        for i in range(len(sound_levels)):
            current_baseline = float(baseline_levels[i]) + 1.0
            initial_threshold = current_baseline + dynamic_threshold
            sustained_threshold = current_baseline + sustained_offset
            lvl = sound_levels[i]

            if not in_event:
                if lvl > initial_threshold:
                    in_event = True
                    event_start = i
                    grace_counter = 0
            else:
                if lvl < sustained_threshold:
                    grace_counter += 1
                    if grace_counter >= grace_frames and event_start is not None:
                        events.append((event_start, i - grace_counter))
                        in_event = False
                        event_start = None
                else:
                    grace_counter = 0

        if in_event:
            events.append((event_start, len(sound_levels) - 1))

        return events

    def extract_segments(self, audio_segment, events, segment_output_path, index):
        segments = []
        segment_counter = 1

        if os.path.exists(segment_output_path):
            for filename in os.listdir(segment_output_path):
                file_path = os.path.join(segment_output_path, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            print(f"Cleared existing files in: {segment_output_path}")

        for (start_idx, end_idx) in events:
            start_time_sec = max(0, start_idx - 1)
            end_time_sec = min(len(audio_segment) / 1000.0, end_idx + 1)
            duration_sec = end_time_sec - start_time_sec

            if duration_sec < 1:
                print(f"Skipping short segment ({duration_sec:.2f}s)")
                continue

            print(f"Segment {segment_counter}: Start={start_time_sec}s, "
                  f"End={end_time_sec}s, Duration={duration_sec}s")

            segments.append((start_time_sec, end_time_sec))

            start_ms = int(start_time_sec * 1000)
            end_ms = int(end_time_sec * 1000)
            extracted = audio_segment[start_ms:end_ms]

            segment_filename = os.path.join(segment_output_path, f"segment_{index}_{segment_counter}.wav")
            extracted.export(segment_filename, format="wav")
            print(f"Saved segment to: {segment_filename}")

            segment_counter += 1

        return segments