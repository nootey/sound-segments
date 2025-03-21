import os

import matplotlib.pyplot as plt
import numpy as np


class AudioPlotter:
    def __init__(self):
        print("AudioPlotter initialized")

    def plot_segment_levels(self, output_path, index, laeq_1s, laf_1s, laf_full, sample_rate):
        time_seconds = np.arange(len(laeq_1s))
        time_full = np.arange(len(laf_full)) / sample_rate

        plt.figure(figsize=(12, 6))
        plt.plot(time_seconds, laeq_1s, label="LAeq [dBA/s]", marker='o')
        plt.plot(time_seconds, laf_1s, label="LAF [dBA/s]", linestyle='--', alpha=0.7)
        plt.plot(time_full, laf_full, label="LAF (full res)", linewidth=0.5, alpha=0.4)

        plt.axhline(np.mean(laeq_1s), color="gray", linestyle="--", label="LAeq avg")
        plt.axhline(np.mean(laf_full), color="orange", linestyle=":", label="LAF avg")

        plt.title(f"Segment {index + 1} - SPL Time Series")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Sound Level (dBA)")
        plt.legend(loc="upper right")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"segment_{index + 1}_spl_time.png"))

    def plot_laf_full_waveform(self, output_path, index, laf_full, sample_rate):
        time_full = np.arange(len(laf_full)) / sample_rate
        plt.figure(figsize=(12, 4))
        plt.plot(time_full, laf_full, label="LAF (full resolution)", linewidth=0.5)
        plt.xlabel("Time (s)")
        plt.ylabel("dBA")
        plt.title("Full-resolution LAF waveform")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"segment_{index + 1}_laf_full.png"))

    def plot_summary_bar(self, output_path, index, laeq_avg, laeq_alt, laf_avg):
        labels = ['LAeq avg', 'LAeq alt', 'LAF avg']
        values = [laeq_avg, laeq_alt, laf_avg]

        plt.figure(figsize=(6, 5))
        bars = plt.bar(labels, values, color=["steelblue", "slategray", "orange"])
        plt.ylabel("dBA")
        plt.title("Average Levels Summary")
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f'{yval:.1f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f"segment_{index + 1}_summary.png"))