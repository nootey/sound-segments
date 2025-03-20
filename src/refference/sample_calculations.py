import json
import numpy as np
import pandas as pd
from datetime import datetime
from pydub import AudioSegment
import scipy.signal as signal

REF_DB_SPL = 98


def load_audio(file_path):
    """ Load MP3 file and convert to normalized numpy array """
    audio = AudioSegment.from_mp3(file_path).set_channels(1)  # Convert to mono
    samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
    sample_rate = audio.frame_rate

    # Normalize samples to range -1 to 1
    samples /= np.max(np.abs(samples))

    return samples, sample_rate


def a_weighting_filter(sample_rate):
    """ Design an A-weighting filter """
    f1, f2, f3, f4, f5 = 20.6, 107.7, 737.9, 12194, 12194
    A1000 = 1.9997
    numer = [(2 * np.pi * f5) ** 2 * (10 ** (A1000 / 20)), 0, 0, 0, 0]
    denom = np.convolve(
        [1, 4 * np.pi * f4, (2 * np.pi * f4) ** 2],
        [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2]
    )
    denom = np.convolve(
        np.convolve(denom, [1, 2 * np.pi * f3]),
        [1, 2 * np.pi * f2]
    )
    return signal.bilinear(numer, denom, fs=sample_rate)


def c_weighting_filter(sample_rate):
    """ Design a C-weighting filter """
    f1, f2 = 20.6, 12194
    numer = [(2 * np.pi * f2) ** 2, 0, 0]
    denom = np.convolve([1, 4 * np.pi * f2, (2 * np.pi * f2) ** 2], [1, 4 * np.pi * f1, (2 * np.pi * f1) ** 2])
    return signal.bilinear(numer, denom, fs=sample_rate)


def apply_weighting(signal_data, sample_rate, weighting_type="A"):
    """ Apply A or C weighting filter """
    if weighting_type == "A":
        b, a = a_weighting_filter(sample_rate)
    elif weighting_type == "C":
        b, a = c_weighting_filter(sample_rate)
    else:
        raise ValueError("Invalid weighting type. Use 'A' or 'C'.")

    return signal.lfilter(b, a, signal_data)


def calculate_laeq(signal_data, sample_rate):
    """ Compute LAeq per second with proper dB SPL reference """
    chunk_size = sample_rate  # One-second chunks
    num_chunks = len(signal_data) // chunk_size
    laeq_values = []

    for i in range(num_chunks):
        chunk = signal_data[i * chunk_size:(i + 1) * chunk_size]
        rms = np.sqrt(np.mean(chunk ** 2))
        if rms > 0:
            laeq = 20 * np.log10(rms / (2e-5)) + 1.5
        else:
            laeq = -np.inf  # Silence case
        laeq_values.append(laeq)

    return np.array(laeq_values)


def calculate_lcpeak(signal_data, sample_rate):
    """ Compute LCpeak as the highest C-weighted level in the signal """
    chunk_size = sample_rate  # One-second chunks
    num_chunks = len(signal_data) // chunk_size
    lcpeak_values = []

    for i in range(num_chunks):
        chunk = signal_data[i * chunk_size:(i + 1) * chunk_size]
        peak = np.max(chunk)
        if peak > 0:
            lcpeak = 20 * np.log10(peak / (2e-5)) + 2
        else:
            lcpeak = -np.inf
        lcpeak_values.append(lcpeak)

    return np.array(lcpeak_values)


def process_audio(file_path):
    """ Process MP3 file to compute LAeq, LCeq, LCF, and LAF per second """
    samples, sample_rate = load_audio(file_path)

    # Apply filters
    la_weighted = apply_weighting(samples, sample_rate, "A")
    lc_weighted = apply_weighting(samples, sample_rate, "C")

    # Calculate per-second levels
    laeq = calculate_laeq(la_weighted, sample_rate)
    lceq = calculate_laeq(lc_weighted, sample_rate)
    laf = laeq.copy()
    lcf = lceq.copy()
    lcpeak = calculate_lcpeak(lc_weighted, sample_rate)

    df = pd.DataFrame({
        "LAeq": laeq,
        "LCeq": lceq,
        "LAF": laf,
        "LCF": lcf,
        "LCpeak": lcpeak
    })

    return df


def main():
    recording_name = "sample"
    audio_file_path = f"data/{recording_name}.mp3"
    output_json_path = f"data/metadata_calc_{recording_name}.json"

    df = process_audio(audio_file_path)

    LAeq_values = df["LAeq"].values
    LCpeak_values = df["LCpeak"].values
    LA_max = np.max(LAeq_values)
    LA_min = np.min(LAeq_values)
    LC_peak_max = np.max(LCpeak_values)

    percentiles = [99, 95, 90, 80, 70, 60, 50, 40, 30, 20, 10, 5, 1]
    method = "inverted_cdf"
    LA_percentiles = {f"LA{p}": np.percentile(LAeq_values, 100 - p, method=method) for p in percentiles}

    per_second_data = {f"{key} [s]": ", ".join(map(str, values)) for key, values in df.to_dict(orient="list").items()}

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    json_output = {
        "data": [
            {
                "Timestamp": current_timestamp,
                "LAF max": LA_max,
                "LAF min": LA_min,
                "LCpeak avg": LC_peak_max,
                **LA_percentiles,
                **per_second_data
            }
        ]
    }

    with open(output_json_path, "w") as f:
        json.dump(json_output, f, indent=4)

    print(f"Results saved to {output_json_path}")


if __name__ == "__main__":
    main()
