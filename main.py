import json
import os
from datetime import datetime, timedelta

import numpy as np
from pydub import AudioSegment
from scipy.signal import sosfilt

from src.internal.audio_percentiles import AudioPercentiles
from src.internal.audio_plotter import AudioPlotter
from src.internal.audio_processor import AudioProcessor
from src.internal.audio_segments import AudioSegments


def plot_options(plotter, output_path, processor, segment_data, segment_signal, index, sample_rate):

    laeq_1s = list(map(float, segment_data["LAeq [dBA] [s]"].split(",")))
    laf_1s = list(map(float, segment_data["LAF [dBA] [s]"].split(",")))
    laeq_avg = segment_data["LAeq avg [dBA]"]
    laeq_alt = segment_data["LAeq avg alt [dBA]"]
    laf_avg = segment_data["LAF avg [dBA]"]

    # Recalculate full-resolution LAF for plotting
    segment = processor.highpass_filter(segment_signal, sample_rate)
    filtered = sosfilt(processor.apply_a_weighting_filter(sample_rate), segment)
    squared = filtered ** 2
    _, laf_full = processor.calculate_laf(squared, sample_rate)

    plotter.plot_segment_levels(output_path, index, laeq_1s, laf_1s, laf_full, sample_rate)
    plotter.plot_laf_full_waveform(output_path, index, laf_full, sample_rate)
    plotter.plot_summary_bar(output_path, index, laeq_avg, laeq_alt, laf_avg)

def generate_metadata(audio_file_path, metadata_output_path, samples=None, sample_rate=None):

    if os.path.exists(metadata_output_path):
        print("Metadata already exists, skipping generation.")
        with open(metadata_output_path, "r") as f:
            existing_data = json.load(f)
        return existing_data["data"]

    print("Generating metadata ...")

    ap = AudioProcessor()
    perc = AudioPercentiles()

    if samples is None or sample_rate is None:
        wav_path = ap.ensure_wav_format(audio_file_path)
        samples, sample_rate = ap.load_audio(wav_path)

    segment_samples = sample_rate * 60
    segments = [samples[i:i + segment_samples] for i in range(0, len(samples), segment_samples)]

    output_data = []
    base_timestamp = datetime.now().replace(second=0, microsecond=0)

    for i, segment in enumerate(segments):
        print(f"\nProcessing segment: {i+1}")
        segment_time = base_timestamp + timedelta(seconds=i * 60)

        print(f"Calculating LAeq")
        segment_data = ap.process_segment(segment, sample_rate, segment_time, i)

        if segment_data:
            print(f"Calculating percentiles")
            with_percentile_data = perc.calculate_percentiles(segment_data)

            segment_data.update(with_percentile_data)
            output_data.append(segment_data)

    json_result = {"data": output_data}

    with open(metadata_output_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\nMetadata saved to: {metadata_output_path}")

    return output_data

def generate_segments(metadata_path, segment_path, recording_name, audio_clip, sr, ap = None):

    if ap is None:
        ap = AudioPlotter()

    a = AudioSegments()
    data_by_index = a.generate_dataframe(metadata_path)

    audio_segment = AudioSegment(
        audio_clip.tobytes(),
        frame_rate=sr,
        sample_width=audio_clip.dtype.itemsize,
        channels=1
    )

    for index, df in data_by_index.items():
        if "LAeq" not in df.columns or "LA90" not in df.columns:
            continue

        dynamic_threshold = a.calculate_dynamic_threshold(df)
        events = a.detect_events(
            sound_levels=df["LAeq"].values,
            baseline_levels=df["LA90"].astype(float).values,
            dynamic_threshold=dynamic_threshold,
        )

        if not events:
            print("No significant sound events detected.")
            continue

        segments = a.extract_segments(audio_segment, events, segment_path, index)

        # Normalize for plotting
        plot_audio = audio_clip.astype(np.float32)
        plot_audio /= np.max(np.abs(plot_audio))

        plot_path = f"results/plots/{recording_name}/LAeq/"
        os.makedirs(plot_path, exist_ok=True)

        ap.plot_waveform_with_segments(
            y=plot_audio,
            sr=sr,
            segments=segments,
            title=f"Waveform: Baseline LA90 with offset {dynamic_threshold} dB",
            save_path=os.path.join(plot_path, f"LA90_{dynamic_threshold}db.png"),
        )

def main():
    print("Sound segments")

    recording_name = "sample"
    audio_file_path = f"data/{recording_name}"
    metadata_path = f"data/metadata_{recording_name}.json"
    plot_output_path = f"results/plots/{recording_name}"
    segment_output_path = f"results/segments/{recording_name}"

    os.makedirs(plot_output_path, exist_ok=True)
    os.makedirs(segment_output_path, exist_ok=True)

    ap = AudioProcessor()

    wav_path = ap.ensure_wav_format(audio_file_path)
    norm_samples, sr = ap.load_audio(wav_path)

    audio_segment = AudioSegment.from_file(wav_path).set_channels(1)
    raw_samples = np.array(audio_segment.get_array_of_samples(), dtype=np.int16)

    metadata = generate_metadata(audio_file_path, metadata_path, samples=norm_samples, sample_rate=sr)

    print(f"Plotting informational graphs ...")
    plotter = AudioPlotter()

    segment_duration_samples = sr * 60
    for index, data in enumerate(metadata):
        segment_signal = norm_samples[index * segment_duration_samples : (index + 1) * segment_duration_samples]
        plot_options(plotter, plot_output_path, ap, data, segment_signal, index, sr)

    print("\nSegmenting audio ...")
    generate_segments(
        ap=plotter,
        metadata_path=metadata_path,
        segment_path=segment_output_path,
        recording_name=recording_name,
        audio_clip=raw_samples,
        sr=sr
    )

    print("Done")


if __name__ == "__main__":
    main()