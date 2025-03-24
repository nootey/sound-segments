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
        return

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

            # print(f"Plotting")
            # plot_options(plotter, plot_output_path, ap, segment_data, segment, i, sample_rate)

    json_result = {"data": output_data}

    with open(metadata_output_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\nMetadata saved to: {metadata_output_path}")

def generate_segments(metadata_path, segment_path, recording_name, audio_clip, sr):

    a = AudioSegments()
    ap = AudioPlotter()
    data_by_index = a.generate_dataframe(metadata_path)

    audio_segment = AudioSegment(
        audio_clip.tobytes(),
        frame_rate=sr,
        sample_width=audio_clip.dtype.itemsize,
        channels=1
    )

    for index, df in data_by_index.items():

        dynamic_threshold = a.calculate_dynamic_threshold(df)
        sustained_threshold_offset = 5
        sound_columns = ["LAeq"]
        la_params = ["LA90"]

        for sound_column in sound_columns:
            if sound_column not in df.columns:
                print(f"Skipping {sound_column} — not found in data.")
                continue

            for la_param in la_params:
                if la_param not in df.columns:
                    print(f"Skipping {la_param} (not in metadata).")
                    continue

                la_baseline_values = df[la_param].astype(float).values
                sound_levels = df[sound_column].values

                print(f"Testing {sound_column} with offset {dynamic_threshold} vs {la_param} baseline")

                events = []
                in_event = False
                event_start = None

                frame_duration = 1  # 1 second per LAeq point
                frame_rate = 1 / frame_duration
                grace_period_sec = 0.5
                grace_frames = int(grace_period_sec * frame_rate)
                grace_counter = 0

                for i in range(len(sound_levels)):
                    current_baseline = float(la_baseline_values[i]) + 1.0  # +1dB "safety buffer"
                    initial_threshold = current_baseline + dynamic_threshold
                    sustained_threshold = current_baseline + sustained_threshold_offset
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

                if not events:
                    print("No significant sound events detected.")

                segments = []
                segment_counter = 1

                for (start_idx, end_idx) in events:
                    start_time_sec = max(0, start_idx - 1)
                    end_time_sec = min(len(sound_levels), end_idx + 1)
                    duration_sec = end_time_sec - start_time_sec

                    if duration_sec < 1:
                        print(f"Skipping short segment ({duration_sec:.2f}s)")
                        continue

                    print(f"Segment {segment_counter}: Start={start_time_sec}s, "
                          f"End={end_time_sec}s, Duration={duration_sec}s")

                    segments.append((start_time_sec, end_time_sec))

                    # Export audio segment
                    start_ms = int(start_time_sec * 1000)
                    end_ms = int(end_time_sec * 1000)
                    extracted = audio_segment[start_ms:end_ms]

                    segment_filename = os.path.join(segment_path, f"segment_{index}_{segment_counter}.wav")
                    os.makedirs(segment_path, exist_ok=True)
                    extracted.export(segment_filename, format="wav")
                    print(f"Saved segment to: {segment_filename}")

                    segment_counter += 1

                # Save plot
                plot_path = f"results/plots/{recording_name}/{sound_column}/"
                os.makedirs(plot_path, exist_ok=True)

                plot_audio = audio_clip.astype(np.float32)
                plot_audio /= np.max(np.abs(plot_audio))

                print("Generating waveform plot")

                ap.plot_waveform_with_segments(
                    y=plot_audio,
                    sr=sr,
                    segments=segments,
                    title=f"Waveform: Baseline {la_param} with offset {dynamic_threshold} dB",
                    save_path=os.path.join(plot_path, f"{la_param}_{dynamic_threshold}db.png"),
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

    generate_metadata(audio_file_path, metadata_path, samples=norm_samples, sample_rate=sr)

    print("\nSegmenting audio ...")
    generate_segments(
        metadata_path=metadata_path,
        segment_path=segment_output_path,
        recording_name=recording_name,
        audio_clip=raw_samples,
        sr=sr
    )

    print("Done")



if __name__ == "__main__":
    main()