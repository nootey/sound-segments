import json
import os
from datetime import datetime, timedelta

from scipy.signal import sosfilt

from src.internal.audio_percentiles import AudioPercentiles
from src.internal.audio_plotter import AudioPlotter
from src.internal.audio_processor import AudioProcessor

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

def main():
    print("Sound segments")

    ap = AudioProcessor()
    perc = AudioPercentiles()
    plotter = AudioPlotter()

    recording_name = "sample"
    audio_file_path = f"data/{recording_name}"
    metadata_output_path = f"data/metadata_{recording_name}.json"
    plot_output_path = f"results/plots/{recording_name}"
    if not os.path.exists(plot_output_path):
        os.makedirs(plot_output_path)

    # Ensure WAV format
    wav_path = ap.ensure_wav_format(audio_file_path)

    # Load the audio file
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

            print(f"Plotting")
            plot_options(plotter, plot_output_path, ap, segment_data, segment, i, sample_rate)

    json_result = {"data": output_data}

    with open(metadata_output_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\nMetadata saved to: {metadata_output_path}")


if __name__ == "__main__":
    main()