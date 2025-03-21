import json
from datetime import datetime, timedelta

from src.internal.audio_percentiles import AudioPercentiles
from src.internal.audio_processor import AudioProcessor

def main():
    print("Sound segments")

    ap = AudioProcessor()
    perc = AudioPercentiles()

    recording_name = "sample"
    audio_file_path = f"data/{recording_name}"
    output_path = f"data/metadata_{recording_name}.json"

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

    json_result = {"data": output_data}

    with open(output_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\nMetadata saved to: {output_path}")


if __name__ == "__main__":
    main()