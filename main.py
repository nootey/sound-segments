import json
from datetime import datetime, timedelta

import numpy as np
from pydub import AudioSegment
from scipy.signal import sosfilt
from src.internal.audio_processor import AudioProcessor

def main():
    print("Sound segments")

    ap = AudioProcessor()

    recording_name = "sample"
    audio_file_path = f"data/{recording_name}"
    output_path = f"data/metadata_{recording_name}.json"

    # Ensure WAV format
    wav_path = ap.ensure_wav_format(audio_file_path)

    # Load the audio file
    samples, sample_rate = ap.load_audio(wav_path)

    segment_samples = sample_rate * 60
    segments = [samples[i:i + segment_samples] for i in range(0, len(samples), segment_samples)]

    print("Applying A-weighting filter")
    sos = ap.apply_a_weighting_filter(sample_rate)

    output_data = []
    base_timestamp = datetime.now().replace(second=0, microsecond=0)

    for i, segment in enumerate(segments):
        print(f"Processing segment: {i+1}")
        segment_time = base_timestamp + timedelta(seconds=i * 60)
        segment_data = ap.process_segment(segment, sample_rate, sos, segment_time, i)
        output_data.append(segment_data)

    json_result = {"data": output_data}


    with open(output_path, "w") as f:
        json.dump(json_result, f, indent=2)

    print(f"\nLAeq JSON written to: {output_path}")


if __name__ == "__main__":
    main()