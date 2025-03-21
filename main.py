import numpy as np
from pydub import AudioSegment
from scipy.signal import sosfilt
from src.internal.audio_processor import AudioProcessor

def main():
    print("Sound segments")

    ap = AudioProcessor()

    recording_name = "sample"
    audio_file_path = f"data/{recording_name}"

    # Ensure WAV format
    wav_path = ap.ensure_wav_format(audio_file_path)

    # Load the audio file
    samples, sample_rate = ap.load_audio(wav_path)

    segment_samples = sample_rate * 60
    segments = [samples[i:i + segment_samples] for i in range(0, len(samples), segment_samples)]

    # Apply A-weighting - The A-weighting filter simulates how human ears perceive loudness
    # it gives less weight to very low and very high frequencies, and more to midrange frequencies.
    sos = ap.apply_a_weighting_filter(sample_rate)

    # Process segments
    for i, segment in enumerate(segments):
        filtered = sosfilt(sos, segment)
        squared = filtered ** 2

        print(f"Filtered segment {i + 1}, length: {len(filtered)} samples \n")
        print(f"First 10 squared samples (segment {i + 1}): {squared[:10]}")

        smoothed = ap.apply_exponential_moving_average(squared, sample_rate)
        print(f"First 10 smoothed samples (segment {i + 1}): {smoothed[:10]}")

        # Apply RMS
        rms_pressure = ap.apply_root_mean_square(smoothed)
        print(f"First 10 root mean square pressure (segment {i + 1}){rms_pressure[:10]}")

        spl = ap.pressure_to_db(rms_pressure)
        print(f"First 10 SPLs (segment {i + 1}){spl[:10]} \n")

if __name__ == "__main__":
    main()