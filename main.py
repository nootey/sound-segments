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
    for i, segment in enumerate(segments):
        filtered = sosfilt(sos, segment)
        print(f"Filtered segment {i+1}, length: {len(filtered)} samples")


if __name__ == "__main__":
    main()