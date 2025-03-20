from pydub import AudioSegment

from src.internal.audio_processor import AudioProcessor

def main():
    print("Sound segments")

    ap = AudioProcessor()

    recording_name = "sample"
    audio_file_path = f"data/{recording_name}.mp3"

    # Load the audio file
    try:
        recording = AudioSegment.from_file(audio_file_path)
        segments = ap.ensure_recording_length(recording)

        print(segments)

    except Exception as e:
        print(f"Error loading audio file: {e}")

if __name__ == "__main__":
    main()