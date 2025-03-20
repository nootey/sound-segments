from pydub import AudioSegment


class AudioProcessor:
    def __init__(self):
        print("AudioProcessor initialized")

    def ensure_recording_length(self, recording: AudioSegment):
        duration_seconds = len(recording) / 1000  # Convert from ms to seconds
        if duration_seconds <= 60:
            print(f"Audio is {duration_seconds:.2f} seconds long. No split needed.")
            return [recording]

        # Split into 60-second segments
        print(f"Audio is {duration_seconds:.2f} seconds long. Splitting into 60s chunks.")
        segments = []
        for i in range(0, len(recording), 60000):  # 60000 ms = 60 seconds
            segments.append(recording[i:i + 60000])
        return segments