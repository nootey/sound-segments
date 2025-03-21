import os

import numpy as np
from pydub import AudioSegment
from numpy import log10, pi
from scipy.signal import bilinear_zpk, freqs, sosfilt, zpk2sos, zpk2tf

class AudioProcessor:
    def __init__(self):
        print("AudioProcessor initialized")

    def load_audio(self, file_path):
        """ Load MP3 file and convert to normalized numpy array """
        try:
            audio = AudioSegment.from_mp3(file_path).set_channels(1)  # Convert to mono
            samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
            sample_rate = audio.frame_rate

            # Normalize samples to range -1 to 1
            samples /= np.max(np.abs(samples))

            return samples, sample_rate

        except Exception as e:
            raise e

    def ensure_wav_format(self, file_path: str) -> str:
        """
        Ensures the audio file is in .wav format.
        """
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext not in ['.mp3', '.wav']:
            raise ValueError(f"Unsupported file format: {ext}. Only MP3 and WAV are supported.")

        if ext == '.wav':
            return file_path

        # If it's an MP3, convert to WAV
        audio = AudioSegment.from_mp3(file_path)
        wav_path = file_path.replace('.mp3', '.wav')

        print(f"Converting MP3 to WAV: {wav_path}")
        audio.export(wav_path, format="wav")

        return wav_path

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

    # "Forked" from https://github.com/endolith/waveform-analysis/blob/master/waveform_analysis/weighting_filters/ABC_weighting.py
    def ABC_weighting(self, curve='A'):
        """
        Analog weighting filter with A, B, or C curve.
        """
        if curve not in 'ABC':
            raise ValueError('Curve type not supported')

        z = [0, 0]
        p = [-2 * pi * 20.598997057568145,
             -2 * pi * 20.598997057568145,
             -2 * pi * 12194.21714799801,
             -2 * pi * 12194.21714799801]
        k = 1

        if curve == 'A':
            p.append(-2 * pi * 107.65264864304628)
            p.append(-2 * pi * 737.8622307362899)
            z.append(0)
            z.append(0)

        elif curve == 'B':
            p.append(-2 * pi * 10 ** 2.2)
            z.append(0)

        # Normalize to 0 dB at 1 kHz for all curves
        b, a = zpk2tf(z, p, k)
        k /= abs(freqs(b, a, [2 * pi * 1000])[1][0])

        return np.array(z), np.array(p), k

    def A_weighting(self, fs, output='ba'):
        """
        Designs a digital A-weighting filter for sampling frequency `fs`.
        """
        z, p, k = self.ABC_weighting('A')

        # Use the bilinear transformation to get the digital filter.
        z_d, p_d, k_d = bilinear_zpk(z, p, k, fs)

        if output == 'zpk':
            return z_d, p_d, k_d
        elif output in {'ba', 'tf'}:
            return zpk2tf(z_d, p_d, k_d)
        elif output == 'sos':
            return zpk2sos(z_d, p_d, k_d)
        else:
            raise ValueError(f"'{output}' is not a valid output form.")

    def apply_a_weighting_filter(self, fs):
        """
        Return the given signal after passing through a digital A-weighting filter
        """
        try:
            return  self.A_weighting(fs, output='sos')
        except Exception as e:
            raise e

