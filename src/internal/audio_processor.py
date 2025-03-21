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

    def get_file_with_extension(self, file_path: str) -> str:
        """
        Checks if the file has an extension. If not, looks for the file with an extension in the filesystem.
        """
        if '.' in file_path and os.path.isfile(file_path):
            return file_path

        # Try known extensions
        for ext in ['.wav', '.mp3']:  # Prioritize .wav first
            possible_path = file_path + ext
            if os.path.isfile(possible_path):
                return possible_path

        raise FileNotFoundError(f"File '{file_path}' not found with either .mp3 or .wav extension.")

    def ensure_wav_format(self, file_path: str) -> str:
        """
        Ensures the audio file is in .wav format.
        If input is MP3 and a corresponding WAV exists, it uses the WAV.
        If not, converts the MP3 to WAV.
        """
        file_with_ext = self.get_file_with_extension(file_path)
        base_name, ext = os.path.splitext(file_with_ext)
        ext = ext.lower()

        if ext == '.wav':
            return file_with_ext

        elif ext == '.mp3':
            wav_path = base_name + '.wav'

            # If WAV version already exists, use it
            if os.path.isfile(wav_path):
                print(f"WAV version already exists: {wav_path}")
                return wav_path

            # Otherwise, convert MP3 to WAV
            print(f"Converting MP3 to WAV: {wav_path}")
            audio = AudioSegment.from_mp3(file_with_ext)
            audio.export(wav_path, format="wav")
            return wav_path

        else:
            raise ValueError(f"Unsupported file format: {ext}. Only MP3 and WAV are supported.")

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

    def apply_exponential_moving_average(self, power_signal, sample_rate, tau=1.0):
        """
        Apply a low-pass filter (exponential moving average) to simulate a real sound level meter response.

        Parameters:
            power_signal (np.ndarray): Squared, A-weighted audio samples.
            sample_rate (int): Sample rate of the audio.
            tau (float): Time constant (in seconds). For LAeq, often 1s.

        Returns:
            np.ndarray: Smoothed (filtered) power signal.
        """
        try:
            # Calculate the smoothing factor alpha based on the time constant tau
            alpha = 1.0 / (tau * sample_rate + 1.0)

            # Preallocate the output array
            smoothed = np.zeros_like(power_signal)

            # Initialize the first value (assume zero start)
            smoothed[0] = power_signal[0]

            # Apply the recursive filter: y[n] = alpha * x[n] + (1 - alpha) * y[n-1]
            for i in range(1, len(power_signal)):
                smoothed[i] = alpha * power_signal[i] + (1 - alpha) * smoothed[i - 1]

            return smoothed
        except Exception as e:
            raise e

    def apply_root_mean_square(self, smoothed_signal):
        """
        To get back to something proportional to sound pressure, take the square root of that smoothed power.
        """
        try:
            return np.sqrt(smoothed_signal)
        except Exception as e:
            raise e

    def pressure_to_db(self, rms_pressure, ref_pressure=20e-6):
        """
        Convert RMS sound pressure to decibels (SPL).

        Parameters:
            rms_pressure (np.ndarray): RMS A-weighted pressure signal.
            ref_pressure (float): Reference pressure in Pascals. Default is 20 µPa for air.

        Returns:
            np.ndarray: Sound Pressure Level (SPL) in dB.
        """
        try:
            # Avoid log(0) by setting a very small floor
            rms_pressure = np.maximum(rms_pressure, 1e-12)

            # Apply the SPL formula
            spl = 20 * np.log10(rms_pressure / ref_pressure)

            return spl
        except Exception as e:
            raise e

    def compute_LAeq(self, squared_smoothed_signal, ref_pressure=20e-6):
        """
        Compute LAeq in dB.

        Parameters:
            squared_smoothed_signal (np.ndarray): Smoothed, squared A-weighted signal (represents power).
            ref_pressure (float): Reference pressure in Pascals. Default is 20 µPa.

        Returns:
            float: LAeq in decibels.
        """
        # Compute mean power over time
        mean_power = np.mean(squared_smoothed_signal)

        # Avoid log(0) by setting a floor
        mean_power = max(mean_power, 1e-20)

        # Convert to LAeq using: LAeq = 10 * log10(mean_power / p0^2)
        LAeq = 10 * np.log10(mean_power / (ref_pressure ** 2))

        return LAeq

    def process_segment(self, segment, sample_rate, sos, timestamp, index):
        """
        Processes a 60-second audio segment:
        - Applies A-weighting
        - Computes full-segment LAeq
        - Computes 1-second LAeq values
        - Formats everything as a dictionary
        """

        # Duration of this segment (in seconds)
        duration_sec = len(segment) / sample_rate
        if duration_sec < 1.0:
            print(f"     Skipping segment {index} — too short to process")
            return None

        # Step 1: Filter + Square
        print("     Squaring the samples")
        filtered = sosfilt(sos, segment)
        squared = filtered ** 2

        # Step 2: Smooth
        print("     Applying low-pass filter")
        smoothed = self.apply_exponential_moving_average(squared, sample_rate)
        overall_la_eq = self.compute_LAeq(smoothed)

        # Step 3: Calculate per-second LAeqs
        print("     Calculating sound level")
        samples_per_second = sample_rate
        per_second_la_eqs = []

        for sec in range(60):
            start = sec * samples_per_second
            end = start + samples_per_second
            if end > len(segment):
                break

            sec_filtered = sosfilt(sos, segment[start:end])
            sec_squared = sec_filtered ** 2
            laeq_1s = self.compute_LAeq(sec_squared)
            per_second_la_eqs.append(round(laeq_1s, 1))

        # Build output dict
        return {
            "Index": index,
            "Timestamp": timestamp.strftime("%Y-%m-%d %H:%M"),
            "Duration [s]": round(duration_sec, 2),
            "LAeq avg [dBA]": round(overall_la_eq, 1),
            "LAeq [dBA] [s]": ", ".join(str(val) for val in per_second_la_eqs)
        }
