# imports
import numpy as np
import librosa


sampling_rate = 16000  # Desired sampling rate
duration_ms = 1000  # Desired duration in milliseconds
n_mfcc= 40
n_mels = 40

class AudioPipeline:
    """Pipeline for basic audio preprocessing and feature extraction.

    Parameters can be customized at construction; defaults use the module-level
    constants defined above.
    """
    def __init__(self, sampling_rate=sampling_rate, duration_ms=duration_ms, n_mfcc=n_mfcc, n_mels=n_mels):
        self.sampling_rate = sampling_rate
        self.duration_ms = duration_ms
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels

    def load_audio(self, file_path):
        audio, sr = librosa.load(file_path, sr=self.sampling_rate)
        audio = librosa.util.fix_length(audio, size=int(self.sampling_rate * self.duration_ms / 1000))
        return audio

    def resample_audio(self, audio, original_sr, target_sr=None):
        if target_sr is None:
            target_sr = self.sampling_rate
        if original_sr != target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
        return audio

    def normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val
        return audio

    def extract_mfcc(self, audio):
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sampling_rate, n_mfcc=self.n_mfcc)
        return mfccs

    def extract_log_mel_spectrogram(self, audio):
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=self.sampling_rate, n_mels=self.n_mels)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        return log_mel_spectrogram

