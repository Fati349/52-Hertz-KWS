# imports
import numpy as np
import librosa
import yaml

class AudioPipeline:
    """Pipeline for basic audio preprocessing and feature extraction.

    Parameters can be customized at construction; defaults use the module-level
    constants defined above.
    """
    def __init__(self,  config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        mfcc_cfg = self.config["features"]["mfcc"]
        self.n_mfcc      = mfcc_cfg["n_mfcc"]
        self.hop_length  = int(mfcc_cfg["stride_ms"] * self.sampling_rate / 1000)  # 320
        self.n_fft       = int(mfcc_cfg["window_ms"] * self.sampling_rate / 1000)  # 640

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
        mfccs = librosa.feature.mfcc(
                y=audio,
                sr=self.sampling_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,          # 640
                hop_length=self.hop_length, # 320
                center=False               # OBLIGATOIRE pour avoir 49 frames
            )
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)   # normalisation
        return mfccs.astype(np.float32)
    

    def extract_mfcc_from_path(self, path):
        path  = path.decode("utf-8")
        audio = self.load_audio(path)
        audio = self.normalize_audio(audio)
        mfccs = self.extract_mfcc(audio)
        return mfccs.T[..., np.newaxis]          # [n_mfcc, n_frames, 1]

    def extract_log_mel_from_path(self, path):
        path  = path.decode("utf-8")
        audio = self.load_audio(path)
        audio = self.normalize_audio(audio)
        log_mel = self.extract_log_mel(audio)
        return log_mel[..., np.newaxis]        # [n_mels, n_frames, 1]

