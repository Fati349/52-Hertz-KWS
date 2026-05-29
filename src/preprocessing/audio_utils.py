import numpy as np
import librosa
import yaml

class AudioPipeline:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        audio_cfg   = self.config["audio"]
        mel_cfg     = self.config["features"]["log_mel_spectrogram"]  

        # Audio
        self.sample_rate  = audio_cfg["sample_rate"]          
        self.duration_ms  = audio_cfg["duration_ms"]          
        self.n_samples    = int(self.sample_rate * self.duration_ms / 1000)  

        # Mel spectrogram
        self.n_mels      = mel_cfg["n_mels"]                  
        self.n_fft       = int(mel_cfg["window_ms"] * self.sample_rate / 1000)  
        self.hop_length  = int(mel_cfg["stride_ms"] * self.sample_rate / 1000)  
        self.n_frames    = mel_cfg["n_frames"]                
        self.fmin        = mel_cfg.get("fmin", 0)
        self.fmax        = mel_cfg.get("fmax", self.sample_rate // 2)  
        self.center      = mel_cfg.get("center", True)       

    # ─────────────────────────────────────────────
    # Chargement
    # ─────────────────────────────────────────────
    def load_audio(self, file_path):
        """Charge et fixe la durée. librosa.load resample déjà à sample_rate."""
        audio, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        audio = librosa.util.fix_length(audio, size=self.n_samples)
        return audio

    def normalize_audio(self, audio):
        """Normalisation peak [-1, 1]."""
        max_val = np.max(np.abs(audio))
        return audio / max_val if max_val > 0 else audio

    # ─────────────────────────────────────────────
    # Augmentation (crucial avec peu de données)
    # ─────────────────────────────────────────────
    def augment_audio(self, audio, noise_level=0.005, max_shift_ms=100):
        """Time shift + bruit gaussien."""
        # Time shift aléatoire
        max_shift = int(max_shift_ms * self.sample_rate / 1000)
        shift = np.random.randint(-max_shift, max_shift)
        audio = np.roll(audio, shift)
        if shift > 0:
            audio[:shift] = 0
        else:
            audio[shift:] = 0

        # Bruit de fond léger
        noise = np.random.randn(len(audio)) * noise_level
        audio = audio + noise
        return audio.astype(np.float32)

    def augment_spectrogram(self, log_mel, freq_mask=8, time_mask=10):
        """SpecAugment : masquage fréquence + temps sur le mel spectrogram.
        Très efficace pour KWS avec peu de données.
        log_mel shape : [n_mels, n_frames]
        """
        log_mel = log_mel.copy()
        n_mels, n_frames = log_mel.shape

        # Masque fréquentiel
        f = np.random.randint(0, freq_mask)
        f0 = np.random.randint(0, n_mels - f)
        log_mel[f0:f0 + f, :] = 0

        # Masque temporel
        t = np.random.randint(0, time_mask)
        t0 = np.random.randint(0, n_frames - t)
        log_mel[:, t0:t0 + t] = 0

        return log_mel

    # ─────────────────────────────────────────────
    # Extraction features
    # ─────────────────────────────────────────────
    def extract_log_mel(self, audio):
        """
        Retourne un log-mel spectrogram normalisé.
        Shape de sortie 
        """
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            fmin=self.fmin,
            fmax=self.fmax,
            center=self.center,
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)  # échelle log (dB)

        # Normalisation par instance (moyenne / écart-type)
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
            # ── Fix : force la dimension temporelle à n_frames ──────────────────
        if log_mel.shape[1] > self.n_frames:
            log_mel = log_mel[:, :self.n_frames]          # crop
        elif log_mel.shape[1] < self.n_frames:
            pad = self.n_frames - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad))) # zero-pad

        return log_mel.astype(np.float32)   
  

    # ─────────────────────────────────────────────
    # Points d'entrée (pour tf.data ou DataLoader)
    # ─────────────────────────────────────────────
    def extract_log_mel_from_path(self, path, augment=False):
        """
        Point d'entrée principal pour l'entraînement.
        Retourne shape [n_mels, n_frames, 1] = [64, 50, 1]
        """
        if isinstance(path, bytes):
            path = path.decode("utf-8")

        audio = self.load_audio(path)
        audio = self.normalize_audio(audio)
        if augment:
            audio = self.augment_audio(audio)

        log_mel = self.extract_log_mel(audio)        

        if augment:
            log_mel = self.augment_spectrogram(log_mel)  

        return log_mel[..., np.newaxis]              

    def extract_log_mel_for_inference(self, path):
        """Point d'entrée pour l'inférence (pas d'augmentation)."""
        return self.extract_log_mel_from_path(path, augment=False)