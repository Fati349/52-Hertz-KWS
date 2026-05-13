import os
import numpy as np
import tensorflow as tf
import yaml
from sklearn.model_selection import train_test_split
from src.preprocessing.audio_utils import AudioPipeline


class KWSDataset:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.batch_size  = self.config["training"]["batch_size"]
        self.val_split   = self.config["training"]["val_split"]
        self.test_split  = self.config["training"]["test_split"]
        self.feature     = self.config["model"]["feature"]
        self.input_shape = self.config["model"]["input_shape"]

        audio_cfg = self.config["audio"]
        self.audio_pipeline = AudioPipeline(
            sampling_rate = audio_cfg["sample_rate"],
            duration_ms   = audio_cfg["duration_ms"],
            n_mfcc        = audio_cfg["n_mfcc"],
            n_mels        = audio_cfg["n_mels"],
            n_fft         = audio_cfg["n_fft"],
            hop_length    = audio_cfg["hop_length"],
        )


    def load_paths_and_labels(self, data_dir):
        """
        Structure attendue :
            data_dir/
                keyword/
                non_keyword/
        """
        paths, labels = [], []
        for label, folder in enumerate(["non_keyword", "keyword"]):
            folder_path = os.path.join(data_dir, folder)
            for fname in os.listdir(folder_path):
                if fname.endswith(".wav"):
                    paths.append(os.path.join(folder_path, fname))
                    labels.append(label)
        return np.array(paths), np.array(labels)

    # split

    def split(self, paths, labels):
        X_train, X_temp, y_train, y_temp = train_test_split(
            paths, labels,
            test_size=self.val_split + self.test_split,
            stratify=labels, random_state=42
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=self.test_split / (self.val_split + self.test_split),
            stratify=y_temp, random_state=42
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    # tf.data

    def _process_path(self, path, label):
        # ajouter un nouveau modèle = ajouter un elif ici + la méthode dans audio_utils.py
        if self.feature == "mfcc":
            func = self.audio_pipeline.extract_mfcc_from_path
        elif self.feature == "log_mel":
            func = self.audio_pipeline.extract_log_mel_from_path

        features = tf.numpy_function(func=func, inp=[path], Tout=tf.float32)
        features.set_shape(self.input_shape)
        return features, label

    def _augment(self, features, label):
        # bruit gaussien
        noise    = tf.random.normal(shape=tf.shape(features), stddev=0.05)
        features = features + noise

        # time masking
        t_size  = tf.random.uniform((), minval=0, maxval=10, dtype=tf.int32)
        t_start = tf.random.uniform((), minval=0,
                                    maxval=tf.shape(features)[1] - t_size,
                                    dtype=tf.int32)
        mask = tf.concat([
            tf.ones ([tf.shape(features)[0], t_start, 1]),
            tf.zeros([tf.shape(features)[0], t_size,  1]),
            tf.ones ([tf.shape(features)[0], tf.shape(features)[1] - t_start - t_size, 1])
        ], axis=1)
        features = features * mask

        return features, label

    def build_tf_dataset(self, paths, labels, augment=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(self._process_path, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._augment, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(1000).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    # entrée principale

    def prepare(self, data_dir):
        """
        Usage :
            ds = KWSDataset("configuration/config.yaml")
            train_ds, val_ds, test_ds = ds.prepare("data/raw")
        """
        paths, labels = self.load_paths_and_labels(data_dir)
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = self.split(paths, labels)

        train_ds = self.build_tf_dataset(X_train, y_train, augment=True)
        val_ds   = self.build_tf_dataset(X_val,   y_val,   augment=False)
        test_ds  = self.build_tf_dataset(X_test,  y_test,  augment=False)

        print(f"Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
        return train_ds, val_ds, test_ds