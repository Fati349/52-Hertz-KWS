import tensorflow as tf
import numpy as np
import yaml
from audio_utils import AudioPipeline


pipeline = AudioPipeline("config.yaml")

def load_and_preprocess(path, label, augment=False):
    """Wrapper tf.py_function pour intégrer le pipeline audio dans tf.data."""
    def _py_fn(p, aug):
        feat = pipeline.extract_log_mel_from_path(p.numpy(), augment=aug.numpy())
        return feat.astype(np.float32)

    feat = tf.py_function(
        func=_py_fn,
        inp=[path, augment],
        Tout=tf.float32
    )
    feat.set_shape([64, 50, 1])   # OBLIGATOIRE pour que Keras connaisse le shape
    return feat, label

# Construction des datasets
def make_dataset(paths, labels, augment=False, batch_size=64):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: load_and_preprocess(p, l, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)