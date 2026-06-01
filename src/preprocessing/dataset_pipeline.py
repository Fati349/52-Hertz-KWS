import tensorflow as tf
import numpy as np
import yaml
from pathlib import Path
from preprocessing import AudioPipeline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configuration" / "config.yaml"
pipeline = AudioPipeline(str(CONFIG_PATH))

def load_and_preprocess(path, label, augment=False):
    def _py_fn(p, aug):
        feat = pipeline.extract_log_mel_from_path(p.numpy(), augment=aug.numpy())
        return feat.astype(np.float32)
    feat = tf.py_function(func=_py_fn, inp=[path, augment], Tout=tf.float32)
    feat.set_shape([64, 50, 1])
    return feat, label

def make_dataset(paths, labels, batch_size, augment=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: load_and_preprocess(p, l, augment),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if augment:
        ds = ds.shuffle(buffer_size=len(paths))   # shuffle TOUT le train
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)