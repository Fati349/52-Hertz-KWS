import tensorflow as tf


def build_transfer_model_yamnet(keras_path, num_classes=3):
    base_model = tf.keras.models.load_model(keras_path, compile=False)

    for layer in base_model.layers:
        layer.trainable = False   # backbone gelé

    x = base_model[0].output   # <- sortie du backbone (256)

    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)

    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="new_head"
    )(x)

    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="yamnet256"
    )

    return model