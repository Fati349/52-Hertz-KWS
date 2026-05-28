import tensorflow as tf


def build_transfer_model(keras_path, num_classes=3):
    base_model = tf.keras.models.load_model(keras_path, compile=False)
    backbone_out = base_model.get_layer("flatten").output

    for layer in base_model.layers:
        layer.trainable = False   # backbone 100% gelé

    x = backbone_out
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="head_output")(x) 

    model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name="miniresnet_kws")
    return model
