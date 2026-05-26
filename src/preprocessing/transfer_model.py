import tensorflow as tf

def build_transfer_model(keras_path: str, num_classes: int = 3):

    # ── 1. Charger le modèle pré-entraîné ─────────────────────────────────
    base_model = tf.keras.models.load_model(keras_path, compile=False)

    # ── 2. Extraire la sortie du backbone (avant new_head) ────────────────
    backbone_out = base_model.get_layer("flatten").output

    # ── 3. Vérifier que le backbone est bien gelé ─────────────────────────
    #       Le .keras _tl a déjà 91 072 params non-trainable -> on s'assure
    #       que c'est bien le cas et on force pour être explicite
    for layer in base_model.layers:
        if layer.name != "new_head":          # exclure l'ancienne tête
            layer.trainable = False

    # Vérification
    frozen  = [l.name for l in base_model.layers if not l.trainable]
    trainab = [l.name for l in base_model.layers if l.trainable]
    print(f" Gelées  ({len(frozen)})  : {frozen}")
    print(f" Actives ({len(trainab)}) : {trainab}")

    # ── 4. Nouvelle tête pour tes 3 classes ───────────────────────────────
    #       Pas de Dense(64) intermédiaire nécessaire : 3584 → 3
    #       avec peu de data, moins de paramètres = moins d'overfitting
    x = backbone_out
    x = tf.keras.layers.Dropout(0.3, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        num_classes,
        activation="softmax",
        name="head_output"                    # ne pas réutiliser "new_head"
    )(x)

    # ── 5. Modèle final ───────────────────────────────────────────────────
    model = tf.keras.Model(
        inputs=base_model.input,
        outputs=outputs,
        name="miniresnet_kws"
    )

    # ── 6. Compiler ───────────────────────────────────────────────────────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ── Instanciation + vérification ──────────────────────────────────────────
model = build_transfer_model("miniresnetv1_s1_64x50_tl.keras", num_classes=3)
model.summary()

# Doit afficher :
# Trainable params:     ~10 755  (Dropout + Dense 3584→3)
# Non-trainable params:  91 072  (backbone intact)