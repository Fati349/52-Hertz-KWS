"""
K-Fold stratifiée (k=5) pour training_v2s1.
Remplace les cellules : Split / Class weights / tf.data / Model / Callbacks / Training / Evaluation
Les cellules Imports et Config restent identiques.
"""

# ─── Imports (identiques au notebook + 2 ajouts) ──────────────────────────────
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "/home/f24achba/Documents/Edge AI/52-Hertz-KWS/src")))

import tensorflow as tf
import numpy as np
import glob, json
from sklearn.model_selection import StratifiedKFold          # ← AJOUT 1
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
from preprocessing import AudioPipeline
from preprocessing.dataset_pipeline import make_dataset
from preprocessing.transfer_model import build_transfer_model

# ─── Config (identique au notebook) ──────────────────────────────────────────
DATA_DIR   = "/home/f24achba/Documents/Edge AI/52-Hertz-KWS/data/raw"
CLASSES    = ["talky", "silence", "unknown"]
BATCH_SIZE = 16
EPOCHS     = 50
LR         = 1e-3
KERAS_PATH = "/home/f24achba/Documents/Edge AI/52-Hertz-KWS/src/models/miniresnetv2_s1_64x50_tl.keras"
CONFIG_YAML= "/home/f24achba/Documents/Edge AI/52-Hertz-KWS/configuration/config.yaml"

K_FOLDS    = 5                                               # ← AJOUT 2
SEED       = 42

# ─── Chargement des chemins + labels (identique au notebook) ─────────────────
pipeline   = AudioPipeline(CONFIG_YAML)
all_paths, all_labels = [], []

for label_idx, class_name in enumerate(CLASSES):
    wavs = glob.glob(os.path.join(DATA_DIR, class_name, "*.wav"))
    all_paths  += wavs
    all_labels += [label_idx] * len(wavs)
    print(f"  {class_name:10s} → {len(wavs)} fichiers")

all_paths  = np.array(all_paths)
all_labels = np.array(all_labels)
print(f"\nTotal : {len(all_paths)} samples\n")


# ══════════════════════════════════════════════════════════════════════════════
#  REMPLACE les cellules : Split / tf.data / Model / Callbacks / Training / Eval
# ══════════════════════════════════════════════════════════════════════════════

# ─── K-Fold stratifiée ───────────────────────────────────────────────────────
#
#  Principe :
#    • StratifiedKFold divise les N samples en 5 blocs de taille égale
#      en respectant les proportions de chaque classe dans chaque bloc.
#    • Fold i  →  test  = bloc i  (≈20% des données)
#                 val   = bloc i-1 (≈20%)
#                 train = les 3 blocs restants (≈60%)
#    • Chaque sample passe exactement 1 fois dans le test set.
#
#  Différence avec tes 10 runs random :
#    • Là, chaque run tirait un split aléatoire → même sample pouvait
#      tomber dans train à chaque fois ou dans test à chaque fois.
#    • Ici, on garantit une couverture exhaustive et équilibrée.

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
fold_results = []

print("=" * 55)
print(f"  K-Fold stratifiée — {K_FOLDS} folds")
print(f"  LR={LR}  |  Epochs={EPOCHS}  |  Batch={BATCH_SIZE}")
print("=" * 55)

for fold_idx, (trainval_idx, test_idx) in enumerate(skf.split(all_paths, all_labels)):

    print(f"\n[Fold {fold_idx+1}/{K_FOLDS}]")

    # ── Séparer train+val  /  test ────────────────────────────────────────
    X_trainval = all_paths[trainval_idx]
    y_trainval = all_labels[trainval_idx]
    X_test     = all_paths[test_idx]
    y_test     = all_labels[test_idx]

    # ── Découper val depuis train (20% de trainval = ~16% total) ─────────
    # On utilise un second StratifiedKFold à 5 folds pour prendre
    # exactement 1 bloc comme val → cohérent avec la logique k-fold
    inner_skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED+fold_idx)
    train_sub_idx, val_sub_idx = next(inner_skf.split(X_trainval, y_trainval))

    X_train = X_trainval[train_sub_idx]
    y_train = y_trainval[train_sub_idx]
    X_val   = X_trainval[val_sub_idx]
    y_val   = y_trainval[val_sub_idx]

    print(f"  Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")

    # ── Class weights (identique au notebook) ────────────────────────────
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))

    # ── tf.data pipeline (identique au notebook) ─────────────────────────
    train_ds = make_dataset(X_train, y_train, BATCH_SIZE, augment=True)
    val_ds   = make_dataset(X_val,   y_val,   BATCH_SIZE, augment=False)
    test_ds  = make_dataset(X_test,  y_test,  BATCH_SIZE, augment=False)

    # ── Modèle — reconstruit à chaque fold (poids frais) ─────────────────
    tf.keras.backend.clear_session()                         # ← libère la VRAM
    model = build_transfer_model(KERAS_PATH, num_classes=3)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ── Callbacks (identiques au notebook, checkpoint par fold) ──────────
    ckpt_path = f"best_model_fold{fold_idx+1}.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_accuracy",
            save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=12,   # ← augmenté (était 8)
            restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=4, min_lr=1e-6, verbose=0
        ),
    ]

    # ── Entraînement (identique au notebook) ─────────────────────────────
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=0
    )

    # ── Évaluation sur le meilleur modèle (identique au notebook) ────────
    model = tf.keras.models.load_model(ckpt_path)
    loss, test_acc = model.evaluate(test_ds, verbose=0)

    best_epoch   = int(np.argmax(history.history["val_accuracy"])) + 1
    best_val_acc = float(np.max(history.history["val_accuracy"]))

    # ── Classification report ─────────────────────────────────────────────
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    y_true = np.concatenate([y for _, y in test_ds], axis=0)
    report = classification_report(
        y_true, y_pred,
        target_names=CLASSES,
        output_dict=True
    )

    result = {
        "fold":          fold_idx + 1,
        "test_accuracy": round(float(test_acc),    4),
        "val_accuracy":  round(best_val_acc,       4),
        "best_epoch":    best_epoch,
        "f1_talky":      round(report["talky"]["f1-score"],   4),
        "f1_silence":    round(report["silence"]["f1-score"], 4),
        "f1_unknown":    round(report["unknown"]["f1-score"], 4),
    }
    fold_results.append(result)

    print(f"  ✓ test={result['test_accuracy']:.4f}  "
          f"val={result['val_accuracy']:.4f}  "
          f"epoch={result['best_epoch']}  "
          f"f1=[{result['f1_talky']:.3f} / {result['f1_silence']:.3f} / {result['f1_unknown']:.3f}]")

    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)


# ─── Statistiques finales ─────────────────────────────────────────────────────
test_accs = np.array([r["test_accuracy"] for r in fold_results])

stats = {
    "mean":   round(float(np.mean(test_accs)),   4),
    "std":    round(float(np.std(test_accs)),    4),
    "median": round(float(np.median(test_accs)), 4),
    "min":    round(float(np.min(test_accs)),    4),
    "max":    round(float(np.max(test_accs)),    4),
}

print("\n" + "=" * 55)
print("  RÉSULTATS K-FOLD")
print("=" * 55)
for r in fold_results:
    print(f"  Fold {r['fold']} | test={r['test_accuracy']:.4f}  "
          f"val={r['val_accuracy']:.4f}  epoch={r['best_epoch']:3d}")
print("-" * 55)
print(f"  Mean   : {stats['mean']:.4f}")
print(f"  Std    : {stats['std']:.4f}  ← cible < 0.02")
print(f"  Median : {stats['median']:.4f}")
print(f"  Min / Max : {stats['min']:.4f} / {stats['max']:.4f}")
print("=" * 55)

stability = "Stable" if stats["std"] < 0.02 else f"Instable (std={stats['std']:.4f})"
print(f"\n  Stabilité : {stability}")

# ─── Sauvegarde JSON ──────────────────────────────────────────────────────────
output = {
    "method":     "StratifiedKFold",
    "k":          K_FOLDS,
    "config":     {"lr": LR, "epochs": EPOCHS, "batch_size": BATCH_SIZE},
    "folds":      fold_results,
    "statistics": stats,
}
with open("results_kfold.json", "w") as f:
    json.dump(output, f, indent=2)
print("  → Résultats sauvegardés dans results_kfold.json")