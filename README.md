# 52-Hertz-KWS : Reconnaissance du mot "talky"

## Introduction

Ce projet a pour objectif de générer le modèle de reconnaissance de mots-clés **ResNet v2 S1**, destiné à être déployé sur MCU (microcontrôleur). Le modèle est repris du [ModelZoo STMicroelectronics](https://github.com/STMicroelectronics/stm32ai-modelzoo/tree/main/audio_event_detection/miniresnetv2), puis adapté par **transfer learning** afin de générer un fichier `.tflite` optimisé pour le déploiement sur MCU.

## Modèle initial

Le modèle de base est entraîné sur le dataset **ESC-10**. Il prend en entrée un mel-spectrogramme de taille `(64, 50, 1)` et produit en sortie un vecteur d'embedding de taille `2048`.

Dans ce projet, il est réutilisé pour reconnaître 3 classes :
- **talky**
- **silence**
- **unknown**

## Étapes de traitement

Le projet se décompose en trois phases principales :

1. **Prétraitement** des données d'entraînement (conversion audio → mel-spectrogrammes)
2. **Transfer learning** pour générer le nouveau modèle adapté aux 3 classes
3. **Post-traitement** pour alléger et convertir le modèle final en un format `.tflite` déployable sur microcontrôleur

## Architecture du projet

```
52-Hertz-KWS/
├── data/                                 # Données audio brutes a ajouter
    ├── talky/*.wav
    ├── silence/*.wav
    └── unknown/*.wav                                 
├── preprocessing/                        # Scripts de génération des mel-spectrogrammes
├── resnet v2 s1/ 
    ├── Tests/
    ├── Tests_results/
    └── training_v2s1_deploiement.ipynb   # Fichier notebook contenant les scripts de transfer learning, 
                                            des tests de deploiement et postprocessing  
├── models/                               # Modèles entraînés 
    ├── model_MCU/
        └── final_model_deploy_int8.tflite # Modèle final a déployer
    └── utils/
        ├── final_model_deploy.keras       # Modèle final en float32 a transformer en int8
        ├── miniresnetv2_s1_64x50_tl.keras # Modèle initial pris de ModelZoo
        └── model_v2s1-chargement.ipynb    # Fichier donnant un résumé des infos sur les couches du modèle 
├── requirements.txt
├── config.yaml                           
└── README.md
```

## Installation

### Prérequis

- Python 3.11 ou supérieur
- pip
- git
- Jupyter Notebook ou JupyterLab (ou l'extension Jupyter de VS Code)

### Étapes

1. **Cloner le dépôt**

```bash
git clone https://github.com/<votre-utilisateur>/52-Hertz-KWS.git
cd 52-Hertz-KWS
```

2. **Créer un environnement virtuel**

```bash
python3 -m venv venv
source venv/bin/activate   # Sous Linux / macOS
venv\Scripts\activate      # Sous Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Vérifier l'installation**

```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### Sélection du kernel Jupyter

Pour que le notebook utilise bien l'environnement virtuel (et toutes ses dépendances : `tensorflow`, `scikit-learn`, etc.), il faut enregistrer ce venv comme kernel Jupyter.

1. **Installer `ipykernel` dans le venv** (le venv doit être activé) :

```bash
pip install ipykernel
```

2. **Enregistrer le venv comme kernel Jupyter** :
```bash
python -m ipykernel install --user --name=52-hertz-kws --display-name "Python (52-Hertz-KWS)"
```

3. **Sélectionner le kernel dans le notebook** :
   - Jupyter Notebook/Lab : menu **Kernel → Change Kernel → Python (52-Hertz-KWS)**
   - VS Code : cliquer sur le sélecteur de kernel en haut à droite du notebook → choisir **Python (52-Hertz-KWS)**


4. **Vérifier que le bon environnement est utilisé**, en exécutant dans une cellule du notebook :
```python
import sys
print(sys.executable)
```
 
Le chemin affiché doit pointer vers `venv/bin/python` (ou `venv\Scripts\python.exe` sous Windows).

## Préparation des données
 
Avant d'exécuter le notebook, placer les fichiers audio `.wav` dans `data/`, organisés par classe (un sous-dossier par classe, nommé exactement comme dans `CLASSES`) :
 
```
data/
├── talky/
│   ├── fichier1.wav
│   └── ...
├── silence/
│   ├── fichier1.wav
│   └── ...
└── unknown/
    ├── fichier1.wav
    └── ...
```

## Configuration avant exécution
 
Le notebook `training_v2s1_deploiement.ipynb` contient plusieurs chemins à adapter manuellement dans la cellule "Configuration" avant de l'exécuter :
 
| Variable | Description |
|---|---|
| `DATA_DIR` | Chemin vers le dossier `data/` contenant les sous-dossiers par classe |
| `KERAS_PATH` | Chemin vers le modèle pré-entraîné `miniresnetv2_s1_64x50_tl.keras` (dans `models/`) |
| Chemin passé à `AudioPipeline(...)` | Chemin vers le fichier `configuration/config.yaml` |
| `DEPLOY_PATH` | Chemin de sauvegarde du modèle entraîné (`.keras`). Par défaut sauvegardé dans le répertoire courant : pointer vers `models/` pour respecter l'arborescence du projet |
 
Si vous êtes sur Linux/macOS, décommenter également la cellule d'import ajoutant le dossier `src` au `PYTHONPATH`, en adaptant le chemin `<Chemin vers le dossier du projet>`.

## Utilisation
 
Le projet s'exécute entièrement depuis le notebook Jupyter, cellule par cellule, dans l'ordre :
 
```bash
jupyter notebook "resnet v2 s1/training_v2s1_deploiement.ipynb"
```
 
Le notebook enchaîne :
 
1. **Chargement et split des données** (85 % entraînement / 15 % validation, stratifié)
2. **Transfer learning** sur le modèle ResNet v2 S1 pré-entraîné, avec pondération des classes pour gérer le déséquilibre du dataset
3. **Évaluation** du modèle final (accuracy, classification report)
4. **Conversion et quantification `.tflite`** (int8) pour le déploiement sur MCU
À la fin de l'exécution, deux fichiers sont générés (au chemin défini par `DEPLOY_PATH`) :
- `final_model_deploy.keras` : modèle Keras complet
- `final_model_deploy_int8.tflite` : modèle quantifié int8, prêt pour le déploiement sur MCU


### Performance de référence
 
Le modèle est attendu autour de **88,9 % ± 2,0 %** d'accuracy en validation (référence obtenue par validation croisée k-fold). Une accuracy sensiblement inférieure lors d'un nouvel entraînement peut indiquer un problème de données (déséquilibre des classes, volume insuffisant) ou de configuration.

