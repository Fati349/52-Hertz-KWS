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
├── data/                  # Données audio brutes a ajouter
├── preprocessing/         # Scripts de génération des mel-spectrogrammes
├── resnet v2 s1/          # Scripts de transfer learning, des tests de deploiement et postprocessing
├── models/                 # Modèles entraînés 
├── requirements.txt
└── README.md
```

## Installation

### Prérequis

- Python 3.8 ou supérieur
- pip
- git

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

## Utilisation

```bash
# 1. Prétraitement des données
python preprocessing/prepare_data.py

# 2. Transfer learning
python training/train.py

# 3. Post-traitement / conversion vers .tflite
python postprocessing/convert_to_tflite.py
```

Le fichier final `.tflite` sera généré dans le dossier `models/`, prêt à être déployé sur MCU.