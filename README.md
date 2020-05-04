# Projet Reconnaissance faciale
---

## Moteur de reconnaissance faciale

### Setup
1. Télécharger et installer CUDA Toolkit 9.0 et NVIDIA cuDNN v7.6.4
* [CUDA Toolkit download](https://developer.nvidia.com/cuda-90-download-archive)
* [NVIDIA CuDNN download](https://developer.nvidia.com/rdp/cudnn-archive) => [Tutoriel d'installation](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

2. Télécharger et installer [Python 3.6](https://www.python.org/downloads/release/python-368/)

3. Installer les dépendances du projet via la commande : `pip install -r requirements.txt`

### Téléchargement des modèles de reconnaissance
Pour lancer correctement le programme, vous devez préalablement télécharger un des modèles de reconnaissance faciale pré-entrainés, disponibles [ici](https://github.com/deepinsight/insightface/wiki/Model-Zoo). Il suffit ensuite de dézipper le dossier et de préciser son url dans le fichier config.json du projet.

Si vous utilisez le programme sur un ordinateur portable, nous vous conseillons de télécharger les modèles LResNet50E-IR ou  LResNet34E-IR. Avec une meilleure configuration matérielle, ne pas hésiter à télécharger le modèle LResNet100E-IR pour de meilleures performances :)

### Setup du fichier config
La configuration du projet s'effectue dans le fichier config.json. Avant lancement du programme, il est conseillé de vérifier les informations contenues dans ce fichier.

### Lancement du programme
Dans le répertoire racine, utiliser la commande suivante pour lancer le programme : `python main_function.py -c config.json`.

