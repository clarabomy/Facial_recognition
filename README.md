# Projet Reconnaissance faciale
---

## Moteur de reconnaissance faciale

### Setup
1. Télécharger et installer CUDA Toolkit 9.0 et NVIDIA cuDNN v7.6.4
* [CUDA Toolkit download](https://developer.nvidia.com/cuda-90-download-archive)
* [NVIDIA CuDNN download](https://developer.nvidia.com/rdp/cudnn-archive)
* [Tutoriel d'installation](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html)

2. Télécharger et installer [Python 3.6](https://www.python.org/downloads/release/python-368/)

3. Installer les dépendances du projet via la commande : `pip install -r requirements.txt`

### Téléchargement des modèles de reconnaissance
Pour lancer correctement le programme, vous devez préalablement télécharger un des modèles de reconnaissance faciale pré-entrainé, disponibles [ici](https://github.com/deepinsight/insightface/wiki/Model-Zoo).

Il suffit ensuite de dézipper le dossier et de préciser son url dans le fichier config.json du projet.


### Lancement du programme
Le programme se lance via la commande : `python main_function.py`

