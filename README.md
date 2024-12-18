# AudioSentimentAnalysis
Projet de sentiment analysis sur le corpus audio RAVDESS

## Auteurs
- AROUN Jeevya
- NGAUV Nicolas
- THEZENAS Anissa

## Usage
### 1. Télécharger le dataset

Le dataset RAVDESS Speech et Song est disponible sur Kaggle :
```console
#!/bin/bash
curl -L -o ~/Downloads/ravdess-emotional-speech-audio.zip\
  https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-speech-audio

curl -L -o ~/Downloads/ravdess-emotional-song-audio.zip\
  https://www.kaggle.com/api/v1/datasets/download/uwrfkaggler/ravdess-emotional-song-audio
```
**ATTENTION** Il faut obligatoirement supprimer les dossiers `audio_speech_actors_01-24/` et `audio_song_actors_01-24/` présents dans les dossiers téléchargés car ils contiennent des doublons et peuvent donc influencer les résultats obtenus sur le test set.
Placer le dataset dans le dossier `data/`.

## CNN Nicolas 🧑🏻‍💻

## LSTM Jeevya 👩🏽‍💻

## SVM (Support Vector Machine) Anissa👩🏾‍💻

### Objectif

Le modèle SVM est utilisé pour effectuer une classification des émotions détectées dans le corpus audio. SVM est un algorithme d’apprentissage supervisé qui sépare les différentes classes en maximisant la marge entre celles-ci grâce à un hyperplan optimal.

### Rapport de classification :

| **Classe** | **Précision** | **Rappel** | **F1-Score** | **Support** |
|------------|--------------:|-----------:|-------------:|------------:|
| 0          | 0.66          | 0.81       | 0.72         | 31          |
| 1          | 0.81          | 0.74       | 0.77         | 34          |
| 2          | 0.80          | 0.67       | 0.73         | 42          |
| 3          | 0.64          | 0.57       | 0.61         | 40          |
| 4          | 0.62          | 0.71       | 0.66         | 45          |

### Moyenne et Accuracy :

| **Métrique**      | **Valeur** |
|-------------------:|----------:|
| **Accuracy**      | 0.69      |
| **Macro Average** | 0.70      |
| **Weighted Avg**  | 0.70      |

