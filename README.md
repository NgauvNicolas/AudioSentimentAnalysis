# AudioSentimentAnalysis
Projet de sentiment analysis sur le corpus audio RAVDESS
Article de référence : Singh, P. Nagrath, P. (2022). Vocal Analysis and Sentiment Discernment using AI. *Fusion: Practice and Applications*, (), 100-109. DOI: https://doi.org/10.54216/FPA.070204

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

### 2. Lancer les scripts
```
python3 train_with_LSTM.py
python3 train_with_CNN_regularization_scheduler.py
python3 script_svm.py
```

## LSTM Jeevya 👩🏽‍💻
Nous avons créé un script python qui permet d'entrainer un modèle LSTM sur la reconnaissance de sentiments sur des extraits audio parlés et chantés provenant du corpus RAVDESS Speech & Song.
Comme dans le papier de référence, nous utilisons uniquement les labels 'sad', 'calm', 'happy', 'angry' et 'fearful' pour essayer d'avoir une meilleur accuracy en éliminant les émotions les plus difficile à discerner.
On utilise un split train/test de respectivement 80% et 20% du corpus. L'entrainement se fait sur 70 epochs. Les hyperparamètres utilisés ne sont pas précisés dans l'article donc nous testons par nous même des valeurs. Nous obtenons avec notre modèle **66% d'accuracy** sur le test set. Dans les prochaines étapes, nous comptons utiliser GridSearch pour essayer de trouver de meilleurs hyperparamètres et améliorer l'accuracy du modèle.

## CNN Nicolas 🧑🏻‍💻

Conformément à l'article, on utilise un modèle de **réseau de neurones convolutifs** (ou réseau de neurones à convolution) entrainé sur une partie du dataset RAVDESS (80%) pour effectuer une classification des sentiments détectés dans une autre partie du dataset RAVDESS (20%), afin d'évaluer le modèle CNN.

Les CNN détectent des motifs dans des matrices comme les **MFCC**, à travers des couches convolutionnelles, d'activation et de pooling. Cela permet de construire des modèles capables de classifier les données avec efficacité.

Ici, on a essayé de jouer avec la régularisation (Dropout ou L2) et un apprentissage plus lent et progressif (avec un learning rate scheduler) pour essayer de stabiliser l'entraînement, d'éviter le sur-apprentissage et d'améliorer la capacité de généralisation du modèle.

### Résultats :

![CNN Training Validation Metrics](plots/CNN_training_validation_metrics.png)

**Training Accuracy** : 98,20%

**Validation Accuracy** : 77,13%

**Training Loss** : 0,3042

**Validation Loss** : 0,8233


![CNN Confusion Matrix](plots/CNN_confusion_matrix.png)

**Structure générale** :

Les lignes représentent les labels réels, et les colonnes représentent les labels prédits.<br>
Les valeurs diagonales indiquent les prédictions correctes, et les autres cellules montrent les erreurs de classification.

**Analyse par classe** :

Classe "angry" (angry → ligne 1) :<br>
70 prédictions correctes.<br>
17 prédictions incorrectes : 10 instances classées comme "fearful" et 7 comme "happy".<br>
Interpétation et discussion : Le modèle semble confondre "angry" avec des émotions comme "fearful" ou "happy", ce qui peut s'expliquer par des similitudes acoustiques entre des émotions intenses comme la colère et le bonheur ou encore la peur (le corpus étant composé de voix parlées et chantées par des acteurs, et donc plus préparées que spontanées). À noter que la confusion entre "angry" et "fearful" provient de l'apport de l'apprentissage avec les voix chantées, car lors des tests avec les voix parlées seulement, nous n'avions pas ce résultat.

Classe "calm" (calm → ligne 2) :<br>
59 prédictions correctes.<br>
11 prédictions incorrectes : 8 instances classées comme "sad" et 3 comme "happy".<br>
Interpétation et discussion : Le modèle confond parfois le calme ("calm") avec des émotions négatives comme "sad". Cela est compréhensible, car une voix calme peut être difficile à distinguer d'une voix légèrement triste ou posée dans le contexte audio (les spectres audio peuvent être similaires). Pour les petites confusion entre "calm" et "happy", toutes les expressions de la joir ne sont pas forcément associées à des hauteurs vocales élevées, et si les variations de pitch dans "happy" ne sont pas marquées (par exemple, une expression de bonheur tranquille ou réservé), elles peuvent être perçues comme similaires à une voix calme.

Classe "fearful" (fearful → ligne 3) :<br>
52 prédictions correctes.<br>
17 prédictions incorrects : 13 instances confondues avec "sad" et 4 avec "angry".<br>
Interpétation et discussion : Cela montre une confusion notable entre la peur ("fearful") et la tristesse ("sad"), souvent liée à des signaux audio de faible intensité ou tonalité similaire : certaines caractéristiques acoustiques associées à la peur (comme un ton bas ou tremblant) peuvent ressembler à celles d'une tristesse prononcée. Pour la confusion entre la peur ("fearful") et la colère ("angry"), elle vient certainement de l'apport des voix chantées (car lors de tests avec seulement les voix parlées, nous n'avions pas ce résultat) : la prosodie lors du chant ou de la voix parlée est différente, et lorsque qu'on entraine et évalue le modlèle sur un corpus composé de ces 2 types d'expression orale, il y a forcément plus de confusion.

Classe "happy" (happy → ligne 4) :<br>
58 prédictions correctes.<br>
12 prédictions incorrectes : 9 instances classées comme "fearful", 3 comme "angry" et 3 comme "sad".<br>
Interpétation et discussion : Les confusions avec "fearful" peuvent indiquer une confusion dans les fréquences audio plus aiguës. On peut aussi penser à la variation rapide du pitch : les variations fréquentes ou abruptes du ton sont communes dans les émotions comme la joie, la peur ou la colère. Par exemple, une voix joyeuse peut avoir des montées rapides, tout comme une voix apeurée ou colérique. On peut aussi se dire que cela peut être dû à la diversité des expressions vocales du bonheur, qui peuvent parfois être interprétées comme des émotions intenses (ce que sont la peur et la colère) ou plus mélancoliques (ce qui peut expliquer la confusion avec "sad").

Classe "sad" (sad → ligne 5) :<br>
51 prédictions correctes.<br>
16 prédictions incorrectes : 9 instances confondues avec "calm", 7 avec "fearful" et 7 avec "happy".<br>
Interpétation et discussion : "Sad" est globalement bien reconnue, mais il existe des confusions significatives avec "Calm". Cette confusion est fréquente dans les modèles de classification audio, car des émotions comme la tristesse et le calme partagent souvent des tonalités douces et des rythmes lents. Pour "sad" et "fearful", ces deux émotions sont souvent exprimées avec des tonalités graves ou basses dans la voix, ce qui peut expliquer la confusion. Les confusions entre "sad" et "happy" peuvent sembler contre-intuitives étant donné que ces deux émotions ont une valence émotionnelle opposée (négative pour "sad", positive pour "happy")... Cependant, elles peuvent survenir dans un modèle de classification audio à cause de l'overlap qu'il peut parfois y avoir dans les spectres acoustiques : les descripteurs acoustiques utilisés par les modèles, comme les MFCC, se concentrent principalement sur les propriétés spectrales (fréquence et énergie). Ils peuvent ne pas capturer les différences subtiles de valence émotionnelle.



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

