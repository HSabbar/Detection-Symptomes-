# Detection-Symptomes-

## Contexte : 
Les données extrait à partir de base de données de NetSoins : entre 30/12/2019 et 03/02/2020 (NetSoins-TRANSMISSIONS-APP).
Nous avons construit le jeu de données en utilisant les opérateurs logiques et les regex (les expressions régulières) en particulier le pattern matching LIKE et NOT LIKE  avec des groupes de mots spécifique qui décrit les symptômes.
[référence à documentation](https://www.postgresql.org/docs/9.5/functions-matching.html)

## Objectif : 
L’objectif de ce travail est de prédire des symptômes potentiels sur les transmissions NetSoins, nous pensons que ce travail peut-être utile pour :
* Prendre des mesures de sécurité avant la maladie. 
* Détecter rapidement des incidents (chutes fréquentes, grippe …)
* Améliorer la vie de nos résidents.
* Augmenter l’espérance de vie.

## Méthode : 

### Pre-Processing : 
Pour réaliser ce travail nous avons utilisé les regex (les expressions régulières) pour le nettoyer du texte :
* Suppression des caractères spéciaux **`['.;:!*=\%?,<>\"()\[\]]`**
* Suppression des balises html restantes dans les transmissions.
* Suppression des espaces en trop.
*la suppression des stopwords, Ce sont les mots très courants dans la langue étudiée ("et", "à", "le"...) qui n'apportent pas de valeur informative pour la compréhension du sens d'un document et corpus. Il sont très fréquents et ralentissent notre travail.

#### Exemple de clean texte : 
```
str = "trés aggressive lors des soins<span style=""text-decoration: underline; background-color: green;"">grande difficulté a la changé frappe++ crie"
new_str = trés aggressive lors des soins grande difficulté a la changé frappe crie 
```
### Normaliser les données :
Normaliser le texte signifie le convertir en un format standard plus pratique avant de le transformer en fonctionnalités pour des algorithmes d’apprentissage. cette étape est considéré comme la conversion d'un langage humain en une forme lisible par la machine.
Un tokenizer sépare le texte en une liste de séquence de mot, qui correspondent à des mots, puis on transforme grâce la fonctionne [`pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)`](https://keras.io/preprocessing/sequence/) les listes des mots en tableaux de formes Numpy avec les mêmes longueurs

### Train test split : 

les données que nous utilisons sont généralement divisées en données d’apprentissage et données de test. L’ensemble d'apprentissage contient une sortie connue et le modèle apprend sur ces données afin d'être généralisé à d'autres données ultérieurement. Nous avons le sous-ensemble des données de test afin de tester la prédiction de notre modèle sur ce sous-ensemble.

### Créer le modèle :

```
MAX_NB_WORDS = 50000 # Le nombre maximum de mots à utiliser.
EMBEDDING_DIM = 100  # int >= 0: Dimension de Embedding dense.

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[1]))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(60, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

```

* La première couche est le [`Embedding`](https://keras.io/layers/embeddings/) initialise d'abord le vecteur de embedding au hasard, puis utilise l'optimiseur de réseau pour le mettre à jour de la même manière que pour toute autre couche réseau en **keras**. cette couche utilise `MAX_NB_WORDS = 100` vecteurs de longueur pour représenter chaque mot, Le `MAX_NB_WORDS = 50000` représente la taille de notre vocabulaire, [`input_length=X_train.shape[1]`](https://keras.io/layers/core/) détermine la taille de chaque séquence d'entrée.
* SpatialDropout1D effectue le Dropout variationnel dans les modèles PNL, l'avantage d'ajouter SpatialDropout par rapport au dropout des keras normaux, dans le SpatialDropout  des canaux embedding complets sont supprimés tandis que le ropout embedding Keras normal supprime tous les canaux pour des mots entiers, et parfois perdre un ou plusieurs mots peut corrompre complètement la signification. ([`keras.layers.SpatialDropout1D(rate)`](https://keras.io/layers/core/) :  **`Dropout`** : consiste à définir au hasard une fraction `rate`  d'unités d'entrée à 0 à chaque mise à jour pendant le temps de formation, ce qui permet d'éviter un surapprentissage.) 
* La couche suivante est la couche LSTM avec 60 unités de mémoire.
* La couche de sortie doit créer 7 valeurs de sortie, une pour chaque classe.
* La fonction d'activation est `softmax` pour la classification multi-classes.
* Parce qu'il s'agit d'un problème de classification multi-classes, `categorical_crossentropy` est utilisé comme fonction de loss.



