# Projet Deep Learning : Classification de sons pensées par analyse de signaux EEG

Menasria Racim, Cavina Théo

Ce repo contient l'ensemble des script utilisés lors de l'étude des deux dataset, FEIS et HablaImaginada. Les deux dataset ont été étudiés séparemment,possèdent chacun leur propre sous dossier et sont indépendants. 

### Pour FEIS : 
* **model** : contient les scripts liés à la conception et à l'entrainement des modèles. 
* **visualisation** : contient des scripts d'exploration des données et de visualisation. 
* **preprocessing** : contient le script permettant d'appliquer la pipeline de preprocessing des données, comprennant le filtrage, la normalisation et la wavelet scattering transformation

### Pour HablaImaginada
* **figures** : contient des statistiques du dataset et des résultats des modèles
* **gridsearch** : contient les scripts pour effectuer des recherche d'hyper-paramètres optimaux pour différents modèles en fonction de différentes méthodes de pré-traitement (swt : stationnary wavelet transformation, dwt : Daubechies wavelet transformation, rien : moyenné dans le temps 
)
* **preprocess** : contient différentes pipeline de preprocessing (dwt, swt, _)
* **train** : contient les scripts liés à la conception et entrainement des modèles 
* **visualisation** : contient les scripts de visualisation des statistiques des données

### Remarque

Nous n'avons pas ajouté les données à cause de leur taille, elles peuvent être téléchargées : 

* FEIS : https://zenodo.org/records/3554128
* HablaImaginada : https://www.kaggle.com/competitions/habla-imaginada-clasificacion-senales-EEG/overview
