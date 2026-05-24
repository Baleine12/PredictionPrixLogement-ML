# Prédiction du prix des logements avec Machine Learning

## Présentation du projet

Ce projet a pour objectif de prédire le prix d'un logement à partir de plusieurs caractéristiques.

La base utilisée est simulée avec Python. Chaque ligne représente un logement avec des informations comme la surface, le nombre de pièces, la distance au centre, l'ancienneté du bâtiment ou encore le quartier.
L'idée est de reproduire une situation classique en analyse de données. Estimer une valeur à partir de plusieurs variables explicatives.

## Objectif

L'objectif est de prédire la variable `prix`
Cette variable correspond au prix estimé d'un logement.

## Variables utilisées

La base contient plusieurs variables :

`surface` : surface du logement 
`nombre_pieces` : nombre de pièces 
`distance_centre` : distance par rapport au centre 
`anciennete_batiment` : ancienneté du bâtiment 
`etage` : étage du logement 
 `balcon` : présence ou non d'un balcon 
`parking` : présence ou non d'une place de parking 
`quartier` : type de quartier 
`prix` : prix du logement.

## Méthode

Le projet suit plusieurs étapes :

 création d'une base de logements simulée 
 analyse descriptive des données 
 visualisation du prix selon certaines variables 
 préparation des variables pour la modélisation 
 estimation d'une régression linéaire 
 estimation d'un modèle Random Forest 
 comparaison des performances des modèles 
 analyse de l'importance des variables 
 export des résultats et des graphiques

## Modèles utilisés

Deux modèles sont comparés :

- une régression linéaire 
- un Random Forest Regressor

Les modèles sont évalués avec trois indicateurs :

- MAE : erreur absolue moyenne 
- RMSE : racine de l'erreur quadratique moyenne 
- R² : part de la variation du prix expliquée par le modèle

## Résultats

La base contient 1 500 logements simulés.

Dans ce projet, la régression linéaire obtient de meilleurs résultats que le Random Forest :

- MAE : environ 19 963 
- RMSE : environ 25 201 
- R² : environ 0,974

Le Random Forest obtient aussi de bons résultats, avec un R² d'environ 0,952, mais son erreur est plus élevée.
Ce résultat est cohérent, car la base simulée a été construite avec une relation assez linéaire entre les caractéristiques des logements et leur prix.

## Interprétation

La variable la plus importante est la surface du logement.
Cela signifie que dans cette simulation la surface explique une grande partie des différences de prix entre les logements.

Les autres variables importantes sont :

- la distance au centre 
- le type de quartier 
- l'ancienneté du bâtiment 
- le nombre de pièces.

Ces résultats sont cohérents avec une lecture simple du marché immobilier. Un logement plus grand, plus proche du centre, situé dans un meilleur quartier et plus récent a généralement un prix plus élevé.

## Fichiers générés

Le programme génère plusieurs fichiers :

- `donnees/logements_simules.csv` : base de logements simulée 
- `sorties/comparaison_modeles.csv` : comparaison des modèles 
- `sorties/importance_variables.csv` : importance des variables 
- `sorties/predictions_prix.csv` : prix réels et prix prédits 
- `sorties/graphiques/prix_selon_surface.png` 
- `sorties/graphiques/prix_selon_distance.png` 
- `sorties/graphiques/prix_moyen_quartier.png` 
- `sorties/graphiques/importance_variables.png`
- `sorties/graphiques/prix_reels_vs_predits.png`
