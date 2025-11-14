# <div align="center">Classification distribuée par un arbre de décision</div>

<div align="center">

**Devoir 2 - 8INF919**

*Gabin Vrillault - VRIG05020300*

---

</div>

## Table des matières

- [Travail demandé](#section-2)
- [Conclusion](#conclusion)


## Section 2: Travail demandé

### Sous-section 2.1

Synthèse de MR-tree.


### Sous-section 2.2
#### *Sous-sous-section 2.2.1*
(élaborer différents scénarios de surcharge selon le protocole d’expérimentation décrit dans l’article, en s’appuyant sur les deux configurations suivantes :)
1. Mode local
  Ma machine locale (MacBook Air M1) possède 7 coeurs. J'ai donc réalisé l'expérience en utilisant l'échelle de 1 coeur pour 250Ko de données.
  J'ai testé des facteurs d'échelle de 1, 8, 32, 50, 64, 100 et 128, ce qui correspond à des tailles de jeu de données allant de 1.75 Mo à 224 Mo.
  Je n'ai pas pu aller au-delà de 128 car la mémoire de mon ordinateur était insuffisante pour traiter des ensembles de données plus volumineux.

2. Mode cluster
  Sur Calcul-Québec, j'ai utilisé un cluster avec 4 nœuds de calcul, avec une échelle de 1 noeud pour 10 Mo de données.
  J'ai testé des facteurs d'échelle de 1, 8, 32, 64, 128, 256, 512 et 1024, ce qui correspond à des tailles de jeu de données allant de 40 Mo à 40 Go.

J'ai utilisé le jeu de données "Adult" pour les deux configurations. C'est un jeu de données qui contient des informations démographiques et est utilisé pour faire des prédictions sur le revenu des individus.
Ce dataset fait environ 3.8 Mo, j'ai donc dû le dupliquer plusieurs fois pour atteindre les tailles de données souhaitées. Ici le but n'est pas d'obtenir des résultats précis, mais plutôt de tester la scalabilité de l'algorithme MR-Tree donc la duplication des données est acceptable.

#### *Sous-sous-section 2.2.2*
Graphiques et tableaux des résultats obtenus.
["Voir les graphiques et tableaux dans les fichiers joints."](./graphs_and_tables.md)

On peut observer que dans les deux configurations, on obtient des résultats similaires à ceux présentés dans l'article original.
L'algorithme MR-Tree montre une bonne scalabilité avec l'augmentation de la taille des données le temps d'exécution augmente beaucoup moins rapidement que la taille des données.

### Sous-section 2.3
(Pour montrer la nécessité de la distribution de l’apprentissage dans le  cas  du  BigData, expérimenter le scénario suivant :)

#### *Sous-sous-section 2.3.1*
évaluation du temps d'entrainement de l'algorithme MR-ID3 en mode cluster avec 2, 3 et 4 nœuds et un même volume de données ["NF-ToN-IoT-v2"](https://huggingface.co/datasets/Nora9029/NF-ToN-IoT-v2) de 1.98 Go qui est un jeu de donnée de trafic réseau pour la détection d'intrusions.
| 2 Noeuds  | 3 Noeuds  | 4 Noeuds  |
|-----------|-----------|-----------|
| 00:45:12  | 00:32:45  | 00:25:30  |

#### *Sous-sous-section 2.3.2*
(Est-ce que  la performance du  modèle appris augmente  du fait de l’entraînement sur un volume important de données ? Justifier votre réponse ?)
Non, la performance du modèle n'augmente pas nécessairement avec un volume de données plus important.

### Sous-section 2.4
(Montrez,  en  faisant  la  comparaison  avec  scikit-learn  en  python  et  MLlib  en 
pyspark, pour un jeu  de données  important, que le  recours à une validation croisée 
distribuée pour trouver le meilleur modèle,  est indispensable. Vous pouvez-réutilisez 
le dataset de la question 2.3. Justifier votre réponse)

En utilisant le dataset "NF-ToN-IoT-v2" de 1.98 Go, j'ai comparé les performances de scikit-learn en Python et MLlib en PySpark pour entraîner un modèle de classification.