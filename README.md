# speckles-interferometry
Test on speckles interferometry for binaries stars resolution

# Running and installation

main.py calculate autocorrelation mean from fit files in directory

postprocess.py is a streamlite program that will load the results from main.py and calculate a mean filter for removing noise, then try to find an ellipse that fit the figure, use the angle of this ellipse to slice the image to obtain a 1D curve on which I use gradient to isolate peaks and then calculate the distance between peaks.

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/streamlite.png "streamlit") 

Other scripts are tests and ML test models (WIP)


# Principe de l'analyse


## Image de base


L'image obtenue par un téléscope avec une focale élevée d'une étoile double dans un cas idéal devrait ressembler à deux tâches d'Airy distinctes comme sur le schéma-ci dessous. Ces deux tâches sont liées au phénomène de diffraction engendré par le système optique permettant l'observation. 

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/airy.png "Etoiles doubles") 

La présence de l'atmosphère "dégrade" l'image. On peut considérer que l'atmosphère est composée de "bulles" plus ou moins grandes ayant des indices de réfraction différents, engendrant la présence de multiples trajets pour la lumière.

Pour réduire au maximum ces perturbations, on prend des temps de pose très courts pour "figer" l'atmosphère, et on multiplie les images (de 1000 à 3000 images pour un système d'étoiles). Les images reçues ressemble à cela : 

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/cou619.png "Etoiles doubles") 

## Autocorrélation

Afin d'extraire les informations utiles, à savoir l'angle formé par les 2 étoiles et la distance angulaire, on réalise l'intercorrélation de chaque image, et on réalise la somme des images. C'est ce que fait le script main.py, en passant calculant la densité spectrale de puissance (qui par le théorème de  Wiener-Khinchin est analogue à l'auto corrélation de l'image).

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/a1453.png "Auto corrélation") 

Pour diminuer le bruit, on applique un filtre médian :

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/a1453_mean_filtering.png "Auto corrélation") 


## Suppression du pic central

Le problème est que l'image est alors composé de 3 pics. Le pic central d'autocorrélation, et les deux pics secondaires liés à la corrélation de l'étoile principale et l'étoile secondaire. Ces 3 pics circulaires se confondent rendant difficile de trouver les 3 centres nécessaires pour calculer distance et angle. 

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/double.png "pics corrélation")

Le pic central est ... centré sur l'image. On connaît donc très bien son centre. Reste à déterminer le centre d'au moins l'un des deux autres pics. Pour cela, le programme calcule l'ellipse qui englobe le mieux l'image obtenue. Puis, en il calcule la perpendiculaire à l'axe principal de l'ellipse et extrait un vecteur contenant les valeurs de l'image sur cette perpendiculaire.

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/doubleellipse.png "Auto corrélation") 

Ce vecteur contient en fait la contribution du pic central à l'image. Le script transforme ensuite ce vecteur en un masque circulaire qu'il applique à l'image :

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/doubleequation.png "Auto corrélation") 

Une fois cela fait, il ne reste qu'a détecter les pics secondaires, calculer la distance et l'angle...

## En pratique

Pour reprendre l'exemple précédent : 

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/a1453_mean_filtering.png "Auto corrélation") 

Voici le masque circulaire obtenue via le vecteur perpendiculaire à l'axe principal de l'ellipse : 

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/a1453_central_contribution.png "Auto corrélation") 

Et voici l'image obtenue en soustrayant le masque. 

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/a1453_result.png "Auto corrélation") 

En traçant le graph correspondant à l'image obtenue le long de l'axe principal de l'ellipse, on retrouver bien les deux pics secondaires :

![Alt text](https://raw.githubusercontent.com/air01a/speckles-interferometry/main/image/a1453_graph.png "Auto corrélation") 

