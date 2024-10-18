# Projet de Traitement d'Images Sous-Marines

Le but du projet est d’implémenter une méthode qui vise à résoudre les problèmes d’altération des couleurs causés par l’atténuation et la dispersion de la lumière dans l’eau, qui affectent particulièrement les composantes rouges, jaunes et vertes du spectre visible.

## Méthode Proposée
### Transformation en espace $l\alpha\beta$
L’image est d’abord convertie de l’espace RGB à l’espace Lab, qui est un espace perceptuel basé sur la perception humaine des couleurs. Cet espace sépare la luminance (l) des composants chromatiques ($\alpha$ pour jaune-bleu, $\beta$ pour rouge-vert).

### Hypothèse du Monde Gris
La méthode repose sur l’hypothèse du “monde gris”, qui suppose que la moyenne des couleurs dans une image devrait être neutre (grise). Cette hypothèse permet de réaliser un équilibrage des blancs en ajustant les composantes chromatiques autour d’un point blanc dans l’espace Lab.

### Correction Chromatique
Les composantes $\alpha$ et $\beta$ (chromatiques) sont ajustées pour centrer leurs distributions autour du point blanc.

### Amélioration du Contraste
Le contraste de l’image est amélioré en egalisant l’histogramme de luminance (l) pour utiliser toute la gamme dynamique disponible.

## Défis des Images Sous-Marines
Les images sous-marines sont fortement affectées par les propriétés physiques de la lumière lorsqu'elle traverse l'eau. Les principaux défis incluent :
- **Atténuation de la lumière** : La lumière est rapidement absorbée par l’eau, limitant la distance de visibilité à environ 20 mètres en eau claire et à seulement 5 mètres en eau trouble.
- **Diffusion de la lumière** : L’eau provoque la diffusion de la lumière, en particulier avec des particules en suspension (neige marine), qui entraîne un voile flou et un faible contraste.
- **Perte de couleur** : À mesure que la profondeur augmente, les couleurs disparaissent en fonction de leur longueur d’onde. Le rouge disparaît à 3 mètres, l’orange à 5 mètres, et le jaune à environ 10 mètres, laissant une dominante bleue dans les images.

## Travaux Réalisés
Nous avons tout d’abord tenté d’appliquer l’hypothèse du monde gris sur une image codée en RGB sans effectuer le changement de coordonnées vers l’espace colorimétrique $l\alpha\beta$. Comme il est suggéré dans l’article, nous avons également testé cette méthode avec la médiane des canaux pour comparer son efficacité par rapport à celle utilisant la moyenne, mais aucune différence n'a été constaté.

Nous avons tenté d’estimer les composantes de la source illuminant la scène grâce à la méthode des patchs blancs, On identifie les pixels les plus lumineux d'une image, généralement les 1 à 5 % les plus lumineux. Ces pixels sont supposés correspondre à des surfaces blanches ou presque blanches dans la scène.

Nous avons également changé de perspective en essayant d’appliquer la normalisation sur la valeur de la teinte dans l’espace HSL, mais cela s’est avéré peu concluant,car l'espace ne bénéficie que d'une seule dimension chromatique.
Enfin, nous nous sommes concentrés sur le cœur de l’article en élaborant des fonctions pour passer dans l’espace Lab, puis pour revenir dans l’espace RGB, sans encore effectuer la correction de manière concluante.

## Conclusion et Perspectives


L'article **"A New Color Correction Method for Underwater Imaging"** (2015), par **G. Bianco, M. Muzzupappa, F. Bruno, R. Garcia, et L. Neumann**, présente une méthode de correction des couleurs pour les images sous-marines. Cette méthode est basée sur l’utilisation de l’espace colorimétrique $l\alpha\beta$ pour améliorer la qualité des images capturées dans des environnements aquatiques, en corrigeant les distorsions dues à l'éclairage et aux conditions spécifiques de l’eau. L'objectif est d'obtenir des couleurs réalistes sans nécessiter de connaissances précises sur les paramètres physiques du milieu.

### Méthodologie Proposée
Le processus de correction des couleurs repose sur deux hypothèses :
- **Hypothèse du Monde Gris** : l’image moyenne doit être grise, ce qui permet de balancer les couleurs.
- **Illumination Uniforme** : l'éclairage est supposé constant sur toute l'image.

L’espace colorimétrique $l\alpha\beta$ est utilisé pour séparer la luminance des composantes chromatiques, et la correction est effectuée en recentrant les composantes chromatiques autour du point blanc tout en améliorant la luminance par une égalisation de l'histogramme.

### Résultats Expérimentaux
Après traitement, les composantes chromatiques sont recentrées autour du point blanc, supprimant ainsi les dominantes de couleur indésirables. L’amélioration du contraste est obtenue en étirant l’histogramme de luminance.
