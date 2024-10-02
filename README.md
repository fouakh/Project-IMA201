L’article **"Underwater Image Processing: State of the Art of Restoration and Image Enhancement Methods"** (2010) par **Raimondo Schettini** et **Silvia Corchs** fournit une analyse approfondie des méthodes actuelles utilisées pour traiter les images sous-marines :

### 1. **Défis des images sous-marines** :
Les images sous-marines sont fortement affectées par les propriétés physiques de la lumière lorsqu'elle traverse l'eau. Les principaux défis incluent :
- **Atténuation de la lumière** : La lumière est rapidement absorbée par l’eau, limitant la distance de visibilité à environ 20 mètres en eau claire et à seulement 5 mètres en eau trouble.
- **Diffusion de la lumière** : L’eau provoque la diffusion de la lumière, en particulier avec des particules en suspension (neige marine), qui entraîne un voile flou et un faible contraste.
- **Perte de couleur** : À mesure que la profondeur augmente, les couleurs disparaissent en fonction de leur longueur d’onde. Le rouge disparaît à 3 mètres, l’orange à 5 mètres, et le jaune à environ 10 mètres, laissant une dominante bleue dans les images.

### 2. **Restauration des images sous-marines** :
La **restauration d'image** vise à inverser les dégradations de l'image en utilisant des modèles physiques pour reproduire le processus de formation de l’image.
- **Modèle de Jaffe-McGlamery** : Il est utilisé pour modéliser l’image sous-marine en tenant compte des trois composants principaux de la lumière captée par la caméra :
  1. **Composant direct** : La lumière qui atteint directement l’objet et revient à la caméra.
  2. **Composant diffusé vers l’avant** : La lumière réfléchie par l'objet qui est légèrement diffusée avant d'atteindre la caméra.
  3. **Composant diffusé vers l’arrière (backscatter)** : La lumière qui rebondit sur des particules en suspension avant de revenir vers la caméra, créant un voile sur l’image.
  
- **Méthodes de déconvolution** : Utilisées pour restaurer les images floues en tenant compte des propriétés optiques de l’eau. Ces méthodes requièrent des informations sur les propriétés optiques du milieu sous-marin (coefficients d’atténuation et de diffusion), ce qui est difficile à obtenir avec précision.

### 3. **Amélioration des images** :
L’**amélioration des images** vise à améliorer la qualité visuelle sans nécessairement modéliser les causes physiques des dégradations :
- **Amélioration du contraste** : Les méthodes d'étirement du contraste permettent d’accentuer les différences entre les différentes parties de l’image, en particulier dans les images sous-marines où les objets peuvent être peu visibles à cause de l'atténuation de la lumière.
- **Correction de la couleur** : Plusieurs méthodes cherchent à rétablir les couleurs d’origine en modifiant les composantes RGB (rouge, vert, bleu) pour compenser la perte de couleur due à la profondeur. Par exemple :
  - **Iqbal et al. (2007)** ont proposé une méthode de correction des couleurs basée sur un modèle de couleur HSI (Hue, Saturation, Intensity) pour améliorer la saturation et restaurer la teinte originale.
  - **Petit et al. (2009)** ont introduit une méthode basée sur des quaternions pour corriger la teinte en fonction de l’atténuation de la lumière.

### 4. **Problèmes d'éclairage** :
L'éclairage est un défi majeur dans les environnements sous-marins :
- **Éclairage artificiel** : Utiliser des lumières sous-marines pour éclairer les scènes crée souvent des points lumineux non uniformes, avec un centre suréclairé et des bords sombres.
- **Méthodes de correction de l'éclairage** :
  - **Filtrage homomorphique** : Transforme les composantes d’illumination en une composante additive et applique un filtre passe-haut pour équilibrer la lumière, supprimant ainsi les variations d'éclairage tout en augmentant le contraste.
  - **Égalisation locale de l'histogramme** : Applique une transformation pour améliorer le contraste dans chaque petite région de l’image, atténuant ainsi les effets de l’éclairage non uniforme.

### 5. **Correction de la couleur sous-marine** :
Les images sous-marines souffrent souvent d’un déséquilibre colorimétrique à cause de la perte des couleurs à différentes profondeurs. Les techniques de correction visent à ajuster les canaux de couleur pour restaurer des teintes naturelles.
- **Algorithmes basés sur la perception** : Par exemple, l'algorithme ACE (Automatic Color Enhancement) simule des mécanismes d’adaptation visuelle pour corriger les couleurs de manière automatique, comme proposé par **Chambah et al. (2004)**.

### 6. **Évaluation de la qualité des images** :
Il est difficile d'évaluer objectivement la qualité des images sous-marines restaurées ou améliorées, car il n’existe souvent pas d’image de référence parfaite avec laquelle comparer. Les métriques classiques comme le **PSNR** (rapport signal sur bruit de pic) ne suffisent pas toujours.
- **Mesure de la netteté** : Une approche consiste à évaluer la netteté des bords dans l’image via la transformée en ondelettes, comme le propose **Hou et Weidemann (2007)**.
- **Indices de robustesse** : Certains chercheurs, comme **Arnold-Bos et al. (2005)**, ont développé des critères basés sur la forme de l’histogramme des gradients de l’image pour évaluer la qualité après amélioration.

### 7. **Conclusion et perspectives** :
L’article conclut qu’il reste encore des défis majeurs dans le traitement des images sous-marines. Bien que plusieurs algorithmes aient été développés pour améliorer la visibilité et restaurer les couleurs, il manque encore une **base de données commune** de référence pour comparer objectivement les résultats des différentes méthodes. L’évolution des technologies d’imagerie optique et les études sur la vision des animaux marins pourraient également ouvrir de nouvelles perspectives pour améliorer la qualité des images sous-marines.

L’article **"A New Color Correction Method for Underwater Imaging"** (2015), par **G. Bianco, M. Muzzupappa, F. Bruno, R. Garcia, et L. Neumann**, présente une méthode innovante de correction des couleurs pour les images sous-marines. Cette méthode est basée sur l’utilisation de l’espace colorimétrique lαβ pour améliorer la qualité des images capturées dans des environnements aquatiques, en corrigeant les distorsions dues à l'éclairage et aux conditions spécifiques de l’eau. L'objectif est d'obtenir des couleurs réalistes sans nécessiter de connaissances précises sur les paramètres physiques du milieu.

1. **Problème de la correction des couleurs sous-marines** :
   - Sous l'eau, les couleurs sont altérées par la diffusion et l'absorption de la lumière à différentes longueurs d'onde, ce qui cause une perte des couleurs (le rouge disparaît à 5 mètres, l'orange à 7,5 mètres, etc.).
   - Les images deviennent souvent bleues ou vertes, avec une faible visibilité et un faible contraste. 
   - La correction des couleurs est essentielle pour des applications telles que la photogrammétrie 3D, la navigation sous-marine et la documentation.

2. **Méthodologie proposée** :
   - Le processus de correction des couleurs repose sur deux hypothèses :
     - **Hypothèse du monde gris** : l’image moyenne doit être grise, ce qui permet de balancer les couleurs.
     - **Illumination uniforme** : l'éclairage est supposé constant sur toute l'image.
   - L’espace colorimétrique **lαβ**, développé pour modéliser la perception visuelle humaine, est utilisé pour séparer la luminance (composante achromatique) des composantes chromatiques (opposants jaune-bleu et rouge-vert).
   - La correction des couleurs est effectuée en recentrant les composantes chromatiques α et β autour du point blanc (balance des blancs), tandis que la luminance est améliorée par une égalisation de l'histogramme pour augmenter le contraste.

3. **Résultats expérimentaux** :
   - Deux images sous-marines capturées dans différents environnements ont été utilisées pour tester la méthode :
     - Une image d'un mur de briques à Baia (Naples, Italie) avec une composante verdâtre.
     - Une image d'une amphore avec une dominante bleue (Turquie).
   - Après traitement, les composantes chromatiques sont recentrées autour du point blanc, supprimant ainsi les dominantes de couleur indésirables.
   - L’amélioration du contraste est obtenue en étirant l’histogramme de luminance après avoir supprimé 1 % des valeurs aux extrémités.

4. **Comparaison avec d'autres méthodes** :
   - La méthode est comparée à d’autres espaces colorimétriques (CIELab, YCrCb, CIECAM97) et à des algorithmes de correction automatique comme le "Gray World" en espace RGB.
   - Les résultats montrent que l’espace lαβ offre une meilleure correction des couleurs sous-marines en supprimant efficacement les dominantes bleues et vertes.

5. **Applications pratiques** :
   - La méthode a été appliquée à un ensemble d'images sous-marines pour la reconstruction 3D d’un site archéologique à Kaulon (Italie).
   - La correction des couleurs améliore la qualité des textures appliquées sur les modèles 3D.

### Conclusion :
L'article propose une méthode efficace pour corriger les couleurs des images sous-marines en utilisant l’espace lαβ, qui permet de traiter les images en temps réel grâce à un faible coût computationnel. Les résultats expérimentaux montrent une nette amélioration des images en termes de réalisme colorimétrique et de contraste, et la méthode est adaptée à des applications pratiques telles que la photogrammétrie sous-marine. Des recherches futures pourraient inclure des ajustements pour tenir compte des conditions d'éclairage non uniforme et des comparaisons plus poussées avec d'autres méthodes de correction des couleurs.