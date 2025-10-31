<div align="center">
    <p>
        <img src="figures/banner-draw.png">
    </p>


<div>

[![Licence](https://img.shields.io/pypi/l/ultralytics)](LICENSE)
[![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/tiazden)
[![HuggingFace Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHichTala%2Fdraw2&query=%24.downloads&logo=huggingface&label=downloads&color=%23FFD21E)](https://huggingface.co/HichTala/draw2)
[![OBS Plugin](https://img.shields.io/badge/-obs_plugin-302E31?logo=obsstudio&labelColor=555&color=%23302E31)](https://github.com/HichTala/draw2-obsplugin)
[![WandB](https://img.shields.io/badge/visualize_in-W%26B-yellow?logo=weightsandbiases&color=%23FFBE00)](https://wandb.ai/hich_/draw)
[![Medium](https://img.shields.io/badge/Medium-12100E?style=flat&logo=medium&logoColor=white)](https://medium.com/@hich.tala.phd/how-i-trained-a-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-6ea71da007fd)


[🇬🇧 English](README.md)

</div>

</div>

DRAW est le tout premier détecteur d'objets entraîné à détecter les cartes Yu-Gi-Oh! dans tous types d'images, 
et en particulier dans les images de duels.

Avec cette nouvelle version, **DRAW 2** va au-delà de son prédécesseur. Il est plus précis, plus robuste 
et beaucoup plus simple à utiliser.
Il comprend désormais un [plugin OBS](https://github.com/HichTala/draw2-obsplugin) qui permet aux utilisateurs 
d'intégrer de manière transparente le détecteur directement dans leurs streams ou leurs vidéos ;
et ceux **sans avoir de compétences techniques particulières**.
Le plugin peut afficher les cartes détectées en temps réel pour une expérience visuelle améliorée pour les spectateurs.

D'autres travaux existent (voir [Projets connexes](#div-aligncenterprojets-connexesdiv)) mais aucun n'est capable de reconnaître des cartes pendant un duel.

Ce projet est sous licence [GNU Affero General Public License v3.0](LICENCE) ; toutes les contributions sont les bienvenues.

---
## <div align="center">📄Documentation</div>

Si vous souhaitez simplement utiliser le plugin, veuillez vous référer à la [page du plugin OBS](https://github.com/HichTala/draw2-obsplugin).
Dans ce cas, aucune installation n’est nécessaire à partir de ce repo.
La documentation ci-dessous s'adresse aux personnes qui souhaitent utiliser le détecteur en dehors d'OBS, ce qui nécessite certaines compétences techniques.

### 🛠️ Installation

Vous avez besoin d'installer Python. L'installation de Python ne sera pas détaillée ici, vous pouvez vous référer à la [documentation](https://www.python.org/).

Nous devons d'abord installer pytorch. Il est recommandé d'utiliser un gestionnaire de paquets tel que [miniconda](https://docs.conda.io/projects/miniconda/en/latest/). 
Veuillez vous référer à la [documentation](https://docs.conda.io/projects/miniconda/en/latest/).

Lorsque tout est prêt, vous pouvez lancer la commande suivante pour installer pytorch :
```shell
python -m pip install torch torchvision
```
Si vous voulez utiliser votre gpus pour faire tourner le tout plus rapidement, veuillez vous référer à la [documentation](https://pytorch.org/get-started/locally/).

Ensuite, il vous suffit de cloner le repo et d'installer les `requirements`:
```Shell
git clone https://github.com/HichTala/draw
cd draw
python -m pip install -r requirements.txt
```

Cloner le repo n'est pas obligatoire si vous avez déjà tous les requirements installés, vous pouvez simplement exécuter :
```Shell
python -m pip install git+https://github.com/HichTala/draw2.git
```

Votre installation est maintenant terminée.

### 🚀 Utilisation

Une fois l'installation terminée, vous pouvez utiliser le détecteur en exécutant la commande suivante :

```Shell
python -m draw
```

Vous pour ajouter le flag `--help` pour afficher toutes les options disponibles. :
```Shell
python -m draw --help
```

Les options les plus importantes sont les suivantes :
- `--source`: Chemin vers l'image, la vidéo ou l'indice de la webcam (par default, `0` pour la webcam).
- `--save`: Chemin où sauvegarder la vidéo.
- `--show`: Pour afficher la vidéo en temps réel.
- `--display-card`: Pour afficher l'image de la carte détectée.
- `--deck-list`: Chemin vers un fichier ydk contenant la deck lists (permet d'améliorer la précision).
- `--fps`: FPS de la vidéo à sauvegarder (par default, 60).

---
## <div align="center">💡Inspiration</div>

Ce projet a été inspiré par un projet du créateur [SuperZouloux](https://www.youtube.com/watch?v=64-LfbggqKI) 
donnant vie aux cartes _Yu-Gi-Oh!_ à l'aide d'un hologramme. Son projet utilise des puces insérées sous les protections
de chaque carte, qui sont lues par le tapis de jeu, ce qui permet de reconnaître les cartes.

L'insertion des puces dans les protections est non seulement laborieuse, mais pose également un autre problème : 
les cartes face cachée sont lues de la même manière que les cartes face visible. 
Un détecteur automatique est donc une solution tout à fait adaptée.

Bien que ce projet ait été découragé par _KONAMI_ <sup>®</sup>, l'éditeur du jeu (ce qui est tout à fait compréhensible),
on peut néanmoins imaginer un tel système pour afficher les cartes jouées lors d'un duel retransmit en direct, 
pour permettre aux spectateurs de lire les cartes.

---
## <div align="center">🔗Projets connexes</div>

Bien qu'à ma connaissance `draw` soit le premier détecteur capable de localiser et de détecter des cartes _Yu-Gi-Oh!_ dans un environnement de duel, 
d'autres travaux existent et ont été une source d'inspiration pour ce projet. Il convient donc de les mentionner proprement.

[Yu-Gi-Oh ! NEURON](https://www.konami.com/games/eu/fr/products/yugioh_neuron/) est une application officielle développée par _KONAMI_ <sup>®</sup>. Elle est dotée de nombreuses fonctionnalités, dont la reconnaissance des cartes. L'application est capable de reconnaître un total de 20 cartes à la fois, ce qui reste très honorable. L'inconvénient est que les cartes doivent être de bonne qualité pour être reconnues, ce qui n'est pas forcément le cas dans un contexte de duel. De plus, elle n'est pas intégrable, la seule et unique façon de l'utiliser est donc d'utiliser l'application.

[yugioh one shot learning](https://github.com/vanstorm9/yugioh-one-shot-learning) fait par `vanstorm9` est un programme de classification des cartes Yu-Gi-Oh!. Il utilise un réseau de neurones siamois pour entraîner son modèle. Il donne des résultats très impressionnants sur des images de bonne qualité, mais pas très bons sur des images de moins bonne qualité,
et il ne peut pas localiser les cartes.

[Yolov11](https://github.com/ultralytics/ultralytics) est la dernière version de la très célèbre famille `yolo` de modèles de détection d'objets qui permet d'utiliser des boxes orientées. Est-il vraiment nécessaire de le présenter aujourd'hui ? Il représente l'état de l'art en matière de modèle de détection d'objets en temps réel.

[ViT](https://arxiv.org/pdf/2010.11929.pdf) est un modèle pré-entraîné pour la classification d'images basé sur l'architecture Vision Transformer. 
Il s'appuie entièrement sur des mécanismes d'attention pour traiter les fragments d'images au lieu d'utiliser des couches convolutives. 
Il convient bien à notre tâche, car des versions pré-entraînées sur des ensembles de données à grande échelle tels que ImageNet-21K sont disponibles. 
Cela est particulièrement pertinent pour notre cas d'utilisation, car il permet de traiter un grand nombre de catégories visuelles similaires aux plus de 13 000 cartes uniques présentes dans _Yu-Gi-Oh!_.


[SpellTable](https://spelltable.wizards.com/) est une application gratuite conçue et réalisée par `Jonathan Rowny` et son équipe pour jouer à _Magic : The Gathering_ à distance. 
Elle permet au joueur de cliquer sur une carte sur le flux de n'importe quel joueur pour l'identifier rapidement. 
Il a quelques similitudes avec `draw` puisqu'il rend possible la localisaton et la reconnaissance de n'importe quelle carte à partir d'une base de données de 17 000 cartes.
L'idée est proche de ce projet, mais elle n'en est pas à l'origine.

---
## <div align="center">💬Contact</div>

Vous pouvez me joindre sur Twitter [@tiazden](https://twitter.com/tiazden) ou par mail [hich.tala.phd@gmail.com](mailto:hich.tala.phd@gmail.com).

---
## <div align="center">⭐Historique des Stars</div>

<a href="https://www.star-history.com/#HichTala/draw2&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=HichTala/draw2&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=HichTala/draw2&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=HichTala/draw2&type=date&legend=top-left" />
 </picture>
</a>
