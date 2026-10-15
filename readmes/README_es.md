<div align="center">
    <p>
        <img src="../docs/assets/banner.png" alt="Banner del proyecto DRAW2">
    </p>


<div>

[![Web Demo](https://img.shields.io/badge/🌐_Web_Demo-Try_It_Now-blue)](https://hichtala.github.io/draw2)

[![Licence](https://img.shields.io/pypi/l/ultralytics)](../LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/HichTala/draw2?logoColor=%23181717)](https://github.com/HichTala/draw2)
[![Twitter](https://img.shields.io/badge/-twitter-000?logo=x&labelColor=555)](https://twitter.com/hichtala)
[![HuggingFace Downloads](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2FHichTala%2Fdraw2&query=%24.downloads&logo=huggingface&label=downloads&color=%23FFD21E)](https://huggingface.co/HichTala/draw2)
[![Plugin](https://img.shields.io/badge/-plugin_for_obs-302E31?logo=obsstudio&labelColor=555&color=%23302E31)](https://github.com/HichTala/draw2-plugin)
[![WandB](https://img.shields.io/badge/visualize_in-W%26B-yellow?logo=weightsandbiases&color=%23FFBE00)](https://wandb.ai/hich_/draw)
[![Medium](https://img.shields.io/badge/-Medium-12100E?style=flat&logo=medium&labelColor=555)](https://medium.com/@hich.tala.phd/how-i-trained-again-my-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-5c567a320b0a)

[🇬🇧 English](../README.md) | [🇫🇷 Français](README_fr.md) | [🇯🇵 日本語](README_jp.md)


</div>

</div>

> 🌐 **¿Buscas una forma sencilla de probar DRAW 2?**  
> Visita nuestra **[demo web](https://hichtala.github.io/draw2)** oficial. ¡No necesitas instalar nada!

DRAW 2 (siglas de **D**etect and **R**ecognize **A** **W**ide range of cards, versión 2) es un detector de objetos
entrenado para detectar cartas de _Yu-Gi-Oh!_ en todo tipo de imágenes, y en particular en imágenes de duelos.

Con esta nueva versión, **DRAW 2** va más allá que su predecesor. Es más preciso, más robusto y mucho más fácil de usar.
Ahora incluye un [plugin para OBS](https://github.com/HichTala/draw2-plugin) que permite integrar el detector
directamente en transmisiones en vivo o en videos grabados, **sin necesidad de conocimientos técnicos especiales**.
El plugin puede mostrar en tiempo real las cartas detectadas para mejorar la experiencia de los espectadores.
(Este es un plugin de terceros para OBS Studio. No está afiliado ni respaldado por el proyecto OBS).

Existen otros trabajos (consulta [Trabajos relacionados](#trabajos-relacionados)), pero ninguno es capaz de reconocer
cartas durante un duelo.

Este proyecto está bajo la licencia [GNU Affero General Public License v3.0](../LICENSE); todas las contribuciones son
bienvenidas.

---

## <div align="center">📄Documentación</div>

Si solo quieres usar el plugin, consulta la [página del plugin](https://github.com/HichTala/draw2-plugin).
No necesitas instalar nada de este repositorio.
La documentación que sigue está dirigida a quienes quieran usar el detector fuera de OBS, y requiere algunos
conocimientos de programación.

### 🛠️ Instalación

Necesitas tener Python instalado. La instalación de Python no se detalla aquí; puedes consultar
la [documentación](https://www.python.org/).

Se recomienda usar un gestor de paquetes como [miniconda](https://docs.conda.io/projects/miniconda/en/latest/).
Consulta la [documentación](https://docs.conda.io/projects/miniconda/en/latest/).

Una vez que todo esté listo, solo tienes que clonar el repositorio e instalar las dependencias:

```shell
git clone https://github.com/HichTala/draw2
cd draw2
python -m pip install .
```

Si no quieres clonar el repositorio y ya tienes todos los requisitos instalados, puedes ejecutar directamente:

```shell
python -m pip install git+https://github.com/HichTala/draw2.git
```

La instalación está completa.

### 🚀 Uso

Una vez terminada la instalación, puedes usar el detector ejecutando el siguiente comando:

```shell
python -m draw
```

Puedes usar la opción `--help` para ver todas las opciones disponibles:

```shell
python -m draw --help
```

Estas son las opciones más importantes:

- `--source`: ruta a tu imagen o video, o índice de la cámara web (por defecto `0`, la cámara web).
- `--save`: ruta donde guardar el resultado.
- `--save-images`: guarda las imágenes de las cartas y las fotos de las cartas detectadas.
- `--show`: muestra el resultado en una ventana.
- `--display-card`: muestra las cartas detectadas sobre el resultado.
- `--deck-list`: ruta a un archivo ydk con la lista de cartas de tu deck, para mejorar el reconocimiento.
- `--fps`: FPS del video guardado (por defecto 60).
- `--language`: idioma en el que se muestran los nombres de las cartas, entre EN (por defecto), FR, IT, DE y PT.

---

## <div align="center">💡Inspiración</div>

Este proyecto está inspirado en la idea del creador de contenido [SuperZouloux](https://www.youtube.com/watch?v=64-LfbggqKI)
de un holograma que da vida a las cartas de _Yu-Gi-Oh!_.
Su proyecto utiliza chips insertados bajo los protectores (sleeves) de cada carta,
que son leídos por el tapete de juego y permiten reconocer las cartas.

Insertar los chips en los protectores no solo es laborioso, sino que además plantea otro problema:
las cartas boca abajo se leen igual que las boca arriba.
Por eso, un detector automático es una solución muy adecuada.

Aunque _KONAMI_ <sup>®</sup>, el editor del juego, desalentó este proyecto (lo cual es bastante comprensible),
podemos imaginar un sistema así usándose para mostrar las cartas jugadas durante un duelo en vivo,
de modo que los espectadores puedan leerlas.

---

## <div align="center">🔗Trabajos relacionados</div>

Aunque, hasta donde sé, `draw` es el primer detector capaz de localizar y reconocer cartas de _Yu-Gi-Oh!_ en un entorno
de duelo, existen otros trabajos que sirvieron de inspiración para este proyecto. Vale la pena mencionarlos aquí.

[Yu-Gi-Oh! NEURON](https://www.konami.com/games/eu/fr/products/yugioh_neuron/) es una aplicación oficial desarrollada por
_KONAMI_ <sup>®</sup>.
Tiene muchas funciones, entre ellas el reconocimiento de cartas. La aplicación puede reconocer hasta 20 cartas a la vez,
lo cual es bastante bueno.
La desventaja es que las cartas deben estar en buen estado para ser reconocidas, lo que no siempre ocurre en el contexto
de un duelo.
Además, no se puede integrar en otras herramientas, así que la única forma de usarla es a través de la propia aplicación.

[yugioh one shot learning](https://github.com/vanstorm9/yugioh-one-shot-learning), de `vanstorm9`, es un programa de
clasificación de cartas de Yu-Gi-Oh! que permite reconocerlas. Utiliza una red siamesa para entrenar su modelo de
clasificación. Da resultados muy impresionantes en imágenes de buena calidad, pero no tanto en imágenes de baja calidad,
y no puede localizar las cartas.

[Yolov11](https://github.com/ultralytics/ultralytics) es la última versión de la famosísima familia `yolo` de modelos de
detección de objetos, y admite cajas delimitadoras orientadas (oriented bounding boxes).
Creo que hoy no necesita presentación: representa el estado del arte en detección de objetos en tiempo real.

[ViT](https://arxiv.org/pdf/2010.11929.pdf) es un modelo preentrenado de clasificación de imágenes basado en la
arquitectura Vision Transformer.
Se basa por completo en mecanismos de atención para procesar fragmentos (patches) de la imagen, en lugar de usar capas
convolucionales.
Se adapta bien a nuestra tarea, ya que existen versiones preentrenadas en conjuntos de datos a gran escala como
ImageNet-21K.
Esto es especialmente relevante en nuestro caso, porque permite manejar una gran cantidad de categorías visuales, como
las más de 13 000 cartas únicas de _Yu-Gi-Oh!_.

[SpellTable](https://spelltable.wizards.com/) es una aplicación gratuita diseñada y desarrollada por `Jonathan Rowny` y
su equipo para jugar _Magic: The Gathering_ en papel a distancia.
Permite hacer clic en una carta en la cámara de cualquier jugador para identificarla rápidamente.
Tiene cierta similitud con `draw`, ya que puede localizar y reconocer cualquier carta de una base de datos integrada de
17 000 cartas.
La idea es cercana a la de este proyecto, aunque no fue su origen.

---

## <div align="center">🔍Descripción del método</div>

Escribí una entrada de blog en Medium que explica el proceso principal, desde la recolección de datos hasta la
predicción final. Puedes leerla
[aquí](https://medium.com/@hich.tala.phd/how-i-trained-again-my-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-5c567a320b0a).
Si tienes alguna pregunta, no dudes en abrir un issue.

[![Medium](https://img.shields.io/badge/-Medium-12100E?style=flat&logo=medium&labelColor=555)](https://medium.com/@hich.tala.phd/how-i-trained-again-my-model-to-detect-and-recognise-a-wide-range-of-yu-gi-oh-cards-5c567a320b0a)

---

## <div align="center">💬Contacto</div>

Puedes contactarme en Twitter [@hichtala](https://twitter.com/hichtala) o por correo electrónico
en [hich.tala.phd@gmail.com](mailto:hich.tala.phd@gmail.com).

---

## <div align="center">⭐Historial de estrellas</div>

<div align="center">
<a href="https://www.star-history.com/?repos=HichTala%2Fdraw2&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=HichTala/draw2&type=date&theme=dark&legend=top-left&sealed_token=Tj24ueb9rFzr_2PF0RNMdSlbPvsVOwdhhTBjutgQ1SUYji8RrcmHiv_ocH9Awcj-YtsbRvsN3HQ3CXrdvIlH9__zptdgBmaVrYYtCTtq8NSwB3pc8qUZcGcM0Ez6xCI8WWYDgM0H9ZiRDUrW24Rmut3k2YSh3Oe003GY7SfNSXCDbUniUCvwNbTd5ejR" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=HichTala/draw2&type=date&legend=top-left&sealed_token=Tj24ueb9rFzr_2PF0RNMdSlbPvsVOwdhhTBjutgQ1SUYji8RrcmHiv_ocH9Awcj-YtsbRvsN3HQ3CXrdvIlH9__zptdgBmaVrYYtCTtq8NSwB3pc8qUZcGcM0Ez6xCI8WWYDgM0H9ZiRDUrW24Rmut3k2YSh3Oe003GY7SfNSXCDbUniUCvwNbTd5ejR" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=HichTala/draw2&type=date&legend=top-left&sealed_token=Tj24ueb9rFzr_2PF0RNMdSlbPvsVOwdhhTBjutgQ1SUYji8RrcmHiv_ocH9Awcj-YtsbRvsN3HQ3CXrdvIlH9__zptdgBmaVrYYtCTtq8NSwB3pc8qUZcGcM0Ez6xCI8WWYDgM0H9ZiRDUrW24Rmut3k2YSh3Oe003GY7SfNSXCDbUniUCvwNbTd5ejR" />
 </picture>
</a>
</div>
