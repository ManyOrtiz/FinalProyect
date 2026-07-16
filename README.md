# Clasificación de Videos Fight / NonFight con ResNet18

## Descripción del proyecto

Este es un proyecto académico de visión por computadora desarrollado en Python. El objetivo principal es clasificar videos de vigilancia en dos categorías:

- **Fight** / Pelea
- **NonFight** / No pelea

El proyecto utiliza el dataset **RWF-2000**, el cual contiene videos reales de vigilancia etiquetados como Fight y NonFight. El dataset está disponible públicamente en Kaggle:

https://www.kaggle.com/datasets/vulamnguyen/rwf2000

El dataset no se incluye dentro de este repositorio debido a su tamaño.
---

## Objetivo

El objetivo del proyecto es construir un flujo completo de clasificación de videos, incluyendo:

1. Extracción de frames a partir de videos.
2. Organización de los frames por clase.
3. División del dataset en entrenamiento, validación y prueba.
4. Entrenamiento de un clasificador basado en ResNet18 usando transfer learning.
5. Evaluación del modelo entrenado mediante métricas de clasificación.
6. Inferencia sobre videos nuevos.

---

## Tecnologías utilizadas

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Scikit-learn
- PIL / Pillow

---

## Estructura del proyecto

```text
FinalProyect/
│
├── extract_frames.py       # Extrae frames de los videos originales
├── split_dataset.py        # Divide los frames en train / validation / test
├── train_resnet.py         # Entrena el clasificador con ResNet18
├── evaluate_resnet.py      # Evalúa el modelo con métricas de clasificación
├── video_inference.py      # Realiza inferencia sobre un video nuevo
├── .gitignore
└── README.md




