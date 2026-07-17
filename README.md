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
```

## 1. Extracción de frames
El archivo **extract_frames.py** lee los videos originales y extrae un frame cada cierto intervalo.
En este proyecto se extrae 1 frame cada 10 frames. Esto ayuda a reducir la cantidad de imágenes generadas y evita guardar demasiados frames consecutivos que son visualmente muy similares.

## 2. División del dataset
El archivo **split_dataset.py** toma los frames extraídos y los divide en tres conjuntos:
- 70% entrenamiento
- 15% validación
- 15% prueba

También se utiliza una semilla fija para que la división sea reproducible.

## 3. Entrenamiento del modelo

El archivo **train_resnet.py** entrena un modelo basado en ResNet18 usando transfer learning.

Configuración principal:
```
Modelo: ResNet18 preentrenada en ImageNet
Clases: Fight / NonFight
Épocas: 8
Batch size: 32
Learning rate: 1e-4
Optimizador: Adam
Función de pérdida: CrossEntropyLoss
Tamaño de entrada: 224x224
```
En este proyecto se congelan las capas convolucionales principales de ResNet18 y se entrena únicamente la capa final para adaptarla a las dos clases del problema.

## 4. Evaluación del modelo

El archivo **evaluate_resnet.py** carga el modelo entrenado y lo evalúa usando métricas como:

- Accuracy
- Matriz de confusión
- Precision
- Recall
- F1-score

Estas métricas permiten analizar el desempeño del modelo más allá de la precisión general, ya que ayudan a identificar si el modelo se equivoca más en una clase que en otra.

## 5. Inferencia en video

El archivo **video_inference.py** carga el modelo entrenado y realiza predicciones sobre un video nuevo.

El modelo predice frame por frame. Para evitar resultados inestables por frames aislados, se utiliza un promedio móvil de las predicciones recientes.
