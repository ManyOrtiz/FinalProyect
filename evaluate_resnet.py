"""
evaluacion del modelo resnet18_fight_nofight.pth
Muestra:
- accuracy en test
- matriz de confusión
- precision, recall y F1 por clase
"""

import os
import torch
import torch.nn as nn

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", DEVICE)

DATA_DIR = "data_split"
BATCH_SIZE = 32
MODEL_PATH = "resnet18_fight_nofight.pth"

def get_test_loader():
    #se crea el DataLoader del conjunto de prueba usando las mismas transformaciones que en el entrenamiento
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],#noralización
            std=[0.229, 0.224, 0.225]
        ),
    ])

    #se cargan las imagenes de test
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, "test"),
        transform_test
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    class_names = test_dataset.classes
    print("Clases:", class_names)
    print("Tamaño test:", len(test_dataset))

    return test_loader, class_names

def load_model(num_classes=2):
    #se carga la arquitectura ResNet18 preentrenada, reemplaza la ultima capa y carga los pesos entrenados
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Congelamos todo menos la última capa (igual que en entrenamiento)
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    return model

def evaluate():
    #se evalua el modelo en el conjunto de prueba y muestra metricas clave
    test_loader, class_names = get_test_loader()
    model = load_model(num_classes=len(class_names))

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Accuracy global
    acc = (sum(int(p == y) for p, y in zip(all_preds, all_labels))
           / len(all_labels))
    print(f"\nAccuracy (recalculado) en test: {acc:.4f}")

    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_preds)
    print("\nMatriz de confusión (filas = verdad, columnas = predicción):")
    print(cm)

    # Precision, recall, F1 por clase
    print("\nReporte de clasificación:")
    print(classification_report(all_labels, all_preds,
                                target_names=class_names,
                                digits=4))

if __name__ == "__main__":
    evaluate()
