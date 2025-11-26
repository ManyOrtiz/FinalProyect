"""
Entrenamiento del clasificador Fight / NonFight usando ResNet18 pre-entrenada
y guarda el mejor modelo
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", DEVICE)

DATA_DIR = "data_split"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 8

def get_dataloaders():
    """
    Carga datasets con transformaciones tipo ImageNet
    Genera DataLoaders para train, val y test
    """
    data_transforms = {
        "train": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "val": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
        "test": transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]),
    }

    image_datasets = {
        x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
        for x in ["train", "val", "test"]
    }

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        for x in ["train", "val", "test"]
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ["train", "val", "test"]}

    print("Tamaños:", dataset_sizes)

    return dataloaders, dataset_sizes

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer):
    """
    se entrena el modelo por EPOCHS.
    y se guarda los mejores pesos según accuracy en validación.
    """
    best_acc = 0.0
    best_weights = model.state_dict()

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                #solo se calculan gradientes en train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            #se guarda el mejor modelo
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = model.state_dict()

    model.load_state_dict(best_weights)
    print("\nMejor accuracy en val:", best_acc.item())
    #se guarda el modelo FINAL
    torch.save(model.state_dict(), "resnet18_fight_nofight.pth")
    print("Modelo guardado como resnet18_fight_nofight.pth")

def evaluate(model, dataloader, dataset_size):
    #Evalúa el modelo solo en test.
    model.eval()
    correct = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels)

    acc = correct.double() / dataset_size
    print("\nAccuracy en test:", acc.item())

def main():
    dataloaders, dataset_sizes = get_dataloaders()
    # Cargar ResNet18 + nueva capa final
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    train_model(model, dataloaders, dataset_sizes, criterion, optimizer)

    print("\nEvaluando en TEST...")
    evaluate(model, dataloaders["test"], dataset_sizes["test"])

if __name__ == "__main__":
    main()


