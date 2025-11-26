import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from collections import deque

MODEL_PATH = "resnet18_fight_nofight.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Transformaciones igual que en entrenamiento
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def load_model():
    """
    Carga ResNet18 con los pesos entrenados
    Congela capas convolucionales y reemplaza la final
    """
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # se congelan las capas convolucionales
    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)  # Fight / NonFight

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    return model

def predict_proba(model, frame):
    """
    Recibe un frame (BGR), lo prepara y devuelve probabilidad de FIGHT y NO FIGHT
    """
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = transform(img_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]  # [p_fight, p_nofight]

    return probs  # [p_fight, p_nofight]

def main():
    model = load_model()

    # se carga video a analizar
    cap = cv2.VideoCapture("video_test3.avi")
    if not cap.isOpened():
        print("No se pudo abrir el video.")
        return

    # Buffer para suavizar predicciones en el tiempo
    history_len = 15
    fight_history = deque(maxlen=history_len)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        probs = predict_proba(model, frame)
        p_fight, p_nofight = probs[0], probs[1]
        fight_history.append(p_fight)

        # Promedio de probabilidad de pelea en los últimos N frames
        avg_fight = np.mean(fight_history) if fight_history else p_fight

        # Umbral para decidir pelea/no pelea
        if avg_fight > 0.55:
            label = "FIGHT"
            color = (0, 0, 255)
        else:
            label = "NO FIGHT"
            color = (0, 255, 0)

        text = f"{label}  p_fight={p_fight:.2f}  avg={avg_fight:.2f}"

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        cv2.imshow("Fight Detection - ResNet18", frame)

       
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

        
        if cv2.getWindowProperty("Fight Detection - ResNet18", cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
