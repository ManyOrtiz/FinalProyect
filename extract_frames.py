"""
Extrae frames de los videos de RWF-2000 (Kaggle) y los guarda como imagenes
"""

import cv2
import os

DATA_RAW_DIR = "RWF-2000"   # carpeta que ya descomprimiste
OUTPUT_DIR   = "frames"     # aquí se guardan las imágenes

# Cada cuántos frames guardo una imagen
FRAME_STEP = 10  # 1 de cada 10 frames 

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def extract_from_split(split_name, class_name):
    """
    Procesa una carpeta (train o val) de una clase (Fight o NonFight)
    Lee cada video, extrae frames y los guarda como imágenes
    """
    input_dir  = os.path.join(DATA_RAW_DIR, split_name, class_name)
    output_dir = os.path.join(OUTPUT_DIR, split_name, class_name)
    ensure_dir(output_dir)

    for fname in os.listdir(input_dir):
        #solo se procesan archivos de video
        if not fname.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
            continue

        video_path = os.path.join(input_dir, fname)
        print(f"[{split_name}/{class_name}] Procesando: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("  -> No se pudo abrir, se salta.")
            continue

        frame_idx = 0
        saved_idx = 0
        base_name = os.path.splitext(fname)[0]

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % FRAME_STEP == 0:
                out_name = f"{base_name}_f{frame_idx:05d}.jpg"
                out_path = os.path.join(output_dir, out_name)
                cv2.imwrite(out_path, frame)
                saved_idx += 1

            frame_idx += 1

        cap.release()
        print(f"  -> Frames guardados: {saved_idx}")

def main():
    for split in ["train", "val"]:
        for class_name in ["Fight", "NonFight"]:
            extract_from_split(split, class_name)

if __name__ == "__main__":
    main()

