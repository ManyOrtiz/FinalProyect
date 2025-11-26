"""
se crea la division train / val / test a partir de los frames extraidos
Se mezclan imágenes de los splits originales de RWF-2000 y se redistribuyen con proporciones 70/15/15
"""

import os
import shutil
import random

SOURCE_DIR = "frames"
OUTPUT_DIR = "data_split"

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15
}

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def split_class(split_name):
    """
    Toma imagenes de 'frames/train' y 'frames/val'
    y devuelve listas de paths para Fight y NonFight.
    """
    src_fight = os.path.join(SOURCE_DIR, split_name, "Fight")
    src_nonfight = os.path.join(SOURCE_DIR, split_name, "NonFight")

    fight_imgs = [os.path.join(src_fight, f) for f in os.listdir(src_fight)]
    nonfight_imgs = [os.path.join(src_nonfight, f) for f in os.listdir(src_nonfight)]

    return fight_imgs, nonfight_imgs

def distribute_images(img_list, class_name):
    """
    Mezcla y distribuye imágenes según SPLITS
    Copia archivos a data_split/train, val y test
    """
    random.shuffle(img_list)
    n = len(img_list)

    n_train = int(SPLITS["train"] * n)
    n_val   = int(SPLITS["val"] * n)

    splits = {
        "train": img_list[:n_train],
        "val": img_list[n_train:n_train+n_val],
        "test": img_list[n_train+n_val:]
    }

    for split, files in splits.items():
        dst_dir = os.path.join(OUTPUT_DIR, split, class_name)
        ensure_dir(dst_dir)
        print(f"Copiando {len(files)} imágenes -> {dst_dir}")

        for src in files:
            fname = os.path.basename(src)
            shutil.copy2(src, os.path.join(dst_dir, fname))

def main():
    random.seed(42)

    total_fight = []
    total_nonfight = []

    for split in ["train", "val"]:
        fight, nonfight = split_class(split)
        total_fight.extend(fight)
        total_nonfight.extend(nonfight)

    distribute_images(total_fight, "Fight")
    distribute_images(total_nonfight, "NonFight")

    print("\n¡Split completado!")

if __name__ == "__main__":
    main()


