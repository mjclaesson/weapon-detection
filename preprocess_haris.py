import numpy as np
import zipfile
import os
import tensorflow as tf
from tensorflow import keras as keras
from tensorflow.keras import layers
from tensorflow import data as tf_data

zip_path = "HARIS.zip"
extract_path = "data/HARIS/"

# Make sure the destination folder exists
os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction complete!")

import shutil

HARIS_image_dir = "data/HARIS/HARIS_Weapon_Detection_Dataset_v2/HARIS_Weapon_Detection_Dataset/train/images"
HARIS_label_dir = "data/HARIS/HARIS_Weapon_Detection_Dataset_v2/HARIS_Weapon_Detection_Dataset/train/labels"

output_weapon = "data/HARIS_classification_train/weapon"
output_no_weapon = "data/HARIS_classification_train/no_weapon"

os.makedirs(output_weapon, exist_ok=True)
os.makedirs(output_no_weapon, exist_ok=True)

for img_file in os.listdir(HARIS_image_dir):
    if not img_file.endswith(".jpg"):
        continue

    base_name = os.path.splitext(img_file)[0]
    label_file = os.path.join(HARIS_label_dir, base_name + ".txt")
    img_path = os.path.join(HARIS_image_dir, img_file)

    # Case 1: label file exists
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            content = f.read().strip()

        if content:  # has coordinates → class 1
            shutil.copy(img_path, os.path.join(output_weapon, img_file))
        else:  # empty → class 2
            shutil.copy(img_path, os.path.join(output_no_weapon, img_file))

    else:
        # No label file → class 2
        shutil.copy(img_path, os.path.join(output_no_weapon, img_file))

import shutil

HARIS_image_dir = "data/HARIS/HARIS_Weapon_Detection_Dataset_v2/HARIS_Weapon_Detection_Dataset/val/images"
HARIS_label_dir = "data/HARIS/HARIS_Weapon_Detection_Dataset_v2/HARIS_Weapon_Detection_Dataset/val/labels"

output_weapon = "data/HARIS_classification_validation/weapon"
output_no_weapon = "data/HARIS_classification_validation/no_weapon"

os.makedirs(output_weapon, exist_ok=True)
os.makedirs(output_no_weapon, exist_ok=True)

for img_file in os.listdir(HARIS_image_dir):
    if not img_file.endswith(".jpg"):
        continue

    base_name = os.path.splitext(img_file)[0]
    label_file = os.path.join(HARIS_label_dir, base_name + ".txt")
    img_path = os.path.join(HARIS_image_dir, img_file)

    # Case 1: label file exists
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            content = f.read().strip()

        if content:  # has coordinates → class 1
            shutil.copy(img_path, os.path.join(output_weapon, img_file))
        else:  # empty → class 2
            shutil.copy(img_path, os.path.join(output_no_weapon, img_file))

    else:
        # No label file → class 2
        shutil.copy(img_path, os.path.join(output_no_weapon, img_file))

HARIS_images_train = "data/HARIS_classification_train/"
num_skipped = 0
for folder_name in ("weapon", "no_weapon"):
    folder_path = os.path.join(HARIS_images_train, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")

HARIS_images_validation = "data/HARIS_classification_validation/"
num_skipped = 0
for folder_name in ("weapon", "no_weapon"):
    folder_path = os.path.join(HARIS_images_validation, folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = b"JFIF" in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print(f"Deleted {num_skipped} images.")