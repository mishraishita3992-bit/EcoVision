import os
import shutil
import random
from PIL import Image

# -----------------------------
# SETTINGS
# -----------------------------
RAW_DIR = "dataset/raw"
OUTPUT_DIR = "dataset"

CLASSES = {
    "1_polyethylene_PET": "PET",
    "2_high_density_polyethylene_PE-HD": "HDPE",
    "4_low_density_polyethylene_PE-LD": "LDPE",
    "5_polypropylene_PP": "PP"
}

# 70% training, 15% validation, 15% testing
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

random.seed(42)


# -----------------------------
# CHECK IMAGE
# -----------------------------
def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
for split in ["train", "validation", "test"]:
    for class_name in CLASSES.values():
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, class_name),
            exist_ok=True
        )


# -----------------------------
# PROCESS EACH CLASS
# -----------------------------
total_images = 0

for source_folder, class_name in CLASSES.items():

    source_path = os.path.join(RAW_DIR, source_folder)

    if not os.path.exists(source_path):
        print(f"ERROR: Folder not found: {source_path}")
        continue

    images = []

    for filename in os.listdir(source_path):
        filepath = os.path.join(source_path, filename)

        if filename.lower().endswith(IMAGE_EXTENSIONS):
            if is_valid_image(filepath):
                images.append(filepath)
            else:
                print(f"Skipping invalid image: {filepath}")

    random.shuffle(images)

    total = len(images)

    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"\n{class_name}")
    print(f"Total      : {total}")
    print(f"Training   : {len(train_images)}")
    print(f"Validation : {len(val_images)}")
    print(f"Testing    : {len(test_images)}")

    # Copy files
    for i, filepath in enumerate(train_images):
        extension = os.path.splitext(filepath)[1]
        destination = os.path.join(
            OUTPUT_DIR,
            "train",
            class_name,
            f"{class_name}_{i}{extension}"
        )
        shutil.copy2(filepath, destination)

    for i, filepath in enumerate(val_images):
        extension = os.path.splitext(filepath)[1]
        destination = os.path.join(
            OUTPUT_DIR,
            "validation",
            class_name,
            f"{class_name}_{i}{extension}"
        )
        shutil.copy2(filepath, destination)

    for i, filepath in enumerate(test_images):
        extension = os.path.splitext(filepath)[1]
        destination = os.path.join(
            OUTPUT_DIR,
            "test",
            class_name,
            f"{class_name}_{i}{extension}"
        )
        shutil.copy2(filepath, destination)

    total_images += total


print("\n==============================")
print("DATASET PREPARATION COMPLETE")
print("==============================")
print(f"Total valid images: {total_images}")
print("Dataset created in: dataset/") 