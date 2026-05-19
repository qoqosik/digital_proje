import random
import shutil
from pathlib import Path

DATASET_DIR = Path("dataset")
SOURCE_IMAGES = DATASET_DIR / "train" / "images"
SOURCE_LABELS = DATASET_DIR / "train" / "labels"

TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
images = [p for p in SOURCE_IMAGES.iterdir() if p.suffix.lower() in image_extensions]

if not images:
    raise RuntimeError("No images found in dataset/train/images")

random.shuffle(images)

total = len(images)
train_count = int(total * TRAIN_RATIO)
valid_count = int(total * VALID_RATIO)

splits = {
    "train_new": images[:train_count],
    "valid": images[train_count:train_count + valid_count],
    "test": images[train_count + valid_count:],
}

for split_name in ["train_new", "valid", "test"]:
    (DATASET_DIR / split_name / "images").mkdir(parents=True, exist_ok=True)
    (DATASET_DIR / split_name / "labels").mkdir(parents=True, exist_ok=True)

for split_name, split_images in splits.items():
    for image_path in split_images:
        label_path = SOURCE_LABELS / f"{image_path.stem}.txt"

        shutil.copy2(image_path, DATASET_DIR / split_name / "images" / image_path.name)

        if label_path.exists():
            shutil.copy2(label_path, DATASET_DIR / split_name / "labels" / label_path.name)
        else:
            print(f"Warning: label not found for {image_path.name}")

old_train = DATASET_DIR / "train"
backup_train = DATASET_DIR / "train_original"

if backup_train.exists():
    shutil.rmtree(backup_train)

old_train.rename(backup_train)
(DATASET_DIR / "train_new").rename(DATASET_DIR / "train")

print("Dataset split completed.")
print(f"Total images: {total}")
print(f"Train: {len(splits['train_new'])}")
print(f"Valid: {len(splits['valid'])}")
print(f"Test: {len(splits['test'])}")
print("Original train folder saved as dataset/train_original")