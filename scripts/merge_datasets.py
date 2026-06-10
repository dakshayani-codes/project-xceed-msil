# merge_datasets.py

from pathlib import Path
import shutil
import random
import yaml
from collections import defaultdict

# =========================================================

# CONFIG

# =========================================================

DATASETS_ROOT = Path("datasets")

DATASET_FOLDERS = [
"new_data",
"custom_raw",
"custom_photos",
"public_curated"
]

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

CLASS_NAMES = [
    "proper_belt",
    "no_belt",
    "clipped_behind",
    "decoy"
]

PREFIXES = {
"new_data": "newdata",
"custom_raw": "raw",
"custom_photos": "photos",
"public_curated": "public"
}

# =========================================================

# OUTPUT STRUCTURE

# =========================================================

FINAL_DATASET = DATASETS_ROOT / "final_dataset"

IMAGES_TRAIN = FINAL_DATASET / "images" / "train"
IMAGES_VAL = FINAL_DATASET / "images" / "val"
IMAGES_TEST = FINAL_DATASET / "images" / "test"

LABELS_TRAIN = FINAL_DATASET / "labels" / "train"
LABELS_VAL = FINAL_DATASET / "labels" / "val"
LABELS_TEST = FINAL_DATASET / "labels" / "test"

ALL_OUTPUT_DIRS = [
IMAGES_TRAIN,
IMAGES_VAL,
IMAGES_TEST,
LABELS_TRAIN,
LABELS_VAL,
LABELS_TEST
]

# =========================================================

# CREATE OUTPUT FOLDERS

# =========================================================

for folder in ALL_OUTPUT_DIRS:
    folder.mkdir(parents=True, exist_ok=True)

# =========================================================

# COLLECT VALID PAIRS

# =========================================================

valid_pairs = []
skipped_files = []

for dataset_name in DATASET_FOLDERS:


    dataset_path = DATASETS_ROOT / dataset_name

    if not dataset_path.exists():
        print(f"[WARNING] Missing dataset: {dataset_name}")
        continue

    prefix = PREFIXES[dataset_name]

    for class_folder in dataset_path.iterdir():

        if not class_folder.is_dir():
            continue

        class_name = class_folder.name

        for image_path in class_folder.iterdir():

            if image_path.name.startswith("."):
                continue

            if image_path.name == "classes.txt":
                continue

            if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
                continue

            label_path = image_path.with_suffix(".txt")

            if not label_path.exists():

                skipped_files.append(str(image_path))
                continue

            new_base_name = f"{prefix}_{image_path.stem}"

            if any(p["new_name"] == new_base_name for p in valid_pairs):
                print(f"[DUPLICATE] Skipping duplicate: {new_base_name}")
                continue

            valid_pairs.append({
                "image": image_path,
                "label": label_path,
                "new_name": new_base_name,
                "class_name": class_name
            })


# =========================================================

# SHUFFLE + SPLIT

# =========================================================

random.seed(RANDOM_SEED)
random.shuffle(valid_pairs)

total = len(valid_pairs)

train_end = int(total * TRAIN_RATIO)
val_end = train_end + int(total * VAL_RATIO)

train_pairs = valid_pairs[:train_end]
val_pairs = valid_pairs[train_end:val_end]
test_pairs = valid_pairs[val_end:]

# =========================================================

# COPY FUNCTION

# =========================================================

class_counts = defaultdict(int)

def copy_pairs(pairs, image_dir, label_dir):
    for pair in pairs:

        image_src = pair["image"]
        label_src = pair["label"]

        image_ext = image_src.suffix.lower()

        image_dst = image_dir / f"{pair['new_name']}{image_ext}"
        label_dst = label_dir / f"{pair['new_name']}.txt"

        shutil.copy2(image_src, image_dst)
        shutil.copy2(label_src, label_dst)

        class_counts[pair["class_name"]] += 1


# =========================================================

# COPY DATA

# =========================================================

copy_pairs(train_pairs, IMAGES_TRAIN, LABELS_TRAIN)
copy_pairs(val_pairs, IMAGES_VAL, LABELS_VAL)
copy_pairs(test_pairs, IMAGES_TEST, LABELS_TEST)

# =========================================================

# GENERATE data.yaml

# =========================================================

yaml_data = {
"path": str(FINAL_DATASET.resolve()),
"train": "images/train",
"val": "images/val",
"test": "images/test",
"names": CLASS_NAMES
}

with open(FINAL_DATASET / "data.yaml", "w") as f:
    yaml.dump(yaml_data, f, sort_keys=False)

# =========================================================

# FINAL REPORT

# =========================================================

print("=" * 60)
print("FINAL DATASET CREATED")
print("=" * 60)

print(f"Total images : {total}")
print(f"Train images : {len(train_pairs)}")
print(f"Val images   : {len(val_pairs)}")
print(f"Test images  : {len(test_pairs)}")

print("\nCLASS DISTRIBUTION")

for class_name, count in class_counts.items():
    print(f"{class_name:<20} : {count}")

print(f"\nSkipped orphan files: {len(skipped_files)}")

print("\nDataset saved to:")
print(FINAL_DATASET.resolve())
