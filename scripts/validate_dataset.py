# validate_dataset.py

from pathlib import Path
from collections import defaultdict

# =========================================================
# CONFIG
# =========================================================

DATASET_PATH = Path("datasets/final_dataset")

VALID_CLASS_IDS = {0, 1, 2, 3}

CLASS_NAMES = {
    0: "proper_belt",
    1: "no_belt",
    2: "clipped_behind",
    3: "decoy"
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png"]

# =========================================================
# STORAGE
# =========================================================

issues = []

class_counts = defaultdict(int)

# =========================================================
# VALIDATE LABEL FILE
# =========================================================

def validate_label_file(label_path):

    with open(label_path, "r") as f:
        lines = f.readlines()

    # EMPTY LABEL FILE
    if len(lines) == 0:

        issues.append(
            f"[EMPTY LABEL] {label_path}"
        )

        return

    # CHECK EACH LINE
    for line_num, line in enumerate(lines, start=1):

        parts = line.strip().split()

        # YOLO FORMAT MUST HAVE 5 VALUES
        if len(parts) != 5:

            issues.append(
                f"[MALFORMED] {label_path} line {line_num}"
            )

            continue

        try:

            class_id = int(parts[0])

            # INVALID CLASS ID
            if class_id not in VALID_CLASS_IDS:

                issues.append(
                    f"[INVALID CLASS ID] {label_path}"
                )

                continue

            # COORDINATES
            coords = list(map(float, parts[1:]))

            for coord in coords:

                if coord < 0 or coord > 1:

                    issues.append(
                        f"[INVALID COORD] {label_path}"
                    )

            # COUNT CLASSES
            class_counts[class_id] += 1

        except Exception:

            issues.append(
                f"[PARSE ERROR] {label_path}"
            )

# =========================================================
# CHECK TRAIN / VAL / TEST
# =========================================================

for split in ["train", "val", "test"]:

    image_dir = DATASET_PATH / "images" / split
    label_dir = DATASET_PATH / "labels" / split

    # CHECK IF DIRECTORIES EXIST
    if not image_dir.exists():

        issues.append(
            f"[MISSING IMAGE DIR] {image_dir}"
        )

        continue

    if not label_dir.exists():

        issues.append(
            f"[MISSING LABEL DIR] {label_dir}"
        )

        continue

    image_files = []

    # COLLECT IMAGE FILES
    for ext in IMAGE_EXTENSIONS:

        image_files.extend(
            image_dir.glob(f"*{ext}")
        )

    # CHECK EACH IMAGE
    for image_path in image_files:

        label_path = label_dir / f"{image_path.stem}.txt"

        # MISSING LABEL
        if not label_path.exists():

            issues.append(
                f"[MISSING LABEL] {image_path}"
            )

        else:

            validate_label_file(label_path)

# =========================================================
# FINAL REPORT
# =========================================================

print("=" * 60)
print("DATASET VALIDATION REPORT")
print("=" * 60)

# ISSUES
if len(issues) == 0:

    print("No issues found.")

else:

    print(f"Total issues found: {len(issues)}\n")

    for issue in issues:
        print(issue)

# CLASS DISTRIBUTION
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

for class_id in sorted(class_counts.keys()):

    print(
        f"{CLASS_NAMES[class_id]:<20} : {class_counts[class_id]}"
    )

print("\nValidation complete.")