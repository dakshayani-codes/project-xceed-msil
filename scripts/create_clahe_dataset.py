import cv2
import os
import shutil

SRC_ROOT = os.path.expanduser(
    "~/Desktop/ir_review/ir_data"
)

DST_ROOT = os.path.expanduser(
    "~/Desktop/ir_review/ir_data_clahe"
)

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

classes = [
    "proper_belt",
    "no_belt",
    "clipped_behind",
    "decoy"
]

for cls in classes:

    src_dir = os.path.join(SRC_ROOT, cls)
    dst_dir = os.path.join(DST_ROOT, cls)

    os.makedirs(dst_dir, exist_ok=True)

    for file in os.listdir(src_dir):

        src_path = os.path.join(src_dir, file)
        dst_path = os.path.join(dst_dir, file)

        # Copy labels unchanged
        if file.endswith(".txt"):
            shutil.copy2(src_path, dst_path)
            continue

        # Process images
        if file.lower().endswith(".jpg"):

            img = cv2.imread(src_path)

            gray = cv2.cvtColor(
                img,
                cv2.COLOR_BGR2GRAY
            )

            gray = clahe.apply(gray)

            gray = cv2.cvtColor(
                gray,
                cv2.COLOR_GRAY2BGR
            )

            cv2.imwrite(dst_path, gray)

print("CLAHE dataset created successfully.")
