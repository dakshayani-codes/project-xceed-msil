# train_yolo.py

from ultralytics import YOLO

# =========================================================

# CONFIG

# =========================================================

DATA_YAML = "datasets/final_dataset/data.yaml"

MODEL_NAME = "models/yolov8n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 16

# =========================================================

# LOAD MODEL

# =========================================================

model = YOLO(MODEL_NAME)

# =========================================================

# TRAIN

# =========================================================

results = model.train(


    data=DATA_YAML,

    epochs=EPOCHS,

    imgsz=IMAGE_SIZE,

    batch=BATCH_SIZE,

    patience=15,

    workers=4,

    project="runs",

    name="seatbelt_detection",

    exist_ok=True,

    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,

    fliplr=0.5,

    mosaic=1.0,

    mixup=0.1,

    degrees=5,

    translate=0.1,

    scale=0.3,

    shear=2,

    perspective=0.0005,

    cache=True


)

# =========================================================

# VALIDATE

# =========================================================

metrics = model.val()

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(metrics)

print("\nBest model saved automatically.")
print("Check:")
print("runs/detect/seatbelt_detection/weights/best.pt")
