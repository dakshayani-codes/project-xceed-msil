import cv2
import os
import sys

video_path = sys.argv[1]
output_folder = sys.argv[2]
frame_interval = int(sys.argv[3]) if len(sys.argv) > 3 else 10

os.makedirs(output_folder, exist_ok=True)

video_name = os.path.splitext(os.path.basename(video_path))[0]

cap = cv2.VideoCapture(video_path)

count = 0
saved = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if count % frame_interval == 0:
        filename = f"{video_name}_frame_{count:05d}.jpg"
        cv2.imwrite(os.path.join(output_folder, filename), frame)
        saved += 1

    count += 1

cap.release()

print(f"Saved {saved} frames from {video_name}")
