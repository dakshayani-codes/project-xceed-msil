import cv2
import os

# Change this each time {proper_belt, no_belt, clipped_behind, decoy}
CLASS_NAME = "proper_belt"

save_dir = f"datasets/custom/{CLASS_NAME}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)

count = len(os.listdir(save_dir))

print(f"\nCapturing {CLASS_NAME} images")
print("SPACE = capture image")
print("Q = quit\n")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Capture Window", frame)

    key = cv2.waitKey(1)

    if key == ord(' '):
        filename = f"{save_dir}/{CLASS_NAME}_{count:04d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved: {filename}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\nTotal captured: {count}")
