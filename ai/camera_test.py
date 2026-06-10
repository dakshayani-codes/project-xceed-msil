from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={"size": (1280, 720), "format": "RGB888"}
)

picam2.configure(config)

picam2.start()

time.sleep(2)

while True:

    frame = picam2.capture_array()

    cv2.imwrite("camera_view.jpg", frame)

    print("Saved camera_view.jpg")

    time.sleep(2)
