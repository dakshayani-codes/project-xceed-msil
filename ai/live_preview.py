from picamera2 import Picamera2
import cv2
import time

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (1280, 720),
        "format": "RGB888"
    }
)

picam2.configure(config)
picam2.set_controls({
    "ScalerCrop": (500, 300, 3600, 2000)
})
picam2.start()

time.sleep(2)

print("Recording preview...")
print("Press CTRL+C to stop")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

video = cv2.VideoWriter(
    "preview.mp4",
    fourcc,
    10.0,
    (1280, 720)
)

try:

    while True:

        # DO NOT CONVERT COLORS
        frame = picam2.capture_array()

        # GUIDE BOX
        cv2.rectangle(
            frame,
            (320, 120),
            (960, 650),
            (0, 255, 0),
            3
        )

        # TEXT
        cv2.putText(
            frame,
            "TORSO SHOULD FIT INSIDE GREEN BOX",
            (220, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # SAVE VIDEO
        video.write(frame)

        time.sleep(0.1)

except KeyboardInterrupt:

    print("Stopping...")

finally:

    video.release()

    picam2.stop()

    print("Saved preview.mp4")
