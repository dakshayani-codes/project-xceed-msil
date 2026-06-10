# detect.py — Project Xceed production inference
# Pi 5 + NoIR V2 + YOLOv8n ONNX — targets ≥20 FPS at IMG_SIZE=320

import numpy as np
import onnxruntime as ort
from picamera2 import Picamera2
import time
import sys
import os
import cv2
import threading
import requests

sys.path.append(os.path.expanduser('~/project-xceed'))
from hardware.gpio_alert import alert_set, alert_off

# ── CONFIG ───────────────────────────────────────────
MODEL_PATH  = os.path.expanduser('~/project-xceed/ai/best320.onnx')

# FIX 1: 320 targets ≥20 FPS on Pi 5.
IMG_SIZE    = 320

CAM_W, CAM_H = 1280, 720

CLASSES = ['proper_belt', 'no_belt', 'clipped_behind', 'decoy']

# FIX 2: Updated thresholds — calibrated to per-class mAP50
CLASS_CONF_THRESHOLDS = {
    'proper_belt':    0.18,
    'no_belt':        0.10,
    'clipped_behind': 0.35,
    'decoy':          0.20,
}

ALERT_CLASSES      = ['no_belt', 'clipped_behind', 'decoy']
FRAME_BUFFER_SIZE  = 3     # consecutive frames required before alert fires
CLEAN_BUFFER_SIZE = 16  # OFF: 16 consecutive clean frames     (~667ms at 24fps)
NMS_IOU_THRESHOLD  = 0.45

# FPS display rolling window
FPS_WINDOW = 30

API_URL = "http://localhost:8000/violation"

def post_event(class_name: str, confidence: float, alert: bool):
    """Non-blocking POST to FastAPI — never stalls inference."""
    def _send():
        try:
            requests.post(API_URL, json={
                "class_name": class_name,
                "confidence": round(confidence, 3),
                "alert":      alert,
            }, timeout=0.3)
        except Exception:
            pass
    threading.Thread(target=_send, daemon=True).start()


# ── PREPROCESS ───────────────────────────────────────
def preprocess(frame):
    # frame is RGB888 → already RGB, no cvtColor needed for model
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img


# ── POSTPROCESS + NMS ────────────────────────────────
def postprocess(outputs):
    output     = outputs[0][0]
    boxes      = output[:4, :]
    scores     = output[4:, :]
    best_class = np.argmax(scores, axis=0)
    best_score = np.max(scores, axis=0)

    # Per-class threshold filtering
    valid_indices = [
        i for i in range(len(best_score))
        if best_score[i] > CLASS_CONF_THRESHOLDS[CLASSES[int(best_class[i])]]
    ]
    if not valid_indices:
        return []

    # FIX 3: scale factors — model output is in IMG_SIZE space
    x_scale = CAM_W / IMG_SIZE
    y_scale = CAM_H / IMG_SIZE

    boxes_for_nms = []
    confidences   = []
    class_ids     = []

    for idx in valid_indices:
        class_id           = int(best_class[idx])
        score              = float(best_score[idx])
        x_c, y_c, w, h    = boxes[:, idx]

        x1 = int((x_c - w / 2) * x_scale)
        y1 = int((y_c - h / 2) * y_scale)
        x2 = int((x_c + w / 2) * x_scale)
        y2 = int((y_c + h / 2) * y_scale)

        x1 = max(0, min(x1, CAM_W - 1))
        y1 = max(0, min(y1, CAM_H - 1))
        x2 = max(0, min(x2, CAM_W - 1))
        y2 = max(0, min(y2, CAM_H - 1))

        bw, bh = x2 - x1, y2 - y1
        if bw < 20 or bh < 20:
            continue

        boxes_for_nms.append([x1, y1, bw, bh])
        confidences.append(score)
        class_ids.append(class_id)

    if not boxes_for_nms:
        return []

    indices = cv2.dnn.NMSBoxes(
        boxes_for_nms, confidences, 0.15, NMS_IOU_THRESHOLD
    )
    if len(indices) == 0:
        return []

    detections = []
    for idx in indices.flatten():
        x1, y1, bw, bh = boxes_for_nms[idx]
        detections.append({
            'class':      CLASSES[class_ids[idx]],
            'confidence': confidences[idx],
            'box':        [x1, y1, x1 + bw, y1 + bh],
        })

    return sorted(detections, key=lambda d: d['confidence'], reverse=True)


# ── MAIN ─────────────────────────────────────────────
def main():
    print("Loading ONNX model...")
    session    = ort.InferenceSession(MODEL_PATH)
    input_name = session.get_inputs()[0].name
    print(f"  Input : {input_name}  shape={session.get_inputs()[0].shape}")
    print(f"  Output: {session.get_outputs()[0].shape}")

    print("Starting camera...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (CAM_W, CAM_H), "format": "RGB888"}
    )
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    print("Camera ready.")

    # FIX 4: XVID + .avi — mp4v codec is unreliable on Pi OpenCV builds
    out_path     = os.path.expanduser('~/project-xceed/test_results/onnx320/output320.avi')
    fourcc       = cv2.VideoWriter_fourcc(*'XVID')
    # FIX 5: VIDEO_FPS matched to target capture rate (no sleep → ~20fps)
    video_writer = cv2.VideoWriter(out_path, fourcc, 20.0, (CAM_W, CAM_H))

    # FIX 6: verify VideoWriter opened — silent failure is common on Pi
    if not video_writer.isOpened():
        print("ERROR: VideoWriter failed to open. Check codec/path/disk space.")
        picam2.stop()
        return

    frame_buffer  = []
    frame_count   = 0
    alert_active  = False
    clean_buffer  = []
    current_alert = None   # which class is currently alerting

    # FPS rolling measurement
    frame_times   = []

    print(f"Inference running at IMG_SIZE={IMG_SIZE} — Ctrl+C to stop")
    print("-" * 50)

    try:
        while True:
            t0 = time.time()

            # Capture — frame is RGB (RGB888), correct for YOLO
            frame = picam2.capture_array()

            img        = preprocess(frame)
            outputs    = session.run(None, {input_name: img})
            detections = postprocess(outputs)
            current_class = 'none'
            current_conf  = 0.0

            frame_bgr = frame.copy()

            # Draw top detection only
            if detections:
                det        = detections[0]
                class_name = det['class']
                confidence = det['confidence']
                x1, y1, x2, y2 = det['box']

                current_class = class_name
                current_conf  = confidence

                color = (0, 0, 255) if class_name in ALERT_CLASSES else (0, 255, 0)

                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)

                label   = f"{class_name} {confidence:.2f}"
                # FIX 8: guard label so it never renders above y=0
                label_y = max(y1 - 10, 20)
                cv2.putText(
                    frame_bgr, label, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
                )

            # ── Temporal filter ──────────────────────────────────────
            frame_buffer.append(current_class)
            if len(frame_buffer) > FRAME_BUFFER_SIZE:
                frame_buffer.pop(0)
 
            # Track consecutive clean frames for OFF hysteresis
            if current_class not in ALERT_CLASSES:
                clean_buffer.append(current_class)
                if len(clean_buffer) > CLEAN_BUFFER_SIZE:
                    clean_buffer.pop(0)
            else:
                clean_buffer.clear()  # any violation resets clean counter
 
            if len(frame_buffer) == FRAME_BUFFER_SIZE:
                all_violation = all(c in ALERT_CLASSES for c in frame_buffer)
                all_clean     = len(clean_buffer) == CLEAN_BUFFER_SIZE
 
                # ALERT ON — requires FRAME_BUFFER_SIZE consecutive violation frames
                if all_violation and not alert_active:
                    alert_set(current_class)
                    alert_active  = True
                    current_alert = current_class
                    print(f"*** ALERT ON  — {current_class} ***")
 
                # ALERT OFF — requires CLEAN_BUFFER_SIZE consecutive clean frames
                elif alert_active and all_clean:
                    alert_off()
                    alert_active  = False
                    current_alert = None
                    print("*** ALERT OFF ***")
 
                # CLASS CHANGE — update buzzer pattern without re-triggering ON/OFF
                elif alert_active and all_violation and current_class != current_alert:
                    alert_set(current_class)
                    current_alert = current_class
                    print(f"*** ALERT CLASS → {current_class} ***")

            # ── On-screen overlays ───────────────────────────────────
            if alert_active:
                cv2.putText(
                    frame_bgr, "ALERT!", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
                )

            # ── FPS counter ──────────────────────────────────────────
            elapsed = time.time() - t0
            frame_times.append(elapsed)
            if len(frame_times) > FPS_WINDOW:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))

            cv2.putText(
                frame_bgr, f"FPS: {avg_fps:.1f}", (CAM_W - 130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2
            )

            # ── Write frame ──────────────────────────────────────────
            if frame_bgr.shape[:2] == (CAM_H, CAM_W):
                video_writer.write(frame_bgr)
            else:
                print(f"WARNING: bad frame shape {frame_bgr.shape} — skipped")

            # ── Logging ──────────────────────────────────────────────
            # ── POST TO BACKEND ──────────────────────────────────────
            post_event(current_class, current_conf, alert_active)

            if frame_count % 30 == 0:
                det_str = f"{current_class} ({current_conf:.2f})" if detections else "no detection"
                print(f"Frame {frame_count:5d} | {det_str:30s} | {avg_fps:.1f} FPS")

            frame_count += 1
            # FIX 10: NO sleep — let Pi run at full inference speed
            # Pi 5 at IMG_SIZE=320 should achieve 20-30 FPS without sleep

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        alert_off()
        video_writer.release()
        picam2.stop()
        if frame_count > 0 and frame_times:
            avg = 1.0 / (sum(frame_times) / len(frame_times))
            print(f"\nAvg FPS: {avg:.1f}  Total frames: {frame_count}")

        # Auto-convert .avi → .mp4
        mp4_path = out_path.replace('.avi', '.mp4')
        print("Converting to mp4...")
        ret = os.system(
            f'ffmpeg -i "{out_path}" -vcodec libx264 -pix_fmt yuv420p '
            f'-crf 23 "{mp4_path}" -y 2>/dev/null'
        )
        if ret == 0 and os.path.exists(mp4_path):
            os.remove(out_path)   # delete .avi, keep only .mp4
            print(f"Video saved: {mp4_path}")
        else:
            print(f"ffmpeg conversion failed — keeping .avi: {out_path}")
        print("Done.")


if __name__ == "__main__":
    main()


