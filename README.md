# Project XCEED – Real-Time Seat Belt Detection System

> **Maruti Suzuki India Limited (MSIL) — Project Xceed Internship Program**  
> Edge AI · YOLOv8 · Raspberry Pi 5 · Fully Offline · Real-Time

---

## Overview

Project XCEED is a real-time seat belt compliance monitoring system developed as part of the Maruti Suzuki India Limited internship program. The system uses a YOLOv8-based computer vision pipeline deployed on a Raspberry Pi 5 to detect seat belt usage and identify unsafe occupant behaviour in real time.

The solution operates fully offline, is optimised for edge deployment, and provides both a software monitoring dashboard and GPIO-based hardware alerts. It was validated across multiple lighting conditions, clothing contrasts, and seat belt misuse scenarios.

---

## Problem Statement

Build a camera-based seat belt monitoring system capable of:

- Detecting seat belt compliance in real time
- Identifying non-compliance scenarios including misuse and evasion
- Triggering immediate warnings upon violation detection
- Running on low-compute edge hardware without GPU acceleration
- Operating without internet connectivity during inference
- Remaining robust across varied lighting conditions and pose variations

---

## Detection Classes

| Class | Description |
|---|---|
| `proper_belt` | Seat belt worn correctly across the shoulder and torso |
| `no_belt` | Seat belt entirely absent |
| `clipped_behind` | Belt fastened but routed behind the occupant |
| `decoy` | Belt-like objects — straps, lanyards, accessories — that are not the actual belt |
| `none` | No valid detection in the current frame |

---

## Key Features

### Edge Deployment
- Raspberry Pi 5 (2 GB) — no cloud dependency
- ONNX Runtime CPU inference at **18–25 FPS**
- Fully offline; zero network calls during operation
- GPIO-controlled LED and buzzer hardware alerts

### Alert System
Each violation class triggers a distinct buzzer pattern so the alert type is identifiable without looking at the dashboard:

| State | LED | Buzzer |
|---|---|---|
| `proper_belt` | OFF | Silent |
| `no_belt` | ON | Slow repeating beep |
| `clipped_behind` | ON | Double beep pattern |
| `decoy` | ON | Slow spaced beep |

Alert stability is managed by hysteresis logic — **Frame Buffer (FBS = 3)** to activate and **Clean Buffer (CBS = 16)** to deactivate — eliminating flickering from single-frame misclassifications.

### Monitoring Dashboard
The Streamlit dashboard provides:
- Live detection class and confidence score
- Active alert state
- Violation log with timestamps
- Session statistics (total inferences, alert events, per-class breakdown)
- Database-backed traceability

### Violation Logging
All detection events are persisted to `xceed.db` (SQLite):

| Field | Description |
|---|---|
| `timestamp` | UTC time of detection |
| `class_name` | Predicted class |
| `confidence` | Model confidence score |
| `alert_status` | Whether an alert was active |

> `xceed.db` is excluded from version control and is generated at runtime.

During validation, the system logged **28,159 inference records** and **10,641 alert events** across all test conditions.

---

## Repository Structure

```text
project-xceed-msil/
│
├── ai/                              # AI subsystem
│   ├── detect.py                    # Real-time inference loop (main runtime)
│   ├── train_yolo.py                # YOLOv8 training script
│   ├── extract_frames.py            # Video-to-frame dataset extractor
│   ├── capture_custom.py            # Pi Camera data collection utility
│   ├── camera_test.py               # Camera connectivity test
│   └── live_preview.py              # Live camera preview utility
│
├── backend/
│   └── main.py                      # FastAPI REST API — detection logging and dashboard data
│
├── frontend/
│   └── dashboard.py                 # Streamlit monitoring dashboard
│
├── hardware/
│   └── gpio_alert.py                # GPIO LED and buzzer controller
│
├── datasets/                        # Active training datasets
│   ├── classes.txt                  # Class label definitions
│   ├── final_dataset/               # Combined RGB dataset (production training)
│   │   ├── data.yaml                # ✓ tracked
│   │   ├── images/                  # ✗ excluded — large binary assets
│   │   └── labels/                  # ✗ excluded — large binary assets
│   ├── final_dataset_ir_raw/        # Raw infrared dataset
│   │   ├── data.yaml                # ✓ tracked
│   │   ├── images/                  # ✗ excluded
│   │   └── labels/                  # ✗ excluded
│   └── final_dataset_ir_clahe/      # CLAHE-enhanced infrared dataset
│       ├── data.yaml                # ✓ tracked
│       ├── images/                  # ✗ excluded
│       └── labels/                  # ✗ excluded
│
├── models/                          # Model artifacts
│   ├── best320.onnx                 # ✓ tracked — production deployment model (320×320)
│   ├── best_final.pt                # ✓ tracked — final PyTorch checkpoint (for retraining)
│   └── yolov8n.pt                   # ✓ tracked — base YOLOv8n weights
│
├── scripts/                         # Dataset preparation utilities
│   ├── merge_datasets.py            # Merge RGB dataset sources
│   ├── merge_datasets_ir_raw.py     # Merge raw IR dataset sources
│   ├── merge_datasets_ir_clahe.py   # Merge CLAHE IR dataset sources
│   ├── create_clahe_dataset.py      # Apply CLAHE to raw IR frames
│   └── validate_dataset.py          # Verify annotation integrity and class balance
│
├── demo_recordings/                 # Per-scenario validation recordings
│   ├── 01_bright_white/             # Daylight, white shirt — all four classes
│   │   ├── proper_belt/             # ✗ video files excluded
│   │   ├── no_belt/
│   │   ├── clipped_behind/
│   │   └── decoy/
│   ├── 02_bright_black/             # Daylight, black shirt — all four classes
│   ├── 03_dim_white/                # Active cabin, white shirt — all four classes
│   ├── 04_dim_black/                # Active cabin, black shirt — all four classes
│   ├── 05_full_demo/
│   │   └── output.mp4               # ✗ excluded
│   └── 06_offline_proof/
│       └── output.mp4               # ✗ excluded
│
├── media/                           # Additional media assets
│   ├── deployment_tests/
│   │   ├── output_master.mp4        # ✗ excluded
│   │   └── output320_detect.mp4     # ✗ excluded
│   └── model_comparison/
│       ├── best320.mp4              # ✗ excluded
│       ├── ir_raw.mp4               # ✗ excluded
│       └── ir_clahe.mp4             # ✗ excluded
│
├── docs/                            # Documentation assets — all tracked
│   ├── System Architecture.jpeg
│   ├── Hardware Architecture.jpeg
│   ├── Alert Logic Flowchart.jpeg
│   ├── MODEL_DEVELOPMENT.jpg
│   ├── model_comparison.jpg
│   ├── dashboard.jpg
│   ├── hardware_setup.jpg
│   ├── gpio_wiring.jpg
│   └── ir_illuminator.jpg
│
├── reports/                         # Submitted project reports — all tracked
│   ├── Initial Plan Report — Project Xceed.pdf
│   ├── Mid_Progress_Report_FINAL.pdf
│   └── DAKSHAYANI_SHARMA_XCEED.pdf
│
├── archive/                         # Legacy development artifacts
│   ├── dataset.yaml
│   ├── datasets/                    # ✗ excluded — raw collected data before merging
│   ├── training_history/            # ✗ excluded — per-phase YOLOv8 run outputs
│   └── videos/                      # ✗ excluded — source videos for dataset extraction
│
├── master.sh                        # System orchestration entry point
├── requirements.txt                 # Python dependency specification
└── README.md
```

> **Note on excluded assets:**  
> Dataset images/labels, video recordings, and intermediate training weights are excluded from this repository via `.gitignore` due to file size. `best320.onnx`, `best_final.pt`, and `yolov8n.pt` are tracked. All other excluded assets are available in the project submission package provided to MSIL.

---

## Hardware

| Component | Specification | Role |
|---|---|---|
| Raspberry Pi 5 | 2 GB, ARM Cortex-A76 | Edge inference host |
| Pi Camera Module V2 NoIR | 8 MP, Sony IMX219 | Frame acquisition (no IR cut filter) |
| Active Buzzer | 5V, GPIO-controlled | Class-differentiated audio alert |
| Red LED | 5mm, 2V forward | Visual violation indicator |
| BC547 NPN Transistor | hFE ~200 | GPIO current switch for buzzer |
| 220Ω Resistor | LED branch | LED current limiting |
| 1kΩ Resistor | Transistor base | GPIO pin protection |
| IR Illuminator | 48 LED, 850nm | Passive-cabin illumination (<1 lux) |

The Pi Camera V2 NoIR was selected because it lacks an infrared cut filter, enabling imaging under near-zero-lux conditions when paired with the IR illuminator.

---

## Software Stack

| Layer | Technology |
|---|---|
| Model Training | YOLOv8n (Ultralytics) |
| Deployment Runtime | ONNX Runtime (CPU) |
| Image Processing | OpenCV 4.x |
| Backend API | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Database | SQLite |
| Hardware Control | RPi.GPIO |

---

## Installation and Deployment

### 1. Clone the Repository

```bash
git clone https://github.com/dakshayani-codes/project-xceed-msil.git
cd project-xceed-msil
```

### 2. Create and Activate a Virtual Environment

```bash
python3 -m venv xceed-env
source xceed-env/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the Deployment Model

The ONNX deployment model is tracked in this repository at:

```
models/best320.onnx
```

If it is missing after cloning (e.g. due to a shallow clone or separate distribution), obtain it from the project submission package and place it at the path above before proceeding.

### 5. Launch the Full System

```bash
bash master.sh
```

`master.sh` automatically:
- Activates the virtual environment
- Starts the FastAPI backend on port 8000
- Starts the Streamlit dashboard on port 8501
- Launches the real-time inference pipeline
- Initialises the GPIO hardware alert subsystem

### 6. Open the Dashboard

From the Raspberry Pi:
```
http://localhost:8501
```

From another device on the same network:
```
http://<raspberry-pi-ip>:8501
```
Find your Pi's IP address by running on the Pi:
```
hostname -I
```

---

## Model Development

### Training Phase 1 — RGB Baseline
Trained on a curated public seat belt dataset. Established the detection baseline and exposed deployment limitations: class imbalance, clothing-colour sensitivity, and weak clipped-behind recall.

### Training Phase 2 — Final RGB Model
Dataset expanded with custom images captured using the Pi Camera V2 NoIR, manually photographed scenarios, and video-extracted frames. Targeted coverage of under-represented conditions: no-belt on white shirt, clipped-behind geometry, and decoy objects common in the Indian market (dupattas, lanyards, bag straps). Produced `best_final.pt`, exported to `best320.onnx`.

> `best_final.pt` is tracked in this repository. `best320.onnx` is the tracked production ONNX artifact.

### Training Phase 3 — Infrared Experiments
Two infrared models were obtained through transfer learning by fine-tuning the final RGB model (`best_final.pt`) on infrared datasets:

- `best_ir_raw.onnx` — fine-tuned on raw infrared imagery
- `best_ir_clahe.onnx` — fine-tuned on CLAHE-enhanced infrared imagery

Both models were trained for 15 additional epochs and evaluated for passive-cabin operation under near-zero-lux conditions. CLAHE improved local contrast at belt-torso boundaries and outperformed the raw IR model. Both IR models remain experimental; `best320.onnx` is the production deployment model. IR ONNX exports are not tracked in this repository.

---

## Performance

| Model | mAP50 | mAP50-95 | FPS (Pi 5) | Tracked | Status |
|---|---|---|---|---|---|
| RGB Final (`best320.onnx`) | 0.899 | 0.603 | 18–25 | ✓ | **Production** |
| IR Raw (`best_ir_raw.onnx`) | 0.956 | 0.658 | ~22 | ✗ | Experimental |
| IR CLAHE (`best_ir_clahe.onnx`) | 0.952 | 0.670 | ~22 | ✗ | Experimental |

`best320.onnx` was selected for deployment because it provides the best real-world balance of throughput, robustness across lighting conditions, and operational reliability. The higher mAP values of the IR models reflect in-distribution validation only; real-world IR performance is lower due to domain shift from the RGB training distribution.

---

## Validation

The system was tested across the following conditions:

| Condition | Lux Range | Test Scenarios |
|---|---|---|
| Daylight | ~10,000–25,000 lux | White shirt, black shirt — all four classes |
| Active cabin | ~50–100 lux | White shirt, black shirt — all four classes |
| Passive cabin (IR) | < 1 lux | IR illuminator — proper_belt and no_belt |

Decoy objects tested include bag straps, lanyards, cables, and dupattas.

> A representative end-to-end demonstration video (demo_recordings/05_full_demo/output.mp4) is included in this repository. Complete scenario-wise validation recordings are available in the project submission package provided to MSIL.

---

## Reproducibility

The project can be reproduced using:

```bash
pip install -r requirements.txt
bash master.sh
```

`requirements.txt` defines all Python dependencies required for deployment. `master.sh` orchestrates backend startup, dashboard initialisation, inference execution, and hardware alert activation. These two files serve distinct and complementary roles: `requirements.txt` enables environment portability; `master.sh` enables operational orchestration.

All experiments, datasets, training history, and deployment artifacts are documented within this repository and the accompanying project reports in `reports/`.

---

## Known Limitations

- Very low-contrast combinations (black belt on black shirt) remain challenging in dim light
- Dupattas and diagonal garments occasionally misclassify as `proper_belt`
- `clipped_behind` detection is geometry-dependent and sensitive to camera angle
- Third-row passenger monitoring was not validated (single-camera field of view)
- Infrared models show residual domain shift compared to the RGB deployment model

---

## Demonstration Assets

The `docs/` folder contains all tracked visual assets:

| Asset | File |
|---|---|
| System architecture diagram | `docs/System Architecture.jpeg` |
| Hardware architecture diagram | `docs/Hardware Architecture.jpeg` |
| Alert logic flowchart | `docs/Alert Logic Flowchart.jpeg` |
| Model development workflow | `docs/MODEL_DEVELOPMENT.jpg` |
| Model comparison visual | `docs/model_comparison.jpg` |
| Dashboard screenshot | `docs/dashboard.jpg` |
| Hardware setup photo | `docs/hardware_setup.jpg` |
| GPIO wiring schematic | `docs/gpio_wiring.jpg` |
| IR illuminator photo | `docs/ir_illuminator.jpg` |

Video demonstrations are excluded from version control. See the project submission package for full recordings.

---

## Author

**Dakshayani Sharma**  
Project Xceed Program  
Maruti Suzuki India Limited (MSIL), 2026