<p align="center">
  <h1 align="center">🛡️ AI-Powered Smart Surveillance for Accident & Violence Detection</h1>
  <p align="center">
    A real-time dual-pipeline surveillance system that detects road accidents and violent incidents from video feeds, with automated email alerts to hospitals and police.
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/YOLOv12n-Ultralytics-00FFFF?logo=yolo&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-green" />
  <img src="https://img.shields.io/badge/Platform-Windows-0078D6?logo=windows&logoColor=white" />
</p>

---

## 📌 Overview

This project implements a **unified AI surveillance system** that simultaneously runs two detection pipelines on a single video stream:

| Pipeline               | Model                            | What it Detects                                                                    |
| ---------------------- | -------------------------------- | ---------------------------------------------------------------------------------- |
| **Accident Detection** | YOLOv12n + Rule-based heuristics | Vehicle collisions via IoU overlap, sudden speed changes, and temporal consistency |
| **Violence Detection** | Custom `ViolenceCNNPoseModel`    | Violent behavior via multi-modal fusion of visual, pose, and motion features       |

When a detection exceeds **70% confidence**, the system automatically sends **email alerts** — to hospitals & police for accidents, and to police for violence.

---

## 🏗️ Architecture

### System Pipeline

```
Video Input → Frame Acquisition
                 │
       ┌─────────┴─────────┐
       ▼                     ▼
  Accident Pipeline      Violence Pipeline
  ┌──────────────┐     ┌───────────────────┐
  │ YOLOv12n     │     │ ViolenceCNNPoseModel│
  │ Vehicle Det. │     │ (16-frame window)  │
  │ IoU Collision│     │                    │
  │ Speed Change │     │ ┌─Visual (ResNet34)│
  │ Temporal     │     │ ├─Pose (MLP+LSTM)  │
  │ Consistency  │     │ └─Motion (FrameDiff)│
  └──────┬───────┘     └────────┬──────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
            Alert Decision Engine
          (confidence ≥ 0.7 → alert)
                    │
                    ▼
          SMTP Email Alerts
     ┌──────────┬──────────┐
     ▼          ▼          ▼
  Hospitals   Police    Display
```

### ViolenceCNNPoseModel — Multi-Modal Deep Learning

The custom violence detection model fuses **three complementary feature streams**:

| Branch     | Input                   | Architecture                                   | Output Dim |
| ---------- | ----------------------- | ---------------------------------------------- | ---------- |
| **Visual** | RGB frames (112×112)    | ResNet-34 → Bi-LSTM (2-layer) → Self-Attention | 512        |
| **Pose**   | 17 keypoints × 2 coords | MLP → Bi-LSTM (2-layer) → Self-Attention       | 256        |
| **Motion** | Frame differences       | 2-layer CNN (optical flow approximation)       | 128        |

These are concatenated into an **896-dim fusion vector** and classified through a fully-connected head with LayerNorm and dropout.

---

## 📊 Results

### Violence Detection (RWF-2000 Dataset)

| Metric            | Value      |
| ----------------- | ---------- |
| **Accuracy**      | **74.90%** |
| AUC-ROC           | 0.8317     |
| Precision (macro) | 0.7516     |
| Recall (macro)    | 0.7478     |
| F1-Score (macro)  | 0.7476     |
| Specificity       | 0.8049     |
| Sensitivity       | 0.6907     |

**Confusion Matrix** (482 test samples):

|                          | Pred: Non-Violence | Pred: Violence |
| ------------------------ | ------------------ | -------------- |
| **Actual: Non-Violence** | 198 (TN)           | 48 (FP)        |
| **Actual: Violence**     | 73 (FN)            | 163 (TP)       |

**5-Fold Cross-Validation:** Accuracy = 0.7491 ± 0.0736

### Accident Detection (Balanced Accident Video Dataset)

| Metric                      | Value      |
| --------------------------- | ---------- |
| **Detection Rate (Recall)** | **72.89%** |
| Correctly Detected          | 164 / 225  |
| Avg Confidence (detected)   | 57.73      |
| Avg Confidence (missed)     | 0.31       |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA support (tested on RTX 3050 Laptop)
- [Git](https://git-scm.com/)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/smart-surveillance.git
cd smart-surveillance

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python numpy matplotlib
```

### Download Model Weights

| File                         | Description                                           |
| ---------------------------- | ----------------------------------------------------- |
| `best_violence_model_v3.pth` | Trained ViolenceCNNPoseModel weights                  |
| `yolo12n.pt`                 | YOLOv12 nano weights (auto-downloaded by Ultralytics) |

Place the violence model weights in the project root directory.

---

## ⚙️ Configuration

Before running, configure the following in `combined_detection.py`:

### 1. Email Settings (for automated alerts)

```python
EMAIL_SENDER = "your_email@gmail.com"
EMAIL_PASSWORD = "your_app_password"   # Use Gmail App Password (not regular password)
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
```

> **Note:** Enable 2-Factor Authentication on your Google account, then generate an [App Password](https://myaccount.google.com/apppasswords).

### 2. Alert Recipients

```python
HOSPITAL_CONTACTS = {
    "City Hospital": "hospital@example.com",
}
POLICE_CONTACTS = {
    "Police Station": "police@example.com",
}
```

### 3. Video Source

```python
VIDEO_PATH = r"path\to\your\video.mp4"
# Or use webcam:
VIDEO_PATH = 0
```

---

## ▶️ Usage

### Run the Combined Detection System

```bash
python combined_detection.py
```

Press **`q`** to quit.

### Run Violence Detection Only (Inference)

```bash
python inference.py
```

### Run Accident Detection Only

```bash
python inference2.py
```

### Display Output

The real-time video window shows:

- Accident status & confidence score
- Violence status & confidence score
- Alert indicators when triggered
- FPS counter & frame number
- Bounding boxes around detected vehicles

---

## 📁 Project Structure

```
├── combined_detection.py          # Main: dual-pipeline detection + email alerts
├── model.py                       # ViolenceCNNPoseModel architecture definition
├── inference.py                   # Standalone accident detection (YOLOv12n)
├── inference2.py                  # Standalone violence detection inference
├── train.ipynb                    # Violence model training notebook
├── train2.ipynb                   # Alternative training notebook
├── evaluate.ipynb                 # Model evaluation notebook
├── evaluation_metrics.ipynb       # Comprehensive metrics computation
├── evaluation_metrics_inference.ipynb  # Accident detection evaluation
├── evaluation_metrics_summary.csv      # Violence detection results
├── inference_evaluation_metrics_summary.csv  # Accident detection results
├── generate_images.py             # Documentation figure generation
├── count_videos.py                # Dataset video counting utility
├── best_violence_model_v3.pth     # Best trained model weights
├── yolo12n.pt                     # YOLOv12 nano weights
├── Project_Documentation.md       # Full academic documentation
├── SETUP_GUIDE.md                 # Detailed setup instructions
└── README.md                      # This file
```

---

## 🔧 Tunable Parameters

### Accident Detection

| Parameter                | Default | Description                         |
| ------------------------ | ------- | ----------------------------------- |
| `CONF_THRESHOLD`         | 0.4     | YOLO detection confidence           |
| `COLLISION_IOU`          | 0.05    | IoU threshold for collision         |
| `MIN_ACCIDENT_FRAMES`    | 3       | Frames needed to confirm accident   |
| `SPEED_CHANGE_THRESHOLD` | 20      | Pixel displacement for speed change |
| `PROXIMITY_THRESHOLD`    | 80      | Max pixel distance between vehicles |
| `YOLO_INTERVAL`          | 2       | Run YOLO every N frames             |

### Violence Detection

| Parameter        | Default | Description                           |
| ---------------- | ------- | ------------------------------------- |
| `SEQ_LEN`        | 16      | Number of frames per inference window |
| Frame resolution | 112×112 | Input size for ViolenceCNNPoseModel   |

### Alert System

| Parameter              | Default | Description                         |
| ---------------------- | ------- | ----------------------------------- |
| `CONFIDENCE_THRESHOLD` | 0.7     | Minimum confidence to trigger alert |
| `ALERT_COOLDOWN`       | 150     | Frames between repeated alerts      |

---

## 📚 Datasets Used

| Dataset                                                                               | Task               | Size                          | Source                          |
| ------------------------------------------------------------------------------------- | ------------------ | ----------------------------- | ------------------------------- |
| [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) | Violence Detection | 2,000 videos (Fight/NonFight) | Real-world surveillance footage |
| Balanced Accident Video Dataset                                                       | Accident Detection | 225 test videos               | Dashcam & traffic cameras       |

---

## 🧪 Training Details

| Parameter       | Value                                      |
| --------------- | ------------------------------------------ |
| Optimizer       | AdamW (lr=1e-3)                            |
| Loss            | CrossEntropyLoss                           |
| Scheduler       | ReduceLROnPlateau (factor=0.5, patience=2) |
| Epochs          | 25                                         |
| Best Checkpoint | Epoch 15 (Val Acc: 77.99%)                 |
| Batch Size      | 10 (train) / 5 (val)                       |
| GPU             | NVIDIA RTX 3050 Laptop                     |

---

## 🔮 Future Enhancements

- [ ] GPS coordinate integration for precise location tagging
- [ ] SMS / WhatsApp alert integration
- [ ] Database logging for historical event tracking
- [ ] Web dashboard for real-time monitoring
- [ ] Multi-camera support
- [ ] Fine-tuned YOLOv12 on accident-specific datasets
- [ ] Pose estimation with MediaPipe for real keypoints (currently zeros)

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLO framework
- [RWF-2000](https://github.com/mchengny/RWF2000-Video-Database-for-Violence-Detection) for the violence detection dataset
- [PyTorch](https://pytorch.org/) and [torchvision](https://github.com/pytorch/vision) for the deep learning framework
- ResNet-34 pretrained weights from ImageNet

---

<p align="center">
  <b>Built for smarter, safer cities.</b><br>
  If you find this useful, consider giving it a ⭐
</p>
