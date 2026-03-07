import torch
import cv2
import numpy as np
from torchvision.transforms import v2
from model import ViolenceCNNPoseModel

# ---------------- CONFIG ----------------
VIDEO_PATH = r"C:\Users\gowth\OneDrive\Desktop\mini project\datasets\Balanced Accident Video Dataset\train\major\2019_06_25_cuted.mp4"
MODEL_PATH = "best_violence_model_v3.pth"
SEQ_LEN = 16
DEVICE = "cpu"
# ---------------------------------------


# Transform (MUST match training)
transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((112, 112)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load model
model = ViolenceCNNPoseModel()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()


cap = cv2.VideoCapture(VIDEO_PATH)

frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Show video
    cv2.imshow("Violence Detection", frame)

    # Prepare frame for model
    frame_tensor = transform(frame)
    frames.append(frame_tensor)

    # Keep only last SEQ_LEN frames
    if len(frames) > SEQ_LEN:
        frames.pop(0)

    # Run inference only when enough frames
    if len(frames) == SEQ_LEN:
        video_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0)
        video_tensor = video_tensor.to(DEVICE)

        # No pose → zeros
        pose_tensor = torch.zeros(1, SEQ_LEN, 34).to(DEVICE)

        with torch.no_grad():
            logits = model(video_tensor, pose_tensor)
            prob = torch.softmax(logits, dim=1)[0]

            pred = prob.argmax().item()
            confidence = prob[pred].item()

        label = "VIOLENCE" if pred == 1 else "NON-VIOLENCE"
        print(f"{label} ({confidence:.2f})")

    # Press q to quit
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
