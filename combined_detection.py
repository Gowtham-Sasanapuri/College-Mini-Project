import cv2
import time
import math
import itertools
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import v2
from model import ViolenceCNNPoseModel
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ==================================================
# CONFIGURATION
# ==================================================
VIDEO_PATH = r"C:\Users\gowth\OneDrive\Desktop\mini project\datasets\Dataset\real life violence situations\Real Life Violence Dataset\Violence\V_48.mp4"
# VIDEO_PATH = r"C:\Users\gowth\OneDrive\Desktop\mini project\datasets\Balanced Accident Video Dataset\test\major\aug_15_2019_08_22_cuted.mp4"

# YOLO parameters for accident detection
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
YOLO_INTERVAL = 2   # YOLO every N frames

# Accident logic parameters
COLLISION_IOU = 0.05
MIN_ACCIDENT_FRAMES = 3
SPEED_CHANGE_THRESHOLD = 20
MIN_VEHICLES_FOR_ACCIDENT = 2
PROXIMITY_THRESHOLD = 80  # pixels

# Violence detection parameters
VIOLENCE_MODEL_PATH = "best_violence_model_v3.pth"
SEQ_LEN = 16
DEVICE = "cpu"

# Alert thresholds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence to send alerts

# Contact information (replace with actual contacts)
HOSPITAL_CONTACTS = {
    "City Hospital": "bhuvanbalaji32@gmail.com",
    "Emergency Center": "bhuvanbalaji32@gmail.com"
}

POLICE_CONTACTS = {
    "Police Station Central": "bhuvanbalaji32@gmail.com",
    "Emergency Response": "bhuvanbalaji32@gmail.com"
}

# Email configuration (configure your SMTP settings)
EMAIL_SENDER = "gowthamsasanapuri.personal@gmail.com"
EMAIL_PASSWORD = "evur tpgk hryi ogfm"  # You need to generate this from Google
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# ==================================================
# ALERT FUNCTIONS
# ==================================================
def send_alert_email(recipients, subject, message):
    """
    Send alert email to specified recipients
    """
    try:
        # Create message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = ', '.join(recipients.values())
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        text = msg.as_string()
        server.sendmail(EMAIL_SENDER, list(recipients.values()), text)
        server.quit()
        
        print(f"✓ Alert sent to: {', '.join(recipients.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed to send alert: {str(e)}")
        return False


def send_accident_alert(frame_id, confidence):
    """
    Send accident alert to hospitals and police stations
    """
    subject = "🚨 ACCIDENT DETECTED - IMMEDIATE RESPONSE REQUIRED"
    message = f"""
ALERT: ACCIDENT DETECTED

Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Frame ID: {frame_id}
Confidence Score: {confidence:.2f}

This is an automated alert from the AI Detection System.
Immediate response and medical assistance may be required.

Location: [GPS coordinates would be added here]
    """
    
    print(f"\n{'='*60}")
    print("🚨 ACCIDENT ALERT")
    print(f"{'='*60}")
    
    # Send to hospitals
    send_alert_email(HOSPITAL_CONTACTS, subject, message)
    
    # Send to police
    send_alert_email(POLICE_CONTACTS, subject, message)
    
    print(f"{'='*60}\n")


def send_violence_alert(frame_id, confidence):
    """
    Send violence alert to police stations
    """
    subject = "⚠️ VIOLENCE DETECTED - POLICE INTERVENTION REQUIRED"
    message = f"""
ALERT: VIOLENCE DETECTED

Time: {time.strftime('%Y-%m-%d %H:%M:%S')}
Frame ID: {frame_id}
Confidence Score: {confidence:.2f}

This is an automated alert from the AI Detection System.
Immediate police intervention may be required.

Location: [GPS coordinates would be added here]
    """
    
    print(f"\n{'='*60}")
    print("⚠️ VIOLENCE ALERT")
    print(f"{'='*60}")
    
    # Send to police only
    send_alert_email(POLICE_CONTACTS, subject, message)
    
    print(f"{'='*60}\n")


# ==================================================
# HELPER FUNCTIONS FOR ACCIDENT DETECTION
# ==================================================
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter

    return inter / union if union > 0 else 0


def near_any_other(c1, centroids, threshold):
    for c2 in centroids:
        if c1 != c2:
            if math.hypot(c1[0] - c2[0], c1[1] - c2[1]) < threshold:
                return True
    return False


# ==================================================
# MODEL LOADING
# ==================================================
print("Loading models...")

# Load YOLO for accident detection
accident_model = YOLO("yolo12n.pt")
print("✓ YOLO model loaded")

# Load violence detection model
violence_transform = v2.Compose([
    v2.ToImage(),
    v2.Resize((112, 112)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

violence_model = ViolenceCNNPoseModel()
violence_model.load_state_dict(torch.load(VIOLENCE_MODEL_PATH, map_location=DEVICE))
violence_model.to(DEVICE)
violence_model.eval()
print("✓ Violence detection model loaded")

# ==================================================
# VIDEO SETUP
# ==================================================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

fps_video = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps_video) if fps_video > 0 else 30

# ==================================================
# STATE VARIABLES
# ==================================================
frame_id = 0
prev_centroids = []
accident_counter = 0
last_results = None
prev_time = time.time()

# Violence detection state
violence_frames = []
violence_label = "NON-VIOLENCE"
violence_confidence = 0.0

# Alert tracking (to avoid sending multiple alerts)
last_accident_alert_frame = -1000
last_violence_alert_frame = -1000
ALERT_COOLDOWN = 150  # frames between alerts

print("\n" + "="*60)
print("COMBINED DETECTION SYSTEM STARTED")
print("="*60)
print("Press 'q' to quit\n")

# ==================================================
# MAIN LOOP
# ==================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    original_frame = frame.copy()
    frame_resized = cv2.resize(frame, (416, 416))

    # ============================================
    # ACCIDENT DETECTION
    # ============================================
    if frame_id % YOLO_INTERVAL == 0:
        last_results = accident_model(frame_resized, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    results = last_results
    boxes = []
    centroids = []

    # Vehicle detection
    if results:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = accident_model.names[cls_id]

                if label in ["car", "bus", "truck", "motorcycle"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    centroids.append((cx, cy))

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame_resized, (cx, cy), 4, (0, 255, 255), -1)

    # Collision check
    collision_detected = False
    for a, b in itertools.combinations(boxes, 2):
        if compute_iou(a, b) > COLLISION_IOU:
            collision_detected = True

    # Speed change check
    speed_change_detected = False
    if len(centroids) >= MIN_VEHICLES_FOR_ACCIDENT and prev_centroids:
        min_len = min(len(prev_centroids), len(centroids))
        for i in range(min_len):
            dx = centroids[i][0] - prev_centroids[i][0]
            dy = centroids[i][1] - prev_centroids[i][1]
            speed = math.hypot(dx, dy)

            if speed > SPEED_CHANGE_THRESHOLD and near_any_other(
                centroids[i], centroids, PROXIMITY_THRESHOLD
            ):
                speed_change_detected = True

    # Temporal consistency
    if collision_detected or speed_change_detected:
        accident_counter += 1
    else:
        accident_counter = max(accident_counter - 1, 0)

    accident_detected = accident_counter >= MIN_ACCIDENT_FRAMES
    
    # Calculate accident confidence (0-1 scale)
    accident_confidence = min(accident_counter / (MIN_ACCIDENT_FRAMES + 5), 1.0)

    # ============================================
    # VIOLENCE DETECTION
    # ============================================
    frame_tensor = violence_transform(original_frame)
    violence_frames.append(frame_tensor)

    if len(violence_frames) > SEQ_LEN:
        violence_frames.pop(0)

    if len(violence_frames) == SEQ_LEN:
        video_tensor = torch.stack(violence_frames).permute(1, 0, 2, 3).unsqueeze(0)
        video_tensor = video_tensor.to(DEVICE)
        pose_tensor = torch.zeros(1, SEQ_LEN, 34).to(DEVICE)

        with torch.no_grad():
            logits = violence_model(video_tensor, pose_tensor)
            prob = torch.softmax(logits, dim=1)[0]
            pred = prob.argmax().item()
            violence_confidence = prob[pred].item()

        violence_label = "VIOLENCE" if pred == 1 else "NON-VIOLENCE"

    violence_detected = violence_label == "VIOLENCE"

    # ============================================
    # ALERT SYSTEM
    # ============================================
    # Send accident alert if confidence >= 0.7
    if accident_detected and accident_confidence >= CONFIDENCE_THRESHOLD:
        if frame_id - last_accident_alert_frame > ALERT_COOLDOWN:
            send_accident_alert(frame_id, accident_confidence)
            last_accident_alert_frame = frame_id

    # Send violence alert if confidence >= 0.7
    if violence_detected and violence_confidence >= CONFIDENCE_THRESHOLD:
        if frame_id - last_violence_alert_frame > ALERT_COOLDOWN:
            send_violence_alert(frame_id, violence_confidence)
            last_violence_alert_frame = frame_id

    # ============================================
    # FPS CALCULATION
    # ============================================
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ============================================
    # DISPLAY OVERLAYS
    # ============================================
    # Accident status
    accident_status = "ACCIDENT DETECTED" if accident_detected else "NORMAL TRAFFIC"
    accident_color = (0, 0, 255) if accident_detected else (0, 255, 0)
    
    cv2.putText(frame_resized, f"Accident: {accident_status}", (15, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, accident_color, 2)
    cv2.putText(frame_resized, f"Accident Conf: {accident_confidence:.2f}", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Violence status
    violence_color = (0, 0, 255) if violence_detected else (0, 255, 0)
    cv2.putText(frame_resized, f"Violence: {violence_label}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, violence_color, 2)
    cv2.putText(frame_resized, f"Violence Conf: {violence_confidence:.2f}", (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Alert indicators
    if accident_confidence >= CONFIDENCE_THRESHOLD and accident_detected:
        cv2.putText(frame_resized, "⚠ ALERT SENT TO HOSPITALS & POLICE", (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    if violence_confidence >= CONFIDENCE_THRESHOLD and violence_detected:
        cv2.putText(frame_resized, "⚠ ALERT SENT TO POLICE", (15, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # FPS and frame info
    cv2.putText(frame_resized, f"FPS: {int(fps_display)}", (15, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(frame_resized, f"Frame: {frame_id}", (15, 235),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Update memory
    prev_centroids = centroids.copy()

    # Display
    cv2.imshow("Combined Detection System - Accident & Violence", frame_resized)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# ==================================================
# CLEANUP
# ==================================================
cap.release()
cv2.destroyAllWindows()
print("\n" + "="*60)
print("DETECTION SYSTEM STOPPED")
print("="*60)
