import cv2
import time
import math
import itertools
from ultralytics import YOLO

# ==================================================
# CONFIGURATION
# ==================================================
VIDEO_PATH = r"C:\Users\gowth\OneDrive\Desktop\mini project\datasets\Balanced Accident Video Dataset\test\major\aug_15_2019_08_22_cuted.mp4"

# YOLO parameters
CONF_THRESHOLD = 0.4
IOU_THRESHOLD = 0.5
YOLO_INTERVAL = 2   # YOLO every N frames (performance + stability)

# Accident logic parameters
COLLISION_IOU = 0.05
MIN_ACCIDENT_FRAMES = 2
SPEED_CHANGE_THRESHOLD = 20
MIN_VEHICLES_FOR_ACCIDENT = 2
PROXIMITY_THRESHOLD = 80  # pixels

# ==================================================
# LOAD YOLO MODEL
# ==================================================
model = YOLO("yolo12n.pt")

# ==================================================
# HELPER FUNCTIONS
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

# ==================================================
# MAIN LOOP
# ==================================================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_id += 1
    frame = cv2.resize(frame, (416, 416))

    # ---------------- YOLO INFERENCE ----------------
    if frame_id % YOLO_INTERVAL == 0:
        last_results = model(frame, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)

    results = last_results

    boxes = []
    centroids = []

    # ---------------- VEHICLE DETECTION ----------------
    if results:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                if label in ["car", "bus", "truck", "motorcycle"]:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    boxes.append((x1, y1, x2, y2))

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    centroids.append((cx, cy))

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

    # ---------------- COLLISION CHECK ----------------
    collision_detected = False
    for a, b in itertools.combinations(boxes, 2):
        if compute_iou(a, b) > COLLISION_IOU:
            collision_detected = True

    # ---------------- SPEED CHANGE CHECK ----------------
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

    # ---------------- TEMPORAL CONSISTENCY ----------------
    if collision_detected or speed_change_detected:
        accident_counter += 1
    else:
        accident_counter = max(accident_counter - 1, 0)

    accident_detected = accident_counter >= MIN_ACCIDENT_FRAMES

    if accident_detected:
        print(f"[ALERT] Accident detected at frame {frame_id} | confidence={accident_counter}")

    # ---------------- FPS ----------------
    curr_time = time.time()
    fps_display = 1 / (curr_time - prev_time)
    prev_time = curr_time

    # ---------------- OVERLAYS ----------------
    status = "ACCIDENT DETECTED" if accident_detected else "NORMAL TRAFFIC"
    color = (0, 0, 255) if accident_detected else (0, 255, 0)

    cv2.putText(frame, status, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(frame, f"Confidence: {accident_counter}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"FPS: {int(fps_display)}", (15, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"Frame: {frame_id}", (15, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # ---------------- UPDATE MEMORY ----------------
    prev_centroids = centroids.copy()

    # ---------------- DISPLAY ----------------
    cv2.imshow("AI Road Accident Detection System", frame)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        break

# ==================================================
# CLEANUP
# ==================================================
cap.release()
cv2.destroyAllWindows()