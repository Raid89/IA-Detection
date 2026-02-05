import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO
from collections import deque, defaultdict
from pathlib import Path

# =============================
# CONFIG
# =============================
WIN = 32
D = 34
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "models/classifier/pose_cls.pt"

POSE_MODEL = "yolov8n-pose.pt"
TRACKER = "bytetrack.yaml"

SCORE_THRESHOLD = 0.6     # probabilidad mínima de suspect
CONSEC_WINDOWS = 10         # ventanas seguidas para alerta
EMA_ALPHA = 0.4            # suavizado de score

# =============================
# MODELO CLASIFICADOR
# =============================
class TCNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(D, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B,D,WIN)
        x = self.net(x).squeeze(-1)
        return self.fc(x)

# =============================
# NORMALIZACIÓN POSE
# =============================
L_SH, R_SH, L_HIP, R_HIP = 5, 6, 11, 12

def normalize_kps(kps_17x3, min_conf=0.35):
    xy = kps_17x3[:, :2].astype(np.float32)
    cf = kps_17x3[:, 2].astype(np.float32)

    if (cf[[L_SH, R_SH, L_HIP, R_HIP]].mean()) < min_conf:
        return None

    hip = (xy[L_HIP] + xy[R_HIP]) / 2.0
    shoulder = (xy[L_SH] + xy[R_SH]) / 2.0

    scale = np.linalg.norm(shoulder - hip)
    if scale < 1e-3:
        return None

    xy = (xy - hip) / (scale + 1e-6)
    return xy.reshape(-1).astype(np.float32)  # (34,)

# =============================
# MAIN
# =============================
def infer_video(video_path):
    # cargar modelos
    pose_model = YOLO(POSE_MODEL)
    clf = TCNClassifier().to(DEVICE)
    clf.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    clf.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("No pude abrir el video")

    # buffers por persona
    pose_buffers = defaultdict(lambda: deque(maxlen=WIN))
    consec_hits = defaultdict(int)
    ema_score = defaultdict(float)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model.track(
            frame,
            persist=True,
            tracker=TRACKER,
            conf=0.25,
            iou=0.5,
            verbose=False
        )

        r = results[0]
        if r.boxes is None or r.keypoints is None:
            cv2.imshow("infer", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        ids = r.boxes.id
        if ids is None:
            continue

        ids = ids.cpu().numpy().astype(int)
        xy = r.keypoints.xy.cpu().numpy()
        cf = r.keypoints.conf.cpu().numpy()
        boxes = r.boxes.xyxy.cpu().numpy()

        for pid, box, xy_i, cf_i in zip(ids, boxes, xy, cf):
            kps = np.concatenate([xy_i, cf_i[:, None]], axis=1)
            vec = normalize_kps(kps)
            if vec is None:
                continue

            pose_buffers[pid].append(vec)

            if len(pose_buffers[pid]) == WIN:
                x = np.stack(pose_buffers[pid], axis=0)  # (WIN,34)
                xt = torch.tensor(x, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                with torch.no_grad():
                    logits = clf(xt)
                    prob = torch.softmax(logits, dim=1)[0, 1].item()

                # EMA suavizado
                ema_score[pid] = EMA_ALPHA * prob + (1 - EMA_ALPHA) * ema_score[pid]

                if ema_score[pid] >= SCORE_THRESHOLD:
                    consec_hits[pid] += 1
                else:
                    consec_hits[pid] = 0

                # dibujar
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                label = f"ID {pid} | {ema_score[pid]:.2f}"

                if consec_hits[pid] >= CONSEC_WINDOWS:
                    color = (0, 0, 255)
                    label = f"⚠ SHOPLIFTING | ID {pid} | {ema_score[pid]:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow("infer", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# =============================
# CLI
# =============================
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    args = ap.parse_args()

    infer_video(args.video)
