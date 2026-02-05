import os
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

# YOLOv8-Pose usa COCO17 keypoints.
# indices: left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12
def normalize_kps_coco17(kps_17x3: np.ndarray) -> np.ndarray:
    """
    kps shape: (17,3) => x,y,conf en pixeles
    returns: (34,) x,y normalizado y flatten
    """
    xy = kps_17x3[:, :2].copy()

    l_hip, r_hip = xy[11], xy[12]
    l_sh, r_sh = xy[5], xy[6]

    hip = (l_hip + r_hip) / 2.0
    shoulder = (l_sh + r_sh) / 2.0

    xy = xy - hip
    scale = np.linalg.norm(shoulder - hip) + 1e-6
    xy = xy / scale

    return xy.reshape(-1).astype(np.float32)  # 17*2 = 34

def extract(video_path: str, out_dir: str, max_frames: int = None, min_track_len: int = 30):
    os.makedirs(out_dir, exist_ok=True)

    # Modelo pose (se descarga solo la primera vez)
    pose_model = YOLO("yolov8n-pose.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No pude abrir video: {video_path}")

    track_buffers = {}  # person_id -> list of vectors (34,)
    frame_i = 0

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    pbar = tqdm(total=total, desc=os.path.basename(video_path))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_i += 1
        pbar.update(1)

        if max_frames and frame_i > max_frames:
            break

        # Tracking + pose
        results = pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.25,
            iou=0.5,
            verbose=False
        )

        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue

        ids = r.boxes.id
        kpts = r.keypoints

        if ids is None or kpts is None:
            continue

        ids = ids.cpu().numpy().astype(int)

        xy = kpts.xy.cpu().numpy()        # (N, 17, 2) pixel coords
        conf = kpts.conf.cpu().numpy()    # (N, 17)

        for pid, xy_i, conf_i in zip(ids, xy, conf):
            # filtro simple: si la pose es muy mala, skip
            if float(conf_i.mean()) < 0.2:
                continue

            kps = np.concatenate([xy_i, conf_i[:, None]], axis=1).astype(np.float32)  # (17,3)
            vec = normalize_kps_coco17(kps)  # (34,)

            track_buffers.setdefault(pid, []).append(vec)

    cap.release()
    pbar.close()

    # Guardar tracks como npy: un archivo por person-track
    base = os.path.splitext(os.path.basename(video_path))[0]
    saved = 0
    for pid, seq in track_buffers.items():
        if len(seq) < min_track_len:
            continue
        arr = np.stack(seq, axis=0)  # (T, 34)
        out_path = os.path.join(out_dir, f"{base}_id{pid}.npy")
        np.save(out_path, arr)
        saved += 1

    print(f"✅ Guardados {saved} tracks en {out_dir}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--min_track_len", type=int, default=30)
    args = ap.parse_args()
    extract(args.video, args.outdir, args.max_frames, args.min_track_len)
