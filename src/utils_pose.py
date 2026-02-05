import numpy as np

# MediaPipe Pose indices:
# left_shoulder=11 right_shoulder=12 left_hip=23 right_hip=24
def normalize_kps(kps_33x3: np.ndarray) -> np.ndarray:
    """
    kps shape: (33,3) => x,y,visibility in [0..1] relative to crop
    returns: (66,) flattened normalized x,y
    """
    xy = kps_33x3[:, :2].copy()

    hip = (xy[23] + xy[24]) / 2.0
    shoulder = (xy[11] + xy[12]) / 2.0

    xy = xy - hip
    scale = np.linalg.norm(shoulder - hip) + 1e-6
    xy = xy / scale

    return xy.reshape(-1).astype(np.float32)  # 33*2 = 66
