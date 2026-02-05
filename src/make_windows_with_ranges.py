import json
import numpy as np
import cv2
from pathlib import Path

# =========================================
# Resolver raíz del proyecto desde /src
# =========================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = PROJECT_ROOT / "data"

# =========================================
# Paths (TU ESTRUCTURA)
# =========================================
POSES_ROOT = DATA_ROOT / "poses"
POSES_NORMAL = POSES_ROOT / "normal"
POSES_SUSPECT = POSES_ROOT / "suspect"

RAW_ROOT = DATA_ROOT / "raw_videos"
RAW_NORMAL = RAW_ROOT / "normal"
RAW_SUSPECT = RAW_ROOT / "suspect"

RANGES_JSON = DATA_ROOT / "ranges.json"

OUT_ROOT = DATA_ROOT / "windows_dataset"
OUT_NORMAL = OUT_ROOT / "normal"
OUT_SUSPECT = OUT_ROOT / "suspect"

# =========================================
# Parámetros
# =========================================
WIN = 32
STRIDE = 16
MIN_T = 40
OVERLAP_THRESHOLD = 0.5  # 50% de la ventana cubierta por el evento

# Si un video suspect NO está en ranges.json:
# - "heuristic": centro 35%-65% como suspect, resto normal
# - "all_normal": todo a normal (más conservador)
FALLBACK_MODE = "heuristic"
HEUR_CENTER_START = 0.35
HEUR_CENTER_END = 0.65

# =========================================
# Utils
# =========================================
def ensure_dirs():
    OUT_NORMAL.mkdir(parents=True, exist_ok=True)
    OUT_SUSPECT.mkdir(parents=True, exist_ok=True)

def to_windows(seq: np.ndarray, win: int, stride: int):
    T, D = seq.shape
    starts = list(range(0, T - win + 1, stride))
    if not starts:
        return np.empty((0, win, D), dtype=np.float32), np.array([], dtype=np.int32)
    windows = [seq[s:s + win] for s in starts]
    return np.stack(windows).astype(np.float32), np.array(starts, dtype=np.int32)

def window_overlap(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    if inter <= 0:
        return 0.0
    return inter / max(1e-6, (a1 - a0))  # fracción de ventana cubierta

def base_from_npy(stem: str) -> str:
    # "video_001_id3" -> "video_001"
    return stem.split("_id")[0] if "_id" in stem else stem

def find_video_path(base: str, is_suspect: bool) -> Path | None:
    # busca un video cuyo nombre base coincida
    folder = RAW_SUSPECT if is_suspect else RAW_NORMAL
    for ext in (".mp4", ".avi", ".mkv", ".mov"):
        p = folder / f"{base}{ext}"
        if p.exists():
            return p
    return None

def get_fps(video_path: Path, default: float = 30.0) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 1 else default

def save_window(arr: np.ndarray, folder: Path, name: str):
    np.save(folder / name, arr.astype(np.float32))

# =========================================
# Procesos
# =========================================
def process_normal_poses():
    files = list(POSES_NORMAL.glob("*.npy"))
    print(f"📁 Poses NORMAL: {len(files)} archivos")

    total = 0
    for f in files:
        seq = np.load(f)
        if seq.ndim != 2 or seq.shape[1] != 34:
            print(f"⚠️ Skip (shape raro) {f.name}: {seq.shape}")
            continue
        if seq.shape[0] < MIN_T:
            continue

        windows, _ = to_windows(seq, WIN, STRIDE)
        for i in range(windows.shape[0]):
            save_window(windows[i], OUT_NORMAL, f"{f.stem}_w{i:05d}.npy")
        total += windows.shape[0]

    print(f"✅ Normal windows guardadas: {total}")

def process_suspect_poses_with_ranges(ranges: dict):
    files = list(POSES_SUSPECT.glob("*.npy"))
    print(f"📁 Poses SUSPECT: {len(files)} archivos")

    total_sus = 0
    total_norm = 0
    missing_video = 0

    for f in files:
        seq = np.load(f)
        if seq.ndim != 2 or seq.shape[1] != 34:
            print(f"⚠️ Skip (shape raro) {f.name}: {seq.shape}")
            continue
        T = seq.shape[0]
        if T < MIN_T:
            continue

        base = base_from_npy(f.stem)
        video_path = find_video_path(base, is_suspect=True)

        windows, starts = to_windows(seq, WIN, STRIDE)
        if windows.shape[0] == 0:
            continue

        # Si no encuentro el video, no puedo calcular fps -> aplico fallback
        if video_path is None:
            missing_video += 1
            events = []
            fps = 30.0
        else:
            fps = get_fps(video_path, default=30.0)
            # ranges.json debe referenciar el nombre real del archivo (con extensión)
            # Intentamos matchear por base contra cualquier extensión
            # Ej: si en json pusiste "video_001.mp4", ok. Si pusiste "video_001", también lo soportamos.
            events = ranges.get(video_path.name, ranges.get(base, []))

        # Si no hay eventos para ese video, aplica fallback
        if not events:
            if FALLBACK_MODE == "all_normal":
                for i in range(windows.shape[0]):
                    save_window(windows[i], OUT_NORMAL, f"{f.stem}_w{i:05d}.npy")
                total_norm += windows.shape[0]
                continue

            # heuristic
            center_start = int(T * HEUR_CENTER_START)
            center_end = int(T * HEUR_CENTER_END)

            for i, start_frame in enumerate(starts):
                name = f"{f.stem}_w{i:05d}.npy"
                if center_start <= start_frame <= center_end:
                    save_window(windows[i], OUT_SUSPECT, name)
                    total_sus += 1
                else:
                    save_window(windows[i], OUT_NORMAL, name)
                    total_norm += 1
            continue

        # Convertir eventos en segundos -> frames usando fps real
        ev_frames = []
        for e in events:
            s = int(float(e["start"]) * fps)
            t = int(float(e["end"]) * fps)
            if t > s:
                ev_frames.append((s, t))

        for i, start_frame in enumerate(starts):
            win_start = int(start_frame)
            win_end = int(start_frame + WIN)

            is_sus = False
            for es, ee in ev_frames:
                ov = window_overlap(win_start, win_end, es, ee)
                if ov >= OVERLAP_THRESHOLD:
                    is_sus = True
                    break

            name = f"{f.stem}_w{i:05d}.npy"
            if is_sus:
                save_window(windows[i], OUT_SUSPECT, name)
                total_sus += 1
            else:
                save_window(windows[i], OUT_NORMAL, name)
                total_norm += 1

    print(f"✅ Suspect windows guardadas: {total_sus}")
    print(f"✅ Normal (hard negatives) desde suspect: {total_norm}")
    if missing_video:
        print(f"⚠️ No encontré el video para {missing_video} tracks suspect. Usé fallback (fps=30).")

# =========================================
# Main
# =========================================
if __name__ == "__main__":
    ensure_dirs()

    if not RANGES_JSON.exists():
        raise RuntimeError(f"No encontré ranges.json en: {RANGES_JSON}")

    ranges = json.loads(RANGES_JSON.read_text(encoding="utf-8"))

    process_normal_poses()
    process_suspect_poses_with_ranges(ranges)

    print("\n🎯 Dataset final:")
    print(f"   {OUT_ROOT}")
    print(f"   WIN={WIN}, STRIDE={STRIDE}, OVERLAP>={OVERLAP_THRESHOLD}")
    print(f"   FALLBACK_MODE={FALLBACK_MODE}")
