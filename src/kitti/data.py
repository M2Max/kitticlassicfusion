import cv2
import numpy as np

from config import DATA, SPLITS

def load_calib(i):
    calib = {}
    with open(DATA / "calib" / f"{i:06d}.txt") as f:
        for line in f:
            line = line.strip()
            if not line:                        # riga vuota in fondo ai file KITTI
                continue
            key, val = line.split(":", 1)
            calib[key] = np.array([float(x) for x in val.split()])
    return calib

def load_lidar(i):
    path = DATA / "velodyne" / f"{i:06d}.bin"
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)   # x, y, z, riflettanza

def load_image(i, cam=2):
    # Camera 2 (sinistra, quella delle label) o 3 (destra), RGB. La misura cambia fra frame (1224-1242 px)
    path = DATA / f"image_{cam}" / f"{i:06d}.png"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:                                         # OpenCV non solleva eccezioni: restituisce None
        raise FileNotFoundError(f"{path} (scarica le immagini con scripts/download_images.sh)")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)             # OpenCV legge in BGR

def load_stereo_pair(i):
    return load_image(i, 2), load_image(i, 3)

def frame_ids(split):
    # "train", "val", "trainval": frame elencati in ImageSets/<split>.txt
    return [int(line) for line in (SPLITS / f"{split}.txt").read_text().split()]

def load_labels(i):
    labels = []
    with open(DATA / "label_2" / f"{i:06d}.txt") as f:
        for line in f:
            v = line.split()
            if not v:                                       # riga vuota
                continue
            labels.append({
                "type": v[0],
                "truncated": float(v[1]),                   # 0 = intero, 1 = fuori dalla foto
                "occluded": int(v[2]),                      # 0 visibile, 1 in parte, 2 molto, 3 sconosciuto
                "bbox": np.array([float(x) for x in v[4:8]]),   # left, top, right, bottom (pixel)
                "hwl": np.array([float(x) for x in v[8:11]]),   # altezza, larghezza, lunghezza (metri)
                "loc": np.array([float(x) for x in v[11:14]]),  # centro del FONDO, camera rettificata
                "ry": float(v[14]),                         # rotation_y
            })
    return labels
