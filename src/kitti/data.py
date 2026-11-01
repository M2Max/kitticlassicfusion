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
