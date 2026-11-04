import cv2
import numpy as np


def camera_offset(P):
    # P = K [I | t]: l'ultima colonna e' K t, quindi t = K^-1 P[:, 3] (dove sta la camera rispetto alla camera 0)
    return np.linalg.solve(P[:, :3], P[:, 3])

def stereo_params(calib):
    P2 = calib["P2"].reshape(3, 4)
    P3 = calib["P3"].reshape(3, 4)
    f, cu, cv = P2[0, 0], P2[0, 2], P2[1, 2]                # in KITTI fx = fy
    baseline = camera_offset(P2)[0] - camera_offset(P3)[0]  # la camera 3 sta ~54 cm a destra della 2
    return f, cu, cv, baseline

def disparity_to_depth(disp, f, baseline):
    depth = np.full(disp.shape, np.nan)                     # nan = disparita' non valida
    valid = disp > 0
    depth[valid] = f * baseline / disp[valid]               # triangoli simili: Z = f B / d
    return depth

def depth_to_points(depth, calib, max_depth=64.0):
    P2 = calib["P2"].reshape(3, 4)
    f, cu, cv = P2[0, 0], P2[0, 2], P2[1, 2]
    v, u = np.nonzero(depth <= max_depth)                   # nan <= x e' sempre False: escluse anche le non valide
    z = depth[v, u]
    camera2 = np.stack([(u - cu) * z / f, (v - cv) * z / f, z], axis=1)   # pinhole al contrario
    return camera2 - camera_offset(P2)                      # X2 = X0 + t2: punti in camera 0 rettificata

def remove_flying_pixels(depth, max_jump=0.05):
    # Ai bordi degli oggetti lo stereo mescola primo piano e sfondo: nel 3 x 3 la profondita' salta
    kernel = np.ones((3, 3), np.uint8)
    valid = np.isfinite(depth)
    farthest = cv2.dilate(np.where(valid, depth, 0).astype(np.float32), kernel)      # massimo dei vicini
    nearest = cv2.erode(np.where(valid, depth, np.inf).astype(np.float32), kernel)   # minimo dei vicini
    flying = valid & ((farthest - nearest) > max_jump * np.where(valid, depth, 1))
    clean = depth.copy()
    clean[flying] = np.nan
    return clean
