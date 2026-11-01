"""Pipeline classica di L1 su una nuvola qualsiasi in coordinate LiDAR (LiDAR, stereo o fusa)."""
import numpy as np

from config import IMG_W
from kitti.transforms import lidar_to_rectcam
from pipeline.bev import build_grid, label_points, labeling
from pipeline.boxes import find_boxes, grow_car
from pipeline.classify import classify
from pipeline.floor import ransac_floor


def in_camera_fov(box, calib):
    center = np.array([[box[0], box[1], box[2] + box[5] / 2]])        # centro della scatola, in LiDAR
    p = lidar_to_rectcam(calib, center)[0]                  # stesso punto in camera rettificata
    if p[2] < 0.1:                                          # z <= 0: dietro la camera, niente divisione
        return False
    u, _, w = calib["P2"].reshape(3, 4) @ np.append(p, 1.0) # un punto solo: forma a colonna, niente trasposte
    return 0 <= u / w < IMG_W                               # colonna del pixel dentro la foto

def proposals(points, calib, min_height=None, max_height=None, min_cell_points=1):
    # Un cluster della griglia BEV = una scatola candidata, se la camera 2 la vede. points in coordinate LiDAR.
    # min/max_height: solo i punti fra queste altezze sopra il suolo; min_cell_points: punti per casella piena.
    # Servono allo stereo, rumoroso; i default lasciano tutto come in L1
    no_floor, n, d = ransac_floor(points)
    height = no_floor @ n + d                               # distanza dal piano del suolo
    keep = np.ones(len(no_floor), dtype=bool)
    if min_height is not None:
        keep &= height >= min_height
    if max_height is not None:
        keep &= height < max_height
    no_floor = no_floor[keep]
    grid, i_r, i_theta, valid_mask = build_grid(no_floor, min_points=min_cell_points)
    clean_labeled, n_clusters = labeling(grid)
    in_range = no_floor[valid_mask]
    point_labels = label_points(clean_labeled, i_r[valid_mask], i_theta[valid_mask])
    boxes, clusters = find_boxes(in_range, point_labels, n_clusters, n, d, return_points=True)
    seen = [k for k, box in enumerate(boxes) if in_camera_fov(box, calib)]     # la camera non vede il resto

    return boxes[seen].reshape(-1, 7), [clusters[k] for k in seen], n, d

def detect_points(points, calib, min_height=None, max_height=None, min_cell_points=1):
    # Scatole candidate + regole su l, w, h
    boxes, _, _, _ = proposals(points, calib, min_height, max_height, min_cell_points)
    return rule_detections(boxes)

def rule_detections(boxes):
    det_boxes, det_classes = [], []
    for box in boxes:
        c = classify(box)
        if c is None:                                       # non è niente
            continue
        if c == "Car":
            box = grow_car(box)                             # la parte nascosta sta dalla parte lontana
        det_boxes.append(box)
        det_classes.append(c)
    scores = np.ones(len(det_boxes))                       # metodo classico: nessuna confidenza, tutte 1
    return np.array(det_boxes).reshape(-1, 7), det_classes, scores
