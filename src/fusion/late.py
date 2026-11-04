"""Fusione late: si tengono le detection del LiDAR confermate da una detection della camera."""
import numpy as np
from scipy.optimize import linear_sum_assignment

GATE = {"Car": 2.0, "Pedestrian": 1.0, "Cyclist": 1.5}     # distanza massima fra i centri (m), per classe del LiDAR
FAR = 1e6                                                   # costo di una coppia impossibile


def associate(lidar_boxes, lidar_classes, camera_boxes):
    # Coppie (lidar, camera) con centri vicini, abbinamento a distanza totale minima (algoritmo ungherese)
    if len(lidar_boxes) == 0 or len(camera_boxes) == 0:
        return []
    dist = np.hypot(lidar_boxes[:, None, 0] - camera_boxes[None, :, 0], lidar_boxes[:, None, 1] - camera_boxes[None, :, 1])
    gate = np.array([GATE[c] for c in lidar_classes])[:, None]
    cost = np.where(dist < gate, dist, FAR)
    rows, cols = linear_sum_assignment(cost)
    return [(i, j) for i, j in zip(rows, cols) if cost[i, j] < FAR]

def confirmed(lidar, camera):
    # lidar, camera: (scatole, classi, confidenze). Scatola e classe restano quelle del LiDAR, piu' precise
    boxes, classes, scores = lidar
    keep = sorted(i for i, _ in associate(boxes, classes, camera[0]))
    return boxes[keep].reshape(-1, 7), [classes[i] for i in keep], scores[keep]
