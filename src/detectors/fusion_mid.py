"""FM: griglia di occupazione fusa LiDAR + stereo, poi scatole e regole come in L1."""
import numpy as np

from detectors.camera_classic import stereo_cloud
from fusion.occupancy import fused_boxes
from kitti.data import load_lidar
from pipeline.classic import in_camera_fov, rule_detections
from pipeline.floor import ransac_floor


def cloud(i):
    return load_lidar(i)[:, :3]

def detect(i, calib):
    lidar = cloud(i)
    _, n, d = ransac_floor(lidar)                           # suolo dal LiDAR, piu' preciso, per tutti e due i sensori
    boxes = fused_boxes(lidar, stereo_cloud(i, calib), n, d)
    seen = np.array([b for b in boxes if in_camera_fov(b, calib)]).reshape(-1, 7)
    return rule_detections(seen)
