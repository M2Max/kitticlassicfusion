"""L1: pipeline classica sulla nuvola del LiDAR (progetto 1)."""
from kitti.data import load_lidar
from pipeline.classic import detect_points


def cloud(i):
    return load_lidar(i)[:, :3]                             # x, y, z (senza riflettanza)

def detect(i, calib):
    return detect_points(cloud(i), calib)
