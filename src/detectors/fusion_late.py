"""FL-c: detection di L1 confermate da quelle di C1."""
from detectors import camera_classic, lidar
from fusion.late import confirmed


def cloud(i):
    return lidar.cloud(i)

def detect(i, calib):
    return confirmed(lidar.detect(i, calib), camera_classic.detect(i, calib))
