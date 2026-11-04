"""FE: pipeline di L1 sulla nuvola fusa LiDAR + stereo."""
from detectors.camera_classic import stereo_cloud
from fusion.early import merge_clouds
from kitti.data import load_calib, load_lidar
from pipeline.classic import detect_points


def fused_cloud(i, calib):
    return merge_clouds(load_lidar(i)[:, :3], stereo_cloud(i, calib))

def cloud(i):
    return fused_cloud(i, load_calib(i))

def detect(i, calib):
    return detect_points(fused_cloud(i, calib), calib)
