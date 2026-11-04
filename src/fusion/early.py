"""Fusione early: LiDAR e stereo in una nuvola sola, lo stereo solo dove il LiDAR e' rado."""
import numpy as np

MIN_RANGE = 30.0                                            # sotto basta il LiDAR, lo stereo vicino guasta i cluster
VOXEL = 0.2                                                 # un punto stereo per cubetto da 20 cm, densita' simile al LiDAR


def voxel_downsample(points, size):
    keys = np.floor(points / size).astype(np.int64)
    _, first = np.unique(keys, axis=0, return_index=True)  # il primo punto di ogni cubetto
    return points[np.sort(first)]

def merge_clouds(lidar, stereo, min_range=MIN_RANGE, voxel=VOXEL):
    far = stereo[np.hypot(stereo[:, 0], stereo[:, 1]) >= min_range]
    return np.vstack([lidar, voxel_downsample(far, voxel)])
