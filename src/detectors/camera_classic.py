"""C1: pipeline classica sulla nuvola ricavata dalla coppia stereo."""
from kitti.data import load_calib, load_stereo_pair
from kitti.transforms import rectcam_to_lidar
from pipeline.classic import detect_points
from stereo.disparity import compute_disparity
from stereo.pseudo_lidar import depth_to_points, disparity_to_depth, remove_flying_pixels, stereo_params

# Scelti su 150 frame di train: la nuvola stereo e' densa e rumorosa
MAX_DEPTH = 40.0                                            # oltre, l'errore di profondita' supera il metro
MAX_JUMP = 0.05                                             # pixel volanti: salto di profondita' > 5% nel 3 x 3
MIN_HEIGHT, MAX_HEIGHT = 0.3, 2.5                           # via suolo spesso e chiome
MIN_CELL_POINTS = 25                                        # casella BEV occupata solo con abbastanza pixel dentro


def stereo_depth(i, calib):
    left, right = load_stereo_pair(i)
    f, _, _, baseline = stereo_params(calib)
    return remove_flying_pixels(disparity_to_depth(compute_disparity(left, right), f, baseline), MAX_JUMP)

def stereo_cloud(i, calib):
    return rectcam_to_lidar(calib, depth_to_points(stereo_depth(i, calib), calib, MAX_DEPTH))

def cloud(i):
    return stereo_cloud(i, load_calib(i))

def detect(i, calib):
    return detect_points(stereo_cloud(i, calib), calib, MIN_HEIGHT, MAX_HEIGHT, MIN_CELL_POINTS)
