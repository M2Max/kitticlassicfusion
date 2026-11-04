import cv2
import numpy as np

MIN_DISPARITY=1
NUM_DISPARITIES=128 # Accetto di perdermi qualcosa sotto i 3 metri di distanza dove probabilmente non c'è granchè
BLOCK_SIZE=7
P1_VALUE= 8 * BLOCK_SIZE ** 2
P2_VALUE= 32 * BLOCK_SIZE ** 2
UNIQUENESS_RATIO=10
SPECKLE_WINDOW_SIZE=100
SPECKLE_RANGE=1
MAX_DIFF=1
MODE=cv2.STEREO_SGBM_MODE_SGBM_3WAY

MATCHER = cv2.StereoSGBM.create(minDisparity=MIN_DISPARITY, 
                        numDisparities=NUM_DISPARITIES, 
                        blockSize=BLOCK_SIZE, 
                        P1=P1_VALUE, 
                        P2=P2_VALUE,
                        uniquenessRatio=UNIQUENESS_RATIO,
                        speckleWindowSize=SPECKLE_WINDOW_SIZE,
                        speckleRange=SPECKLE_RANGE,
                        disp12MaxDiff=MAX_DIFF,
                        mode=MODE)

def compute_disparity(left, right):
    left_gray = cv2.cvtColor(left, cv2.COLOR_RGB2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)

    disparity = MATCHER.compute(left_gray, right_gray).astype(np.float32) / 16

    return disparity