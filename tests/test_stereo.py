"""Stereo con dati sintetici di cui si conosce la risposta."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA                                                      # noqa: E402
from kitti.data import load_calib                                            # noqa: E402
from stereo.disparity import compute_disparity                               # noqa: E402
from stereo.pseudo_lidar import depth_to_points, disparity_to_depth, stereo_params   # noqa: E402

F, CU, CV = 700.0, 5.5, 3.5                                 # focale e centro ottico (immagine piccola 8 x 12)
T2 = np.array([0.06, 0.002, 0.0])                           # camera 2 rispetto alla camera 0 rettificata (m)
BASELINE = 0.54                                             # camera 3 = camera 2 spostata di 0.54 m a destra
K = np.array([[F, 0, CU], [0, F, CV], [0, 0, 1]])


def synthetic_calib():
    P2 = K @ np.c_[np.eye(3), T2]                           # P = K [I | t]
    P3 = K @ np.c_[np.eye(3), T2 - [BASELINE, 0, 0]]
    return {"P2": P2.ravel(), "P3": P3.ravel()}


class TestStereoParams(unittest.TestCase):
    def test_synthetic(self):
        f, cu, cv, b = stereo_params(synthetic_calib())
        np.testing.assert_allclose([f, cu, cv, b], [F, CU, CV, BASELINE])

    @unittest.skipUnless((DATA / "calib").is_dir(), "dataset non trovato")
    def test_real_kitti(self):
        f, cu, cv, b = stereo_params(load_calib(0))
        self.assertTrue(700 < f < 730, f)
        self.assertTrue(0.5 < b < 0.6, b)                    # KITTI: circa 54 cm fra le due camere a colori


class TestDisparityToDepth(unittest.TestCase):
    def test_values(self):
        disp = np.array([[10.0, 37.8, 0.0], [-1.0, 1.0, 378.0]])
        depth = disparity_to_depth(disp, F, BASELINE)
        self.assertEqual(depth.shape, disp.shape)
        ok = disp > 0
        np.testing.assert_allclose(depth[ok] * disp[ok], F * BASELINE)       # Z * d = f * B
        self.assertTrue(np.isnan(depth[~ok]).all())                          # disparita' non valida -> nan


class TestDepthToPoints(unittest.TestCase):
    def setUp(self):
        self.depth = np.full((8, 12), np.nan)
        self.pixels = [(2, 1, 10.0), (9, 6, 20.0), (5, 3, 33.3)]             # (u, v, Z) validi
        for u, v, z in self.pixels:
            self.depth[v, u] = z
        self.depth[0, 0] = 70.0                                              # oltre max_depth: scartato

    def test_reprojects_to_own_pixel(self):
        # Ogni punto, riproiettato con P2, deve ricadere sul suo pixel alla sua profondita'
        pts = depth_to_points(self.depth, synthetic_calib(), max_depth=64.0)
        self.assertEqual(pts.shape, (3, 3))
        pts = pts[np.argsort(pts[:, 2])]                                     # ordine per profondita'
        P2 = synthetic_calib()["P2"].reshape(3, 4)
        for p, (u, v, z) in zip(pts, sorted(self.pixels, key=lambda t: t[2])):
            uvw = P2 @ np.append(p, 1.0)
            np.testing.assert_allclose(uvw[:2] / uvw[2], [u, v], atol=1e-9)
            self.assertAlmostEqual(uvw[2], z)                                # profondita' della camera 2


class TestComputeDisparity(unittest.TestCase):
    def test_shifted_texture(self):
        # Texture casuale; la destra e' la sinistra spostata di D pixel: disparita' attesa D ovunque
        D = 12
        rng = np.random.default_rng(0)
        left = cv2.GaussianBlur(rng.integers(0, 256, (120, 320, 3), dtype=np.uint8), (3, 3), 0)
        right = np.roll(left, -D, axis=1)                    # il pixel u della sinistra sta in u - D a destra
        disp = compute_disparity(left, right)
        self.assertEqual(disp.shape, left.shape[:2])
        self.assertEqual(disp.dtype, np.float32)
        center = disp[20:100, 150:280]                      # lontano dai bordi
        valid = center > 0
        self.assertGreater(valid.mean(), 0.8, "troppi pixel senza disparita'")
        self.assertAlmostEqual(float(np.median(center[valid])), D, delta=0.5)


if __name__ == "__main__":
    unittest.main()
