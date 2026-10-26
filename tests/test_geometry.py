"""Trasformazioni LiDAR <-> camera, piano RANSAC e box fitting con risultati noti."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from kitti.transforms import lidar_to_rectcam, rectcam_to_lidar               # noqa: E402
from pipeline.boxes import box_bev                                           # noqa: E402
from pipeline.floor import plane_from_points                                 # noqa: E402


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def calib(R, t, R0):
    return {"Tr_velo_to_cam": np.hstack([R, np.array(t)[:, None]]).ravel(), "R0_rect": R0.ravel()}


class TestTransforms(unittest.TestCase):
    def test_pure_translation(self):
        c = calib(np.eye(3), [1.0, 2.0, 3.0], np.eye(3))
        np.testing.assert_allclose(lidar_to_rectcam(c, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
                                   [[1, 2, 3], [2, 3, 4]])

    def test_kitti_axes(self):
        # LiDAR x avanti, y sinistra, z su  ->  camera x destra, y giu', z avanti
        R = np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]])
        c = calib(R, [0, 0, 0], np.eye(3))
        np.testing.assert_allclose(lidar_to_rectcam(c, np.array([[10.0, 2.0, 1.0]])), [[-2, -1, 10]])

    def test_round_trip(self):
        rng = np.random.default_rng(0)
        c = calib(rot_z(0.3) @ np.array([[0.0, -1, 0], [0, 0, -1], [1, 0, 0]]), [0.27, -0.08, -0.06], rot_z(0.01))
        pts = rng.uniform(-50, 50, size=(100, 3))
        np.testing.assert_allclose(rectcam_to_lidar(c, lidar_to_rectcam(c, pts)), pts, atol=1e-9)


class TestFloor(unittest.TestCase):
    def test_horizontal_plane(self):
        # tre punti a z = -1.7: normale (0, 0, 1) verso l'alto, d = 1.7
        n, d = plane_from_points(np.array([[0.0, 0, -1.7], [0, 1, -1.7], [1, 0, -1.7]]))
        np.testing.assert_allclose(n, [0, 0, 1])
        self.assertAlmostEqual(d, 1.7)

    def test_collinear(self):
        self.assertIsNone(plane_from_points(np.array([[0.0, 0, 0], [1, 1, 1], [2, 2, 2]])))


class TestBoxBev(unittest.TestCase):
    def test_rotated_rectangle(self):
        # punti sul bordo di un rettangolo 4 x 2 centrato in (5, 3) e girato di 30 gradi
        psi = np.radians(30)
        s = np.linspace(-1, 1, 21)
        local = np.vstack([np.c_[2 * s, np.full_like(s, 1)], np.c_[2 * s, np.full_like(s, -1)],
                           np.c_[np.full_like(s, 2), s], np.c_[np.full_like(s, -2), s]])
        u = np.array([np.cos(psi), np.sin(psi)])
        v = np.array([-np.sin(psi), np.cos(psi)])
        xy = np.array([5.0, 3.0]) + local[:, :1] * u + local[:, 1:] * v
        cx, cy, l, w, got = box_bev(xy)
        np.testing.assert_allclose([cx, cy, l, w], [5, 3, 4, 2], atol=1e-9)
        self.assertAlmostEqual(np.mod(got - psi, np.pi), 0.0)


if __name__ == "__main__":
    unittest.main()
