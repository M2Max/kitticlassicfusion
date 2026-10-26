"""IoU BEV e 3D su scatole verticali con risultati calcolati a mano (auto 4 x 2 x 1.5 m)."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from evaluation.iou import iou_3d, iou_bev                                    # noqa: E402

CAR = np.array([10.0, 2.0, -1.7, 4.0, 2.0, 1.5, 0.0])   # [cx, cy, z_bottom, l, w, h, psi]


def moved(box, **kw):
    b = box.copy()
    for k, v in kw.items():
        b["cx cy z l w h psi".split().index(k)] = v
    return b


class TestIoU(unittest.TestCase):
    def test_identical(self):
        self.assertAlmostEqual(iou_3d(CAR, CAR), 1.0)
        self.assertAlmostEqual(iou_bev(CAR, CAR), 1.0)

    def test_shift_1m_along_length(self):
        # intersezione 3 x 2 x 1.5 = 9, unione 12 + 12 - 9 = 15
        self.assertAlmostEqual(iou_3d(CAR, moved(CAR, cx=11.0)), 0.6)

    def test_shift_1m_along_length_rotated(self):
        # stesso caso con la scatola girata di 30 gradi: lo spostamento segue la lunghezza
        psi = np.radians(30)
        a = moved(CAR, psi=psi)
        b = moved(a, cx=a[0] + np.cos(psi), cy=a[1] + np.sin(psi))
        self.assertAlmostEqual(iou_3d(a, b), 0.6)

    def test_3m_box_aligned_at_one_end(self):
        # da x = 8..12 contro x = 9..12: intersezione 9, unione 12 + 9 - 9 = 12
        self.assertAlmostEqual(iou_3d(CAR, moved(CAR, cx=10.5, l=3.0)), 0.75)

    def test_rotated_90_about_center(self):
        # dall'alto: intersezione 2 x 2 = 4, unione 8 + 8 - 4 = 12
        rot = moved(CAR, psi=np.pi / 2)
        self.assertAlmostEqual(iou_bev(CAR, rot), 1 / 3)
        self.assertAlmostEqual(iou_3d(CAR, rot), 1 / 3)            # stessa altezza: anche in 3D

    def test_no_overlap(self):
        self.assertEqual(iou_3d(CAR, moved(CAR, cx=20.0)), 0.0)
        self.assertEqual(iou_3d(CAR, moved(CAR, z=0.0)), 0.0)       # una sopra l'altra


if __name__ == "__main__":
    unittest.main()
