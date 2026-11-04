"""Fusione late su liste di detection sintetiche: associazione e conferma."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fusion.late import associate, confirmed                                # noqa: E402

CAR = [10.0, 2.0, -1.7, 3.9, 1.6, 1.5, 0.0]                 # [cx, cy, z_bottom, l, w, h, psi]
CAR_CAM = [10.8, 2.3, -1.6, 3.9, 1.6, 1.5, 0.0]             # stessa auto, vista male dallo stereo
PED = [20.0, -5.0, -1.7, 0.8, 0.6, 1.7, 0.0]


def dets(*items):
    # items: (scatola, classe)
    return np.array([b for b, _ in items]).reshape(-1, 7), [c for _, c in items], np.ones(len(items))


class TestAssociate(unittest.TestCase):
    def test_near_pair_only(self):
        lb, lc, _ = dets((CAR, "Car"), (PED, "Pedestrian"))
        self.assertEqual(associate(lb, lc, dets((CAR_CAM, "Car"))[0]), [(0, 0)])

    def test_gate(self):
        far = [CAR[0] + 3.0, *CAR[1:]]                      # 3 m: oltre il gate delle auto (2 m)
        lb, lc, _ = dets((CAR, "Car"))
        self.assertEqual(associate(lb, lc, dets((far, "Car"))[0]), [])

    def test_one_to_one(self):
        # Due detection camera vicine alla stessa auto: abbinata solo la piu' vicina
        lb, lc, _ = dets((CAR, "Car"))
        cb, _, _ = dets((CAR_CAM, "Car"), ([10.1, 2.0, -1.7, 3.9, 1.6, 1.5, 0.0], "Car"))
        self.assertEqual(associate(lb, lc, cb), [(0, 1)])

    def test_empty(self):
        self.assertEqual(associate(np.zeros((0, 7)), [], dets((CAR, "Car"))[0]), [])


class TestConfirmed(unittest.TestCase):
    def test_keeps_lidar_box_and_class(self):
        boxes, classes, scores = confirmed(dets((CAR, "Car")), dets((CAR_CAM, "Cyclist")))
        np.testing.assert_allclose(boxes, [CAR])
        self.assertEqual(classes, ["Car"])
        self.assertEqual(list(scores), [1.0])

    def test_single_sensor_dropped(self):
        # L'auto la vede solo il LiDAR, il pedone solo la camera: nessuna delle due resta
        boxes, classes, _ = confirmed(dets((CAR, "Car")), dets((PED, "Pedestrian")))
        self.assertEqual((boxes.shape, classes), ((0, 7), []))

    def test_both_empty(self):
        boxes, classes, scores = confirmed(dets(), dets())
        self.assertEqual((boxes.shape, classes, len(scores)), ((0, 7), [], 0))


if __name__ == "__main__":
    unittest.main()
