"""Matrice di confusione su record sintetici: ogni casella con un caso costruito a mano, e coerenza con summary_table."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CLASSES, SPLITS                                           # noqa: E402
from evaluation.confusion import COLS, ROWS, confusion_matrix                # noqa: E402
from evaluation.iou import iou_3d                                            # noqa: E402
from evaluation.result_analysis import collect, summary_table                # noqa: E402
from kitti.data import frame_ids                                             # noqa: E402

CAR = [10.0, 2.0, -1.7, 4.0, 1.8, 1.5, 0.0]                 # [cx, cy, z_bottom, l, w, h, psi]
PED = [8.0, -3.0, -1.7, 0.6, 0.6, 1.7, 0.0]
FAR = [40.0, 10.0, -1.7, 4.0, 1.8, 1.5, 0.0]                # lontano da tutto


def record(dets, gts, ignored=None):
    # dets e gts: liste di (scatola, classe)
    return {"frame": 0,
            "det_boxes": np.array([b for b, _ in dets]).reshape(-1, 7), "det_classes": [c for _, c in dets],
            "gt_boxes": np.array([b for b, _ in gts]).reshape(-1, 7), "gt_classes": [c for _, c in gts],
            "gt_ignored": np.array(ignored if ignored is not None else [False] * len(gts), dtype=bool)}

def cell(m, row, col):
    return m[ROWS.index(row), COLS.index(col)]


class TestConfusion(unittest.TestCase):
    def test_correct(self):
        m = confusion_matrix([record([(CAR, "Car"), (PED, "Pedestrian")], [(CAR, "Car"), (PED, "Pedestrian")])], iou_3d)
        self.assertEqual(cell(m, "Car", "Car"), 1)
        self.assertEqual(cell(m, "Pedestrian", "Pedestrian"), 1)
        self.assertEqual(m.sum(), 2)

    def test_wrong_class(self):
        # Posto giusto, classe sbagliata: una sola casella, non un FP + un FN
        m = confusion_matrix([record([(CAR, "Cyclist")], [(CAR, "Car")])], iou_3d)
        self.assertEqual(cell(m, "Car", "Cyclist"), 1)
        self.assertEqual(m.sum(), 1)

    def test_missed_and_invented(self):
        m = confusion_matrix([record([(FAR, "Car")], [(CAR, "Car")])], iou_3d)
        self.assertEqual(cell(m, "Car", "persa"), 1)
        self.assertEqual(cell(m, "sfondo", "Car"), 1)
        self.assertEqual(m.sum(), 2)

    def test_ignored_gt_not_counted(self):
        # Det sopra una GT fuori Moderate, e sopra un Van (classe vicina di Car): nessun errore
        rec = record([(CAR, "Car"), (FAR, "Car")], [(CAR, "Car"), (FAR, "Van")], ignored=[True, False])
        self.assertEqual(confusion_matrix([rec], iou_3d).sum(), 0)

    def test_empty_frame(self):
        self.assertEqual(confusion_matrix([record([], [])], iou_3d).sum(), 0)

    def test_consistent_with_summary(self):
        recs = [record([(CAR, "Car"), (PED, "Cyclist"), (FAR, "Pedestrian")], [(CAR, "Car"), (PED, "Pedestrian")]),
                record([(FAR, "Car")], [(CAR, "Car"), (PED, "Cyclist")])]
        m = confusion_matrix(recs, iou_3d)
        for k, row in enumerate(summary_table(collect(recs, iou_3d))):
            self.assertEqual(m[k, k], row["TP"], row)
            self.assertEqual(m[k].sum() - m[k, k], row["FN"], row)             # riga senza diagonale
            self.assertEqual(m[:, k].sum() - m[k, k], row["FP"], row)          # colonna senza diagonale


@unittest.skipUnless((SPLITS / "val.txt").is_file(), f"split non trovati in {SPLITS} (imposta KITTI_ROOT)")
class TestSplits(unittest.TestCase):
    def test_train_val_disjoint(self):
        train, val = frame_ids("train"), frame_ids("val")
        self.assertEqual((len(train), len(val)), (3712, 3769))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(sorted(train + val), frame_ids("trainval"))


if __name__ == "__main__":
    unittest.main()
