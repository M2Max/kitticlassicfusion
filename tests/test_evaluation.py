"""Valutazione: le label confrontate con se stesse non danno ne' FP ne' FN. Serve il dataset (KITTI_ROOT)."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import CLASSES, DATA                                             # noqa: E402
from evaluation.iou import iou_3d                                            # noqa: E402
from evaluation.result_analysis import collect, summary_table                # noqa: E402
from kitti.data import load_calib                                            # noqa: E402
from kitti.ground_truth import gt_boxes                                      # noqa: E402

FRAMES = range(10)


@unittest.skipUnless((DATA / "label_2").is_dir(), f"dataset non trovato in {DATA} (imposta KITTI_ROOT)")
class TestLabelsAgainstThemselves(unittest.TestCase):
    def test_no_fp_no_fn(self):
        for i in FRAMES:
            with self.subTest(frame=i):
                calib = load_calib(i)
                gt_b, gt_c, gt_ign = gt_boxes(i, calib)
                rec = {"frame": i, "det_boxes": gt_b, "det_classes": gt_c,       # det = tutte le label
                       "gt_boxes": gt_b, "gt_classes": gt_c, "gt_ignored": gt_ign}
                for row in summary_table(collect([rec], iou_3d)):
                    n_valid = sum(1 for k, ign in zip(gt_c, gt_ign) if k == row["class"] and not ign)
                    self.assertEqual(row["FP"], 0, row)
                    self.assertEqual(row["FN"], 0, row)
                    self.assertEqual(row["TP"], n_valid, row)

    def test_frames_have_objects(self):
        # il test sopra deve avere almeno un TP per non passare a vuoto
        n = sum(1 for i in FRAMES for k, ign in zip(*gt_boxes(i, load_calib(i))[1:]) if k in CLASSES and not ign)
        self.assertGreater(n, 0)


if __name__ == "__main__":
    unittest.main()
