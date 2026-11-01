"""Lettura delle immagini stereo. Serve il dataset con image_2 e image_3 (scripts/download_images.sh)."""
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA                                                      # noqa: E402
from kitti.data import load_image, load_stereo_pair                          # noqa: E402

FRAMES = [0, 1, 7480]                                       # primo, uno qualsiasi, ultimo


@unittest.skipUnless((DATA / "image_3").is_dir(), f"immagini non trovate in {DATA} (scripts/download_images.sh)")
class TestImages(unittest.TestCase):
    def test_stereo_pair_same_shape(self):
        for i in FRAMES:
            with self.subTest(frame=i):
                left, right = load_stereo_pair(i)
                self.assertEqual(left.dtype, np.uint8)
                self.assertEqual(left.ndim, 3)
                self.assertEqual(left.shape[2], 3)
                self.assertEqual(left.shape, right.shape)

    def test_rgb_order(self):
        # load_image deve girare i canali rispetto a OpenCV (BGR)
        bgr = cv2.imread(str(DATA / "image_2" / "000000.png"))
        np.testing.assert_array_equal(load_image(0), bgr[:, :, ::-1])

    def test_missing_frame_raises(self):
        with self.assertRaises(FileNotFoundError):
            load_image(99999)


if __name__ == "__main__":
    unittest.main()
