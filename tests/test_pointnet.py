"""PointNet di L2: codifica delle scatole e proprieta' della rete, senza dataset."""
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from networks.pointnet import N_GLOBAL, N_POINTS, PointNet, decode_box, encode_box, local_points   # noqa: E402

PROP = np.array([12.0, -3.0, -1.6, 3.0, 1.5, 1.4, 0.4])    # candidata: [cx, cy, z_bottom, l, w, h, psi]
GT = np.array([12.5, -2.6, -1.7, 4.1, 1.7, 1.5, 0.5])


class TestBoxCoding(unittest.TestCase):
    def test_round_trip(self):
        np.testing.assert_allclose(decode_box(PROP, encode_box(PROP, GT)), GT, atol=1e-6)

    def test_identity(self):
        np.testing.assert_allclose(encode_box(PROP, PROP), np.zeros(7), atol=1e-7)

    def test_local_points(self):
        # Il centro del fondo e' l'origine; un punto davanti di l/2 lungo la lunghezza sta su x = l/2
        tip = PROP[:3] + [1.5 * np.cos(PROP[6]), 1.5 * np.sin(PROP[6]), 1.0]
        np.testing.assert_allclose(local_points(np.array([PROP[:3], tip]), PROP), [[0, 0, 0], [1.5, 0, 1.0]], atol=1e-6)


class TestNetwork(unittest.TestCase):
    def test_shapes_and_order_invariance(self):
        torch.manual_seed(0)
        model = PointNet().eval()
        pts = torch.randn(4, N_POINTS, 3)
        glob = torch.randn(4, N_GLOBAL)
        logits, reg = model(pts, glob)
        self.assertEqual(tuple(logits.shape), (4, 4))
        self.assertEqual(tuple(reg.shape), (4, 7))
        shuffled = pts[:, torch.randperm(N_POINTS)]          # stesso insieme di punti, altro ordine
        torch.testing.assert_close(model(shuffled, glob)[0], logits)


if __name__ == "__main__":
    unittest.main()
