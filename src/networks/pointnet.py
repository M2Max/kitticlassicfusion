"""PointNet di L2: dai punti di un cluster, classe (sfondo, Car, Pedestrian, Cyclist) e correzione della scatola."""
import numpy as np
import torch
from torch import nn

from config import CLASSES, REPO

NET_CLASSES = ["sfondo"] + CLASSES                          # indice 0 = non e' un oggetto
N_POINTS = 128                                              # punti per cluster in ingresso alla rete
N_GLOBAL = 5                                                # l, w, h, distanza, numero di punti
MODEL_PATH = REPO / "data" / "models" / "pointnet_l2.pt"    # pesi + soglie, fuori da git


def local_points(pts, box):
    # Punti nel sistema della scatola: origine al centro del fondo, x lungo la lunghezza, z verso l'alto
    cx, cy, z_bottom, _, _, _, psi = box
    c, s = np.cos(psi), np.sin(psi)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    return np.stack([c * dx + s * dy, -s * dx + c * dy, pts[:, 2] - z_bottom], axis=1).astype(np.float32)

def sample_points(local, rng, n=N_POINTS):
    # Sempre n punti: estratti a caso, con ripetizione se il cluster ne ha meno
    idx = rng.choice(len(local), size=n, replace=len(local) < n)
    return local[idx]

def global_features(box, n_points):
    return np.array([box[3], box[4], box[5], np.hypot(box[0], box[1]) / 40, np.log1p(n_points) / 5], dtype=np.float32)

def wrap_half_pi(a):
    # Angolo in [-pi/2, pi/2): per una scatola l'orientamento conta a meno di 180 gradi
    return (a + np.pi / 2) % np.pi - np.pi / 2

def encode_box(prop, gt):
    # Scatola vera rispetto alla candidata: spostamento nel sistema della candidata, rapporti delle misure, rotazione
    if abs(wrap_half_pi(gt[6] - prop[6])) > np.pi / 4:      # candidata girata di ~90 gradi: stessa scatola vera
        gt = np.array([*gt[:3], gt[4], gt[3], gt[5], gt[6] + np.pi / 2])   # descritta con l e w scambiate
    c, s = np.cos(prop[6]), np.sin(prop[6])
    dx, dy = gt[0] - prop[0], gt[1] - prop[1]
    return np.array([c * dx + s * dy, -s * dx + c * dy, gt[2] - prop[2],
                     np.log(gt[3] / prop[3]), np.log(gt[4] / prop[4]), np.log(gt[5] / prop[5]),
                     wrap_half_pi(gt[6] - prop[6])], dtype=np.float32)

def decode_box(prop, reg):
    # Inverso di encode_box
    c, s = np.cos(prop[6]), np.sin(prop[6])
    box = np.array([prop[0] + c * reg[0] - s * reg[1], prop[1] + s * reg[0] + c * reg[1], prop[2] + reg[2],
                    prop[3] * np.exp(reg[3]), prop[4] * np.exp(reg[4]), prop[5] * np.exp(reg[5]), prop[6] + reg[6]])
    if box[4] > box[3]:                                     # convenzione del progetto: l >= w
        box[3], box[4], box[6] = box[4], box[3], box[6] + np.pi / 2
    return box


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.point_mlp = nn.Sequential(                     # stesso MLP su ogni punto (conv 1x1)
            nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 1), nn.BatchNorm1d(256), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(256 + N_GLOBAL, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU())
        self.cls = nn.Linear(128, len(NET_CLASSES))
        self.reg = nn.Linear(128, 7)

    def forward(self, points, glob):
        # points: (B, N, 3), glob: (B, N_GLOBAL) -> logits (B, 4), correzioni (B, 7)
        feat = self.point_mlp(points.transpose(1, 2)).max(dim=2).values   # max pooling: invariante all'ordine
        x = self.head(torch.cat([feat, glob], dim=1))
        return self.cls(x), self.reg(x)


def load_model(device="cpu"):
    saved = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model = PointNet().to(device)
    model.load_state_dict(saved["weights"])
    return model.eval(), saved["thresholds"]
