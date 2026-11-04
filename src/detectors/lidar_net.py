"""L2: i cluster della pipeline classica passano da una PointNet che decide la classe e corregge la scatola."""
import numpy as np
import torch

from kitti.data import load_lidar
from networks.pointnet import NET_CLASSES, decode_box, global_features, load_model, local_points, sample_points
from pipeline.classic import proposals

MODEL = None                                                # caricato una volta per processo


def cloud(i):
    return load_lidar(i)[:, :3]

def predict(boxes, clusters, model):
    # Probabilita' per classe (M, 4) e correzioni (M, 7) di ogni scatola candidata
    rng = np.random.default_rng(0)                          # stessi punti estratti a ogni run
    points = np.stack([sample_points(local_points(p, b), rng) for b, p in zip(boxes, clusters)])
    glob = np.stack([global_features(b, len(p)) for b, p in zip(boxes, clusters)])
    with torch.no_grad():
        logits, reg = model(torch.from_numpy(points), torch.from_numpy(glob))
    return torch.softmax(logits, dim=1).numpy(), reg.numpy()

def get_model():
    global MODEL
    if MODEL is None:
        torch.set_num_threads(1)                            # run.py lancia gia' un processo per core
        MODEL = load_model()
    return MODEL

def detect(i, calib):
    model, thresholds = get_model()
    boxes, clusters, _, _ = proposals(cloud(i), calib)
    if len(boxes) == 0:
        return np.zeros((0, 7)), [], np.zeros(0)
    probs, reg = predict(boxes, clusters, model)
    best = 1 + np.argmax(probs[:, 1:], axis=1)              # classe di oggetto piu' probabile (0 = sfondo)
    scores = probs[np.arange(len(best)), best]
    keep = [k for k in range(len(best)) if scores[k] >= thresholds[NET_CLASSES[best[k]]]]
    det_boxes = np.array([decode_box(boxes[k], reg[k]) for k in keep]).reshape(-1, 7)
    return det_boxes, [NET_CLASSES[best[k]] for k in keep], scores[keep]
