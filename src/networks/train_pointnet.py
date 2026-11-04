"""Addestra la PointNet di L2 sui cluster dello split train e sceglie le soglie sugli ultimi frame di train.

Uso: .venv/bin/python src/networks/train_pointnet.py      (~20 min i cluster, ~5 min l'addestramento su M4)
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import CLASSES, RESULTS                         # noqa: E402
from evaluation.iou import iou_3d                           # noqa: E402
from evaluation.result_analysis import collect, summary_table   # noqa: E402
from kitti.data import frame_ids, load_calib, load_lidar    # noqa: E402
from kitti.ground_truth import gt_boxes                     # noqa: E402
from networks.pointnet import (MODEL_PATH, N_POINTS, NET_CLASSES, PointNet, decode_box,  # noqa: E402
                               encode_box, global_features, local_points)
from pipeline.classic import proposals                      # noqa: E402

HOLDOUT = 300                                               # ultimi frame di train: solo per scegliere le soglie
MARGIN = 0.3                                                # centro del cluster dentro la scatola vera allargata di tanto
MAX_STORED = 256                                            # punti salvati per candidata (a ogni epoca se ne estraggono N_POINTS)
CACHE = RESULTS / "cache" / "pointnet_train.npz"
THRESHOLDS = np.r_[np.round(np.arange(0.05, 0.96, 0.05), 2), 0.97, 0.98, 0.99]   # le migliori stanno spesso in alto


def assign(box, gt_b):
    # Indice della scatola vera che contiene il centro della candidata (la piu' vicina), oppure None
    best, best_dist = None, np.inf
    for j, g in enumerate(gt_b):
        a, b, _ = local_points(np.array([[box[0], box[1], g[2]]]), g)[0]
        if abs(a) <= g[3] / 2 + MARGIN and abs(b) <= g[4] / 2 + MARGIN and np.hypot(a, b) < best_dist:
            best, best_dist = j, np.hypot(a, b)
    return best

def frame_samples(i):
    calib = load_calib(i)
    boxes, clusters, _, _ = proposals(load_lidar(i)[:, :3], calib)
    gt_b, gt_c, _ = gt_boxes(i, calib)
    rng = np.random.default_rng(i)
    samples = []
    for box, pts in zip(boxes, clusters):
        j = assign(box, gt_b)
        if j is None:
            label, target = 0, np.zeros(7, dtype=np.float32)
        elif gt_c[j] in CLASSES:
            label, target = NET_CLASSES.index(gt_c[j]), encode_box(box, gt_b[j])
        else:                                               # Van, Truck, ...: ne' oggetto ne' sfondo
            continue
        local = local_points(pts, box)
        stored = local[rng.choice(len(local), size=min(len(local), MAX_STORED), replace=False)]
        padded = np.zeros((MAX_STORED, 3), dtype=np.float16)
        padded[:len(stored)] = stored
        samples.append((padded, len(stored), global_features(box, len(pts)), label, target))
    return samples

def build_dataset(frames, rebuild):
    if CACHE.exists() and not rebuild:
        print(f"Candidate gia' pronte in {CACHE}")
        return dict(np.load(CACHE))
    t0 = time.perf_counter()
    with ProcessPoolExecutor() as pool:
        samples = [s for frame in pool.map(frame_samples, frames, chunksize=8) for s in frame]
    data = {"points": np.stack([s[0] for s in samples]), "counts": np.array([s[1] for s in samples]),
            "glob": np.stack([s[2] for s in samples]), "labels": np.array([s[3] for s in samples]),
            "targets": np.stack([s[4] for s in samples])}
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **data)
    print(f"{len(samples)} candidate da {len(frames)} frame in {time.perf_counter() - t0:.0f} s")
    return data

def train(data, epochs, device, batch=512):
    points = torch.from_numpy(data["points"]).to(device)
    counts = torch.from_numpy(data["counts"]).to(device)
    glob = torch.from_numpy(data["glob"]).to(device)
    labels = torch.from_numpy(data["labels"]).long().to(device)
    targets = torch.from_numpy(data["targets"]).to(device)
    freq = np.bincount(data["labels"], minlength=len(NET_CLASSES))
    print("candidate per classe:", dict(zip(NET_CLASSES, freq.tolist())))
    weights = torch.tensor((freq.sum() / np.maximum(freq, 1)) ** 0.5, dtype=torch.float32)   # classi rare pesano di piu'
    ce = nn.CrossEntropyLoss(weight=(weights / weights.mean()).to(device))

    model = PointNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = epochs * ((len(labels) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(labels), device=device)
        total = 0.0
        for k in range(0, len(perm), batch):
            idx = perm[k:k + batch]
            pick = (torch.rand(len(idx), N_POINTS, device=device) * counts[idx, None]).long()   # N_POINTS fra quelli salvati
            pts = points[idx[:, None], pick].float()
            logits, reg = model(pts, glob[idx])
            pos = labels[idx] > 0
            loss = ce(logits, labels[idx])
            if pos.any():
                loss = loss + 2.0 * nn.functional.smooth_l1_loss(reg[pos], targets[idx][pos])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        print(f"  epoca {epoch + 1}/{epochs}: loss {total / len(labels):.4f}", flush=True)
    return model.cpu().eval()

def holdout_frame(i):
    calib = load_calib(i)
    boxes, clusters, _, _ = proposals(load_lidar(i)[:, :3], calib)
    gt_b, gt_c, gt_ign = gt_boxes(i, calib)
    return {"frame": i, "boxes": boxes, "clusters": clusters, "gt_boxes": gt_b, "gt_classes": gt_c, "gt_ignored": gt_ign}

def choose_thresholds(model, frames):
    # Tutte le detection con la loro confidenza, poi per ogni classe la soglia con F1 migliore
    from detectors.lidar_net import predict
    with ProcessPoolExecutor() as pool:
        frames_data = list(pool.map(holdout_frame, frames, chunksize=8))
    candidates = []
    for f in frames_data:
        dets = []
        if len(f["boxes"]):
            probs, reg = predict(f["boxes"], f["clusters"], model)
            for box, p, r in zip(f["boxes"], probs, reg):
                k = 1 + int(np.argmax(p[1:]))
                dets.append((decode_box(box, r), NET_CLASSES[k], p[k]))
        candidates.append((f, dets))

    thresholds = {}
    for c in CLASSES:
        best = (-1.0, 0.5)
        for t in THRESHOLDS:
            records = []
            for f, dets in candidates:
                keep = [d for d in dets if d[1] == c and d[2] >= t]
                records.append({"frame": f["frame"], "det_boxes": np.array([d[0] for d in keep]).reshape(-1, 7),
                                "det_classes": [c] * len(keep), "gt_boxes": f["gt_boxes"],
                                "gt_classes": f["gt_classes"], "gt_ignored": f["gt_ignored"]})
            f1 = next(r["F1"] for r in summary_table(collect(records, iou_3d)) if r["class"] == c)
            if f1 > best[0]:
                best = (f1, float(t))
        thresholds[c] = best[1]
        print(f"  {c}: soglia {best[1]:.2f} -> F1 {best[0]:.3f} sui {len(frames)} frame di controllo")
    return thresholds

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--thresholds-only", action="store_true", help="non riaddestra: ricalcola solo le soglie")
    ap.add_argument("--rebuild", action="store_true", help="ricalcola i cluster anche se sono gia' in cache")
    args = ap.parse_args()
    frames = frame_ids("train")
    if args.thresholds_only:
        from networks.pointnet import load_model
        model, _ = load_model()
        torch.save({"weights": model.state_dict(), "thresholds": choose_thresholds(model, frames[-HOLDOUT:])}, MODEL_PATH)
        print(f"Soglie aggiornate in {MODEL_PATH}")
        return
    data = build_dataset(frames[:-HOLDOUT], args.rebuild)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    model = train(data, args.epochs, device)
    print(f"Addestramento: {time.perf_counter() - t0:.0f} s su {device}")
    thresholds = choose_thresholds(model, frames[-HOLDOUT:])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"weights": model.state_dict(), "thresholds": thresholds}, MODEL_PATH)
    print(f"Modello salvato in {MODEL_PATH}")


if __name__ == "__main__":
    main()
