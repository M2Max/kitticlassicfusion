"""Matrice di confusione della detection: righe classe vera + "sfondo", colonne classe rilevata + "persa".

Diagonale = TP del matching per classe; un FP e un FN di classi diverse sovrapposti diventano "classe sbagliata".
"""
import csv

import matplotlib.pyplot as plt
import numpy as np

from config import CLASSES, IOU_THRESH
from evaluation.result_analysis import class_indices, greedy_match, iou_table, save_fig, score_frame, setup_style

ROWS = CLASSES + ["sfondo"]
COLS = CLASSES + ["persa"]


def frame_confusion(rec, iou_fn):
    m = np.zeros((len(ROWS), len(COLS)), dtype=int)
    left_det, left_gt = [], []                              # FP e FN del matching per classe: (indice, classe)
    for k, c in enumerate(CLASSES):
        det, valid, ignored = class_indices(rec, c)
        dets = rec["det_boxes"][det]
        f = {"dets": dets, "valid": rec["gt_boxes"][valid],
             "iou_v": iou_table(dets, rec["gt_boxes"][valid], iou_fn),
             "iou_i": iou_table(dets, rec["gt_boxes"][ignored], iou_fn)}
        pairs, fp, fn = score_frame(f, IOU_THRESH[c])
        m[k, k] += len(pairs)
        left_det += [(det[i], k) for i in fp]
        left_gt += [(valid[j], k) for j in fn]

    # Secondo giro: gli FP rimasti contro gli FN rimasti, ognuno con la soglia della classe vera
    ious = iou_table(rec["det_boxes"][[i for i, _ in left_det]], rec["gt_boxes"][[j for j, _ in left_gt]], iou_fn)
    thresh = np.array([IOU_THRESH[CLASSES[g]] for _, g in left_gt])
    ious[ious < thresh] = -1.0                              # sotto soglia: coppia non valida
    pairs = greedy_match(ious, 0.0)
    for a, b in pairs:
        m[left_gt[b][1], left_det[a][1]] += 1
    used_d, used_g = {a for a, _ in pairs}, {b for _, b in pairs}
    for a, (_, p) in enumerate(left_det):
        if a not in used_d:
            m[-1, p] += 1                                   # inventata: sfondo scambiato per oggetto
    for b, (_, g) in enumerate(left_gt):
        if b not in used_g:
            m[g, -1] += 1                                   # persa: nessuna det sopra
    return m

def confusion_matrix(records, iou_fn):
    return sum((frame_confusion(rec, iou_fn) for rec in records), np.zeros((len(ROWS), len(COLS)), dtype=int))

def save_confusion(m, out, title):
    with open(out / "confusion.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vera \\ rilevata"] + COLS)
        for name, row in zip(ROWS, m):
            w.writerow([name] + list(row))
    setup_style()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot_confusion(ax, m, title)
    save_fig(fig, out / "confusion.png")

def plot_confusion(ax, m, title):
    # Colore = quota della riga (dove finiscono gli oggetti di quella classe), numero = conteggio
    share = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    ax.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for r in range(m.shape[0]):
        for c in range(m.shape[1]):
            text = "-" if (r, c) == (m.shape[0] - 1, m.shape[1] - 1) else str(m[r, c])   # sfondo/persa: non esiste
            ax.text(c, r, text, ha="center", va="center", fontsize=10, color="white" if share[r, c] > 0.6 else "black")
    ax.set_xticks(range(len(COLS)), COLS, rotation=30)
    ax.set_yticks(range(len(ROWS)), ROWS)
    ax.set_xlabel("rilevata")
    ax.set_ylabel("vera")
    ax.grid(False)
    ax.set_title(title)
